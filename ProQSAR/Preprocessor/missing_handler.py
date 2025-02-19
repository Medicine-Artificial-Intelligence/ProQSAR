import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.exceptions import NotFittedError
from typing import Tuple, Optional
import pickle
import os
import logging


class MissingHandler:
    """
    A class to handle missing data by performing imputation and removing columns with
    missing values exceeding a specified threshold.

    Attributes:
        id_col (Optional[str]): Column name for identifiers.
        activity_col (Optional[str]): Column name for activity values.
        missing_thresh (float): Threshold for missing values to drop columns.
        imputation_strategy (str): Imputation strategy for missing values.
        n_neighbors (int): Number of neighbors for KNN imputation.
        save_method (bool): Whether to save the imputation model.
        save_dir (Optional[str]): Directory to save the imputation model and transformed data.
        save_trans_data (bool): Whether to save the transformed data after imputation.
        trans_data_name (str): Name of the transformed data file.
    """

    def __init__(
        self,
        id_col: Optional[str] = None,
        activity_col: Optional[str] = None,
        missing_thresh: float = 40.0,
        imputation_strategy: str = "mean",
        n_neighbors: int = 5,
        save_method: bool = False,
        save_dir: Optional[str] = "Project/MissingHandler",
        save_trans_data: bool = False,
        trans_data_name: str = "mh_trans_data",
    ):
        """
        Initializes the MissingHandler object with the given parameters.
        """
        self.id_col = id_col
        self.activity_col = activity_col
        self.missing_thresh = missing_thresh
        self.imputation_strategy = imputation_strategy
        self.n_neighbors = n_neighbors
        self.save_method = save_method
        self.save_dir = save_dir
        self.save_trans_data = save_trans_data
        self.trans_data_name = trans_data_name
        self.binary_imputer = None
        self.non_binary_imputer = None

    @staticmethod
    def calculate_missing_percent(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the percentage of missing values for each column in the DataFrame.

        Parameters:
        data (pd.DataFrame): The DataFrame to analyze.

        Returns:
        pd.DataFrame: A DataFrame containing the percentage of missing values for each column.
        """
        missing_percent = (data.isnull().sum() / len(data)) * 100
        return pd.DataFrame({"MissingPercent": missing_percent}).reset_index(
            names="ColumnName"
        )

    @staticmethod
    def _get_imputer(
        data_to_impute: pd.DataFrame,
        imputation_strategy: str = "mean",
        n_neighbors: int = 5,
    ) -> Tuple[SimpleImputer, Optional[SimpleImputer]]:
        """
        Fits and optionally saves imputation models for binary and non-binary columns.

        Parameters:
        data_to_impute (pd.DataFrame): Data for which imputers are to be created.
        imputation_strategy (str): The strategy for imputing non-binary columns.
        n_neighbors (int): The number of neighbors to consider for KNN imputation.

        Returns:
        Tuple[SimpleImputer, Optional[SimpleImputer]]: The fitted binary and non-binary imputers.
        """
        binary_cols = [
            col
            for col in data_to_impute.columns
            if data_to_impute[col].dropna().isin([0, 1]).all()
        ]
        data_binary = data_to_impute[binary_cols]
        data_non_binary = data_to_impute.drop(columns=binary_cols, errors="ignore")

        # Fit imputation transformer for binary columns
        binary_imputer = None
        if binary_cols:
            binary_imputer = SimpleImputer(strategy="most_frequent").fit(data_binary)

        # Fit imputation transformer for non-binary columns
        imputer_dict = {
            "mean": SimpleImputer(strategy="mean"),
            "median": SimpleImputer(strategy="median"),
            "mode": SimpleImputer(strategy="most_frequent"),
            "knn": KNNImputer(n_neighbors=n_neighbors),
            "mice": IterativeImputer(estimator=BayesianRidge(), random_state=42),
        }

        non_binary_imputer = None
        if not data_non_binary.empty:
            if imputation_strategy in imputer_dict:
                non_binary_imputer = imputer_dict[imputation_strategy].fit(
                    data_non_binary
                )
            else:
                raise ValueError(
                    f"Unsupported imputation strategy {imputation_strategy}. Choose from"
                    + "'mean', 'median', 'mode', 'knn', or 'mice'."
                )

        return binary_imputer, non_binary_imputer

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fits the imputation models to the data and optionally saves the configuration and models.

        Parameters:
        data (pd.DataFrame): The data on which to fit the imputation models.

        Returns:
        Tuple[SimpleImputer, Optional[SimpleImputer]]: The fitted binary and non-binary imputers.
        """
        try:
            data_to_impute = data.drop(
                columns=[self.id_col, self.activity_col], errors="ignore"
            )

            missing_percent_df = self.calculate_missing_percent(data_to_impute)
            self.drop_cols = missing_percent_df[
                missing_percent_df["MissingPercent"] > self.missing_thresh
            ]["ColumnName"].tolist()
            data_to_impute.drop(columns=self.drop_cols, inplace=True)

            self.binary_cols = [
                col
                for col in data_to_impute.columns
                if data_to_impute[col].dropna().isin([0, 1]).all()
            ]

            self.binary_imputer, self.non_binary_imputer = self._get_imputer(
                data_to_impute,
                imputation_strategy=self.imputation_strategy,
                n_neighbors=self.n_neighbors,
            )

            if self.save_method:
                if self.save_dir and not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir, exist_ok=True)
                with open(f"{self.save_dir}/missing_handler.pkl", "wb") as file:
                    pickle.dump(self, file)
                logging.info(
                    f"MissingHandler method saved at: {self.save_dir}/missing_handler.pkl"
                )

        except Exception as e:
            logging.error(f"Error in fitting LowVarianceHandler: {e}")
            raise

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in the input DataFrame using the fitted models.

        This method applies the learned imputation strategies to the input DataFrame and returns the imputed DataFrame.

        Parameters:
        ----------
        data : pd.DataFrame
            DataFrame with missing values to be imputed. Should have the same structure as during fitting.

        Returns:
        -------
        pd.DataFrame
            DataFrame with missing values imputed.

        Raises:
        ------
        NotFittedError
            If imputation models have not been fitted.
        """
        try:
            if self.binary_imputer is None and self.non_binary_imputer is None:
                raise NotFittedError(
                    "MissingHandler is not fitted yet. Call 'fit' before using this method."
                )

            data_to_impute = data.drop(
                columns=[self.id_col, self.activity_col], errors="ignore"
            )
            data_to_impute.drop(columns=self.drop_cols, inplace=True, errors="ignore")

            data_binary = data_to_impute[self.binary_cols]
            data_non_binary = data_to_impute.drop(
                columns=self.binary_cols, errors="ignore"
            )

            imputed_data_binary = (
                pd.DataFrame(
                    self.binary_imputer.transform(data_binary),
                    columns=data_binary.columns,
                )
                if self.binary_imputer
                else data_binary
            )

            imputed_data_non_binary = (
                pd.DataFrame(
                    self.non_binary_imputer.transform(data_non_binary),
                    columns=data_non_binary.columns,
                )
                if self.non_binary_imputer
                else data_non_binary
            )

            columns_to_include = []
            if self.id_col is not None:
                columns_to_include.append(self.id_col)
            if self.activity_col is not None:
                columns_to_include.append(self.activity_col)

            transformed_data = pd.concat(
                [
                    data[columns_to_include],
                    imputed_data_binary,
                    imputed_data_non_binary,
                ],
                axis=1,
            )

            if self.save_trans_data:
                if self.save_dir and not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir, exist_ok=True)
                if os.path.exists(f"{self.save_dir}/{self.trans_data_name}.csv"):
                    base, ext = os.path.splitext(self.trans_data_name)
                    counter = 1
                    new_filename = f"{base} ({counter}){ext}"

                    while os.path.exists(f"{self.save_dir}/{new_filename}.csv"):
                        counter += 1
                        new_filename = f"{base} ({counter}){ext}"

                    csv_name = new_filename

                else:
                    csv_name = self.trans_data_name

                transformed_data.to_csv(f"{self.save_dir}/{csv_name}.csv")
                logging.info(
                    f"Transformed data saved at: {self.save_dir}/{csv_name}.csv"
                )

            return transformed_data

        except Exception as e:
            logging.error(f"Error in transforming data: {e}")
            raise

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the imputation models to the data and transforms the data using the fitted models.

        Parameters:
        data (pd.DataFrame): The data to fit and transform.

        Returns:
        pd.DataFrame: The transformed (imputed) DataFrame.
        """
        self.fit(data)
        return self.transform(data)
