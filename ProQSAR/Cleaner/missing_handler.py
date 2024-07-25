import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from typing import Tuple, Optional, List
import pickle
import os


class MissingHandler:
    def __init__(
        self,
        id_col: str,
        activity_col: str,
        missing_thresh: float = 40.0,
        imputation_strategy: str = "mean",
        n_neighbors: int = 5,
        save_dir: Optional[str] = None,
    ):
        """
        Initializes the MissingHandler with necessary configuration.

        Parameters:
        - id_col (str): Column name that contains the ID of the entries.
        - activity_col (str): Column name that represents the activity label.
        - missing_thresh (float): Threshold percentage above which columns are dropped.
        - imputation_strategy (str): Strategy used for imputing missing values in non-binary columns.
        - n_neighbors (int): Number of neighbors to consider for KNN imputation.
        - save_dir (Optional[str]): Directory to save the imputation models; None if not saving.
        """
        self.id_col = id_col
        self.activity_col = activity_col
        self.missing_thresh = missing_thresh
        self.imputation_strategy = imputation_strategy
        self.n_neighbors = n_neighbors
        self.save_dir = save_dir

    @staticmethod
    def calculate_missing_percent(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the percentage of missing values for each column in the DataFrame.

        Parameters:
        - data (pd.DataFrame): The DataFrame to analyze.

        Returns:
        pd.DataFrame
            A DataFrame containing the percentage of missing values for each column.
            The DataFrame has two columns:
            - 'ColumnName': The name of the columns in the original DataFrame.
            - 'MissingPercent': The percentage of missing values in each column.
        """
        missing_percent = (data.isnull().sum() / len(data)) * 100
        return pd.DataFrame(
            {"ColumnName": data.columns, "MissingPercent": missing_percent}
        )

    @staticmethod
    def _get_imputer(
        data_to_impute: pd.DataFrame,
        imputation_strategy: str = "mean",
        n_neighbors: int = 5,
        save_dir: Optional[str] = None,
    ) -> Tuple[SimpleImputer, Optional[SimpleImputer]]:
        """
        Fits and optionally saves imputation models for binary and non-binary columns.

        Parameters:
        - data_to_impute (pd.DataFrame): Data for which imputers are to be created.
        - imputation_strategy (str): The strategy for imputing non-binary columns.
        - n_neighbors (int): The number of neighbors to consider for KNN imputation.
        - save_dir (Optional[str]): Directory where the fitted imputers will be saved.

        Returns:
        - Tuple[SimpleImputer, Optional[SimpleImputer]]: The fitted binary and non-binary imputers.
        """
        binary_cols = [
            col
            for col in data_to_impute.columns
            if data_to_impute[col].nunique(dropna=True) <= 2
            and data_to_impute[col].dropna().isin([0, 1]).all()
        ]
        data_binary = data_to_impute[binary_cols]
        data_non_binary = data_to_impute.drop(columns=binary_cols)

        # Fit imputation transformer for binary columns
        binary_imputer = SimpleImputer(strategy="most_frequent")
        if binary_cols:
            binary_imputer.fit(data_binary)
            if save_dir:
                with open(f"{save_dir}/binary_imputer.pkl", "wb") as file:
                    pickle.dump(binary_imputer, file)
                with open(f"{save_dir}/binary_cols.pkl", "wb") as file:
                    pickle.dump(binary_cols, file)

        # Fit imputation transformer for non-binary columns
        non_binary_imputer = None
        if imputation_strategy == "mean":
            non_binary_imputer = SimpleImputer(strategy="mean")
        elif imputation_strategy == "median":
            non_binary_imputer = SimpleImputer(strategy="median")
        elif imputation_strategy == "mode":
            non_binary_imputer = SimpleImputer(strategy="most_frequent")
        elif imputation_strategy == "knn":
            non_binary_imputer = KNNImputer(n_neighbors=n_neighbors)
        elif imputation_strategy == "mice":
            estimator = BayesianRidge()
            non_binary_imputer = IterativeImputer(estimator=estimator, random_state=42)

        if non_binary_imputer:
            non_binary_imputer.fit(data_non_binary)
            if save_dir:
                with open(f"{save_dir}/non_binary_imputer.pkl", "wb") as file:
                    pickle.dump(non_binary_imputer, file)

        return binary_imputer, non_binary_imputer

    def fit(self, data: pd.DataFrame) -> Tuple[SimpleImputer, SimpleImputer]:
        """
        Fits the imputation models to the data and optionally saves the configuration and models.

        Parameters:
        - data (pd.DataFrame): The data on which to fit the imputation models.

        Returns:
        - Tuple[SimpleImputer, SimpleImputer]: The fitted binary and non-binary imputers.
        """
        columns_to_exclude = [self.id_col, self.activity_col]
        data_to_impute = data.drop(columns=columns_to_exclude)

        missing_percent_df = self.calculate_missing_percent(data_to_impute)
        drop_cols = missing_percent_df[
            missing_percent_df["MissingPercent"] > self.missing_thresh
        ]["ColumnName"].tolist()
        data_to_impute.drop(columns=drop_cols, inplace=True)

        if self.save_dir:
            with open(f"{self.save_dir}/columns_to_exclude.pkl", "wb") as file:
                pickle.dump(columns_to_exclude, file)
            with open(f"{self.save_dir}/drop_cols.pkl", "wb") as file:
                pickle.dump(drop_cols, file)

        return self._get_imputer(
            data_to_impute,
            imputation_strategy=self.imputation_strategy,
            n_neighbors=self.n_neighbors,
            save_dir=self.save_dir,
        )

    @staticmethod
    def transform(data: pd.DataFrame, save_dir: str) -> pd.DataFrame:
        """
        Transforms the provided DataFrame using the saved imputers and configurations.

        Parameters:
        - data (pd.DataFrame): Data to be transformed.
        - save_dir (str): Directory where the imputers and configuration files are stored.

        Returns:
        - pd.DataFrame: The transformed (imputed) DataFrame.
        """
        # Load necessary objects
        with open(f"{save_dir}/non_binary_imputer.pkl", "rb") as file:
            non_binary_imputer = pickle.load(file)
        with open(f"{save_dir}/columns_to_exclude.pkl", "rb") as file:
            columns_to_exclude = pickle.load(file)
        with open(f"{save_dir}/drop_cols.pkl", "rb") as file:
            drop_cols = pickle.load(file)
        if os.path.exists(f"{save_dir}/binary_cols.pkl"):
            with open(f"{save_dir}/binary_cols.pkl", "rb") as file:
                binary_cols = pickle.load(file)
            with open(f"{save_dir}/binary_imputer.pkl", "rb") as file:
                binary_imputer = pickle.load(file)
        else:
            binary_cols = []

        data_to_impute = data.drop(columns=columns_to_exclude)
        data_to_impute.drop(columns=drop_cols, inplace=True, errors="ignore")

        data_binary = data_to_impute[binary_cols]
        data_non_binary = data_to_impute.drop(columns=binary_cols, errors="ignore")

        imputed_data_binary = (
            pd.DataFrame(
                binary_imputer.transform(data_binary), columns=data_binary.columns
            )
            if binary_cols
            else data_binary
        )

        imputed_data_non_binary = pd.DataFrame(
            non_binary_imputer.transform(data_non_binary),
            columns=data_non_binary.columns,
        )

        return pd.concat(
            [data[columns_to_exclude], imputed_data_binary, imputed_data_non_binary],
            axis=1,
        )

    def fit_transform(self, data: pd.DataFrame, save_dir: str) -> pd.DataFrame:
        """
        Fits the imputation models to the data and transforms the data using the fitted models.

        Parameters:
        - data (pd.DataFrame): The data to fit and transform.
        - save_dir (str): Directory where the imputers and configuration files are stored.

        Returns:
        - pd.DataFrame: The transformed (imputed) DataFrame.
        """
        self.fit(data)
        return self.transform(data, save_dir)
