import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.exceptions import NotFittedError
from typing import Tuple, Optional
import pickle
import os
import logging


class MissingHandler(BaseEstimator, TransformerMixin):
    """
    Handle missing values by:
      - dropping columns with too many missing values,
      - imputing binary and non-binary columns separately,
      - supporting multiple imputation strategies.

    Supports saving fitted imputers and transformed data for reproducibility.
    """

    def __init__(
        self,
        activity_col: Optional[str] = None,
        id_col: Optional[str] = None,
        missing_thresh: float = 40.0,
        imputation_strategy: str = "mean",
        n_neighbors: int = 5,
        save_method: bool = False,
        save_dir: Optional[str] = "Project/MissingHandler",
        save_trans_data: bool = False,
        trans_data_name: str = "trans_data",
        deactivate: bool = False,
    ):
        """
        Initialize MissingHandler.

        :param activity_col: Column name of activity/target values to preserve.
        :type activity_col: Optional[str]
        :param id_col: Column name of sample identifiers to preserve.
        :type id_col: Optional[str]
        :param missing_thresh: Maximum percentage of missing values allowed per column.
                               Columns above this threshold are dropped. Default is 40.0.
        :type missing_thresh: float
        :param imputation_strategy: Strategy for imputing non-binary columns.
                                    One of {"mean","median","mode","knn","mice"}.
        :type imputation_strategy: str
        :param n_neighbors: Number of neighbors for KNN imputation. Default is 5.
        :type n_neighbors: int
        :param save_method: If True, save the fitted handler object as pickle.
        :type save_method: bool
        :param save_dir: Directory to save handler and transformed data.
        :type save_dir: Optional[str]
        :param save_trans_data: If True, save transformed (imputed) data as CSV.
        :type save_trans_data: bool
        :param trans_data_name: Base filename for saving transformed data.
        :type trans_data_name: str
        :param deactivate: If True, disable the handler (returns input unchanged).
        :type deactivate: bool
        """
        self.activity_col = activity_col
        self.id_col = id_col
        self.missing_thresh = missing_thresh
        self.imputation_strategy = imputation_strategy
        self.n_neighbors = n_neighbors
        self.save_method = save_method
        self.save_dir = save_dir
        self.save_trans_data = save_trans_data
        self.trans_data_name = trans_data_name
        self.deactivate = deactivate
        self.fitted = False
        self.binary_imputer = None
        self.non_binary_imputer = None

    @staticmethod
    def calculate_missing_percent(data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute percentage of missing values per column.

        :param data: Input DataFrame.
        :type data: pandas.DataFrame
        :return: DataFrame with columns ["ColumnName","MissingPercent"].
        :rtype: pandas.DataFrame
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
    ) -> Tuple[Optional[SimpleImputer], Optional[SimpleImputer]]:
        """
        Fit imputers for binary and non-binary columns.

        :param data_to_impute: Data subset for fitting imputers.
        :type data_to_impute: pandas.DataFrame
        :param imputation_strategy: Strategy for non-binary imputation
                                    {"mean","median","mode","knn","mice"}.
        :type imputation_strategy: str
        :param n_neighbors: Number of neighbors for KNN imputation.
        :type n_neighbors: int
        :return: Tuple (binary_imputer, non_binary_imputer).
        :rtype: Tuple[Optional[SimpleImputer], Optional[SimpleImputer]]
        :raises ValueError: If an unsupported strategy is provided.
        """
        binary_cols = [
            col
            for col in data_to_impute.columns
            if data_to_impute[col].dropna().isin([0, 1]).all()
        ]
        data_binary = data_to_impute[binary_cols]
        data_non_binary = data_to_impute.drop(columns=binary_cols, errors="ignore")

        # Binary columns → most frequent
        binary_imputer = None
        if binary_cols:
            binary_imputer = SimpleImputer(strategy="most_frequent").fit(data_binary)
            logging.info("MissingHandler: Binary columns imputed with 'most_frequent'.")

        # Non-binary → configurable strategy
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
                logging.info(
                    f"MissingHandler: Non-binary columns imputed with '{imputation_strategy}'."
                )
            else:
                raise ValueError(
                    f"Unsupported imputation strategy {imputation_strategy}. "
                    "Choose from {'mean','median','mode','knn','mice'}."
                )

        return binary_imputer, non_binary_imputer

    def fit(self, data: pd.DataFrame, y=None) -> "MissingHandler":
        """
        Fit imputers to the dataset.

        :param data: Input DataFrame to fit on.
        :type data: pandas.DataFrame
        :param y: Ignored, present for sklearn compatibility.
        :type y: Optional[pandas.Series]
        :return: Fitted handler.
        :rtype: MissingHandler
        :raises Exception: For unexpected fitting errors.
        """
        if self.deactivate:
            logging.info("MissingHandler is deactivated. Skipping fit.")
            return self

        try:
            data_to_impute = data.drop(
                columns=[self.id_col, self.activity_col], errors="ignore"
            )

            # Drop columns above threshold
            missing_percent_df = self.calculate_missing_percent(data_to_impute)
            self.drop_cols = missing_percent_df[
                missing_percent_df["MissingPercent"] > self.missing_thresh
            ]["ColumnName"].tolist()
            data_to_impute.drop(columns=self.drop_cols, inplace=True)

            # Split binary/non-binary & fit
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
            self.fitted = True

            if self.save_method:
                os.makedirs(self.save_dir, exist_ok=True)
                with open(f"{self.save_dir}/missing_handler.pkl", "wb") as file:
                    pickle.dump(self, file)
                logging.info(
                    f"MissingHandler: Fitted object saved at {self.save_dir}/missing_handler.pkl"
                )

        except Exception as e:
            logging.error(f"Error fitting MissingHandler: {e}")
            raise

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using fitted imputers.

        :param data: Input DataFrame to transform.
        :type data: pandas.DataFrame
        :return: Transformed DataFrame with missing values imputed.
        :rtype: pandas.DataFrame
        :raises NotFittedError: If called before ``fit``.
        :raises Exception: For unexpected transformation errors.
        """
        if self.deactivate:
            self.transformed_data = data
            logging.info("MissingHandler is deactivated. Returning input unchanged.")
            return data

        try:
            if not self.fitted:
                raise NotFittedError(
                    "MissingHandler is not fitted yet. Call 'fit' before transform."
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
                    dtype=np.int64,
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

            transformed_data = pd.concat(
                [
                    data.filter(items=[self.id_col, self.activity_col]),
                    imputed_data_binary,
                    imputed_data_non_binary,
                ],
                axis=1,
            )

            if self.save_trans_data:
                os.makedirs(self.save_dir, exist_ok=True)
                csv_name = self.trans_data_name
                if os.path.exists(f"{self.save_dir}/{csv_name}.csv"):
                    base, ext = os.path.splitext(self.trans_data_name)
                    counter = 1
                    while os.path.exists(
                        f"{self.save_dir}/{base} ({counter}){ext}.csv"
                    ):
                        counter += 1
                    csv_name = f"{base} ({counter}){ext}"
                transformed_data.to_csv(f"{self.save_dir}/{csv_name}.csv", index=False)
                logging.info(
                    f"MissingHandler: Transformed data saved at {self.save_dir}/{csv_name}.csv"
                )

            self.transformed_data = transformed_data
            return transformed_data

        except Exception as e:
            logging.error(f"Error transforming data: {e}")
            raise

    def fit_transform(self, data: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit imputers and transform the dataset in one step.

        :param data: Input DataFrame.
        :type data: pandas.DataFrame
        :param y: Ignored, present for sklearn compatibility.
        :type y: Optional[pandas.Series]
        :return: Transformed DataFrame with imputed values.
        :rtype: pandas.DataFrame
        """
        if self.deactivate:
            logging.info("MissingHandler is deactivated. Returning input unchanged.")
            return data

        self.fit(data)
        return self.transform(data)
