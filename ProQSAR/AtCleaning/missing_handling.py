import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor


class MissingHandler:
    """
    Handles missing data in provided datasets including finding the percentage of missing values and
    applying various imputation strategies.

    Parameters:
    ----------
    data_train: pandas.DataFrame
        Data for training the model.
    data_test: pandas.DataFrame
        Data for external validation.
    id_col: str
        Identifier column name.
    activity_col: str
        Name of the activity column (e.g., pIC50, pChEMBL Value).
    missing_thresh: float, optional
        Threshold for missing value percentage to consider dropping a column (default is 40).
    """

    def __init__(
        self,
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        id_col: str,
        activity_col: str,
        missing_thresh: float = 40,
    ):
        self.data_train = data_train.copy().reset_index(drop=True)
        self.data_test = data_test.copy().reset_index(drop=True)
        self.activity_col = activity_col
        self.id_col = id_col
        self.missing_thresh = missing_thresh
        self.columns = self.data_train.columns

    @staticmethod
    def find_missing_percent(data: pd.DataFrame) -> pd.DataFrame:
        """
        Finds the percentage of missing values for each column in the DataFrame.

        Parameters:
        data: pandas.DataFrame
            The DataFrame to analyze.

        Returns:
        pd.DataFrame
            DataFrame containing the percentage of missing values for each column.
        """
        miss_percent = (data.isnull().sum() / len(data)) * 100
        return pd.DataFrame(
            {"ColumnName": data.columns, "PercentMissing": miss_percent}
        )

    def handle_missing_values(
        self, imputation_strategy: str = "mean", n_neighbors: int = 5
    ):
        """
        Handles missing values in the data based on the specified imputation strategy, excluding id_col and activity_col.

        Parameters:
        imputation_strategy: str, optional
            Strategy for imputation ('mean', 'median', 'mode', 'knn', 'mice'). Default is 'mean'.
        n_neighbors: int, optional
            Number of neighbors for KNNImputer. Default is 5.
        """
        # Exclude id_col and activity_col
        columns_to_exclude = [self.id_col, self.activity_col]
        train_data_to_impute = self.data_train.drop(columns=columns_to_exclude)
        test_data_to_impute = self.data_test.drop(columns=columns_to_exclude)

        # Drop columns with high missing percentage
        miss_df = self.find_missing_percent(train_data_to_impute)
        drop_cols = miss_df[
            miss_df["PercentMissing"] > self.missing_thresh
        ].ColumnName.tolist()
        train_data_to_impute.drop(drop_cols, axis=1, inplace=True)
        test_data_to_impute.drop(drop_cols, axis=1, inplace=True)

        # Choose imputation strategy
        if imputation_strategy == "mean":
            imputer = SimpleImputer(strategy="mean")
        elif imputation_strategy == "median":
            imputer = SimpleImputer(strategy="median")
        elif imputation_strategy == "mode":
            imputer = SimpleImputer(strategy="most_frequent")
        elif imputation_strategy == "knn":
            imputer = KNNImputer(n_neighbors=n_neighbors)
        elif imputation_strategy == "mice":
            estimator = BayesianRidge()  # Default estimator, can be changed
            imputer = IterativeImputer(estimator=estimator, random_state=42)

        # Apply imputation
        imputed_train_data = pd.DataFrame(
            imputer.fit_transform(train_data_to_impute),
            columns=train_data_to_impute.columns,
        )
        imputed_test_data = pd.DataFrame(
            imputer.transform(test_data_to_impute), columns=test_data_to_impute.columns
        )

        # Re-add the excluded columns
        self.data_train = pd.concat(
            [self.data_train[columns_to_exclude], imputed_train_data], axis=1
        )
        self.data_test = pd.concat(
            [self.data_test[columns_to_exclude], imputed_test_data], axis=1
        )

    def fit(self, imputation_strategy: str = "mean", n_neighbors: int = 5):
        """
        Executes the missing data handling process.
        """
        self.handle_missing_values(imputation_strategy, n_neighbors)
        return self.data_train, self.data_test
