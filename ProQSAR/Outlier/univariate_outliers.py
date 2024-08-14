import os
import pickle
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from ProQSAR.Preprocessor.missing_handler import MissingHandler
from typing import Tuple, Optional, List, Dict


class UnivariateOutliersHandler:
    """
    A class for handling univariate outliers in a dataset using various methods.

    Attributes:
    -----------
    id_col : Optional[str]
        The column name of the ID feature.
    activity_col : Optional[str]
        The column name of the activity feature.
    handling_method : str
        The method used to handle outliers. Options are "iqr", "winsorization", "imputation", "power",
        "normal", "uniform".
    imputation_strategy : str
        The strategy used to impute missing values. Default is "mean".
        Options are "mean", "mode", "median", "knn", "mice".
    missing_thresh : float
        The threshold percentage of missing data in a feature for it to be considered bad.
    n_neighbors : int
        The number of neighbors to use for KNN imputation.
    save_dir : Optional[str]
        Directory where fitted parameters and thresholds will be saved.
    """

    def __init__(
        self,
        id_col: Optional[str] = None,
        activity_col: Optional[str] = None,
        handling_method: str = "uniform",
        imputation_strategy: str = "mean",
        missing_thresh: float = 40.0,
        n_neighbors: int = 5,
        save_dir: Optional[str] = None,
    ) -> None:

        self.id_col = id_col
        self.activity_col = activity_col
        self.handling_method = handling_method
        self.imputation_strategy = imputation_strategy
        self.missing_thresh = missing_thresh
        self.n_neighbors = n_neighbors
        self.save_dir = save_dir
        self.iqr_thresholds = None
        self.handler = None
        self.bad = []

        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    @staticmethod
    def _iqr_threshold(data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calculate the IQR thresholds for each column in the dataframe.

        Parameters:
        -----------
        data : pd.DataFrame
            The input dataframe.

        Returns:
        --------
        iqr_thresholds : dict
            A dictionary with column names as keys and thresholds as values.
        """
        iqr_thresholds = {}
        for col in data.columns:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            iqr_thresholds[col] = {"low": low, "high": high}
        return iqr_thresholds

    @staticmethod
    def _feature_quality(
        data: pd.DataFrame,
        id_col: Optional[str] = None,
        activity_col: Optional[str] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Identify good and bad features based on their IQR thresholds.

        Parameters:
        -----------
        data : pd.DataFrame
            The input dataframe.
        id_col : Optional[str]
            The column name of the ID feature.
        activity_col : Optional[str]
            The column name of the activity feature.

        Returns:
        --------
        good : list
            List of good feature column names.
        bad : list
            List of bad feature column names.
        """
        good, bad = [], []
        cols_to_exclude = [id_col, activity_col]
        temp_data = data.drop(columns=cols_to_exclude, errors="ignore")
        non_binary_cols = [
            col
            for col in temp_data.columns
            if not temp_data[col].dropna().isin([0, 1]).all()
        ]

        iqr_thresholds = UnivariateOutliersHandler._iqr_threshold(
            temp_data[non_binary_cols]
        )
        for col, thresh in iqr_thresholds.items():
            low = thresh["low"]
            high = thresh["high"]
            df = temp_data[(temp_data[col] <= high) & (temp_data[col] >= low)]
            remove = temp_data.shape[0] - df.shape[0]
            if remove == 0:
                good.append(col)
            else:
                bad.append(col)

        return good, bad

    @staticmethod
    def _apply_iqr(
        data: pd.DataFrame, iqr_thresholds: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Apply IQR thresholding to remove outliers from the data.

        Parameters:
        -----------
        data : pd.DataFrame
            The input dataframe.
        iqr_thresholds : dict
            The IQR thresholds for each column.

        Returns:
        --------
        data : pd.DataFrame
            The dataframe with outliers removed.
        """
        for col, thresh in iqr_thresholds.items():
            low = thresh["low"]
            high = thresh["high"]
            data = data[(data[col] >= low) & (data[col] <= high)]
        return data

    @staticmethod
    def _apply_winsorization(
        data: pd.DataFrame, iqr_thresholds: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Apply winsorization to cap outliers at the IQR thresholds.

        Parameters:
        -----------
        data : pd.DataFrame
            The input dataframe.
        iqr_thresholds : dict
            The IQR thresholds for each column.

        Returns:
        --------
        data : pd.DataFrame
            The dataframe with outliers capped.
        """
        for col, thresh in iqr_thresholds.items():
            low = thresh["low"]
            high = thresh["high"]
            data[col] = np.where(data[col] < low, low, data[col])
            data[col] = np.where(data[col] > high, high, data[col])
        return data

    @staticmethod
    def _impute_nan(
        data: pd.DataFrame, iqr_thresholds: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Impute NaNs for values outside the IQR thresholds.

        Parameters:
        -----------
        data : pd.DataFrame
            The input dataframe.
        iqr_thresholds : dict
            The IQR thresholds for each column.

        Returns:
        --------
        data : pd.DataFrame
            The dataframe with NaNs imputed for outliers.
        """
        for col, thresh in iqr_thresholds.items():
            low = thresh["low"]
            high = thresh["high"]
            data[col] = np.where(
                (data[col] < low) | (data[col] > high), np.nan, data[col]
            )

        return data

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the outlier handler to the data, identifying bad features and calculating thresholds.

        Parameters:
        -----------
        data : pd.DataFrame
            The input dataframe.
        """
        _, self.bad = self._feature_quality(
            data, id_col=self.id_col, activity_col=self.activity_col
        )

        if self.bad:
            if self.handling_method in ["iqr", "winsorization", "imputation"]:
                self.iqr_thresholds = self._iqr_threshold(data[self.bad])

                if self.handling_method == "imputation":
                    data = self._impute_nan(data, self.iqr_thresholds)
                    self.handler = MissingHandler(
                        id_col=self.id_col,
                        activity_col=self.activity_col,
                        missing_thresh=self.missing_thresh,
                        imputation_strategy=self.imputation_strategy,
                        n_neighbors=self.n_neighbors,
                        save_dir=self.save_dir,
                    )
                    self.handler.fit(data)

            elif self.handling_method == "power":
                self.handler = PowerTransformer().fit(data[self.bad])

            elif self.handling_method == "normal":
                self.handler = QuantileTransformer(output_distribution="normal").fit(
                    data[self.bad]
                )

            elif self.handling_method == "uniform":
                self.handler = QuantileTransformer(output_distribution="uniform").fit(
                    data[self.bad]
                )
            else:
                raise ValueError(f"Unsupported method: {self.handling_method}")

        if self.save_dir:
            with open(f"{self.save_dir}/handling_method.pkl", "wb") as file:
                pickle.dump(self.handling_method, file)
            with open(f"{self.save_dir}/bad_features.pkl", "wb") as file:
                pickle.dump(self.bad, file)

            if self.iqr_thresholds:
                with open(f"{self.save_dir}/iqr_thresholds.pkl", "wb") as file:
                    pickle.dump(self.iqr_thresholds, file)
            if self.handler:
                with open(f"{self.save_dir}/handler.pkl", "wb") as file:
                    pickle.dump(self.handler, file)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted outlier handler.

        Parameters:
        -----------
        data : pd.DataFrame
            The input dataframe.

        Returns:
        --------
        transformed_data : pd.DataFrame
            The transformed dataframe.
        """
        transformed_data = data.copy()

        if self.handling_method in ["iqr", "winsorization", "imputation"]:
            if self.handling_method == "iqr":
                transformed_data = self._apply_iqr(
                    transformed_data, self.iqr_thresholds
                )
            elif self.handling_method == "winsorization":
                transformed_data = self._apply_winsorization(
                    transformed_data, self.iqr_thresholds
                )
            elif self.handling_method == "imputation":
                transformed_data = self._impute_nan(
                    transformed_data, self.iqr_thresholds
                )
                transformed_data = self.handler.transform(transformed_data)
        elif self.handling_method in ["power", "normal", "uniform"]:
            transformed_data[self.bad] = self.handler.transform(
                transformed_data[self.bad]
            )

        return transformed_data

    @staticmethod
    def static_transform(data: pd.DataFrame, save_dir: str) -> pd.DataFrame:
        """
        Transform the data using previously saved handling methods and thresholds.

        Parameters:
        -----------
        data : pd.DataFrame
            The input dataframe.
        saved_dir : str
            Directory where fitted parameters and thresholds are saved.

        Returns:
        --------
        transformed_data : pd.DataFrame
            The transformed dataframe.
        """
        if not os.path.exists(f"{save_dir}/handling_method.pkl"):
            raise NotFittedError(
                "The UnivariateOutliersHandler instance is not fitted yet. Call 'fit' before using this method."
            )

        with open(f"{save_dir}/handling_method.pkl", "rb") as file:
            handling_method = pickle.load(file)
        with open(f"{save_dir}/bad_features.pkl", "rb") as file:
            bad = pickle.load(file)

        if os.path.exists(f"{save_dir}/handler.pkl"):
            with open(f"{save_dir}/handler.pkl", "rb") as file:
                handler = pickle.load(file)

        transformed_data = data.copy()

        if handling_method in ["iqr", "winsorization", "imputation"]:
            with open(f"{save_dir}/iqr_thresholds.pkl", "rb") as file:
                iqr_thresholds = pickle.load(file)

            if handling_method == "iqr":
                transformed_data = UnivariateOutliersHandler._apply_iqr(
                    transformed_data, iqr_thresholds
                )
            elif handling_method == "winsorization":
                transformed_data = UnivariateOutliersHandler._apply_winsorization(
                    transformed_data, iqr_thresholds
                )
            elif handling_method == "imputation":
                imputed_nan_data = UnivariateOutliersHandler._impute_nan(
                    transformed_data, iqr_thresholds
                )
                transformed_data = handler.transform(imputed_nan_data)
        else:
            transformed_data[bad] = handler.transform(transformed_data[bad])

        return transformed_data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the handler and transform the data.

        Parameters:
        - data: Input DataFrame.

        Returns:
        - Transformed DataFrame.
        """
        self.fit(data)
        return self.transform(data)

    @staticmethod
    def compare_outlier_methods(
        data1: pd.DataFrame,
        data2: Optional[pd.DataFrame] = None,
        data1_name: str = "data1",
        data2_name: str = "data2",
        activity_col: Optional[str] = None,
        id_col: Optional[str] = None,
    ):
        """
        Compare the effect of different outlier handling methods between two datasets.

        Parameters:
        - data1: Input DataFrame to fit the handler.
        - data2: Optional second DataFrame to transform.

        Returns:
        - DataFrame showing the number of data points before and after handling, and the number of data points
        removed for each method.
        """
        comparison_data = []
        methods = ["iqr", "winsorization", "imputation", "power", "normal", "uniform"]

        for method in methods:
            handler = UnivariateOutliersHandler(
                id_col=id_col, activity_col=activity_col, handling_method=method
            )
            handler.fit(data1)

            if data2 is None:
                transformed_data1 = handler.transform(data1)
                comparison_data.append(
                    {
                        "Method": method,
                        "Original Rows": data1.shape[0],
                        "After Handling Rows": transformed_data1.shape[0],
                        "Removed Rows": data1.shape[0] - transformed_data1.shape[0],
                    }
                )
                comparison_table = pd.DataFrame(comparison_data)
                comparison_table.name = (
                    f"Comparison of different outlier handling methods on {data1_name}"
                )
            else:
                transformed_data2 = handler.transform(data2)
                comparison_data.append(
                    {
                        "Method": method,
                        "Original Rows": data2.shape[0],
                        "After Handling Rows": transformed_data2.shape[0],
                        "Removed Rows": data2.shape[0] - transformed_data2.shape[0],
                    }
                )
                comparison_table = pd.DataFrame(comparison_data)
                comparison_table.name = (
                    f"Methods fitted on {data1_name} & transformed on {data2_name}"
                )
        return comparison_table
