import os
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from ProQSAR.Preprocessor.missing_handler import MissingHandler
from typing import Tuple, Optional, List, Dict


def _iqr_threshold(data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculates the Interquartile Range (IQR) thresholds for each numeric column in the provided DataFrame.

    Parameters:
    - data: A pandas DataFrame containing numeric data.

    Returns:
    - A dictionary where each key is the column name and each value is another dictionary with "low" and "high"
      thresholds based on the IQR method (1.5 * IQR rule).
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


def _impute_nan(
    data: pd.DataFrame, iqr_thresholds: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Imputes NaN values in the DataFrame based on the IQR thresholds, replacing outliers with NaN.

    Parameters:
    - data: A pandas DataFrame with potential outliers to be imputed.
    - iqr_thresholds: A dictionary containing the IQR thresholds for each column.

    Returns:
    - A DataFrame with outliers replaced by NaN based on the IQR thresholds.
    """
    nan_data = deepcopy(data)
    for col, thresh in iqr_thresholds.items():
        low = thresh["low"]
        high = thresh["high"]
        nan_data[col] = np.where(
            (nan_data[col] < low) | (nan_data[col] > high), np.nan, nan_data[col]
        )

    return nan_data


def _feature_quality(
    data: pd.DataFrame,
    id_col: Optional[str] = None,
    activity_col: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """
    Identifies good and bad features based on the quality of data, removing columns with high outlier presence.

    Parameters:
    - data: A pandas DataFrame to analyze.
    - id_col: Optional; The name of the column that contains IDs, to exclude from analysis.
    - activity_col: Optional; The name of the column that contains activity labels, to exclude from analysis.

    Returns:
    - A tuple of two lists:
        - 'good' features: Columns that do not have outliers based on IQR.
        - 'bad' features: Columns that are identified as having outliers.
    """
    good, bad = [], []
    cols_to_exclude = [id_col, activity_col]
    temp_data = data.drop(columns=cols_to_exclude, errors="ignore")
    non_binary_cols = [
        col
        for col in temp_data.columns
        if not temp_data[col].dropna().isin([0, 1]).all()
    ]

    iqr_thresholds = _iqr_threshold(temp_data[non_binary_cols])
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


class IQRHandler:
    """
    A handler class to manage outliers using the Interquartile Range (IQR) method.

    Attributes:
    - iqr_thresholds: A dictionary of IQR thresholds for columns, set during the fitting process.

    Methods:
    - fit: Calculates the IQR thresholds for the data.
    - transform: Removes rows containing outliers based on the IQR thresholds.
    - fit_transform: Combines fitting and transforming in one method.
    """

    def __init__(self):
        self.iqr_thresholds = None

    def fit(self, data: pd.DataFrame) -> "IQRHandler":
        """
        Fits the IQR handler by calculating the IQR thresholds for the given data.

        Parameters:
        - data: A pandas DataFrame containing the data to calculate IQR thresholds for.

        Returns:
        - The IQRHandler instance (for method chaining).
        """

        self.iqr_thresholds = _iqr_threshold(data)
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data by removing rows containing outliers based on the calculated IQR thresholds.

        Parameters:
        - data: A pandas DataFrame to be transformed.

        Returns:
        - A DataFrame with outliers removed based on IQR thresholds.
        """

        if self.iqr_thresholds is None:
            raise NotFittedError("The 'fit' method must be called before 'transform'.")

        transformed_data = deepcopy(data)

        for col, thresh in self.iqr_thresholds.items():
            low = thresh["low"]
            high = thresh["high"]
            transformed_data = transformed_data[
                (transformed_data[col] >= low) & (transformed_data[col] <= high)
            ]
        return transformed_data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the handler and transforms the data in one step.

        Parameters:
        - data: A pandas DataFrame to be fitted and transformed.

        Returns:
        - Transformed DataFrame with outliers removed.
        """

        self.fit(data)
        return self.transform(data)


class WinsorHandler:
    """
    A handler class to manage outliers by applying Winsorization, i.e., capping outliers at the IQR thresholds.

    Attributes:
    - iqr_thresholds: A dictionary of IQR thresholds for columns, set during the fitting process.

    Methods:
    - fit: Calculates the IQR thresholds for the data.
    - transform: Caps outliers to the IQR thresholds.
    - fit_transform: Combines fitting and transforming in one method.
    """

    def __init__(self):
        self.iqr_thresholds = None

    def fit(self, data: pd.DataFrame) -> "WinsorHandler":
        """
        Fits the WinsorHandler by calculating the IQR thresholds for the given data.

        Parameters:
        - data: A pandas DataFrame to calculate IQR thresholds for.

        Returns:
        - The WinsorHandler instance (for method chaining).
        """
        self.iqr_thresholds = _iqr_threshold(data)
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data by capping outliers at the IQR thresholds.

        Parameters:
        - data: A pandas DataFrame to be transformed.

        Returns:
        - A DataFrame with outliers capped at the IQR thresholds.
        """
        if self.iqr_thresholds is None:
            raise NotFittedError("The 'fit' method must be called before 'transform'.")

        transformed_data = deepcopy(data)

        for col, thresh in self.iqr_thresholds.items():
            low = thresh["low"]
            high = thresh["high"]
            transformed_data[col] = np.where(
                transformed_data[col] < low, low, transformed_data[col]
            )
            transformed_data[col] = np.where(
                transformed_data[col] > high, high, transformed_data[col]
            )

        return transformed_data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the handler and transforms the data in one step.

        Parameters:
        - data: A pandas DataFrame to be fitted and transformed.

        Returns:
        - Transformed DataFrame with outliers capped.
        """
        self.fit(data)
        return self.transform(data)


class ImputationHandler:
    """
    A handler class to manage imputation of missing values and outlier handling.

    Attributes:
    - missing_thresh: The threshold percentage for missing data.
    - imputation_strategy: Strategy for imputing missing values ("mean", "median", etc.).
    - n_neighbors: Number of neighbors to use if imputation strategy is "knn".
    - iqr_thresholds: IQR thresholds for identifying outliers.
    - imputation_handler: An instance of MissingHandler used for imputation.

    Methods:
    - fit: Calculates IQR thresholds and fits the imputation handler.
    - transform: Imputes missing values in the data.
    - fit_transform: Combines fitting and transforming in one method.
    """

    def __init__(
        self,
        missing_thresh: float = 40.0,
        imputation_strategy: str = "mean",
        n_neighbors: int = 5,
    ):

        self.missing_thresh = missing_thresh
        self.imputation_strategy = imputation_strategy
        self.n_neighbors = n_neighbors
        self.iqr_thresholds = None
        self.imputation_handler = None

    def fit(self, data: pd.DataFrame) -> "ImputationHandler":
        """
        Fits the ImputationHandler by calculating IQR thresholds and preparing the imputation strategy.

        Parameters:
        - data: A pandas DataFrame to be used for fitting the handler.

        Returns:
        - The ImputationHandler instance (for method chaining).
        """
        self.iqr_thresholds = _iqr_threshold(data)
        nan_data = _impute_nan(data, self.iqr_thresholds)
        self.imputation_handler = MissingHandler(
            missing_thresh=self.missing_thresh,
            imputation_strategy=self.imputation_strategy,
            n_neighbors=self.n_neighbors,
        )
        self.imputation_handler.fit(nan_data)

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data by imputing missing values.

        Parameters:
        - data: A pandas DataFrame to be transformed.

        Returns:
        - Transformed DataFrame with missing values imputed.
        """
        if self.iqr_thresholds is None or self.imputation_handler is None:
            raise NotFittedError("The 'fit' method must be called before 'transform'.")

        nan_data = _impute_nan(data, self.iqr_thresholds)
        return self.imputation_handler.transform(nan_data)

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the handler and imputes missing values in one step.

        Parameters:
        - data: A pandas DataFrame to be fitted and transformed.

        Returns:
        - Transformed DataFrame with missing values imputed.
        """
        self.fit(data)
        return self.transform(data)


class UnivariateOutliersHandler:
    """
    A class for handling univariate outliers in a dataset using various methods
    (e.g., IQR, Winsorization, imputation, etc.).

    Attributes:
    - activity_col: Optional; The column with activity data to exclude from analysis.
    - id_col: Optional; The column with IDs to exclude from analysis.
    - select_method: The method to use for outlier handling (e.g., "iqr", "winsorization").
    - save_method: Boolean indicating if the method should be saved.
    - save_dir: Directory to save the fitted handler model.
    - save_trans_data: Boolean indicating if transformed data should be saved.
    - trans_data_name: Name for the saved transformed data file.

    Methods:
    - fit: Fits the outlier handler based on the selected method.
    - transform: Transforms the data by handling outliers.
    - fit_transform: Combines fitting and transforming in one method.
    - compare_outlier_methods: Compares the effect of different outlier handling methods on two datasets.
    """

    def __init__(
        self,
        activity_col: Optional[str] = None,
        id_col: Optional[str] = None,
        select_method: str = "uniform",
        imputation_strategy: str = "mean",
        missing_thresh: float = 40.0,
        n_neighbors: int = 5,
        save_method: bool = False,
        save_dir: Optional[str] = "Project/OutlierHandler",
        save_trans_data: bool = False,
        trans_data_name: str = "uo_trans_data",
    ):
        self.activity_col = activity_col
        self.id_col = id_col
        self.select_method = select_method
        self.imputation_strategy = imputation_strategy
        self.missing_thresh = missing_thresh
        self.n_neighbors = n_neighbors
        self.save_method = save_method
        self.save_dir = save_dir
        self.save_trans_data = save_trans_data
        self.trans_data_name = trans_data_name
        self.uni_outlier_handler = None
        self.bad = []

    def fit(self, data: pd.DataFrame) -> "UnivariateOutliersHandler":
        """
        Fits the outlier handler by identifying bad features and selecting the method for outlier handling.

        Parameters:
        - data: A pandas DataFrame containing the data to be processed.

        Returns:
        - The UnivariateOutliersHandler instance (for method chaining).
        """

        _, self.bad = _feature_quality(
            data, id_col=self.id_col, activity_col=self.activity_col
        )
        if not self.bad:
            print("No bad features (outliers) found. Skipping outlier handling.")
            return self

        method_map = {
            "iqr": IQRHandler(),
            "winsorization": WinsorHandler(),
            "imputation": ImputationHandler(
                missing_thresh=self.missing_thresh,
                imputation_strategy=self.imputation_strategy,
                n_neighbors=self.n_neighbors,
            ),
            "power": PowerTransformer(),
            "normal": QuantileTransformer(output_distribution="normal"),
            "uniform": QuantileTransformer(output_distribution="uniform"),
        }

        if self.select_method in method_map:
            self.uni_outlier_handler = method_map[self.select_method].fit(
                data[self.bad]
            )
        else:
            raise ValueError(f"Unsupported method: {self.select_method}")

        if self.save_method:
            if self.save_dir and not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir, exist_ok=True)
            with open(f"{self.save_dir}/uni_outlier_handler.pkl", "wb") as file:
                pickle.dump(self, file)

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data by applying outlier handling to the identified bad features.

        Parameters:
        - data: A pandas DataFrame to be transformed.

        Returns:
        - Transformed DataFrame with outliers handled based on the selected method.
        """
        transformed_data = deepcopy(data)
        if not self.bad:
            print("No bad features (outliers) to handle. Returning original data.")
            return transformed_data

        transformed_data[self.bad] = self.uni_outlier_handler.transform(
            transformed_data[self.bad]
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
            print(f"File have been saved at: {self.save_dir}/{csv_name}.csv")

        return transformed_data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the handler and transform the data in one step.

        Parameters:
        - data: A pandas DataFrame to be fitted and transformed.

        Returns:
        - Transformed DataFrame with outliers handled.
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
        methods_to_compare: List[str] = None,
    ) -> pd.DataFrame:
        """
        Compare the effect of different outlier handling methods between two datasets.

        Parameters:
        - data1: The first dataset to be analyzed.
        - data2: Optional; The second dataset to be analyzed.
        - data1_name: Name for the first dataset.
        - data2_name: Name for the second dataset (if provided).
        - activity_col: Optional; The name of the activity column.
        - id_col: Optional; The name of the ID column.
        - methods_to_compare: List of methods to compare, e.g., ["iqr", "winsorization"].

        Returns:
        - A DataFrame summarizing the effect of each method on the datasets.
        """
        comparison_data = []
        methods = ["iqr", "winsorization", "imputation", "power", "normal", "uniform"]
        methods_to_compare = methods_to_compare or methods

        for method in methods_to_compare:
            uni_outlier_handler = UnivariateOutliersHandler(
                id_col=id_col,
                activity_col=activity_col,
                select_method=method,
            )
            uni_outlier_handler.fit(data1)

            transformed_data1 = uni_outlier_handler.transform(data1)
            comparison_data.append(
                {
                    "Method": method,
                    "Dataset": data1_name,
                    "Original Rows": data1.shape[0],
                    "After Handling Rows": transformed_data1.shape[0],
                    "Removed Rows": data1.shape[0] - transformed_data1.shape[0],
                }
            )
            comparison_table = pd.DataFrame(comparison_data)
            comparison_table.name = f"Methods fitted & transformed on {data1_name}"
            if data2 is not None:

                transformed_data2 = uni_outlier_handler.transform(data2)
                comparison_data.append(
                    {
                        "Method": method,
                        "Dataset": data2_name,
                        "Original Rows": data2.shape[0],
                        "After Handling Rows": transformed_data2.shape[0],
                        "Removed Rows": data2.shape[0] - transformed_data2.shape[0],
                    }
                )
                comparison_table = pd.DataFrame(comparison_data)
                comparison_table.name = f"Methods fitted on {data1_name} & transformed on {data1_name} & {data2_name}"
        return comparison_table
