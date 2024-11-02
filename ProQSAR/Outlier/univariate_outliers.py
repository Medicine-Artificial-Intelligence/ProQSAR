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
    data: pd.DataFrame, 
    iqr_thresholds: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
    
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

    good, bad = [], []
    cols_to_exclude = [id_col, activity_col]
    temp_data = data.drop(columns=cols_to_exclude, errors="ignore")
    non_binary_cols = [
        col
        for col in temp_data.columns
        if not temp_data[col].dropna().isin([0, 1]).all()
    ]

    iqr_thresholds = _iqr_threshold(
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

class IQRHandler:

    def __init__(self):
        self.iqr_thresholds = None
    
    def fit(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:

        self.iqr_thresholds = _iqr_threshold(data)
        return self
    
    def transform(self, data: pd.DataFrame):

        if self.iqr_thresholds is None:
            raise ValueError("The 'fit' method must be called before 'transform'.")
        
        transformed_data = deepcopy(data)
        
        for col, thresh in self.iqr_thresholds.items():
            low = thresh["low"]
            high = thresh["high"]
            transformed_data = data[(data[col] >= low) & (data[col] <= high)]
        return transformed_data
    
    def fit_transform(self, data: pd.DataFrame):
        
        self.fit(data)
        return self.transform(data)

class WinsorHandler:
    
    def __init__(self):
        self.iqr_thresholds = None
        
    def fit(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:

        self.iqr_thresholds = _iqr_threshold(data)
        return self
    
    def transform(self, data: pd.DataFrame):
        
        if self.iqr_thresholds is None:
            raise ValueError("The 'fit' method must be called before 'transform'.")

        transformed_data = deepcopy(data)
                        
        for col, thresh in self.iqr_thresholds.items():
            low = thresh["low"]
            high = thresh["high"]
            transformed_data[col] = np.where(transformed_data[col] < low, low, transformed_data[col])
            transformed_data[col] = np.where(transformed_data[col] > high, high, transformed_data[col])
            
        return transformed_data
    
    def fit_transform(self, data: pd.DataFrame):
        self.fit(data)
        return self.transform(data)

class ImputationHandler:
    
    def __init__(
        self,
        missing_thresh: float = 40.0,
        imputation_strategy: str = "mean",
        n_neighbors: int = 5
        ):
        
        self.missing_thresh = missing_thresh
        self.imputation_strategy = imputation_strategy
        self.n_neighbors = n_neighbors
        self.iqr_thresholds = None
        self.imputation_handler = None
        
    def fit(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:

        self.iqr_thresholds = _iqr_threshold(data)
        nan_data = _impute_nan(data, self.iqr_thresholds)
        self.imputation_handler = MissingHandler(
            missing_thresh=self.missing_thresh,
            imputation_strategy=self.imputation_strategy,
            n_neighbors=self.n_neighbors
        )
        self.imputation_handler.fit(nan_data)
        
        return self
    
    def transform(self, data: pd.DataFrame):
        if self.iqr_thresholds or self.imputation_handler is None:
            raise ValueError("The 'fit' method must be called before 'transform'.")

                
        nan_data = _impute_nan(data, self.iqr_thresholds)
        return self.imputation_handler.transform(nan_data)

    def fit_transform(self, data: pd.DataFrame):
        self.fit(data)
        return self.transform(data)
    
class UnivariateOutliersHandler:
    """
    A class for handling univariate outliers in a dataset using various methods.

    Attributes:
    -----------
    id_col : Optional[str]
        The column name of the ID feature.
    activity_col : Optional[str]
        The column name of the activity feature.
    select_method : str
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
        self.iqr_thresholds = None
        self.outlier_handler = None
        self.bad = []

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the outlier handler to the data, identifying bad features and calculating thresholds.

        Parameters:
        -----------
        data : pd.DataFrame
            The input dataframe.
        """
        _, self.bad = _feature_quality(
            data, id_col=self.id_col, activity_col=self.activity_col
        )
        method_map = {
            "iqr": IQRHandler(),
            "winsorization": WinsorHandler(),
            "imputation": ImputationHandler(
                missing_thresh=self.missing_thresh,
                imputation_strategy=self.imputation_strategy,
                n_neighbors=self.n_neighbors),
            "power": PowerTransformer(),
            "normal": QuantileTransformer(output_distribution="normal"),
            "uniform": QuantileTransformer(output_distribution="uniform"),
        }
        
        if self.bad:
            if self.select_method in method_map:
                self.uni_outlier_handler = method_map[self.select_method].fit(data[self.bad])
            
            else:
                raise ValueError(f"Unsupported method: {self.select_method}")

        if self.save_method:
            if self.save_dir and not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir, exist_ok=True)
            with open(f"{self.save_dir}/uni_outlier_handler_uni.pkl", "wb") as file:
                pickle.dump(self.uni_outlier_handler, file)

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
        transformed_data = deepcopy(data)

        if self.select_method in ["iqr", "winsorization", "imputation"]:
            if self.select_method == "iqr":
                transformed_data = self._apply_iqr(
                    transformed_data, self.iqr_thresholds
                )
            elif self.select_method == "winsorization":
                transformed_data = self._apply_winsorization(
                    transformed_data, self.iqr_thresholds
                )
            elif self.select_method == "imputation":
                transformed_data = self._impute_nan(
                    transformed_data, self.iqr_thresholds
                )
                transformed_data = self.outlier_handler.transform(transformed_data)
                
                
        elif self.select_method in ["power", "normal", "uniform"]:
            transformed_data[self.bad] = self.outlier_handler.transform(
                transformed_data[self.bad]
            )

        return transformed_data

############# fit => return self | save => save self =>>> remove static_transform #############

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
        if not os.path.exists(f"{save_dir}/select_method.pkl"):
            raise NotFittedError(
                "The UnivariateOutliersHandler instance is not fitted yet. Call 'fit' before using this method."
            )

        with open(f"{save_dir}/select_method.pkl", "rb") as file:
            select_method = pickle.load(file)
        with open(f"{save_dir}/bad_features.pkl", "rb") as file:
            bad = pickle.load(file)

        if os.path.exists(f"{save_dir}/outlier_handler.pkl"):
            with open(f"{save_dir}/outlier_handler.pkl", "rb") as file:
                outlier_handler = pickle.load(file)

        transformed_data = deepcopy(data)

        if select_method in ["iqr", "winsorization", "imputation"]:
            with open(f"{save_dir}/iqr_thresholds.pkl", "rb") as file:
                iqr_thresholds = pickle.load(file)

            if select_method == "iqr":
                transformed_data = _apply_iqr(
                    transformed_data, iqr_thresholds
                )
            elif select_method == "winsorization":
                transformed_data = _apply_winsorization(
                    transformed_data, iqr_thresholds
                )
            elif select_method == "imputation":
                imputed_nan_data = _impute_nan(
                    transformed_data, iqr_thresholds
                )
                transformed_data = outlier_handler.transform(imputed_nan_data)
        else:
            transformed_data[bad] = outlier_handler.transform(transformed_data[bad])

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
            outlier_handler = UnivariateOutliersHandler(
                id_col=id_col, activity_col=activity_col, select_method=method
            )
            outlier_handler.fit(data1)

            if data2 is None:
                transformed_data1 = outlier_handler.transform(data1)
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
                transformed_data2 = outlier_handler.transform(data2)
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
