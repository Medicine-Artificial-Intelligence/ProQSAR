import os
import pickle
import logging
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Tuple, Optional, List, Dict
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from ..Clean.missing_handler import MissingHandler


def _iqr_threshold(data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculates the Interquartile Range (IQR) thresholds for each numeric column.

    Outliers are defined using the conventional 1.5 * IQR rule:
    low = Q1 - 1.5 * IQR, high = Q3 + 1.5 * IQR.

    :param data: DataFrame containing numeric columns for which thresholds are computed.
    :type data: pd.DataFrame
    :return: Dictionary mapping column name -> {"low": low_value, "high": high_value}.
    :rtype: Dict[str, Dict[str, float]]
    :raises Exception: Propagates exceptions that occur during calculation.
    """
    try:
        iqr_thresholds = {}
        for col in data.columns:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            iqr_thresholds[col] = {"low": low, "high": high}

        return iqr_thresholds

    except Exception as e:
        logging.error(f"Error in computing IQR thresholds: {e}")
        raise


def _impute_nan(
    data: pd.DataFrame, iqr_thresholds: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Replace values outside IQR thresholds with NaN so they can be later imputed.

    :param data: Input DataFrame potentially containing outliers.
    :type data: pd.DataFrame
    :param iqr_thresholds: Mapping column -> {"low": low_value, "high": high_value}.
    :type iqr_thresholds: Dict[str, Dict[str, float]]
    :return: A deep-copied DataFrame where outlier values are replaced by np.nan.
    :rtype: pd.DataFrame
    :raises Exception: Propagates exceptions that occur during replacement.
    """
    try:
        nan_data = deepcopy(data)
        for col, thresh in iqr_thresholds.items():
            low = thresh["low"]
            high = thresh["high"]
            nan_data[col] = np.where(
                (nan_data[col] < low) | (nan_data[col] > high), np.nan, nan_data[col]
            )
        return nan_data
    except Exception as e:
        logging.error(f"Error in imputing NaN values: {e}")
        raise


def _feature_quality(
    data: pd.DataFrame,
    id_col: Optional[str] = None,
    activity_col: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """
    Identify 'good' and 'bad' features based on univariate outlier presence.

    A feature is considered "bad" when at least one row is outside its IQR
    thresholds (i.e., would be removed by an IQR-based filter). Binary
    (0/1) columns are excluded from assessment.

    :param data: Input DataFrame to analyze.
    :type data: pd.DataFrame
    :param id_col: Optional column name to exclude from analysis (e.g., 'id').
    :type id_col: Optional[str]
    :param activity_col: Optional column name to exclude from analysis (e.g., 'activity').
    :type activity_col: Optional[str]
    :return: Tuple of (good_features, bad_features).
    :rtype: Tuple[List[str], List[str]]
    :raises Exception: Propagates exceptions that occur during analysis.
    """
    try:
        good, bad = [], []
        cols_to_exclude = [id_col, activity_col]
        temp_data = data.drop(columns=cols_to_exclude, errors="ignore")
        non_binary_cols = [
            col
            for col in temp_data.columns
            if not temp_data[col].dropna().isin([0, 1]).all()
        ]

        if not non_binary_cols:
            logging.info("OutlierDetection: No non-binary columns to handle outliers.")

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
    except Exception as e:
        logging.error(f"Error in feature quality assessment: {e}")
        raise


class IQRHandler:
    """
    Handler that removes rows containing univariate outliers based on IQR thresholds.

    Typical use:
        handler = IQRHandler()
        handler.fit(df)
        df_clean = handler.transform(df)

    :ivar iqr_thresholds: dictionary of thresholds created during `fit`.
    :vartype iqr_thresholds: Optional[Dict[str, Dict[str, float]]]
    """

    def __init__(self):
        self.iqr_thresholds: Optional[Dict[str, Dict[str, float]]] = None

    def fit(self, data: pd.DataFrame) -> "IQRHandler":
        """
        Compute IQR thresholds from the provided data.

        :param data: DataFrame used to compute thresholds.
        :type data: pd.DataFrame
        :return: self (fitted handler).
        :rtype: IQRHandler
        """
        self.iqr_thresholds = _iqr_threshold(data)
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows that contain values outside the precomputed IQR thresholds.

        :param data: DataFrame to filter.
        :type data: pd.DataFrame
        :return: Filtered DataFrame with outlier rows removed.
        :rtype: pd.DataFrame
        :raises NotFittedError: If `fit` has not been called.
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
        Fit the handler and immediately transform the same data.

        :param data: DataFrame to fit and transform.
        :type data: pd.DataFrame
        :return: Filtered DataFrame.
        :rtype: pd.DataFrame
        """
        self.fit(data)
        return self.transform(data)


class WinsorHandler:
    """
    Handler that applies Winsorization (capping) using IQR thresholds.

    Typical use:
        wh = WinsorHandler()
        wh.fit(df)
        df_capped = wh.transform(df)

    :ivar iqr_thresholds: dictionary of thresholds created during `fit`.
    :vartype iqr_thresholds: Optional[Dict[str, Dict[str, float]]]
    """

    def __init__(self):
        self.iqr_thresholds: Optional[Dict[str, Dict[str, float]]] = None

    def fit(self, data: pd.DataFrame) -> "WinsorHandler":
        """
        Compute and store IQR thresholds.

        :param data: DataFrame used to compute thresholds.
        :type data: pd.DataFrame
        :return: self (fitted handler).
        :rtype: WinsorHandler
        """
        self.iqr_thresholds = _iqr_threshold(data)
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cap values below/above thresholds to low/high respectively.

        :param data: DataFrame to apply Winsorization to.
        :type data: pd.DataFrame
        :return: Winsorized DataFrame.
        :rtype: pd.DataFrame
        :raises NotFittedError: If `fit` has not been called.
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
        Fit thresholds then apply Winsorization.

        :param data: DataFrame to fit & transform.
        :type data: pd.DataFrame
        :return: Winsorized DataFrame.
        :rtype: pd.DataFrame
        """
        self.fit(data)
        return self.transform(data)


class ImputationHandler:
    """
    Handler that marks univariate outliers as NaN (based on IQR) and imputes them using MissingHandler.

    Typical use:
        ih = ImputationHandler(missing_thresh=40.0, imputation_strategy='mean')
        ih.fit(df)
        df_imputed = ih.transform(df)

    :param missing_thresh: Max allowed percent-missing per column (forwarded to MissingHandler).
    :type missing_thresh: float
    :param imputation_strategy: Imputation strategy passed to MissingHandler ('mean','median','mode','knn','mice').
    :type imputation_strategy: str
    :param n_neighbors: neighbors used for KNN imputation when selected.
    :type n_neighbors: int

    :ivar iqr_thresholds: thresholds used to mark outliers as NaN.
    :vartype iqr_thresholds: Optional[Dict[str, Dict[str, float]]]
    :ivar imputation_handler: fitted MissingHandler instance.
    :vartype imputation_handler: Optional[MissingHandler]
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
        self.iqr_thresholds: Optional[Dict[str, Dict[str, float]]] = None
        self.imputation_handler: Optional[MissingHandler] = None

    def fit(self, data: pd.DataFrame) -> "ImputationHandler":
        """
        Compute IQR thresholds and fit a MissingHandler on the NaN-marked data.

        :param data: DataFrame used to compute thresholds and to fit imputer.
        :type data: pd.DataFrame
        :return: self (fitted ImputationHandler).
        :rtype: ImputationHandler
        """
        self.iqr_thresholds = _iqr_threshold(data)
        nan_data = _impute_nan(data, self.iqr_thresholds)
        self.imputation_handler = MissingHandler(
            missing_thresh=self.missing_thresh,
            imputation_strategy=self.imputation_strategy,
            n_neighbors=self.n_neighbors,
            save_method=None,
        )
        self.imputation_handler.fit(nan_data)

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Replace outliers with NaN according to fitted thresholds and impute them.

        :param data: DataFrame to impute.
        :type data: pd.DataFrame
        :return: Imputed DataFrame.
        :rtype: pd.DataFrame
        :raises NotFittedError: If `fit` has not been called.
        """
        if self.iqr_thresholds is None or self.imputation_handler is None:
            raise NotFittedError("The 'fit' method must be called before 'transform'.")

        nan_data = _impute_nan(data, self.iqr_thresholds)
        return self.imputation_handler.transform(nan_data)

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and impute in one step.

        :param data: DataFrame to fit & impute.
        :type data: pd.DataFrame
        :return: Imputed DataFrame.
        :rtype: pd.DataFrame
        """
        self.fit(data)
        return self.transform(data)


class UnivariateOutliersHandler(BaseEstimator, TransformerMixin):
    """
    High-level univariate outlier handler.

    This class detects features with univariate outliers (via `_feature_quality`)
    and applies one of several handling strategies only to those features:

      - 'iqr'           : remove rows outside IQR thresholds (IQRHandler)
      - 'winsorization' : cap values at thresholds (WinsorHandler)
      - 'imputation'    : set outliers to NaN and impute (ImputationHandler)
      - 'power'         : PowerTransformer()
      - 'normal'        : QuantileTransformer(output_distribution='normal')
      - 'uniform'       : QuantileTransformer(output_distribution='uniform')

    Typical usage:
        uoh = UnivariateOutliersHandler(select_method='iqr', id_col='id')
        uoh.fit(df)
        df_out = uoh.transform(df)

    :param activity_col: Optional column name for activity/target to exclude from detection.
    :type activity_col: Optional[str]
    :param id_col: Optional column name for identifiers to exclude from detection.
    :type id_col: Optional[str]
    :param select_method: Chosen method key (one of the supported methods).
    :type select_method: str
    :param imputation_strategy: Strategy forwarded to ImputationHandler when used.
    :type imputation_strategy: str
    :param missing_thresh: Missing percent threshold forwarded to ImputationHandler.
    :type missing_thresh: float
    :param n_neighbors: KNN neighbors forwarded to ImputationHandler when used.
    :type n_neighbors: int
    :param save_method: If True, saves the fitted handler as a pickle in save_dir.
    :type save_method: bool
    :param save_dir: Directory used for saving pickles/CSVs.
    :type save_dir: Optional[str]
    :param save_trans_data: If True, transformed data is saved to CSV.
    :type save_trans_data: bool
    :param trans_data_name: Filename base for saving transformed CSV.
    :type trans_data_name: str
    :param deactivate: If True, fit/transform become no-ops and input is returned unchanged.
    :type deactivate: bool
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
        trans_data_name: str = "trans_data",
        deactivate: bool = False,
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
        self.deactivate = deactivate
        self.uni_outlier_handler = None
        self.bad = []

    def fit(self, data: pd.DataFrame, y=None) -> "UnivariateOutliersHandler":
        """
        Detect bad features and fit the selected outlier handling strategy.

        :param data: Input DataFrame used to detect bad features and to fit the chosen handler.
        :type data: pd.DataFrame
        :param y: Ignored; present for sklearn compatibility.
        :type y: Optional[pd.Series]
        :return: self (fitted UnivariateOutliersHandler).
        :rtype: UnivariateOutliersHandler
        :raises ValueError: If an unsupported select_method is provided.
        :raises Exception: Propagates unexpected exceptions.
        """
        if self.deactivate:
            logging.info("UnivariateOutliersHandler is deactivated. Skipping fit.")
            return self

        try:
            _, self.bad = _feature_quality(
                data, id_col=self.id_col, activity_col=self.activity_col
            )
            if not self.bad:
                logging.info(
                    "UnivariateOutlierHandler: No bad features found. Skipping outlier handling."
                )
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
                logging.info(
                    f"UnivariateOutliersHandler: Using '{self.select_method}' method."
                )
            else:
                raise ValueError(f"Unsupported method: {self.select_method}")

            if self.save_method:
                if self.save_dir and not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir, exist_ok=True)
                with open(f"{self.save_dir}/uni_outlier_handler.pkl", "wb") as file:
                    pickle.dump(self, file)
                logging.info("UnivariateOutliersHandler saved successfully.")

        except Exception as e:
            logging.error(f"Error during fitting: {e}")
            raise

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted outlier handler to the detected bad features.

        Only the columns flagged in `self.bad` are transformed; the rest of the
        DataFrame is preserved. If the chosen handler returns a numpy array, it
        is coerced into a DataFrame with the original column names.

        :param data: DataFrame to transform.
        :type data: pd.DataFrame
        :return: Transformed DataFrame with outlier handling applied.
        :rtype: pd.DataFrame
        :raises NotFittedError: If the handler has not been fitted.
        :raises Exception: Propagates unexpected exceptions during transformation.
        """
        if self.deactivate:
            self.transformed_data = data
            logging.info(
                "UnivariateOutliersHandler is deactivated. Returning unmodified data."
            )
            return data

        try:
            transformed_data = deepcopy(data)
            if not self.bad:
                self.transformed_data = transformed_data
                logging.info(
                    "UnivariateOutlierHandler: No bad features to handle. Returning original data."
                )
                return transformed_data

            if self.uni_outlier_handler is None:
                raise NotFittedError(
                    "UnivariateOutlierHandler is not fitted yet. Call 'fit' before using this method."
                )

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
                logging.info(
                    f"UnivariateOutlierHandler: Transformed data saved at: {self.save_dir}/{csv_name}.csv"
                )

            self.transformed_data = transformed_data

            return transformed_data

        except Exception as e:
            logging.error(f"Error during transformation: {e}")
            raise

    def fit_transform(self, data: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit the handler and immediately transform the provided data.

        :param data: DataFrame to fit & transform.
        :type data: pd.DataFrame
        :param y: Ignored; present for sklearn compatibility.
        :type y: Optional[pd.Series]
        :return: Transformed DataFrame.
        :rtype: pd.DataFrame
        """
        if self.deactivate:
            logging.info(
                "UnivariateOutliersHandler is deactivated. Returning unmodified data."
            )
            return data

        self.fit(data)
        return self.transform(data)

    @staticmethod
    def compare_univariate_methods(
        data1: pd.DataFrame,
        data2: Optional[pd.DataFrame] = None,
        data1_name: str = "data1",
        data2_name: str = "data2",
        activity_col: Optional[str] = None,
        id_col: Optional[str] = None,
        methods_to_compare: List[str] = None,
        save_dir: Optional[str] = "Project/OutlierHandler",
    ) -> pd.DataFrame:
        """
        Compare a set of univariate outlier handling methods by applying each to `data1`
        and (optionally) `data2` and summarizing how many rows remain / are removed.

        :param data1: Primary DataFrame to evaluate methods on.
        :type data1: pd.DataFrame
        :param data2: Optional secondary DataFrame to evaluate with the same fitted handlers.
        :type data2: Optional[pd.DataFrame]
        :param data1_name: Label used for dataset1 in the output table.
        :type data1_name: str
        :param data2_name: Label used for dataset2 in the output table.
        :type data2_name: str
        :param activity_col: Optional activity/target column to exclude from detection.
        :type activity_col: Optional[str]
        :param id_col: Optional ID column to exclude from detection.
        :type id_col: Optional[str]
        :param methods_to_compare: List of method keys to compare. Defaults to all supported methods.
        :type methods_to_compare: List[str]
        :param save_dir: If provided, the comparison table CSV will be saved here.
        :type save_dir: Optional[str]
        :return: DataFrame summarizing for each method and dataset the row counts before/after handling.
        :rtype: pd.DataFrame
        :raises Exception: Propagates exceptions encountered during comparison.
        """
        try:
            comparison_data = []
            methods = [
                "iqr",
                "winsorization",
                "imputation",
                "power",
                "normal",
                "uniform",
            ]
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

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                comparison_table.to_csv((f"{save_dir}/compare_univariate_methods.csv"))
                logging.info(
                    f"Comparison table of univariate methods saved at: {save_dir}/compare_univariate_methods.csv"
                )

            logging.info("Comparison of univariate methods completed.")
            return comparison_table

        except Exception as e:
            logging.error(f"Error in comparison: {e}")
            raise
