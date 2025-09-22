import os
import pickle
import logging
import pandas as pd
from copy import deepcopy
from typing import Optional
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.base import BaseEstimator, TransformerMixin
from .univariate_outliers import _feature_quality


class KBinHandler(BaseEstimator, TransformerMixin):
    """
    Discretize features identified as univariate outliers using
    :class:`sklearn.preprocessing.KBinsDiscretizer`.

    This handler detects "bad" features via :func:`_feature_quality`,
    fits a KBinsDiscretizer, and replaces bad features with binned
    columns (`Kbin1`, `Kbin2`, ...).

    Typical usage::

        >>> kbin = KBinHandler(activity_col="activity", id_col="id", n_bins=3)
        >>> kbin.fit(df)
        >>> transformed = kbin.transform(df)

    :param activity_col: Name of the activity/target column (if present).
    :type activity_col: Optional[str]
    :param id_col: Name of the identifier column (if present).
    :type id_col: Optional[str]
    :param n_bins: Number of bins to produce. Default is 3.
    :type n_bins: int
    :param encode: Encoding strategy {"ordinal","onehot","onehot-dense"}.
                   Default is "ordinal".
    :type encode: str
    :param strategy: Binning strategy {"uniform","quantile","kmeans"}.
                     Default is "quantile".
    :type strategy: str
    :param save_method: If True, save fitted handler as pickle.
    :type save_method: bool
    :param save_dir: Directory to save pickled handler / CSV outputs.
                     Default is "Project/KBinHandler".
    :type save_dir: Optional[str]
    :param save_trans_data: If True, save transformed data to CSV.
    :type save_trans_data: bool
    :param trans_data_name: Base filename for saving transformed CSV.
                            Default is "trans_data".
    :type trans_data_name: str
    :param deactivate: If True, disable handler and return inputs unchanged.
    :type deactivate: bool

    :ivar kbin: Fitted :class:`KBinsDiscretizer` after :meth:`fit`, or ``None``.
    :vartype kbin: Optional[KBinsDiscretizer]
    :ivar bad: Names of detected univariate outlier features.
    :vartype bad: list[str]
    :ivar transformed_data: Stores the last transformed DataFrame.
    :vartype transformed_data: pandas.DataFrame
    """

    def __init__(
        self,
        activity_col: Optional[str] = None,
        id_col: Optional[str] = None,
        n_bins: int = 3,
        encode: str = "ordinal",
        strategy: str = "quantile",
        save_method: bool = False,
        save_dir: Optional[str] = "Project/KBinHandler",
        save_trans_data: bool = False,
        trans_data_name: str = "trans_data",
        deactivate: bool = False,
    ) -> None:
        self.activity_col = activity_col
        self.id_col = id_col
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.save_method = save_method
        self.save_dir = save_dir
        self.save_trans_data = save_trans_data
        self.trans_data_name = trans_data_name
        self.deactivate = deactivate
        self.kbin = None
        self.bad = []

    def fit(self, data: pd.DataFrame, y=None) -> "KBinHandler":
        """
        Detect univariate outliers and fit KBinsDiscretizer on them.

        Steps:
          1. Call :func:`_feature_quality` to detect "bad" features.
          2. If any, fit :class:`KBinsDiscretizer` on those columns.
          3. Optionally save fitted handler as pickle.

        :param data: Input DataFrame to fit on.
        :type data: pandas.DataFrame
        :param y: Ignored, present for sklearn compatibility.
        :type y: Any
        :return: Fitted handler (self).
        :rtype: KBinHandler
        :raises Exception: If fitting fails unexpectedly.
        """
        if self.deactivate:
            logging.info("KBinHandler is deactivated. Skipping fit.")
            return self

        try:
            _, self.bad = _feature_quality(
                data, id_col=self.id_col, activity_col=self.activity_col
            )

            if not self.bad:
                logging.info("KBinHandler: No univariate outliers found. Skipping.")
                return self

            self.kbin = KBinsDiscretizer(
                n_bins=self.n_bins, encode=self.encode, strategy=self.strategy
            ).fit(data[self.bad])

            if self.save_method:
                os.makedirs(self.save_dir, exist_ok=True)
                with open(f"{self.save_dir}/kbin_handler.pkl", "wb") as file:
                    pickle.dump(self, file)
                logging.info(f"KBinHandler saved at: {self.save_dir}/kbin_handler.pkl")

            return self

        except Exception as e:
            logging.error(f"Error fitting KBinHandler: {e}")
            raise

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted KBinsDiscretizer to detected bad features.

        - If deactivated → return input unchanged.
        - If no bad features detected → return input unchanged.
        - Otherwise → replace bad features with new columns
          ("Kbin1", "Kbin2", ...).

        :param data: Input DataFrame to transform.
        :type data: pandas.DataFrame
        :return: Transformed DataFrame with discretized columns.
        :rtype: pandas.DataFrame
        :raises Exception: If transformation fails unexpectedly.
        """
        if self.deactivate:
            self.transformed_data = data
            logging.info("KBinHandler is deactivated. Returning input unchanged.")
            return data

        try:
            transformed_data = deepcopy(data)

            if not self.bad or transformed_data[self.bad].empty:
                self.transformed_data = transformed_data
                logging.info("KBinHandler: No bad features. Returning input.")
                return transformed_data

            new_bad_data = pd.DataFrame(self.kbin.transform(transformed_data[self.bad]))
            new_bad_data.columns = [
                f"Kbin{i}" for i in range(1, len(new_bad_data.columns) + 1)
            ]
            transformed_data.drop(columns=self.bad, inplace=True)
            transformed_data = pd.concat([transformed_data, new_bad_data], axis=1)

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
                    f"KBinHandler: Transformed data saved at {self.save_dir}/{csv_name}.csv"
                )

            self.transformed_data = transformed_data
            return transformed_data

        except Exception as e:
            logging.error(f"Error transforming data in KBinHandler: {e}")
            raise

    def fit_transform(self, data: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit KBinsDiscretizer on bad features and transform in one call.

        :param data: Input DataFrame to fit and transform.
        :type data: pandas.DataFrame
        :param y: Ignored, present for sklearn compatibility.
        :type y: Any
        :return: Transformed DataFrame with discretized features.
        :rtype: pandas.DataFrame
        """
        if self.deactivate:
            logging.info("KBinHandler is deactivated. Returning input unchanged.")
            return data

        self.fit(data)
        return self.transform(data)
