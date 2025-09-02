import os
import pickle
import logging
import pandas as pd
from copy import deepcopy
from typing import Optional
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.base import BaseEstimator, TransformerMixin
from ProQSAR.Outlier.univariate_outliers import _feature_quality


class KBinHandler(BaseEstimator, TransformerMixin):
    """
    Handler that discretizes features identified as univariate outliers using
    KBinsDiscretizer and appends the resulting binned columns to the dataset.

    Typical usage:
        kbin = KBinHandler(activity_col="activity", id_col="id", n_bins=3)
        kbin.fit(df)
        transformed = kbin.transform(df)

    Parameters
    ----------
    activity_col : Optional[str]
        Name of the activity/target column (if present) used by `_feature_quality`.
    id_col : Optional[str]
        Name of the id column used by `_feature_quality`.
    n_bins : int
        Number of bins to produce (passed to KBinsDiscretizer).
    encode : str
        Encoding strategy for KBinsDiscretizer (default "ordinal").
    strategy : str
        Binning strategy for KBinsDiscretizer (default "quantile").
    save_method : bool
        If True, save the fitted KBinHandler object (pickle) to `save_dir`.
    save_dir : Optional[str]
        Directory to store saved objects / transformed data (default "Project/KBinHandler").
    save_trans_data : bool
        If True, save the transformed DataFrame to CSV.
    trans_data_name : str
        Base filename for transformed CSV (default "trans_data").
    deactivate : bool
        If True, the handler is deactivated and fit/transform become no-ops.

    Attributes
    ----------
    kbin : Optional[KBinsDiscretizer]
        Fitted KBinsDiscretizer after calling `fit`, or None if not applicable.
    bad : list
        Names of features identified as "bad" (univariate outliers) to be discretized.
    transformed_data : pd.DataFrame
        Stores the last transformed DataFrame after transform() is called.
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
        """
        Initialize the KBinHandler with the specified configuration.
        """
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
        Fit the KBinsDiscretizer to features identified as "bad" (univariate outliers).

        Procedure:
        - Use `_feature_quality` to identify bad features. `_feature_quality`
          is expected to return a tuple where the second element is a list of bad
          feature names.
        - If bad features exist, instantiate and fit KBinsDiscretizer on those
          columns.
        - Optionally save the fitted handler as a pickle.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame used to detect bad features and to fit the discretizer.
        y : ignored
            Present for sklearn compatibility.

        Returns
        -------
        KBinHandler
            The fitted handler (self).

        Raises
        ------
        Exception
            Unexpected exceptions are logged and re-raised.
        """
        if self.deactivate:
            logging.info("KBinHandler is deactivated. Skipping fit.")
            return self

        try:
            _, self.bad = _feature_quality(
                data, id_col=self.id_col, activity_col=self.activity_col
            )

            if not self.bad:
                logging.info(
                    "KBinHandler: No bad features (univariate outliers) found. Skipping KBin handling."
                )
                return self

            if self.bad:
                self.kbin = KBinsDiscretizer(
                    n_bins=self.n_bins, encode=self.encode, strategy=self.strategy
                ).fit(data[self.bad])

            if self.save_method:
                if self.save_dir and not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir, exist_ok=True)
                with open(f"{self.save_dir}/kbin_handler.pkl", "wb") as file:
                    pickle.dump(self, file)
                logging.info(f"KBin handler saved at: {self.save_dir}/kbin_handler.pkl")

            return self

        except Exception as e:
            logging.error(f"Error in fitting: {e}")
            raise

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the provided DataFrame by discretizing the previously-detected
        bad features and appending the resulting 'Kbin' columns.

        Procedure:
        - If deactivated, return the input unchanged.
        - If no bad features were detected during fit, return the original data.
        - Otherwise, apply the fitted KBinsDiscretizer and replace the original
          bad columns with new columns named "Kbin1", "Kbin2", ...

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with Kbin columns. The result is also stored in
            `self.transformed_data`.

        Raises
        ------
        Exception
            Unexpected exceptions are logged and re-raised.
        """
        if self.deactivate:
            self.transformed_data = data
            logging.info("KBinHandler is deactivated. Returning unmodified data.")
            return data

        try:
            transformed_data = deepcopy(data)
            if not self.bad or transformed_data[self.bad].empty:
                self.transformed_data = transformed_data
                logging.info(
                    "KBinHandler: No bad features (outliers) to handle. Returning original data."
                )
                return transformed_data

            new_bad_data = pd.DataFrame(self.kbin.transform(transformed_data[self.bad]))
            new_bad_data.columns = [
                "Kbin" + str(i) for i in range(1, len(new_bad_data.columns) + 1)
            ]
            transformed_data.drop(columns=self.bad, inplace=True)
            transformed_data = pd.concat([transformed_data, new_bad_data], axis=1)

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
                    f"KBinHandler: Transformed data saved at: {self.save_dir}/{csv_name}.csv"
                )

            self.transformed_data = transformed_data

            return transformed_data

        except Exception as e:
            logging.error(f"Error in transforming the data: {e}")
            raise

    def fit_transform(self, data: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Convenience method for fitting and transforming in a single call.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame.
        y : ignored
            Present for sklearn compatibility.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame after fitting and applying the discretizer.
        """
        if self.deactivate:
            logging.info("KBinHandler is deactivated. Returning unmodified data.")
            return data

        self.fit(data)
        return self.transform(data)
