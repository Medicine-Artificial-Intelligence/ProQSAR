import os
import pickle
import logging
import pandas as pd
from copy import deepcopy
from typing import Optional
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import BaseEstimator, TransformerMixin
from ProQSAR.Outlier.univariate_outliers import _feature_quality


class KBinHandler(BaseEstimator, TransformerMixin):
    """
    A handler for detecting and transforming univariate outliers in data using KBinsDiscretizer.

    This class identifies "bad" features (columns with univariate outliers) and applies a
    discretization transformation to them. It also allows for saving the transformation model
    and the transformed data.

    Attributes:
        activity_col (Optional[str]): The name of the activity column to exclude from handling.
        id_col (Optional[str]): The name of the ID column to exclude from handling.
        n_bins (int): The number of bins to use in KBinsDiscretizer.
        encode (str): The encoding method for transformed bins ('ordinal', 'onehot', or 'onehot-dense').
        strategy (str): The binning strategy ('uniform', 'quantile', or 'kmeans').
        save_method (bool): Whether to save the fitted model.
        save_dir (Optional[str]): Directory path where the model and data should be saved.
        save_trans_data (bool): Whether to save the transformed data to a CSV file.
        trans_data_name (str): The name of the CSV file for transformed data.
        deactivate (bool): Flag to deactivate the process.
        kbin (Optional[KBinsDiscretizer]): The fitted KBinsDiscretizer object.
        bad (list[str]): List of columns identified as having outliers.
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
        Fit the KBinsDiscretizer to features with univariate outliers.

        Args:
            data (pd.DataFrame): The dataset to fit the model to.

        Returns:
            KBinHandler: Returns self for chaining.
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
                    "No bad features (univariate outliers) found. Skipping KBin handling."
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
                logging.info("KBin handler saved successfully.")

            logging.info("Model fitting complete.")
            return self

        except Exception as e:
            logging.error(f"Error in fitting: {e}")
            raise

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the discretization transformation to the data.

        Args:
            data (pd.DataFrame): The dataset to transform.

        Returns:
            pd.DataFrame: The transformed dataset with "bad" features discretized.
        """
        if self.deactivate:
            logging.info("KBinHandler is deactivated. Returning unmodified data.")
            return data

        try:
            transformed_data = deepcopy(data)
            if not self.bad or transformed_data[self.bad].empty:
                logging.info(
                    "No bad features (outliers) to handle. Returning original data."
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
                    f"Transformed data saved at: {self.save_dir}/{csv_name}.csv"
                )
            return transformed_data

        except Exception as e:
            logging.error(f"Error in transforming the data: {e}")
            raise

    def fit_transform(self, data: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit the KBinsDiscretizer to the data and transform it in a single step.

        Args:
            data (pd.DataFrame): The dataset to fit and transform.

        Returns:
            pd.DataFrame: The transformed dataset with "bad" features discretized.
        """
        if self.deactivate:
            logging.info("KBinHandler is deactivated. Returning unmodified data.")
            return data

        self.fit(data)
        return self.transform(data)

    def setting(self, **kwargs):
        valid_keys = self.__dict__.keys()
        for key in kwargs:
            if key not in valid_keys:
                raise KeyError(f"'{key}' is not a valid attribute of KBinHandler.")
        self.__dict__.update(**kwargs)

        return self
