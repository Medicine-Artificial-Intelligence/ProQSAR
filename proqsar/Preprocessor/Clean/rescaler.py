import pandas as pd
from typing import Optional
from copy import deepcopy
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
)
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import os
import logging


class Rescaler(BaseEstimator, TransformerMixin):
    """
    Rescale (normalize or standardize) numerical columns in a dataset.

    This class provides scaling methods such as Min-Max scaling,
    Standard scaling, and Robust scaling. It excludes identifier and
    activity columns, automatically detects non-binary columns for
    scaling, and optionally saves both the fitted scaler and
    transformed data.

    :param activity_col: Column name containing activity labels to exclude from scaling.
    :type activity_col: Optional[str]
    :param id_col: Column name containing unique identifiers to exclude from scaling.
    :type id_col: Optional[str]
    :param select_method: Scaling method to use. Options are
                          ``"MinMaxScaler"``, ``"StandardScaler"``, ``"RobustScaler"``, or ``"None"``.
                          Default is ``"MinMaxScaler"``.
    :type select_method: str
    :param save_method: Whether to save the fitted rescaler model after fitting. Default is ``False``.
    :type save_method: bool
    :param save_dir: Directory where the rescaler model and transformed data will be saved.
                     Default is ``"Project/Rescaler"``.
    :type save_dir: Optional[str]
    :param save_trans_data: Whether to save the transformed data as a CSV file. Default is ``False``.
    :type save_trans_data: bool
    :param trans_data_name: Base name for the transformed data file. Default is ``"trans_data"``.
    :type trans_data_name: str
    :param deactivate: If True, disables scaling and returns unmodified data. Default is ``False``.
    :type deactivate: bool

    **Example**

    .. code-block:: python

        import pandas as pd
        from proqsar.Preprocessor.rescaler import Rescaler

        df = pd.DataFrame({
            "id": [1, 2, 3],
            "feature1": [0.1, 0.5, 0.9],
            "feature2": [10, 20, 30],
            "activity": [1.2, 3.4, 2.1]
        })

        rescaler = Rescaler(activity_col="activity", id_col="id", select_method="StandardScaler")
        df_scaled = rescaler.fit_transform(df)

        print(df_scaled)
    """

    def __init__(
        self,
        activity_col: Optional[str] = None,
        id_col: Optional[str] = None,
        select_method: str = "MinMaxScaler",
        save_method: bool = False,
        save_dir: Optional[str] = "Project/Rescaler",
        save_trans_data: bool = False,
        trans_data_name: str = "trans_data",
        deactivate: bool = False,
    ):
        self.id_col = id_col
        self.activity_col = activity_col
        self.select_method = select_method
        self.save_method = save_method
        self.save_dir = save_dir
        self.save_trans_data = save_trans_data
        self.trans_data_name = trans_data_name
        self.deactivate = deactivate
        self.non_binary_cols = None
        self.rescaler = None
        self.fitted = False

    @staticmethod
    def _get_scaler(select_method: str) -> object:
        """
        Get a scaler object based on the selected method.

        :param select_method: Scaling method (``"MinMaxScaler"``, ``"StandardScaler"``, ``"RobustScaler"``).
        :type select_method: str

        :return: The scaler object corresponding to the method.
        :rtype: object

        :raises ValueError: If the provided method is unsupported.
        """
        rescalers_dict = {
            "MinMaxScaler": MinMaxScaler(),
            "StandardScaler": StandardScaler(),
            "RobustScaler": RobustScaler(),
        }

        try:
            rescaler = rescalers_dict[select_method]
        except KeyError:
            raise ValueError(
                f"Unsupported select_method {select_method}. Choose from "
                + "'MinMaxScaler', 'StandardScaler', 'RobustScaler'."
            )
        return rescaler

    def fit(self, data: pd.DataFrame, y=None) -> "Rescaler":
        """
        Fit the rescaler on the dataset.

        Non-binary columns (not exclusively 0/1) are detected and used
        for fitting the scaler.

        :param data: Dataset to fit on.
        :type data: pd.DataFrame
        :param y: Ignored, included for compatibility with scikit-learn pipelines.
        :type y: any, optional

        :return: Fitted Rescaler object.
        :rtype: Rescaler

        :raises Exception: If an error occurs during fitting.
        """
        if self.deactivate:
            logging.info("Rescaler is deactivated. Skipping fit.")
            return self

        try:
            temp_data = data.drop(columns=[self.id_col, self.activity_col])
            self.non_binary_cols = [
                col
                for col in temp_data.columns
                if not temp_data[col].dropna().isin([0, 1]).all()
            ]

            if self.non_binary_cols:
                logging.info(f"Rescaler: Using '{self.select_method}' method.")
                self.rescaler = self._get_scaler(self.select_method).fit(
                    data[self.non_binary_cols]
                )
            else:
                logging.info("Rescaler: No non-binary columns to rescale.")

            self.fitted = True

            if self.save_method:
                if self.save_dir and not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir, exist_ok=True)
                with open(f"{self.save_dir}/rescaler.pkl", "wb") as file:
                    pickle.dump(self, file)
                logging.info(f"Rescaler model saved to {self.save_dir}/rescaler.pkl")

        except Exception as e:
            logging.error(f"Error during fitting the rescaler: {e}")
            raise

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataset using the fitted rescaler.

        :param data: Dataset to transform.
        :type data: pd.DataFrame

        :return: Transformed dataset.
        :rtype: pd.DataFrame

        :raises NotFittedError: If the rescaler has not been fitted yet.
        :raises Exception: If an error occurs during transformation.
        """
        if self.deactivate:
            self.transformed_data = data
            logging.info("Rescaler is deactivated. Returning unmodified data.")
            return data

        try:
            if not self.fitted:
                raise NotFittedError(
                    "Rescaler is not fitted yet. Call 'fit' before using this model."
                )

            transformed_data = deepcopy(data)
            if not self.non_binary_cols:
                logging.info(
                    "Rescaler: No non-binary columns to scale. Returning unchanged data."
                )
            else:
                transformed_data[self.non_binary_cols] = self.rescaler.transform(
                    data[self.non_binary_cols]
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
                    f"Rescaler: Transformed data saved at {self.save_dir}/{csv_name}.csv"
                )

            self.transformed_data = transformed_data
            return transformed_data

        except NotFittedError as e:
            logging.error(f"Error: {e}")
            raise
        except Exception as e:
            logging.error(f"Error during transforming the data: {e}")
            raise

    def fit_transform(self, data: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        :param data: Dataset to fit and transform.
        :type data: pd.DataFrame
        :param y: Ignored, included for compatibility with scikit-learn pipelines.
        :type y: any, optional

        :return: Transformed dataset.
        :rtype: pd.DataFrame
        """
        if self.deactivate:
            logging.info("Rescaler is deactivated. Returning unmodified data.")
            return data

        self.fit(data)
        return self.transform(data)

    def setting(self, **kwargs):
        """
        Update settings of the Rescaler object.

        :param kwargs: Keyword arguments mapping attribute names to new values.
        :type kwargs: dict

        :return: Updated Rescaler object.
        :rtype: Rescaler

        :raises KeyError: If a provided key is not a valid attribute of Rescaler.
        """
        valid_keys = self.__dict__.keys()
        for key in kwargs:
            if key not in valid_keys:
                raise KeyError(f"'{key}' is not a valid attribute of Rescaler.")
        self.__dict__.update(**kwargs)
        return self
