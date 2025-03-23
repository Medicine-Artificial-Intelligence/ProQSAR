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
    A class to perform rescaling (normalization or standardization) on numerical data columns.

    This class provides functionality for scaling data using different methods such as Min-Max Scaling,
    Standard Scaling, or Robust Scaling, and also supports saving the scaling method and transformed data.

    Attributes
    ----------
    id_col : Optional[str]
        The column name that contains the unique IDs to exclude from the scaling operation.
    activity_col : Optional[str]
        The column name that contains activity labels to exclude from the scaling operation.
    select_method : str
        The method used for scaling the data. Options are 'MinMaxScaler', 'StandardScaler', 'RobustScaler', or 'None'.
    save_method : bool
        Whether to save the fitted rescaler object after fitting.
    save_dir : Optional[str]
        Directory to save the fitted rescaler object. Default is "Project/Rescaler".
    save_trans_data : bool
        Whether to save the transformed data to a CSV file.
    trans_data_name : str
        The name of the transformed data file to save (if `save_trans_data` is True).
    non_binary_cols : Optional[list]
        List of columns that are non-binary and should be rescaled.
    deactivate : bool
        Flag to deactivate the process.
    rescaler : Optional[object]
        The rescaler object (e.g., MinMaxScaler) fitted to the data.

    Methods
    -------
    fit(data: pd.DataFrame) -> None
        Fits the rescaler to the data.
    transform(data: pd.DataFrame) -> pd.DataFrame
        Transforms the data using the fitted rescaler.
    fit_transform(data: pd.DataFrame) -> pd.DataFrame
        Fits the rescaler and transforms the data in one step.
    _get_scaler(select_method: str) -> object
        Returns the appropriate scaler object based on the provided rescaling method.
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
        """
        Initializes the Rescaler object.

        Parameters
        ----------
        activity_col : Optional[str]
            The column name containing activity labels to exclude from scaling.
        id_col : Optional[str]
            The column name containing unique identifiers to exclude from scaling.
        select_method : str, default 'MinMaxScaler'
            The rescaling method to use. Options are 'MinMaxScaler', 'StandardScaler', 'RobustScaler', or 'None'.
        save_method : bool, default True
            Whether to save the fitted rescaler model after fitting.
        save_dir : Optional[str], default 'Project/Rescaler'
            Directory where the fitted rescaler model will be saved.
        save_trans_data : bool, default False
            Whether to save the transformed data to a CSV file.
        trans_data_name : str, default 'rs_trans_data'
            The name for the transformed data file to be saved.
        deactivate : bool
            Flag to deactivate the process.
        """
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
        Returns the appropriate scaler object based on the provided scaling method.

        Parameters
        ----------
        select_method : str
            The method for scaling (e.g., 'MinMaxScaler', 'StandardScaler', 'RobustScaler', or 'None').

        Returns
        -------
        object
            The scaler object corresponding to the specified method.

        Raises
        ------
        ValueError
            If the provided scaling method is not supported.
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
        Fits the rescaler to the provided data, excluding columns specified by `id_col` and `activity_col`.

        Parameters
        ----------
        data : pd.DataFrame
            The data to fit the rescaler to.

        Returns
        -------
        Rescaler: The fitted Rescaler object.
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
                self.rescaler = self._get_scaler(self.select_method).fit(
                    data[self.non_binary_cols]
                )

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
        Transforms the provided data using the fitted rescaler.

        Parameters
        ----------
        data : pd.DataFrame
            The data to transform using the fitted rescaler.

        Returns
        -------
        pd.DataFrame
            The transformed data.
        """
        if self.deactivate:
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
                    "No non-binary columns found in the data. The data remains unchanged."
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
                    f"Transformed data saved at: {self.save_dir}/{csv_name}.csv"
                )

            return transformed_data

        except NotFittedError as e:
            logging.error(f"Error: {e}")
            raise

        except Exception as e:
            logging.error(f"Error during transforming the data: {e}")
            raise

    def fit_transform(self, data: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fits the rescaler to the data and then transforms it.

        Parameters
        ----------
        data : pd.DataFrame
            The data to fit and transform.

        Returns
        -------
        pd.DataFrame
            The fitted and transformed data.
        """
        if self.deactivate:
            logging.info("Rescaler is deactivated. Returning unmodified data.")
            return data

        self.fit(data)
        return self.transform(data)

    def setting(self, **kwargs):
        valid_keys = self.__dict__.keys()
        for key in kwargs:
            if key not in valid_keys:
                raise KeyError(f"'{key}' is not a valid attribute of Rescaler.")
        self.__dict__.update(**kwargs)

        return self
