import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    FunctionTransformer,
)
from typing import Optional, List
import pickle
import os


class Rescale:
    """
    A class for normalizing or transforming the range of data features using various scaling methods.

    Parameters:
    -----------
    data_train : pandas.DataFrame
        Training data.
    id_col : str
        Identifier column name.
    activity_col : str
        Name of the activity column, such as pIC50 or pChEMBL Value.
    scaler_method : str
        Method for scaling ('MinMaxScaler', 'StandardScaler', 'RobustScaler', or 'None' for no scaling).
    save_dir : Optional[str]
        Directory to save the scaler objects. If None, scalers will not be saved.

    Methods:
    --------
    fit()
        Fits the scaler to the training data based on non-binary float columns.
    transform(data: pandas.DataFrame) -> pandas.DataFrame
        Transforms the data using the previously fitted scaler.
    """

    def __init__(
        self,
        data_train: pd.DataFrame,
        id_col: str,
        activity_col: str,
        scaler_method: str = "MinMaxScaler",
        save_dir: Optional[str] = None,
    ) -> None:
        """
        Initializes the Rescale object with training data and configuration for scaling.

        Parameters:
        -----------
        data_train : pandas.DataFrame
            The training data.
        id_col : str
            The column name in `data_train` that identifies each entry.
        activity_col : str
            The column in `data_train` that contains the target or activity data to be excluded from scaling.
        scaler_method : str, optional
            The scaling technique to be used. Default is 'MinMaxScaler'.
        save_dir : Optional[str], optional
            The directory to save the scaler objects. Default is None.
        """
        self.data_train = data_train
        self.id_col = id_col
        self.activity_col = activity_col
        self.scaler_method = scaler_method
        self.save_dir = save_dir
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    @staticmethod
    def _select_scaler(scaler_method: str):
        """
        Selects the appropriate scaler based on the method specified.

        Parameters:
        -----------
        scaler_method : str
            The method of scaling to use.

        Returns:
        --------
        A scaler instance from sklearn.preprocessing, or a FunctionTransformer
        if no scaling is selected.
        """
        scalers = {
            "MinMaxScaler": MinMaxScaler(),
            "StandardScaler": StandardScaler(),
            "RobustScaler": RobustScaler(),
            "None": FunctionTransformer(lambda x: x),  # No operation scaler
        }
        scaler = scalers.get(scaler_method)
        if scaler is None:
            raise ValueError(
                f"Unsupported scaler method {scaler_method}. Choose from"
                + "'MinMaxScaler', 'StandardScaler', 'RobustScaler', or 'None'."
            )
        return scaler

    def fit(self) -> None:
        """
        Fits the scaler to the non-binary float columns in the training data.
        """
        self.non_binary_float_cols = [
            col
            for col in self.data_train.columns
            if self.data_train[col].dtype == float
            and self.data_train[col].nunique() != 2
        ]

        if self.non_binary_float_cols:
            self.scaler = self._select_scaler(self.scaler_method)
            self.scaler.fit(self.data_train[self.non_binary_float_cols])

            if self.save_dir:
                with open(f"{self.save_dir}/scaler.pkl", "wb") as file:
                    pickle.dump(self.scaler, file)
                with open(f"{self.save_dir}/cols.pkl", "wb") as file:
                    pickle.dump(self.non_binary_float_cols, file)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data using the previously fitted scaler.

        Parameters:
        -----------
        data : pandas.DataFrame
            The data to be transformed.

        Returns:
        --------
        pandas.DataFrame
            The transformed data with scaled non-binary float columns.
        """
        if not self.save_dir or not os.path.exists(f"{self.save_dir}/scaler.pkl"):
            raise FileNotFoundError(
                "Scaler not found. Ensure the scaler has been fitted and saved correctly."
            )

        with open(f"{self.save_dir}/scaler.pkl", "rb") as file:
            self.scaler = pickle.load(file)
        with open(f"{self.save_dir}/cols.pkl", "rb") as file:
            self.non_binary_float_cols = pickle.load(file)

        if self.non_binary_float_cols:
            data[self.non_binary_float_cols] = self.scaler.transform(
                data[self.non_binary_float_cols]
            )

        return data
