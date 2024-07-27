import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    FunctionTransformer,
)
import pickle
import os


class Rescaler:
    """
    A class used to rescale data using different scaling methods.

    Attributes
    ----------
    id_col : str
        The column name representing the ID in the data.
    activity_col : str
        The column name representing the activity in the data.
    save_dir : str
        The directory where the scaler and column information will be saved.
    scaler_method : str
        The method used for scaling. Default is "MinMaxScaler".

    Methods
    -------
    fit(data: pd.DataFrame):
        Fits the scaler to the data.
    transform(data: pd.DataFrame, save_dir: str) -> pd.DataFrame:
        Transforms the data using the fitted scaler.
    fit_transform(data: pd.DataFrame) -> pd.DataFrame:
        Fits the scaler to the data and then transforms it.
    """

    def __init__(
        self,
        id_col: str,
        activity_col: str,
        save_dir: str,
        scaler_method: str = "MinMaxScaler",
    ):
        """
        Constructs all the necessary attributes for the Rescaler object.

        Parameters
        ----------
        id_col : str
            The column name representing the ID in the data.
        activity_col : str
            The column name representing the activity in the data.
        save_dir : str
            The directory where the scaler and column information will be saved.
        scaler_method : str, optional
            The method used for scaling (default is "MinMaxScaler").
        """
        self.id_col = id_col
        self.activity_col = activity_col
        self.scaler_method = scaler_method
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    @staticmethod
    def _get_scaler(scaler_method: str) -> object:
        """
        Returns the scaler object based on the scaler method provided.

        Parameters
        ----------
        scaler_method : str
            The method used for scaling.

        Returns
        -------
        object
            The scaler object.

        Raises
        ------
        ValueError
            If the scaler method is not supported.
        """
        scalers_dict = {
            "MinMaxScaler": MinMaxScaler(),
            "StandardScaler": StandardScaler(),
            "RobustScaler": RobustScaler(),
            "None": FunctionTransformer(lambda x: x),  # No operation scaler
        }

        scaler = scalers_dict.get(scaler_method)
        if scaler is None:
            raise ValueError(
                f"Unsupported scaler method {scaler_method}. Choose from"
                + "'MinMaxScaler', 'StandardScaler', 'RobustScaler', or 'None'."
            )

        return scaler

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fits the scaler to the data.

        Parameters
        ----------
        data : pd.DataFrame
            The data to fit the scaler to.
        """
        cols_to_exclude = [self.id_col, self.activity_col]
        temp_data = data.drop(columns=cols_to_exclude)
        non_binary_cols = [
            col
            for col in temp_data.columns
            if not temp_data[col].dropna().isin([0, 1]).all()
        ]

        if non_binary_cols:
            scaler = self._get_scaler(self.scaler_method)
            scaler.fit(data[non_binary_cols])
            with open(f"{self.save_dir}/scaler.pkl", "wb") as file:
                pickle.dump(scaler, file)
            with open(f"{self.save_dir}/non_binary_cols.pkl", "wb") as file:
                pickle.dump(non_binary_cols, file)

        # Mark as fitted
        with open(f"{self.save_dir}/fitted.pkl", "wb") as file:
            pickle.dump(True, file)

    @staticmethod
    def transform(data: pd.DataFrame, save_dir: str) -> pd.DataFrame:
        """
        Transforms the data using the fitted scaler.

        Parameters
        ----------
        data : pd.DataFrame
            The data to transform.
        save_dir : str
            The directory where the scaler and column information are saved.

        Returns
        -------
        pd.DataFrame
            The transformed data.

        Raises
        ------
        FileNotFoundError
            If the scaler has not been fitted.
        """
        if not os.path.exists(f"{save_dir}/fitted.pkl"):
            raise FileNotFoundError("Rescaler method must be fitted before transform.")

        non_binary_cols = []
        scaler = None
        rescaled_data = data.copy()
        if os.path.exists(f"{save_dir}/non_binary_cols.pkl"):
            with open(f"{save_dir}/non_binary_cols.pkl", "rb") as file:
                non_binary_cols = pickle.load(file)
            with open(f"{save_dir}/scaler.pkl", "rb") as file:
                scaler = pickle.load(file)
            rescaled_data[non_binary_cols] = scaler.transform(data[non_binary_cols])

        return rescaled_data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the scaler to the data and then transforms it.

        Parameters
        ----------
        data : pd.DataFrame
            The data to fit and transform.

        Returns
        -------
        pd.DataFrame
            The fitted and transformed data.
        """
        self.fit(data)
        return self.transform(data, self.save_dir)
