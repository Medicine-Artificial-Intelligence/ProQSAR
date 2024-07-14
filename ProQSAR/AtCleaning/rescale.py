import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    FunctionTransformer,
)


class Rescale:
    """
    Rescale class for normalizing or transforming the range of data features.

    Attributes:
    -----------
    data_train : pandas.DataFrame
        Training data.
    data_test : pandas.DataFrame
        Testing data.
    id_col : str
        Identifier column name.
    activity_col : str
        Name of the activity column (e.g., pIC50, pChEMBL Value).
    scaler_method : str
        Method for scaling ('MinMaxScaler', 'StandardScaler', 'RobustScaler', or None for no scaling).

    Methods:
    --------
    fit()
        Applies the chosen scaling method to the data.
    """

    def __init__(
        self,
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        id_col: str,
        activity_col: str,
        scaler_method: str = "MinMaxScaler",
    ) -> None:
        self.data_train = data_train
        self.data_test = data_test
        self.id_col = id_col
        self.activity_col = activity_col
        self.scaler_method = scaler_method
        self.scaler = self._select_scaler()

    def _select_scaler(self):
        """
        Selects and returns the appropriate scaler based on the scaler_method attribute.
        """
        scalers = {
            "MinMaxScaler": MinMaxScaler(),
            "StandardScaler": StandardScaler(),
            "RobustScaler": RobustScaler(),
            "None": FunctionTransformer(lambda x: x),  # No scaling
        }
        return scalers.get(self.scaler_method, FunctionTransformer(lambda x: x))

    def _scale_data(self, data: pd.DataFrame, columns_to_scale: list) -> pd.DataFrame:
        """
        Scales the specified columns of the given DataFrame and returns the scaled DataFrame.
        """
        # Extract the columns to be scaled
        df_to_scale = data[columns_to_scale]

        # Apply scaling
        df_scaled = pd.DataFrame(
            self.scaler.transform(df_to_scale),
            columns=columns_to_scale,
            index=df_to_scale.index,
        )

        # Concatenate scaled columns with the rest of the data
        df_other = data.drop(columns=columns_to_scale)
        return pd.concat([df_other, df_scaled], axis=1)

    def fit(self):
        """
        Fits the scaler to the training data and applies it to both training and testing data.
        """
        # Identify non-binary float columns in the training data
        non_binary_float_cols = [
            col
            for col in self.data_train.columns
            if self.data_train[col].dtype == float
            and self.data_train[col].nunique() != 2
        ]

        # Fit the scaler only on non-binary float columns from the training data
        if non_binary_float_cols:
            self.scaler.fit(self.data_train[non_binary_float_cols])

            # Scale the specified columns in both training and testing data
            self.data_train = self._scale_data(self.data_train, non_binary_float_cols)

            # Ensure that the test data has the same columns for scaling
            test_cols_to_scale = [
                col for col in non_binary_float_cols if col in self.data_test.columns
            ]
            self.data_test = self._scale_data(self.data_test, test_cols_to_scale)

        return self.data_train, self.data_test
