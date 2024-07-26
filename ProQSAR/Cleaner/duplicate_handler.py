import pandas as pd
import pickle
from typing import Optional


class DuplicateHandler:
    def __init__(
        self,
        id_col: str,
        activity_col: str,
        save_dir: Optional[str] = None,
    ):
        """
        Initializes the DuplicateHandler with the necessary configuration.

        Parameters:
        - id_col (str): The name of the column to be used as the identifier.
        - activity_col (str): The name of the column to be used for activity tracking.
        - save_dir (Optional[str]): Directory to save the configuration; None if not saving.
        """
        self.id_col = id_col
        self.activity_col = activity_col
        self.save_dir = save_dir

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fits the duplicate handler by identifying duplicated columns.

        Parameters:
        - data (pd.DataFrame): The data on which to fit the handler.
        """
        cols_to_exclude = [self.id_col, self.activity_col]
        temp_data = data.drop(columns=cols_to_exclude)
        dup_cols = temp_data.columns[temp_data.T.duplicated()].tolist()

        if self.save_dir:
            with open(f"{self.save_dir}/cols_to_exclude.pkl", "wb") as file:
                pickle.dump(cols_to_exclude, file)
            with open(f"{self.save_dir}/dup_cols.pkl", "wb") as file:
                pickle.dump(dup_cols, file)

    @staticmethod
    def transform(data: pd.DataFrame, save_dir: str) -> pd.DataFrame:
        """
        Transforms the provided DataFrame by removing duplicate rows and columns.

        Parameters:
        - data (pd.DataFrame): The data to transform.
        - save_dir (str): Directory where the configuration is saved.

        Returns:
        - pd.DataFrame: The transformed DataFrame with duplicates removed.
        """
        # Load necessary objects
        with open(f"{save_dir}/cols_to_exclude.pkl", "rb") as file:
            cols_to_exclude = pickle.load(file)
        with open(f"{save_dir}/dup_cols.pkl", "rb") as file:
            dup_cols = pickle.load(file)

        # Drop duplicated rows & columns
        temp_data = data.drop(columns=cols_to_exclude)
        dup_rows = temp_data.duplicated()
        data = data[~dup_rows].reset_index(drop=True)
        data = data.drop(columns=dup_cols)

        return data

    def fit_transform(self, data: pd.DataFrame, save_dir: str) -> pd.DataFrame:
        """
        Fits the handler and then transforms the data.

        Parameters:
        - data (pd.DataFrame): The data to fit and transform.
        - save_dir (str): Directory where the configuration is saved.

        Returns:
        - pd.DataFrame: The transformed DataFrame with duplicates removed.
        """
        self.fit(data)
        return self.transform(data, save_dir)
