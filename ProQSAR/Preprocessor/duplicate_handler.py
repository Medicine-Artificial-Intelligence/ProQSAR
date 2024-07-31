import pandas as pd
import pickle
import os


class DuplicateHandler:
    def __init__(
        self, 
        id_col: str, 
        activity_col: str, 
        save_dir: str = "Project/DuplicateHandler"
    ):
        """
        Initializes the DuplicateHandler with the necessary configuration.

        Parameters:
        - id_col (str): The name of the column to be used as the identifier.
        - activity_col (str): The name of the column to be used for activity tracking.
        - save_dir (str): Directory to save the configuration.
        """
        self.id_col = id_col
        self.activity_col = activity_col
        self.save_dir = save_dir
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fits the duplicate handler by identifying duplicated columns.

        Parameters:
        - data (pd.DataFrame): The data on which to fit the handler.
        """
        cols_to_exclude = [self.id_col, self.activity_col]
        temp_data = data.drop(columns=cols_to_exclude)
        dup_cols = temp_data.columns[temp_data.T.duplicated()].tolist()

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
        if os.path.exists(f"{save_dir}/cols_to_exclude.pkl"):
            with open(f"{save_dir}/cols_to_exclude.pkl", "rb") as file:
                cols_to_exclude = pickle.load(file)
            with open(f"{save_dir}/dup_cols.pkl", "rb") as file:
                dup_cols = pickle.load(file)
        else:
            raise FileNotFoundError(
                "DuplicatedHandler must be fitted before transform."
            )

        # Drop duplicated rows & columns
        temp_data = data.drop(columns=cols_to_exclude)
        dup_rows = temp_data.index[temp_data.duplicated()].tolist()
        data = data.drop(index=dup_rows, columns=dup_cols)

        return data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the handler and then transforms the data.

        Parameters:
        - data (pd.DataFrame): The data to fit and transform.
        - save_dir (str): Directory where the configuration is saved.

        Returns:
        - pd.DataFrame: The transformed DataFrame with duplicates removed.
        """
        self.fit(data)
        return self.transform(data, self.save_dir)
