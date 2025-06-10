import pandas as pd
import pickle
import os
import logging
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin


class DuplicateHandler(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        activity_col: Optional[str] = None,
        id_col: Optional[str] = None,
        cols: bool = True,
        rows: bool = True,
        save_method: bool = False,
        save_dir: str = "Project/DuplicateHandler",
        save_trans_data: bool = False,
        trans_data_name: str = "trans_data",
        deactivate: bool = False,
    ):
        """
        Initializes the DuplicateHandler with the necessary configuration.

        Parameters:
        - activity_col (str): The name of the column to be used for activity tracking.
        - id_col (str): The name of the column to be used as the identifier.
        - save_method (bool): Whether to save the fitted duplicate data handler.
        - save_dir (str): Directory to save the configuration.
        - save_trans_data (bool): Whether to save the transformed data.
        - trans_data_name (str): File name for saved transformed data.
        - deactivate (bool): Flag to deactivate the process.
        """
        self.id_col = id_col
        self.activity_col = activity_col
        self.cols = cols
        self.rows = rows
        self.save_method = save_method
        self.save_dir = save_dir
        self.save_trans_data = save_trans_data
        self.trans_data_name = trans_data_name
        self.deactivate = deactivate
        self.dup_cols = None

    def fit(self, data: pd.DataFrame, y=None) -> "DuplicateHandler":
        """
        Fits the duplicate handler by identifying duplicated columns.

        Parameters:
        - data (pd.DataFrame): The data on which to fit the handler.
        """
        if self.deactivate:
            logging.info("DuplicateHandler is deactivated. Skipping fit.")
            return self

        try:
            temp_data = data.drop(
                columns=[self.id_col, self.activity_col], errors="ignore"
            )
            self.dup_cols = temp_data.columns[temp_data.T.duplicated()].tolist()
            logging.info(
                f"DuplicateHandler: Identified duplicate columns: {self.dup_cols}"
            )

            if self.save_method:
                if self.save_dir and not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir, exist_ok=True)
                with open(f"{self.save_dir}/duplicate_handler.pkl", "wb") as file:
                    pickle.dump(self, file)
                logging.info(
                    f"DuplicateHandler saved at: {self.save_dir}/duplicate_handler.pkl"
                )

        except Exception as e:
            logging.error(f"An error occurred while fitting: {e}")
            raise

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the provided DataFrame by removing duplicate rows and columns.

        Parameters:
        - data (pd.DataFrame): The data to transform.

        Returns:
        - pd.DataFrame: The transformed DataFrame with duplicates removed.
        """
        if self.deactivate:
            self.transformed_data = data
            logging.info("DuplicateHandler is deactivated. Returning unmodified data.")
            return data

        try:
            temp_data = data.drop(
                columns=[self.id_col, self.activity_col], errors="ignore"
            )
            if not self.cols:
                self.dup_cols = []

            if not self.rows:
                dup_rows = []
            else:
                dup_rows = temp_data.index[temp_data.duplicated()].tolist()

            transformed_data = data.drop(index=dup_rows, columns=self.dup_cols)
            transformed_data.reset_index(drop=True, inplace=True)

            logging.info(
                f"DuplicateHandler: Dropped duplicate row {dup_rows} &  columns {self.dup_cols}"
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
                    f"DuplicateHandler: Transformed data saved at: {self.save_dir}/{csv_name}.csv"
                )

            self.transformed_data = transformed_data

        except KeyError as e:
            logging.error(f"Column missing in the dataframe: {e}")
            raise ValueError(f"Column {e} not found in the dataframe.")

        except Exception as e:
            logging.error(f"An error occurred while transforming the data: {e}")
            raise

        return transformed_data

    def fit_transform(self, data: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fits the handler and then transforms the data.

        Parameters:
        - data (pd.DataFrame): The data to fit and transform.

        Returns:
        - pd.DataFrame: The transformed DataFrame with duplicates removed.
        """
        if self.deactivate:
            logging.info("DuplicateHandler is deactivated. Returning unmodified data.")
            return data

        self.fit(data)
        return self.transform(data)
