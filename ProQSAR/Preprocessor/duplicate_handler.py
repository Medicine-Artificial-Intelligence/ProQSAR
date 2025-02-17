import pandas as pd
import pickle
import os
import logging
from typing import Optional


class DuplicateHandler:
    def __init__(
        self,
        id_col: Optional[str] = None,
        activity_col: Optional[str] = None,
        save_method: bool = False,
        save_dir: str = "Project/DuplicateHandler",
        save_trans_data: bool = False,
        trans_data_name: str = "dh_trans_data",
    ):
        """
        Initializes the DuplicateHandler with the necessary configuration.

        Parameters:
        - id_col (str): The name of the column to be used as the identifier.
        - activity_col (str): The name of the column to be used for activity tracking.
        - save_method (bool): Whether to save the fitted duplicate data handler.
        - save_dir (str): Directory to save the configuration.
        - save_trans_data (bool): Whether to save the transformed data.
        - trans_data_name (str): File name for saved transformed data.
        """
        self.id_col = id_col
        self.activity_col = activity_col
        self.save_method = save_method
        self.save_dir = save_dir
        self.save_trans_data = save_trans_data
        self.trans_data_name = trans_data_name
        self.dup_cols = None

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fits the duplicate handler by identifying duplicated columns.

        Parameters:
        - data (pd.DataFrame): The data on which to fit the handler.
        """
        try:
            logging.info("Fitting DuplicateHandler model...")
            temp_data = data.drop(
                columns=[self.id_col, self.activity_col], errors="ignore"
            )
            self.dup_cols = temp_data.columns[temp_data.T.duplicated()].tolist()
            logging.info(f"Identified duplicate columns: {self.dup_cols}")

            if self.save_method:
                if self.save_dir and not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir, exist_ok=True)
                with open(f"{self.save_dir}/duplicate_handler.pkl", "wb") as file:
                    pickle.dump(self, file)
                logging.info(
                    f"DuplicateHandler model saved at: {self.save_dir}/duplicate_handler.pkl"
                )

        except Exception as e:
            logging.error(f"An error occurred while fitting the model: {e}")
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
        try:
            logging.info("Transforming data to remove duplicates...")
            temp_data = data.drop(
                columns=[self.id_col, self.activity_col], errors="ignore"
            )
            dup_rows = temp_data.index[temp_data.duplicated()].tolist()
            transformed_data = data.drop(index=dup_rows, columns=self.dup_cols)

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

        except KeyError as e:
            logging.error(f"Column missing in the dataframe: {e}")
            raise ValueError(f"Column {e} not found in the dataframe.")

        except Exception as e:
            logging.error(f"An error occurred while transforming the data: {e}")
            raise

        return transformed_data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the handler and then transforms the data.

        Parameters:
        - data (pd.DataFrame): The data to fit and transform.

        Returns:
        - pd.DataFrame: The transformed DataFrame with duplicates removed.
        """
        self.fit(data)
        return self.transform(data)
