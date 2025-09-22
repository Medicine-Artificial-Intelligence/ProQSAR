import pandas as pd
import pickle
import os
import logging
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin


class DuplicateHandler(BaseEstimator, TransformerMixin):
    """
    A preprocessing transformer to detect and remove duplicate
    columns and rows in a pandas DataFrame.

    Supports saving the fitted handler and transformed data for reproducibility.
    """

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
        Initialize the DuplicateHandler with configuration.

        :param activity_col: Column name for the activity or target variable.
        :type activity_col: Optional[str]
        :param id_col: Column name for the identifier column.
        :type id_col: Optional[str]
        :param cols: Whether to remove duplicate columns.
        :type cols: bool
        :param rows: Whether to remove duplicate rows.
        :type rows: bool
        :param save_method: Save the fitted DuplicateHandler object to disk if True.
        :type save_method: bool
        :param save_dir: Directory to save the handler and transformed data.
        :type save_dir: str
        :param save_trans_data: Save transformed data as CSV if True.
        :type save_trans_data: bool
        :param trans_data_name: Base filename for saved transformed data.
        :type trans_data_name: str
        :param deactivate: If True, the transformer is a no-op and returns input unchanged.
        :type deactivate: bool
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
        Fit the handler by identifying duplicate columns.

        :param data: Input DataFrame to inspect for duplicate columns.
        :type data: pandas.DataFrame
        :param y: Ignored. Present for sklearn compatibility.
        :type y: Optional[pandas.Series]
        :return: The fitted DuplicateHandler instance.
        :rtype: DuplicateHandler
        :raises Exception: If an unexpected error occurs during fitting.
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
        Transform the DataFrame by removing duplicate rows and columns.

        :param data: Input DataFrame to transform.
        :type data: pandas.DataFrame
        :return: Transformed DataFrame with duplicates removed.
        :rtype: pandas.DataFrame
        :raises ValueError: If a required column is missing.
        :raises Exception: For any unexpected error during transformation.
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
                f"DuplicateHandler: Dropped duplicate rows {dup_rows} "
                f"& columns {self.dup_cols}"
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

                transformed_data.to_csv(f"{self.save_dir}/{csv_name}.csv", index=False)
                logging.info(
                    f"DuplicateHandler: Transformed data saved at: "
                    f"{self.save_dir}/{csv_name}.csv"
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
        Fit the handler and then transform the data.

        :param data: Input DataFrame to fit and transform.
        :type data: pandas.DataFrame
        :param y: Ignored. Present for sklearn compatibility.
        :type y: Optional[pandas.Series]
        :return: Transformed DataFrame with duplicates removed.
        :rtype: pandas.DataFrame
        """
        if self.deactivate:
            logging.info("DuplicateHandler is deactivated. Returning unmodified data.")
            return data

        self.fit(data)
        return self.transform(data)
