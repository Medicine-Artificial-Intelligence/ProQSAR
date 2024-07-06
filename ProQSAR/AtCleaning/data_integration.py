import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from copy import deepcopy
from typing import Tuple


class DataIntegration:
    """
    Create a DataFrame from a CSV file, handle missing values (NaN),
    perform target transformation (classification), and split data into training and test sets.

    Parameters:
    - data (pandas.DataFrame): DataFrame containing features and target columns.
    - activity_col (str): Name of the activity column (e.g., pIC50, pChEMBL Value).
    - task_type (str): 'C' for Classification or 'R' for Regression.
    - target_thresh (int): Threshold for numerical-to-binary target transformation.
    - visualize (bool, optional): Whether to visualize the target distribution. Default is True.
    - figsize (tuple, optional): Figure size. Default is (16, 5).

    Returns:
    - data_train (pandas.DataFrame): Training data.
    - data_test (pandas.DataFrame): Test data.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        activity_col: str,
        id_col: str,
        task_type: str = "C",
        target_thresh: float = 7,
        visualize: bool = True,
        figsize: Tuple[int, int] = (16, 5),
    ) -> None:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        if task_type not in ["C", "R"]:
            raise ValueError(
                "task_type must be either 'C' for Classification or 'R' for Regression"
            )
        self.data = deepcopy(data)
        self.activity_col = activity_col
        self.id_col = id_col
        self.task_type = task_type
        self.target_thresh = target_thresh
        self.visualize = visualize
        self.figsize = figsize

    @staticmethod
    def _check_nan(data: pd.DataFrame) -> pd.DataFrame:
        """
        Check for NaN values in a DataFrame and replace them with np.nan in numeric columns.

        Parameters:
        - data (pandas.DataFrame): DataFrame to check for NaN values.

        Returns:
        - data (pandas.DataFrame): DataFrame with NaN values replaced in numeric columns.
        """
        for col in data.columns:
            # Check if the column is numeric (int or float)
            if data[col].dtype.kind in "if":
                data[col] = data[col].apply(lambda x: np.nan if pd.isna(x) else x)
        return data

    @staticmethod
    def _target_bin(data: pd.DataFrame, activity_col: str, thresh: int, task_type: str):
        """
        Perform target transformation for classification.

        Parameters:
        - data (pandas.DataFrame): DataFrame containing the target column.
        - activity_col (str): Name of the activity column.
        - thresh (int): Threshold for target transformation.
        - task_type (str): 'C' for Classification or 'R' for Regression.

        Returns:
        - data (pandas.DataFrame): DataFrame with target transformation applied.
        """

        data[activity_col] = (data[activity_col] >= thresh).astype("int64")
        return data

    @staticmethod
    def handle_duplicates(
        data: pd.DataFrame,
        list_columns: list,
    ) -> pd.DataFrame:
        """
        Handles duplicate rows and columns in a DataFrame.

        Parameters:
        data (pd.DataFrame): The DataFrame to process.

        Returns:
        pd.DataFrame: The processed DataFrame with duplicates removed.
        """
        temp_data = data.drop(columns=list_columns, axis=1)
        duplicates = temp_data.duplicated()
        data = data[~duplicates].reset_index(drop=True)

        # Handling duplicate columns
        duplicated_cols = data.columns[data.T.duplicated()]
        data = data.drop(columns=duplicated_cols)

        return data

    @staticmethod
    def _data_split(data: pd.DataFrame, activity_col: str):
        """
        Split data into training and test sets.

        Parameters:
        - data (pandas.DataFrame): DataFrame containing features and target columns.
        - activity_col (str): Name of the activity column.

        Returns:
        - data_train (pandas.DataFrame): Training data.
        - data_test (pandas.DataFrame): Test data.
        """
        stratify = data[activity_col] if data[activity_col].nunique() == 2 else None
        data_train, data_test = train_test_split(
            data, test_size=0.2, random_state=42, stratify=stratify
        )
        return data_train, data_test

    @staticmethod
    def _visualize_target(
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        activity_col: str,
        task_type: str,
        figsize=(16, 5),
    ) -> None:
        """
        Visualize the target distribution for either classification or regression tasks.

        Parameters:
        - data_train (pd.DataFrame): Training data.
        - data_test (pd.DataFrame): Test data.
        - activity_col (str): Name of the activity column.
        - task_type (str): 'C' for Classification or 'R' for Regression.
        - figsize (tuple, optional): Figure size. Default is (16, 5).

        Returns:
        None
        """
        sns.set(style="whitegrid")
        plt.figure(figsize=figsize)

        if task_type.upper() == "C":
            for i, data in enumerate([data_train, data_test], 1):
                plt.subplot(1, 2, i)
                plt.title(
                    f'{"Training" if i == 1 else "External"} Data',
                    weight="semibold",
                    fontsize=24,
                )
                sns.countplot(data=data, x=activity_col)
                plt.ylabel("Number of Compounds")
                imbalance_ratio = round(
                    (data[activity_col] == 1).sum() / (data[activity_col] == 0).sum(), 3
                )
                plt.xlabel(f"Imbalance ratio: {imbalance_ratio}")

                # Add count labels above each bar
                for p in plt.gca().patches:
                    plt.gca().annotate(
                        f"{int(p.get_height())}",
                        (p.get_x() + p.get_width() / 2.0, p.get_height()),
                        ha="center",
                        va="bottom",
                    )

        elif task_type.upper() == "R":
            for i, data in enumerate([data_train, data_test], 1):
                plt.subplot(1, 2, i)
                plt.title(
                    f'{"Train" if i == 1 else "Test"} Distribution',
                    weight="semibold",
                    fontsize=24,
                )
                sns.histplot(data=data, x=activity_col, kde=True)
                plt.ylabel("Number of Compounds")
                plt.xlabel(f"Number of Compounds: {data.shape[0]}")

        plt.tight_layout()
        plt.show()

    def fit(self) -> Tuple[pd.DataFrame, pd.DataFrame]:

        # 1. Handle NaN values
        data = self._check_nan(self.data)

        # 2. Remove duplicates
        data = self.handle_duplicates(
            data, list_columns=[self.activity_col, self.id_col]
        )

        # 3. Target transformation (Classification)
        if self.task_type.title() == "C":
            data = self._target_bin(
                data, self.activity_col, self.target_thresh, self.task_type
            )

        # 4. Data split
        data_train, data_test = self._data_split(data, self.activity_col)

        # 5. Visualize the target distribution
        if self.visualize:
            self._visualize_target(
                data_train, data_test, self.activity_col, self.task_type, self.figsize
            )

        return data_train, data_test
