import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split


class StratifiedRandomSplitter:
    """
    A class used to split data into training and testing sets with stratification based on the activity column.
    """

    def __init__(
        self,
        activity_col: str,
        smiles_col: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Constructs all the necessary attributes for the StratifiedRandomSplitter object.

        Parameters:
        -----------
        activity_col : str
            The name of the column representing the activity or target label.
        smiles_col : str
            The name of the column containing SMILES strings for molecular data.
        test_size : float, optional
            The proportion of the dataset to include in the test split (default is 0.2).
        random_state : int, optional
            The random seed used by the random number generator (default is 42).
        """
        self.test_size = test_size
        self.random_state = random_state
        self.activity_col = activity_col
        self.smiles_col = smiles_col

    def fit(
        self,
        data: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the data into training and testing sets.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing the features and labels.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            The training and testing sets as pandas DataFrames.
        """
        data_train, data_test = train_test_split(
            data,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=data[self.activity_col],
        )
        return data_train, data_test
