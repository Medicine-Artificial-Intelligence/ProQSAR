import numpy as np
import pandas as pd
from typing import Tuple
from ProQSAR.Splitter.scaffold_utils import generate_scaffold_list, check_scaffold_list


class RandomScaffoldSplitter:
    """
    A class used to split data into training and testing sets based on molecular scaffolds.
    """

    def __init__(
        self,
        activity_col: str,
        smiles_col: str,
        mol_col: str = "mol",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Constructs all the necessary attributes for the RandomScaffoldSplitter object.

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
        self.mol_col = mol_col

    def fit(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the data into training and testing sets based on molecular scaffolds.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing the features and labels.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]:
            The training and testing sets as pandas DataFrames.
        """
        scaffold_lists = generate_scaffold_list(data, self.smiles_col, self.mol_col)
        check_scaffold_list(data, scaffold_lists)

        rng = np.random.RandomState(self.random_state)
        rng.shuffle(scaffold_lists)

        num_molecules = len(data)
        num_test = int(np.floor(self.test_size * num_molecules))

        train_idx, test_idx = [], []

        for group in scaffold_lists:
            if len(test_idx) + len(group) <= num_test:
                test_idx.extend(group)
            else:
                train_idx.extend(group)

        assert (
            len(set(train_idx).intersection(set(test_idx))) == 0
        ), "Train and test indices overlap."

        data_train = data.loc[train_idx]
        data_test = data.loc[test_idx]

        return data_train, data_test
