from typing import List, Tuple
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import logging
import pandas as pd
import numpy as np


class ScaffoldSplitter:
    """
    A class used to split data into training and testing sets based on molecular scaffolds.
    """

    def __init__(
        self,
        activity_col: str,
        smiles_col: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Constructs all the necessary attributes for the ScaffoldSplitter object.

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

    @staticmethod
    def scaffold(data: pd.DataFrame, smiles_col: str) -> List[List[int]]:
        """
        Generates scaffold groups from the SMILES strings and returns a list of scaffold indices.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing the features and labels.
        smiles_col : str
            The name of the column containing SMILES strings.

        Returns:
        --------
        List[List[int]]:
            A list of lists, where each inner list contains the indices of molecules that share the same scaffold.
        """
        scaffolds = {}
        for idx, row in data.iterrows():
            smiles = row[smiles_col]
            try:
                mol = Chem.MolFromSmiles(smiles)
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=False
                )
            except Exception:
                logging.error(f"Failed to convert SMILES to Mol: {smiles}")
                continue
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [idx]
            else:
                scaffolds[scaffold].append(idx)

        scaffold_lists = list(scaffolds.values())
        return scaffold_lists

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
        scaffold_lists = ScaffoldSplitter.scaffold(data, self.smiles_col)
        count_list = [len(i) for i in scaffold_lists]
        if np.array(count_list).sum() != len(data):
            raise ValueError("Failed to generate scaffold groups")
        np.random.seed(self.random_state)
        np.random.shuffle(scaffold_lists)

        num_molecules = len(data)
        num_test = int(np.floor(self.test_size * num_molecules))
        train_idx, test_idx = [], []
        for scaffold_list in scaffold_lists:
            if len(test_idx) + len(scaffold_list) <= num_test:
                test_idx.extend(scaffold_list)
            else:
                train_idx.extend(scaffold_list)

        data_train = data.iloc[train_idx]
        data_test = data.iloc[test_idx]
        return data_train, data_test
