from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Literal, Optional
from ProQSAR.Splitter.stratified_scaffold_kfold import StratifiedScaffoldKFold


class StratifiedScaffoldSplitter:
    """
    A class used to split data into training and testing sets based on molecular scaffolds
    with stratification on the activity column.
    """

    def __init__(
        self,
        activity_col: str,
        smiles_col: str,
        mol_col: str = 'mol',
        random_state: int = 42,
        n_splits: int = 5,
        scaff_based: Literal["median", "mean"] = "median",
        shuffle: bool = True,
    ):
        """
        Constructs all the necessary attributes for the StratifiedScaffoldSplitter object.

        Parameters:
        -----------
        activity_col : str
            The name of the column representing the activity or target label.
        smiles_col : str
            The name of the column containing SMILES strings for molecular data.
        random_state : int, optional
            The random seed used by the random number generator (default is 42).
        n_splits : int, optional
            Number of splits/folds to create (default is 5).
        scaff_based : Literal["median", "mean"], optional
            The strategy to use for scaffold-based splitting (default is 'median').
        shuffle : bool, optional
            Whether to shuffle the data before splitting (default is True).
        """
        self.random_state = random_state
        self.activity_col = activity_col
        self.smiles_col = smiles_col
        self.mol_col = mol_col
        self.n_splits = n_splits
        self.scaff_based = scaff_based
        self.shuffle = shuffle


    @staticmethod
    def get_scaffold_groups(
        data: pd.DataFrame, 
        smiles_col: str, 
        mol_col: Optional[str] = None) -> np.ndarray:
        """
        Generates scaffold groups from the SMILES strings and returns an array of group indices.

        Parameters:
        -----------
        smiles_list : List[str]
            A list of SMILES strings.

        Returns:
        --------
        np.ndarray:
            An array of integers representing scaffold group indices.
        """
        scaffolds = {}
        for idx, row in data.iterrows():
            try:
                if mol_col:
                    mol = row[mol_col]
                else:
                    smiles = row[smiles_col]
                    mol = Chem.rdmolfiles.MolFromSmiles(smiles)

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
        groups = np.full(len(data[smiles_col].to_list()), -1, dtype="i")
        for i, scaff in enumerate(scaffold_lists):
            groups[scaff] = i

        if -1 in groups:
            raise AssertionError("Some molecules are not assigned to a group.")
        
        return groups

    def fit(
        self,
        data: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the data into training and testing sets using stratified scaffolds.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing the features and labels.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]:
            The training and testing sets as pandas DataFrames.
        """
        cv = StratifiedScaffoldKFold(
            n_splits=self.n_splits,
            random_state=self.random_state,
            shuffle=self.shuffle,
            scaff_based=self.scaff_based,
        )
        groups = StratifiedScaffoldSplitter.get_scaffold_groups(data, self.smiles_col, self.mol_col)
        
        y = data[self.activity_col].to_numpy(dtype=float)
        X = data.drop([self.activity_col, self.smiles_col, self.smiles_col], axis=1, errors='ignore').to_numpy()
        train_idx, test_idx = next(cv.split(X, y, groups))
        data_train = data.iloc[train_idx]
        data_test = data.iloc[test_idx]

        return data_train, data_test
