import numpy as np
import pandas as pd
from typing import Tuple
from ProQSAR.Splitter.scaffold_utils import generate_scaffold_dict, check_scaffold_dict


class ScaffoldSplitter:
    """
    Adapted from  https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    Splits a dataset into training and test sets based on Bemis-Murcko scaffolds.
    This deterministic splitting method ensures that molecules with the same scaffold are
    grouped together.
    """

    def __init__(
        self,
        activity_col: str,
        smiles_col: str,
        mol_col: str = "mol",
        test_size: float = 0.2,
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
        self.activity_col = activity_col
        self.smiles_col = smiles_col
        self.mol_col = mol_col
        self.test_size = test_size

    def fit(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

        # generate scaffolds
        all_scaffolds = generate_scaffold_dict(data, self.smiles_col, self.mol_col)
        check_scaffold_dict(data, all_scaffolds)

        # sort scaffolds by length and then by first index
        # This ensures that larger scaffolds are prioritized in the training set
        # and smaller scaffolds are used for testing.
        all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
        all_scaffold_sets = [
            scaffold_set
            for (scaffold, scaffold_set) in sorted(
                all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
            )
        ]

        # get train, valid test indices
        frac_train = 1 - self.test_size
        train_cutoff = frac_train * len(data)

        train_idx, test_idx = [], []
        for scaffold_set in all_scaffold_sets:
            if len(train_idx) + len(scaffold_set) > train_cutoff:
                test_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        assert (
            len(set(train_idx).intersection(set(test_idx))) == 0
        ), "Train and test indices overlap."

        data_train = data.loc[train_idx].copy()
        data_test = data.loc[test_idx].copy()

        return data_train, data_test
