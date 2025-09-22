import pandas as pd
from typing import Tuple
from .scaffold_utils import generate_scaffold_dict, check_scaffold_dict


class ScaffoldSplitter:
    """
    Deterministic scaffold-based splitter using Bemis–Murcko scaffolds.

    This splitter groups molecules by their Bemis–Murcko scaffolds
    and ensures that scaffolds are not split across train/test sets.
    Larger scaffolds are prioritized for the training set, while smaller
    scaffolds are allocated to the test set. Adapted from
    `DeepChem <https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py>`_.

    :param activity_col: Name of the column representing the activity or target label.
    :type activity_col: str
    :param smiles_col: Name of the column containing SMILES strings for molecular data.
    :type smiles_col: str
    :param mol_col: Name of the column containing RDKit Mol objects (if available).
                    Default is ``"mol"``.
    :type mol_col: str
    :param test_size: Proportion of the dataset to include in the test split.
                      Default is ``0.2``.
    :type test_size: float

    **Example**

    .. code-block:: python

        import pandas as pd
        from proqsar.Data.Splitter.scaffold_splitter import ScaffoldSplitter

        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "SMILES": ["CCO", "CCC", "CCN", "CCCl", "CCBr", "CCF"],
            "activity": [1.2, 3.4, 2.1, 0.5, 4.7, 3.3]
        })

        splitter = ScaffoldSplitter(activity_col="activity", smiles_col="SMILES", test_size=0.3)
        train, test = splitter.fit(df)
        print(train.shape, test.shape)
        # e.g., (4, 3) (2, 3)
    """

    def __init__(
        self,
        activity_col: str,
        smiles_col: str,
        mol_col: str = "mol",
        test_size: float = 0.2,
    ):
        self.activity_col = activity_col
        self.smiles_col = smiles_col
        self.mol_col = mol_col
        self.test_size = test_size

    def fit(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into training and testing sets based on Bemis–Murcko scaffolds.

        :param data: Input dataset containing SMILES, Mol objects, and activity labels.
        :type data: pd.DataFrame

        :return: A tuple ``(train_df, test_df)`` where
                 - ``train_df`` is the training dataset,
                 - ``test_df`` is the testing dataset.
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]

        :raises AssertionError: If train and test indices overlap.
        """
        # generate scaffolds
        all_scaffolds = generate_scaffold_dict(data, self.smiles_col, self.mol_col)
        check_scaffold_dict(data, all_scaffolds)

        # sort scaffolds by length and then by first index
        all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
        all_scaffold_sets = [
            scaffold_set
            for (scaffold, scaffold_set) in sorted(
                all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
            )
        ]

        # determine cutoff for train vs. test
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
