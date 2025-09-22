import numpy as np
import pandas as pd
from typing import Tuple
from .scaffold_utils import generate_scaffold_list, check_scaffold_list


class RandomScaffoldSplitter:
    """
    Split data into training and testing sets based on molecular scaffolds.

    This splitter ensures that molecules with the same scaffold are not split
    across train and test sets, thereby preventing scaffold leakage during
    model evaluation.

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
    :param random_state: Random seed for reproducibility of scaffold shuffling.
                         Default is ``42``.
    :type random_state: int

    **Example**

    .. code-block:: python

        import pandas as pd
        from proqsar.Data.Splitter.random_scaffold_splitter import RandomScaffoldSplitter

        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "SMILES": ["CCO", "CCC", "CCN", "CCCl", "CCBr", "CCF"],
            "activity": [1.2, 3.4, 2.1, 0.5, 4.7, 3.3]
        })

        splitter = RandomScaffoldSplitter(
            activity_col="activity",
            smiles_col="SMILES",
            test_size=0.3,
            random_state=0
        )

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
        random_state: int = 42,
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.activity_col = activity_col
        self.smiles_col = smiles_col
        self.mol_col = mol_col

    def fit(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into training and testing sets based on scaffolds.

        :param data: Input dataset containing SMILES, Mol objects, and activity labels.
        :type data: pd.DataFrame

        :return: A tuple ``(train_df, test_df)`` where
                 - ``train_df`` is the training dataset,
                 - ``test_df`` is the testing dataset.
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]

        :raises AssertionError: If train and test indices overlap.
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
