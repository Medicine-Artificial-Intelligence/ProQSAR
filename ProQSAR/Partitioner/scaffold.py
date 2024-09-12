from rdkit import Chem
import logging
from rdkit.Chem.Scaffolds import MurckoScaffold
from typing import Tuple
import pandas as pd
import numpy as np


def scaffold_split(
    data: pd.DataFrame, smiles_col: str, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a scaffold-based split of the dataset, where molecules with similar scaffolds (core structures) are grouped
    together, and the dataset is split into training and test sets based on these scaffold groups. This ensures that
    similar scaffolds do not appear in both the training and test sets, which helps assess model generalization more
    realistically.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset containing molecular data, including the SMILES column that describes the molecular structures.

    smiles_col : str
        The name of the column in `data` containing SMILES strings that define the molecular structures.

    test_size : float, default=0.2
        The proportion of the dataset to include in the test split. A value of 0.2 means 20% of the dataset will
        be used for testing, and 80% will be used for training.

    random_state : int, default=42
        Seed for random number generation used to shuffle the scaffold groups. This ensures reproducibility of the
        split.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing two DataFrames:
        - `data_train`:The training set consisting of rows from the `data` DataFrame corresponding to the training fold.
        - `data_test`:The test set consisting of rows from the `data` DataFrame corresponding to the test fold.

    Example
    -------
    >>> data_train, data_test = scaffold_split(data, smiles_col='smiles', test_size=0.2)
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
    count_list = [len(i) for i in scaffold_lists]
    if np.array(count_list).sum() != len(data):
        raise ValueError("Failed to generate scaffold groups")
    np.random.seed(random_state)
    np.random.shuffle(scaffold_lists)

    num_molecules = len(data)
    num_test = int(np.floor(test_size * num_molecules))
    train_idx, test_idx = [], []
    for scaffold_list in scaffold_lists:
        if len(test_idx) + len(scaffold_list) <= num_test:
            test_idx.extend(scaffold_list)
        else:
            train_idx.extend(scaffold_list)

    data_train = data.iloc[train_idx]
    data_test = data.iloc[test_idx]

    return data_train, data_test
