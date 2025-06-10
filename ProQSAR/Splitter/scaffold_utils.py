import logging
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from typing import Optional, List


def generate_scaffold_dict(
    data: pd.DataFrame, smiles_col: str, mol_col: Optional[str] = None
) -> dict:

    scaffolds = {}
    for idx, row in data.iterrows():
        try:
            smiles = row[smiles_col]
            if mol_col is not None and mol_col in row and row[mol_col] is not None:
                mol = row[mol_col]
            else:
                mol = Chem.rdmolfiles.MolFromSmiles(smiles)

            if mol is None:
                raise ValueError(f"RDKit failed to parse molecule at index {idx}")

            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=True
            )
        except Exception as e:
            logging.error(
                f"Failed to generate scaffold for index {idx} (SMILES: {smiles}): {e}"
            )
            continue

        if scaffold not in scaffolds:
            scaffolds[scaffold] = [idx]
        else:
            scaffolds[scaffold].append(idx)

    return scaffolds


def generate_scaffold_list(
    data: pd.DataFrame, smiles_col: str, mol_col: Optional[str] = None
) -> List[List[int]]:
    """
    Generates a list of scaffold groups from the dataset.

    Parameters:
    -----------
    data : pd.DataFrame
        The dataset containing the features and labels.
    smiles_col : str
        The name of the column containing SMILES strings.
    mol_col : Optional[str]
        The name of the column containing RDKit Mol objects, if available.

    Returns:
    --------
    List[List[int]]:
        A list where each element is a list of indices corresponding to molecules sharing the same scaffold.
    """
    scaffold_dict = generate_scaffold_dict(data, smiles_col, mol_col)
    return list(scaffold_dict.values())


def get_scaffold_groups(
    data: pd.DataFrame, smiles_col: str, mol_col: Optional[str] = None
) -> np.ndarray:
    """
    Converts scaffold groups into an array of group indices.

    Parameters:
    -----------
    data : pd.DataFrame
        The dataset containing the features and labels.
    smiles_col : str
        The name of the column containing SMILES strings.
    mol_col : Optional[str]
        The name of the column containing RDKit Mol objects, if available.

    Returns:
    --------
    np.ndarray:
        An array of integers representing scaffold group indices.
    """
    scaffold_lists = generate_scaffold_list(data, smiles_col, mol_col)
    groups = np.full(len(data[smiles_col].to_list()), -1, dtype="i")
    for i, scaff in enumerate(scaffold_lists):
        groups[scaff] = i

    if -1 in groups:
        raise AssertionError("Some molecules are not assigned to a group.")

    return groups


def check_scaffold_dict(data: pd.DataFrame, scaffold_dict: dict):
    """
    Checks if the scaffold dictionary contains all molecules from the dataset.

    Parameters:
    -----------
    scaffold_dict : dict
        The dictionary containing scaffold groups, where keys are scaffold SMILES and values are lists of indices.
    data : pd.DataFrame
        The dataset containing the features and labels.

    Raises:
    -------
    AssertionError: If the total number of molecules does not match the number of indices in the scaffold dictionary.
    """
    total_molecules = len(data)
    total_indices = sum(len(indices) for indices in scaffold_dict.values())

    if total_indices != total_molecules:
        raise AssertionError(
            "Scaffold dictionary does not contain all molecules from the dataset."
        )


def check_scaffold_list(data: pd.DataFrame, scaffold_lists: List[List[int]]):
    """
    Checks if the scaffold generation was successful by comparing the number of molecules in the scaffold groups
    with the total number of molecules in the dataset.

    Parameters:
    -----------
    data : pd.DataFrame
        The dataset containing the features and labels.
    scaffold_lists : List[List[int]]
        The list of scaffold groups, where each inner list contains indices of molecules sharing the same scaffold.

    Raises:
    -------
    AssertionError: If the total number of molecules does not match the sum of molecules in scaffold groups.
    """
    count_list = [len(group) for group in scaffold_lists]

    if np.array(count_list).sum() != len(data):
        raise AssertionError("Failed to generate scaffold groups for all molecules.")
