import logging
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from typing import Optional, List, Dict


def generate_scaffold_dict(
    data: pd.DataFrame, smiles_col: str, mol_col: Optional[str] = None
) -> Dict[str, List[int]]:
    """
    Generate a dictionary mapping Bemis–Murcko scaffolds to molecule indices.

    Each unique scaffold SMILES string is used as a key, and the value is a list
    of dataset indices corresponding to molecules that share this scaffold.

    :param data: Input dataset containing SMILES and optionally RDKit Mol objects.
    :type data: pd.DataFrame
    :param smiles_col: Name of the column containing SMILES strings.
    :type smiles_col: str
    :param mol_col: Optional name of the column containing RDKit Mol objects.
                    If provided and present in the dataset, these Mol objects are used
                    instead of parsing SMILES. Default is ``None``.
    :type mol_col: Optional[str]

    :return: Dictionary where keys are scaffold SMILES and values are lists of indices.
    :rtype: Dict[str, List[int]]

    :raises ValueError: If RDKit fails to parse a SMILES string into a Mol object.
    """
    scaffolds: Dict[str, List[int]] = {}
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
    Generate scaffold groups as a list of index lists.

    :param data: Input dataset containing SMILES and optionally RDKit Mol objects.
    :type data: pd.DataFrame
    :param smiles_col: Name of the column containing SMILES strings.
    :type smiles_col: str
    :param mol_col: Optional name of the column containing RDKit Mol objects.
                    Default is ``None``.
    :type mol_col: Optional[str]

    :return: List of scaffold groups, where each inner list contains indices of molecules
             sharing the same scaffold.
    :rtype: List[List[int]]
    """
    scaffold_dict = generate_scaffold_dict(data, smiles_col, mol_col)
    return list(scaffold_dict.values())


def get_scaffold_groups(
    data: pd.DataFrame, smiles_col: str, mol_col: Optional[str] = None
) -> np.ndarray:
    """
    Convert scaffold groups into an array of group indices.

    Each molecule in the dataset is assigned a group index corresponding to its scaffold.

    :param data: Input dataset containing SMILES and optionally RDKit Mol objects.
    :type data: pd.DataFrame
    :param smiles_col: Name of the column containing SMILES strings.
    :type smiles_col: str
    :param mol_col: Optional name of the column containing RDKit Mol objects.
                    Default is ``None``.
    :type mol_col: Optional[str]

    :return: Array of integers, where each entry is the scaffold group index
             for the corresponding molecule.
    :rtype: np.ndarray

    :raises AssertionError: If some molecules are not assigned to a scaffold group.
    """
    scaffold_lists = generate_scaffold_list(data, smiles_col, mol_col)
    groups = np.full(len(data[smiles_col].to_list()), -1, dtype="i")
    for i, scaff in enumerate(scaffold_lists):
        groups[scaff] = i

    if -1 in groups:
        raise AssertionError("Some molecules are not assigned to a group.")

    return groups


def check_scaffold_dict(data: pd.DataFrame, scaffold_dict: Dict[str, List[int]]):
    """
    Validate that a scaffold dictionary accounts for all molecules in the dataset.

    :param data: Input dataset containing SMILES and activity labels.
    :type data: pd.DataFrame
    :param scaffold_dict: Dictionary mapping scaffold SMILES to lists of indices.
    :type scaffold_dict: Dict[str, List[int]]

    :raises AssertionError: If the number of molecules in the dataset does not
                            match the total number of indices in the scaffold dictionary.
    """
    total_molecules = len(data)
    total_indices = sum(len(indices) for indices in scaffold_dict.values())

    if total_indices != total_molecules:
        raise AssertionError(
            "Scaffold dictionary does not contain all molecules from the dataset."
        )


def check_scaffold_list(data: pd.DataFrame, scaffold_lists: List[List[int]]):
    """
    Validate that scaffold groups cover all molecules in the dataset.

    :param data: Input dataset containing SMILES and activity labels.
    :type data: pd.DataFrame
    :param scaffold_lists: List of scaffold groups, where each inner list contains
                           indices of molecules sharing the same scaffold.
    :type scaffold_lists: List[List[int]]

    :raises AssertionError: If the sum of molecule counts across all groups does
                            not equal the number of molecules in the dataset.
    """
    count_list = [len(group) for group in scaffold_lists]

    if np.array(count_list).sum() != len(data):
        raise AssertionError("Failed to generate scaffold groups for all molecules.")
