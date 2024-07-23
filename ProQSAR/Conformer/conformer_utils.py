import logging
from rdkit import Chem
from rdkit.Chem.rdDistGeom import ETDG, ETKDG, ETKDGv2, ETKDGv3, srETKDGv3, KDG
from typing import Any

logger = logging.getLogger(__name__)


def _get_embedding_method(force_field_method: str = "ETKDG") -> Any:
    """
    Retrieves the appropriate RDKit embedding method object based on the specified force field method.

    Parameters:
    - force_field_method (str): The force field method identifier. Defaults to 'ETKDG'.
        Supported values are 'ETDG', 'ETKDG', 'ETKDGv2', 'ETKDGv3', 'srETKDGv3', and 'KDG'.

    Returns:
    - Any: An instance of the specified RDKit embedding method class.

    Raises:
    - KeyError: If the specified force field method is not supported.

    Example:
    >>> embedding_method = _get_embedding_method('ETKDG')
    >>> print(type(embedding_method))
    <class 'rdkit.Chem.rdDistGeom.ETKDG'>
    """
    _embedding_methods = {
        "ETDG": ETDG(),
        "ETKDG": ETKDG(),
        "ETKDGv2": ETKDGv2(),
        "ETKDGv3": ETKDGv3(),
        "srETKDGv3": srETKDGv3(),
        "KDG": KDG(),
    }

    if force_field_method not in _embedding_methods:
        raise KeyError(
            f"Unsupported force field method '{force_field_method}'."
            + " Supported methods are: {list(_embedding_methods.keys())}."
        )

    return _embedding_methods[force_field_method]


def _get_num_conformers_from_molecule_size(
    molecule: Chem.Mol,
    max_num_conformers: int = 10,
    min_num_conformers: int = 2,
    decr_num_conformers: float = 0.04,
) -> int:
    """
    Calculates the appropriate number of conformers to generate based on the molecule size.

    Parameters:
    - molecule (Chem.Mol): The molecule for which to determine the number of conformers.
    - max_num_conformers (int): Maximum number of conformers to generate. Defaults to 10.
    - min_num_conformers (int): Minimum number of conformers to generate. Defaults to 2.
    - decr_num_conformers (float): Decrement factor for the number of conformers based on molecule size.
    Defaults to 0.04.

    Returns:
        int: Calculated number of conformers to generate.
    """
    num_atoms = molecule.GetNumAtoms()
    suggested_num = max_num_conformers - int(num_atoms * decr_num_conformers)
    return max(min_num_conformers, min(max_num_conformers, suggested_num))


def _get_max_iter_from_molecule_size(
    molecule: Chem.Mol,
    min_iter: int = 20,
    max_iter: int = 2000,
    incr_iter: int = 10,
) -> int:
    """
    Calculates the maximum number of iterations for molecular force field calculations based on molecule size.

    Parameters:
    - molecule (Chem.Mol): The molecule for which to calculate the maximum number of iterations.
    - min_iter (int, optional): The minimum number of iterations. Default is 20.
    - max_iter (int, optional): The maximum number of iterations. Default is 2000.
    - incr_iter (int, optional): The increment for each atom in the molecule. Default is 10.

    Returns:
    - int: Maximum number of iterations.
    """
    num_atoms = molecule.GetNumAtoms()
    return min(max_iter, min_iter + num_atoms * incr_iter)


def _assert_correct_force_field(force_field: str) -> None:
    """
    Check if the provided force field method is available.

    Parameters:
    - force_field (str): The force field method to check.

    Raises:
    ValueError: If the force field method is not available.
    """
    _available_ff_methods = ["MMFF", "MMFF94", "MMFF94s", "UFF"]
    if force_field not in _available_ff_methods:
        raise ValueError(
            f"`force_field_method` has to be either of: {_available_ff_methods}"
        )


def _assert_has_conformers(molecule: Chem.Mol) -> None:
    """
    Check if the provided molecule has any conformers.

    Parameters:
    - molecule (Chem.Mol): The molecule to check.

    Raises:
    ValueError: If the molecule does not have any conformers.
    """
    if not molecule.GetNumConformers():
        raise ValueError("`molecule` has no conformers.")
