from rdkit import Chem
import numpy as np
import logging
from typing import Union, Optional, List
from ProQSAR.Conformer.conformer_utils import (
    _get_num_conformers_from_molecule_size,
    _get_max_iter_from_molecule_size,
    _assert_correct_force_field,
    _assert_has_conformers,
    _get_embedding_method,
)

logger = logging.getLogger(__name__)


def mol_embed(
    molecule: Chem.Mol,
    num_conformers: Optional[Union[int, str]] = "auto",
    embedding_method: str = "ETKDGv3",
    num_threads: int = 1,
    random_coords_threshold: int = 100,
    random_seed: int = 42,
) -> Chem.Mol:
    """
    Embeds one or more conformers from RDKit molecule object.

    Parameters:
    - molecule (Chem.Mol): The molecule to be embedded.
    - num_conformers (int or str, optional): Number of conformers to generate. If 'auto', the number of
        conformers will depend on the size of the molecule. Default to 'auto'.
    - embedding_method (str, optional): The embedding method to use. Default to 'ETKDGv3'.
    - num_threads (int, optional): Number of threads to use. Default to 1.
    - random_coords_threshold (int, optional): Threshold for using random coordinates. Default to 100.
    - random_seed (int, optional): Seed for random number generator. Default to 42.

    Returns:
    - Chem.Mol: The embedded molecule.
    """

    # Get copy, to leave original molecle untouched
    molecule = Chem.Mol(molecule)

    if num_conformers is None or num_conformers == "auto":
        num_conformers = _get_num_conformers_from_molecule_size(molecule)

    params = _get_embedding_method(embedding_method)

    params.numThreads = num_threads
    params.randomSeed = random_seed

    if molecule.GetNumAtoms() > random_coords_threshold:
        # For large molecules, random coordinates might help
        params.useRandomCoords = True

    success = Chem.rdDistGeom.EmbedMultipleConfs(
        molecule, numConfs=num_conformers, params=params
    )

    if not len(success):
        # Not pretty, but it is desired to force a 3D structure (instead
        # of ignoring the molecule) so that the same molecules exist
        # between the 2d and 3d datasets.
        logger.warning(
            "Could not embed conformers, computing conformer from "
            + "2D coordinates instead."
        )
        Chem.rdDepictor.Compute2DCoords(molecule)

    return molecule


def force_field_minimization(
    molecule: Chem.Mol,
    force_field_method: Optional[str] = "UFF",
    max_iter: Optional[Union[int, str]] = "auto",
    return_energies: bool = False,
    num_threads: int = 1,
    **kwargs,
) -> Union[np.ndarray, None]:
    """
    Performs a force field minimization on embedded conformers of RDKit
    molecule object.

    Parameters:
    - molecule (Chem.Mol): The molecule to be minimized.
    - force_field_method (str, optional): The force field method to use. Default to 'UFF'.
    - max_iter (int or str, optional): Maximum number of iterations for minimization. If 'auto', the number of
        iterations will depend on the size of the molecule. Default to 'auto'.
    - return_energies (bool, optional): Whether to return the energies of the minimized conformers. Default to False.
    - num_threads (int, optional): Number of threads to use. Default to 1.
    **kwargs: Arbitrary keyword arguments.

    Returns:
    - Chem.Mol or np.ndarray: The minimized molecule.
    If `return_energies` is True, also returns the energies of the minimized conformers.
    """

    _assert_correct_force_field(force_field_method)

    # Get copy, to leave original molecle untouched
    molecule = Chem.Mol(molecule)

    if max_iter is None or max_iter == "auto":
        max_iter = _get_max_iter_from_molecule_size(molecule)

    if force_field_method == "MMFF":
        force_field_method = "MMFF94"

    try:
        if force_field_method.startswith("MMFF"):
            # Merck Molecular Force Field (MMFF; specifically MMFF94 or MMFF94s)
            Chem.rdForceFieldHelpers.MMFFSanitizeMolecule(molecule)
            result = Chem.rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
                molecule,
                maxIters=max_iter,
                numThreads=num_threads,
                mmffVariant=force_field_method,
                **kwargs,
            )
        else:
            # Universal Force Field (UFF)
            result = Chem.rdForceFieldHelpers.UFFOptimizeMoleculeConfs(
                molecule, numThreads=num_threads, maxIters=max_iter, *kwargs
            )
    except RuntimeError:
        logger.warnings(
            f"{force_field_method} raised a `RunTimeError`, procedding "
            + f"without {force_field_method} minimization."
        )

        return molecule

    converged = [r[0] != 1 for r in result]

    if not any(converged):
        logger.warning(
            f"{force_field_method} minimization did not converge "
            + f"after {max_iter} iterations, for any of the "
            + f"{len(converged)} conformers."
        )

    if return_energies:
        return molecule, [r[1] for r in result]

    return molecule


def get_lowest_energy_conformer(molecule, force_field_method: str = "UFF"):

    _assert_has_conformers(molecule)

    conformer_ids = [conformer.GetId() for conformer in molecule.GetConformers()]
    energy_lowest = float("inf")
    for conformer_id in conformer_ids:
        energy = compute_force_field_energy(molecule, conformer_id, force_field_method)
        if energy < energy_lowest:
            energy_lowest = energy
            conformer_id_keep = conformer_id

    new_molecule = Chem.Mol(molecule)
    new_molecule.RemoveAllConformers()
    conformer = molecule.GetConformer(conformer_id_keep)
    new_molecule.AddConformer(conformer, assignId=True)
    return new_molecule


def compute_force_field_energy(
    molecule: Chem.Mol, conformer_id: int, force_field_method: str = "UFF"
) -> float:
    """
    Computes the force field energy of a given conformer of an RDKit molecule object.

    Parameters:
    - molecule (Chem.Mol): The molecule for which to compute the force field energy.
    conformer_id (int): The ID of the conformer for which to compute the force field energy.
    - force_field_method (str, optional): The force field method to use. Default to 'UFF'.

    Returns:
    float: The computed force field energy.
    """

    # If the force field method is 'MMFF', change it to 'MMFF94'
    if force_field_method == "MMFF":
        force_field_method = "MMFF94"

    # If the force field method starts with 'MMFF', use the MMFF force field
    if force_field_method.startswith("MMFF"):
        # Get the MMFF properties of the molecule
        mmff_properties = Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties(
            mol=molecule, mmffVariant=force_field_method
        )
        # Get the MMFF force field of the molecule
        force_field = Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField(
            molecule, mmff_properties, confId=conformer_id
        )
    else:
        # If the force field method is not 'MMFF', use the UFF force field
        force_field = Chem.rdForceFieldHelpers.UFFGetMoleculeForceField(
            molecule, confId=conformer_id
        )

    # Calculate and return the energy of the force field
    return force_field.CalcEnergy()
