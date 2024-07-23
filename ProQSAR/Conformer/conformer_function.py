from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers
import numpy as np
import logging
from typing import Union, Optional
from ProQSAR.Conformer.conformer_utils import (
    _get_num_conformers_from_molecule_size,
    _get_max_iter_from_molecule_size,
    _assert_correct_force_field,
    _assert_has_conformers,
    _get_embedding_method,
)


def mol_embed(
    molecule: Chem.Mol,
    num_conformers: Optional[Union[int, str]] = "auto",
    embedding_method: str = "ETKDGv3",
    num_threads: int = 1,
    random_coords_threshold: int = 100,
    random_seed: int = 42,
) -> Chem.Mol:
    """
    Embeds conformers for an RDKit molecule object using specified embedding parameters.

    Parameters:
    - molecule (Chem.Mol): The molecule to be embedded.
    - num_conformers (Optional[Union[int, str]]): The number of conformers to generate.
      If 'auto', the number is determined based on the molecule's size. Defaults to 'auto'.
    - embedding_method (str): The embedding method to use, corresponding to different RDKit embedding strategies.
      Defaults to 'ETKDGv3'.
    - num_threads (int): The number of threads to use for conformer generation. Defaults to 1.
    - random_coords_threshold (int): Atom count threshold above which random coordinates are
    used to initialize embeddings. Defaults to 100.
    - random_seed (int): Seed for the random number generator to ensure reproducibility. Defaults to 42.

    Returns:
    - Chem.Mol: The molecule with embedded conformers, or the original molecule if embedding fails.

    Raises:
    - ValueError: If an invalid number of conformers is provided.
    """
    # Validate num_conformers input
    if isinstance(num_conformers, str) and num_conformers != "auto":
        raise ValueError("num_conformers must be an integer or 'auto'.")

    # Copy the molecule to keep the original untouched
    mol_copy = Chem.Mol(molecule)
    mol_copy = Chem.AddHs(mol_copy)

    # Determine the number of conformers
    if num_conformers == "auto":
        num_conformers = _get_num_conformers_from_molecule_size(mol_copy)

    # Retrieve embedding parameters
    params = _get_embedding_method(embedding_method)
    params.numThreads = num_threads
    params.randomSeed = random_seed
    params.useRandomCoords = mol_copy.GetNumAtoms() > random_coords_threshold

    # Attempt to embed multiple conformers
    success = Chem.rdDistGeom.EmbedMultipleConfs(
        mol_copy, numConfs=num_conformers, params=params
    )

    if not success:
        logging.warning(
            "Failed to embed any conformers; attempting to compute 2D coordinates."
        )
        Chem.rdDepictor.Compute2DCoords(mol_copy)

    return mol_copy


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
            rdForceFieldHelpers.MMFFSanitizeMolecule(molecule)
            result = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
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
        logging.warnings(
            f"{force_field_method} raised a `RunTimeError`, procedding "
            + f"without {force_field_method} minimization."
        )

        return molecule

    converged = [r[0] != 1 for r in result]

    if not any(converged):
        logging.warning(
            f"{force_field_method} minimization did not converge "
            + f"after {max_iter} iterations, for any of the "
            + f"{len(converged)} conformers."
        )

    if return_energies:
        return molecule, [r[1] for r in result]

    return molecule


# def get_lowest_energy_conformer(molecule, force_field_method: str = "UFF"):

#     _assert_has_conformers(molecule)

#     conformer_ids = [conformer.GetId() for conformer in molecule.GetConformers()]
#     energy_lowest = float("inf")
#     for conformer_id in conformer_ids:
#         energy = compute_force_field_energy(molecule, conformer_id, force_field_method)
#         if energy < energy_lowest:
#             energy_lowest = energy
#             conformer_id_keep = conformer_id

#     new_molecule = Chem.Mol(molecule)
#     new_molecule.RemoveAllConformers()
#     conformer = molecule.GetConformer(conformer_id_keep)
#     new_molecule.AddConformer(conformer, assignId=True)
#     return new_molecule


def get_lowest_energy_conformer(
    molecule: Chem.Mol, force_field_method: str = "UFF"
) -> Chem.Mol:
    """
    Identifies the conformer with the lowest energy based on a specified force field method.

    Parameters:
    - molecule (Chem.Mol): The molecule containing multiple conformers.
    - force_field_method (str): The force field method to use for energy calculation. Default is 'UFF'.

    Returns:
    - Chem.Mol: A new RDKit molecule object containing only the lowest energy conformer.

    Raises:
    - ValueError: If the molecule does not contain any conformers.
    """
    _assert_has_conformers(molecule)

    conformer_ids = [conformer.GetId() for conformer in molecule.GetConformers()]
    lowest_energy = float("inf")
    conformer_id_keep = None

    for conformer_id in conformer_ids:
        energy = compute_force_field_energy(molecule, conformer_id, force_field_method)
        if energy < lowest_energy:
            lowest_energy = energy
            conformer_id_keep = conformer_id

    if conformer_id_keep is None:
        raise ValueError("Failed to identify a conformer with finite energy.")

    new_molecule = Chem.Mol(molecule)
    new_molecule.RemoveAllConformers()
    new_molecule.AddConformer(molecule.GetConformer(conformer_id_keep), assignId=True)

    return new_molecule


# def compute_force_field_energy(
#     molecule: Chem.Mol, conformer_id: int, force_field_method: str = "UFF"
# ) -> float:
#     """
#     Computes the force field energy of a given conformer of an RDKit molecule object.

#     Parameters:
#     - molecule (Chem.Mol): The molecule for which to compute the force field energy.
#     conformer_id (int): The ID of the conformer for which to compute the force field energy.
#     - force_field_method (str, optional): The force field method to use. Default to 'UFF'.

#     Returns:
#     float: The computed force field energy.
#     """

#     # If the force field method is 'MMFF', change it to 'MMFF94'
#     if force_field_method == "MMFF":
#         force_field_method = "MMFF94"

#     # If the force field method starts with 'MMFF', use the MMFF force field
#     if force_field_method.startswith("MMFF"):
#         # Get the MMFF properties of the molecule
#         mmff_properties = Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties(
#             mol=molecule, mmffVariant=force_field_method
#         )
#         # Get the MMFF force field of the molecule
#         force_field = Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField(
#             molecule, mmff_properties, confId=conformer_id
#         )
#     else:
#         # If the force field method is not 'MMFF', use the UFF force field
#         force_field = Chem.rdForceFieldHelpers.UFFGetMoleculeForceField(
#             molecule, confId=conformer_id
#         )

#     # Calculate and return the energy of the force field
#     return force_field.CalcEnergy()


def compute_force_field_energy(
    molecule: Chem.Mol, conformer_id: int, force_field_method: str = "UFF"
) -> float:
    """
    Computes the force field energy for a specified conformer of an RDKit molecule object.

    Parameters:
    - molecule (Chem.Mol): The molecule to calculate energy for.
    - conformer_id (int): The ID of the conformer whose energy needs to be computed.
    - force_field_method (str): The force field method to use, defaults to 'UFF'.

    Returns:
    - float: The energy of the specified conformer.

    Raises:
    - RuntimeError: If there is an issue initializing the force field.
    """
    if force_field_method == "MMFF":
        force_field_method = "MMFF94"

    if force_field_method.startswith("MMFF"):
        mmff_properties = rdForceFieldHelpers.MMFFGetMoleculeProperties(
            molecule, mmffVariant=force_field_method
        )
        if not mmff_properties:
            raise RuntimeError(
                f"Failed to initialize MMFF properties for {force_field_method}."
            )
        force_field = rdForceFieldHelpers.MMFFGetMoleculeForceField(
            molecule, mmff_properties, confId=conformer_id
        )
    else:
        force_field = rdForceFieldHelpers.UFFGetMoleculeForceField(
            molecule, confId=conformer_id
        )

    if not force_field:
        raise RuntimeError("Failed to initialize force field.")

    return force_field.CalcEnergy()
