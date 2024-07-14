from rdkit import Chem
from rdkit.Chem.rdDistGeom import ETDG, ETKDG, ETKDGv2, ETKDGv3, srETKDGv3, KDG
import numpy as np
import logging
from typing import Union, Optional, List
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole

logger = logging.getLogger(__name__)


_embedding_method = {
    "ETDG": ETDG(),
    "ETKDG": ETKDG(),
    "ETKDGv2": ETKDGv2(),
    "ETKDGv3": ETKDGv3(),
    "srETKDGv3": srETKDGv3(),
    "KDG": KDG(),
}


_available_ff_methods = ["MMFF", "MMFF94", "MMFF94s", "UFF"]


def _get_num_conformers_from_molecule_size(
    molecule: Chem.Mol,
    max_num_conformers: int = 10,
    min_num_conformers: int = 2,
    decr_num_conformers: int = 0.04,
) -> int:
    """
    Determines the number of conformers to generate based on the size of the molecule.

    Parameters:
    - molecule (Chem.Mol): The molecule for which to determine the number of conformers.
    - max_num_conformers (int, optional): The maximum number of conformers to generate. Default to 10.
    - min_num_conformers (int, optional): The minimum number of conformers to generate. Default to 2.
    - decr_num_conformers (int, optional): The decrement factor for the number of conformers based on the size of the molecule. Default to 0.04.

    Returns:
    int: The number of conformers to generate.
    """

    # Get the number of atoms in the molecule
    num_atoms = molecule.GetNumAtoms()

    # Calculate the number of conformers to generate based on the size of the molecule
    # The number of conformers is the maximum of `min_num_conformers` and the difference between `max_num_conformers` and the product of `num_atoms` and `decr_num_conformers`
    # The `min` function is used to ensure that the number of conformers does not exceed `max_num_conformers - 1`
    return max(
        min_num_conformers,
        int(
            max_num_conformers
            - min(max_num_conformers - 1, num_atoms * decr_num_conformers)
        ),
    )


def _get_max_iter_from_molecule_size(
    molecule: Chem.Mol,
    min_iter: int = 20,
    max_iter: int = 2000,
    incr_iter: int = 10,
) -> int:
    """
    Calculate the maximum number of iterations based on the size of the molecule.

    Parameters:
    - molecule (Chem.Mol): The molecule for which to calculate the maximum number of iterations.
    - min_iter (int, optional): The minimum number of iterations. Default is 20.
    - max_iter (int, optional): The maximum number of iterations. Default is 2000.
    - incr_iter (int, optional): The increment for each atom in the molecule. Default is 10.

    Returns:
    int: The calculated maximum number of iterations.
    """
    return min(max_iter, int(min_iter + incr_iter * molecule.GetNumAtoms()))


# raise error function


def _assert_correct_force_field(force_field):
    """
    Check if the provided force field method is available.

    Parameters:
    - force_field (str): The force field method to check.

    Raises:
    ValueError: If the force field method is not available.
    """
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


# visualize conformer
def visualize_conformers(
    molecule: Chem.Mol,
    force_field_method: str = "MMFF94",
    subImgSize: tuple = (200, 200),
) -> Draw.IPythonConsole.display.Image:
    """
    Visualize the conformers of a molecule.

    Parameters:
    molecule (Chem.Mol): The molecule whose conformers to visualize.
    force_field_method (str, optional): The force field method to use. Default is 'MMFF94'.
    subImgSize (tuple, optional): The size of the sub-images. Default is (200,200).

    Returns:
    Chem.Draw.IPythonConsole.display.Image: The image of the conformers.
    """
    conformers = []  # List to store the conformers of the molecule
    energies = []  # List to store the energies of the conformers

    if force_field_method == "MMFF":  # If the force field method is 'MMFF'
        force_field_method = "MMFF94"  # Change it to 'MMFF94'

    for conformer in molecule.GetConformers():  # For each conformer in the molecule
        new_molecule = Chem.Mol(molecule)  # Create a copy of the molecule
        new_molecule.RemoveAllConformers()  # Remove all conformers from the new molecule
        new_molecule.AddConformer(
            conformer, assignId=True
        )  # Add the current conformer to the new molecule
        conformers.append(
            new_molecule
        )  # Append the new molecule to the list of conformers
        if force_field_method.startswith(
            "MMFF"
        ):  # If the force field method starts with 'MMFF'
            mmff_properties = Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties(
                mol=molecule, mmffVariant=force_field_method
            )  # Get the MMFF properties of the molecule
            ff = Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField(
                molecule, mmff_properties, confId=0
            )  # Get the MMFF force field of the molecule
        else:  # If the force field method does not start with 'MMFF'
            ff = Chem.rdForceFieldHelpers.UFFGetMoleculeForceField(
                new_molecule, confId=0
            )  # Get the UFF force field of the new molecule
        energies.append(
            ff.CalcEnergy()
        )  # Calculate the energy of the conformer and append it to the list of energies
    legends = [
        f"{force_field_method} energy = {energy:.2f}" for energy in energies
    ]  # Create the legends for the conformers
    return Chem.Draw.IPythonConsole.ShowMols(
        conformers, legends=legends, subImgSize=subImgSize
    )  # Show the conformers
