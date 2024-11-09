import logging
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Optional, Union

logger = logging.getLogger(__name__)


class ForceField:
    def __init__(self) -> None:
        pass

    @staticmethod
    def force_field_minimization(
        molecule: Chem.Mol,
        force_field_method: Optional[str] = "MMFF94",
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
        - force_field_method (str, optional): The force field method to use.
        Default to 'MMFF94'.
        - max_iter (int or str, optional): Maximum number of iterations for minimization.
        If 'auto', the number of iterations will depend on the size of the molecule.
        Default to 'auto'.
        - return_energies (bool, optional): Whether to return the energies
        of the minimized conformers. Default to False.
        - num_threads (int, optional): Number of threads to use. Default to 1.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        - Chem.Mol or np.ndarray: The minimized molecule.
        If `return_energies` is True, also returns the energies of
        the minimized conformers.
        """
        ForceField._assert_has_conformers(molecule)
        ForceField._assert_correct_force_field(force_field_method)

        # Get copy, to leave original molecle untouched
        molecule = Chem.Mol(molecule)

        if max_iter is None or max_iter == "auto":
            max_iter = ForceField._get_max_iter_from_molecule_size(molecule)

        if force_field_method == "MMFF":
            force_field_method = "MMFF94"

        try:
            if force_field_method.startswith("MMFF"):
                # Merck Molecular Force Field (MMFF; specifically MMFF94 or MMFF94s)
                AllChem.MMFFSanitizeMolecule(molecule)
                result = AllChem.MMFFOptimizeMoleculeConfs(
                    molecule,
                    maxIters=max_iter,
                    numThreads=num_threads,
                    mmffVariant=force_field_method,
                    **kwargs,
                )
            else:
                # Universal Force Field (UFF)
                result = AllChem.UFFOptimizeMoleculeConfs(
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

    def compute_force_field_energy(
        molecule: Chem.Mol, conformer_id: int, force_field_method: str = "MMFF94"
    ) -> float:
        """
        Computes the force field energy for a specified conformer of an RDKit molecule
        object using MMFF94 or UFF force fields. Additional variants of MMFF can also be
        specified explicitly.

        Parameters:
        ----------
        - molecule (Chem.Mol): The RDKit molecule object to calculate energy for.
        - conformer_id (int): The ID of the conformer whose energy needs to be computed.
        - force_field_method (str, optional): The force field method to use.
        Defaults to 'MMFF94'. Supports 'MMFF94', 'MMFF94s', and 'UFF'.

        Returns:
        -------
        - float: The energy of the specified conformer.

        Raises:
        ------
        RuntimeError
        - If there is an issue initializing the force field or if the conformer ID
        is invalid.
        ValueError
        - If the force field method is not supported or the conformer ID does not exist.
        """
        if force_field_method not in ["MMFF94", "MMFF94s", "UFF"]:
            raise ValueError(f"Unsupported force field method: {force_field_method}")

        if conformer_id >= molecule.GetNumConformers():
            raise ValueError(
                f"Conformer ID {conformer_id}" + " does not exist in the molecule."
            )

        try:
            if force_field_method.startswith("MMFF"):
                mmff_properties = AllChem.MMFFGetMoleculeProperties(
                    molecule, mmffVariant=force_field_method
                )
                if not mmff_properties:
                    raise RuntimeError(
                        f"Failed to initialize MMFF properties for {force_field_method}."
                    )
                force_field = AllChem.MMFFGetMoleculeForceField(
                    molecule, mmff_properties, confId=conformer_id
                )
            elif force_field_method == "UFF":
                force_field = AllChem.UFFGetMoleculeForceField(
                    molecule, confId=conformer_id
                )

            if not force_field:
                raise RuntimeError("Failed to initialize the force field.")

            energy = force_field.CalcEnergy()
            return energy
        except Exception as e:
            logging.error(f"Error computing force field energy: {e}")
            raise RuntimeError(f"Error computing force field energy: {str(e)}")

    @staticmethod
    def get_lowest_energy_conformer(
        molecule: Chem.Mol, force_field_method: str = "MMFF94"
    ) -> Chem.Mol:
        """
        Identifies the conformer with the lowest energy based on a specified
        force field method.

        Parameters:
        - molecule (Chem.Mol): The molecule containing multiple conformers.
        - force_field_method (str): The force field method to use for energy calculation.
        Default is 'UFF'.

        Returns:
        - Chem.Mol: A new RDKit molecule object containing only the
        lowest energy conformer.

        Raises:
        - ValueError: If the molecule does not contain any conformers.
        """
        ForceField._assert_has_conformers(molecule)

        conformer_ids = [conformer.GetId() for conformer in molecule.GetConformers()]
        lowest_energy = float("inf")
        conformer_id_keep = None

        for conformer_id in conformer_ids:
            energy = ForceField.compute_force_field_energy(
                molecule, conformer_id, force_field_method
            )
            if energy < lowest_energy:
                lowest_energy = energy
                conformer_id_keep = conformer_id

        if conformer_id_keep is None:
            raise ValueError("Failed to identify a conformer with finite energy.")

        new_molecule = Chem.Mol(molecule)
        new_molecule.RemoveAllConformers()
        new_molecule.AddConformer(
            molecule.GetConformer(conformer_id_keep), assignId=True
        )

        return new_molecule

    @staticmethod
    def _get_max_iter_from_molecule_size(
        molecule: Chem.Mol,
        min_iter: int = 20,
        max_iter: int = 2000,
        incr_iter: int = 10,
    ) -> int:
        """
        Calculates the maximum number of iterations for molecular force field calculations
        based on molecule size.

        Parameters:
        - molecule (Chem.Mol): The molecule for which to calculate the maximum number of
        iterations.
        - min_iter (int, optional): The minimum number of iterations. Default is 20.
        - max_iter (int, optional): The maximum number of iterations. Default is 2000.
        - incr_iter (int, optional): The increment for each atom in the molecule.
        Default is 10.

        Returns:
        - int: Maximum number of iterations.
        """
        num_atoms = molecule.GetNumAtoms()
        return min(max_iter, min_iter + num_atoms * incr_iter)

    @staticmethod
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

    @staticmethod
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
