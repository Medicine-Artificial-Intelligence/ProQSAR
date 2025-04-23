import pandas as pd
from rdkit import Chem
from ProQSAR.Conformer._embedding import Embeddings
from ProQSAR.Conformer._force_field import ForceField
from typing import Any, Dict, Optional, Union
import logging
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


class ConformationGenerator:
    """
    A class to handle the generation and optimization of conformers for molecules.
    """

    def __init__(self) -> None:
        """
        Initializes the ConformationGenerator instance.
        """
        pass

    @staticmethod
    def _mol_process(
        molecule: Chem.Mol,
        num_conformers: Optional[Union[int, str]] = "auto",
        embedding_method: str = "ETKDGv3",
        num_threads: int = 1,
        random_coords_threshold: int = 100,
        random_seed: int = 42,
        force_field_method: Optional[str] = "MMFF94",
        max_iter: Optional[Union[int, str]] = "auto",
        return_energies: bool = False,
        **kwargs: Any
    ) -> tuple[Chem.Mol, float]:
        """
        Processes a molecule to generate and minimize conformers.

        Parameters:
        - molecule (Chem.Mol): The RDKit molecule object.
        - num_conformers (Optional[Union[int, str]]): Number of conformers to generate or
        'auto'. Defaults to 'auto'.
        - embedding_method (str): Method for embedding. Defaults to 'ETKDGv3'.
        - num_threads (int): Number of threads to use. Defaults to 1.
        - random_coords_threshold (int): Threshold for random coordinates.
        Defaults to 100.
        - random_seed (int): Seed for random number generation. Defaults to 42.
        - force_field_method (Optional[str]): Force field method for minimization.
        Defaults to 'MMFF94'.
        - max_iter (Optional[Union[int, str]]): Maximum iterations for minimization
        or 'auto'. Defaults to 'auto'.
        - return_energies (bool): Whether to return energies. Defaults to False.
        - **kwargs (Any): Additional parameters for minimization.

        Returns:
        - tuple[Chem.Mol, float]: The minimized molecule and its energy.
        """
        try:
            embed = Embeddings.mol_embed(
                molecule,
                num_conformers,
                embedding_method,
                num_threads,
                random_coords_threshold,
                random_seed,
            )
            minimize = ForceField.force_field_minimization(
                embed,
                force_field_method,
                max_iter,
                return_energies,
                num_threads,
                **kwargs
            )
            minimized_mol = ForceField.get_lowest_energy_conformer(
                minimize, force_field_method
            )
            minimized_energy = ForceField.compute_force_field_energy(
                minimized_mol, 0, force_field_method
            )
            logger.info("Successfully processed molecule: %s", molecule)
            return minimized_mol, minimized_energy
        except Exception as e:
            logger.error("Error in _mol_process: %s", str(e))
            raise

    @staticmethod
    def _smiles_process(
        smiles: str,
        num_conformers: Optional[Union[int, str]] = "auto",
        embedding_method: str = "ETKDGv3",
        num_threads: int = 1,
        random_coords_threshold: int = 100,
        random_seed: int = 42,
        force_field_method: Optional[str] = "MMFF94",
        max_iter: Optional[Union[int, str]] = "auto",
        return_energies: bool = False,
        **kwargs: Any
    ) -> tuple[Optional[Chem.Mol], Optional[float]]:
        """
        Processes a SMILES string to generate and minimize conformers.

        Parameters:
        - smiles (str): The SMILES representation of the molecule.
        - num_conformers (Optional[Union[int, str]]): Number of conformers to generate or
        'auto'. Defaults to 'auto'.
        - embedding_method (str): Method for embedding. Defaults to 'ETKDGv3'.
        - num_threads (int): Number of threads to use. Defaults to 1.
        - random_coords_threshold (int): Threshold for random coordinates.
        Defaults to 100.
        - random_seed (int): Seed for random number generation. Defaults to 42.
        - force_field_method (Optional[str]): Force field method for minimization.
        Defaults to 'MMFF94'.
        - max_iter (Optional[Union[int, str]]): Maximum iterations for minimization
        or 'auto'. Defaults to 'auto'.
        - return_energies (bool): Whether to return energies. Defaults to False.
        - **kwargs (Any): Additional parameters for minimization.

        Returns:
        - tuple[Optional[Chem.Mol], Optional[float]]: The minimized molecule and its
        energy. Returns (None, None) if the SMILES is invalid or an error occurs.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.error("Invalid SMILES string: %s", smiles)
                return None, None

            return ConformationGenerator._mol_process(
                mol,
                num_conformers,
                embedding_method,
                num_threads,
                random_coords_threshold,
                random_seed,
                force_field_method,
                max_iter,
                return_energies,
                **kwargs
            )
        except Exception as e:
            logger.error("Error in _smiles_process: %s", str(e))
            return None, None

    @staticmethod
    def _dict_process(
        input_dict: Dict[str, Any],
        smi_col: str = "SMILES",
        num_conformers: Optional[Union[int, str]] = "auto",
        embedding_method: str = "ETKDGv3",
        num_threads: int = 1,
        random_coords_threshold: int = 100,
        random_seed: int = 42,
        force_field_method: Optional[str] = "MMFF94",
        max_iter: Optional[Union[int, str]] = "auto",
        return_energies: bool = False,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Processes an SMILES string from the input dictionary to compute molecular conformers and possibly calculate
        energy differences. The results, including the energy and conformer, are added back to the input dictionary.

        Parameters:
        - input_dict (Dict[str, Any]): Input dictionary containing molecular data.
        - smi_col (str): Key in dictionary for SMILES string. Defaults to 'SMILES'.
        - num_conformers (Optional[Union[int, str]]): Number of conformers to generate or 'auto'.
        - embedding_method (str): Method for molecular embedding.
        - num_threads (int): Number of threads to use for calculations.
        - random_coords_threshold (int): Threshold for using random coordinates in calculations.
        - random_seed (int): Seed for random number generator.
        - force_field_method (Optional[str]): Force field method for energy minimization.
        - max_iter (Optional[Union[int, str]]): Maximum iterations for minimization or 'auto'.
        - return_energies (bool): Flag to decide if energies should be returned.
        - **kwargs (Any): Additional keyword arguments for other processing needs.

        Returns:
        - Dict[str, Any]: The input dictionary enhanced with 'energy' and 'conformer' data.
        """
        data = input_dict.copy()
        smi = data.get(smi_col)
        if smi is None:
            data["energy"] = None
            data["conformer"] = None
            return data

        minimized_mol, minimized_energy = ConformationGenerator._smiles_process(
            smi,
            num_conformers,
            embedding_method,
            num_threads,
            random_coords_threshold,
            random_seed,
            force_field_method,
            max_iter,
            return_energies,
            **kwargs
        )
        data["energy"] = minimized_energy
        data["conformer"] = minimized_mol
        return data

    @classmethod
    def parallel_process(
        cls,
        input_data: Union[list, pd.DataFrame],
        smi_col: str = "SMILES",
        num_conformers: Optional[Union[int, str]] = "auto",
        embedding_method: str = "ETKDGv3",
        num_threads: int = 1,
        random_coords_threshold: int = 100,
        random_seed: int = 42,
        force_field_method: Optional[str] = "MMFF94",
        max_iter: Optional[Union[int, str]] = "auto",
        return_energies: bool = False,
        n_jobs: int = 1,  # Use all available cores
        **kwargs: Any
    ) -> list:
        """
        Processes a list or DataFrame of molecular data in parallel using multiple cores,
        each entry processed to compute energy difference based on SMILES strings.

        Parameters:
        - input_data (Union[list, pd.DataFrame]): List of dictionaries or a DataFrame containing molecular data.
        - smi_col (str): Column or key for the SMILES string.
        - num_conformers, embedding_method, num_threads, random_coords_threshold,
        random_seed, force_field_method, max_iter, return_energies as in _dict_process.
        - n_jobs (int): Number of jobs to run in parallel. -1 uses all cores.

        Returns:
        - list: A list of floats representing the energy differences or NaNs for each input.
        """
        if isinstance(input_data, pd.DataFrame):
            input_data = input_data.to_dict("records")

        results = Parallel(n_jobs=n_jobs)(
            delayed(cls._dict_process)(
                input_dict=record,
                smi_col=smi_col,
                num_conformers=num_conformers,
                embedding_method=embedding_method,
                num_threads=num_threads,
                random_coords_threshold=random_coords_threshold,
                random_seed=random_seed,
                force_field_method=force_field_method,
                max_iter=max_iter,
                return_energies=return_energies,
                **kwargs
            )
            for record in input_data
        )
        return results
