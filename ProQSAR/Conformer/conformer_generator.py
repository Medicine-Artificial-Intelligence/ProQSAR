import pandas as pd
from rdkit import Chem
from joblib import Parallel, delayed
from typing import Dict, Any, List, Union, Optional

from ProQSAR.Conformer import conformer_function


class ConformerGenerator:
    """
    A class used to generate molecular conformers with configurable settings.

    Parameters
    ----------
    num_conformer_candidates : Optional[Union[str, int]]
        Number of conformers to generate, with 'auto' to allow automatic determination.
    embedding_method : str
        The embedding method to use.
    force_field_method : str
        The force field method to use.
    max_iter : Optional[Union[str, int]]
        Maximum number of iterations for generating a conformer.
    keep_hydrogens : bool
        Indicates whether to keep the hydrogens in the selected conformer.
    """

    def __init__(
        self,
        num_conformer_candidates: Optional[Union[str, int]] = "auto",
        embedding_method: str = "ETKDGv2",
        force_field_method: str = "MMFF94",
        max_iter: Optional[Union[str, int]] = "auto",
        keep_hydrogens: bool = False,
    ):
        self.num_conformer_candidates = num_conformer_candidates
        self.embedding_method = embedding_method
        self.force_field_method = force_field_method
        self.max_iter = max_iter
        self.keep_hydrogens = keep_hydrogens

    @staticmethod
    def gen_conformers(
        molecule: Chem.Mol,
        num_conformer_candidates: Optional[Union[str, int]] = "auto",
        embedding_method: str = "ETKDGv2",
        force_field_method: str = "MMFF94",
        max_iter: Optional[Union[str, int]] = "auto",
        keep_hydrogens: bool = False,
    ) -> Optional[Chem.Mol]:
        """
        Generates a conformer for the specified RDKit molecule using provided embedding and force field methods.

        Parameters:
        - molecule (Chem.Mol): The molecule for which to generate a conformer.
        - num_conformer_candidates (Optional[Union[str, int]]): The number or strategy ('auto') to determine
        how many conformer candidates to consider during generation. Defaults to 'auto'.
        - embedding_method (str): The molecular embedding method to use. Defaults to 'ETKDGv2'.
        - force_field_method (str): The force field method to apply for energy minimization. Defaults to 'MMFF94'.
        - max_iter (Optional[Union[str, int]]): The maximum number of iterations for the force field minimization.
        Can be set to 'auto' for automatic determination based on the molecule size. Defaults to 'auto'.
        - keep_hydrogens (bool): Specifies whether to keep hydrogen atoms attached in the final conformer.
        Defaults to False, which means hydrogens are removed.

        Returns:
        - Optional[Chem.Mol]: The generated conformer as an RDKit Mol object, or None if the input is invalid.

        Raises:
        - ValueError: If the input is not a valid RDKit Mol object.
        """
        if not isinstance(molecule, Chem.Mol):
            raise ValueError("Input must be a Chem.Mol object.")

        molecule = Chem.AddHs(molecule)
        molecule = conformer_function.mol_embed(
            molecule, num_conformer_candidates, embedding_method
        )
        molecule = conformer_function.force_field_minimization(
            molecule, force_field_method, max_iter
        )
        molecule = conformer_function.get_lowest_energy_conformer(
            molecule, force_field_method
        )

        if not keep_hydrogens:
            molecule = Chem.RemoveHs(molecule)

        return molecule

    @staticmethod
    def _dict_process(
        input_dict: Dict[Any, Any],
        mol_column: str = "mol",
        num_conformer_candidates: Optional[Union[str, int]] = "auto",
        embedding_method: str = "ETKDGv2",
        force_field_method: str = "MMFF94",
        max_iter: Optional[Union[str, int]] = "auto",
        keep_hydrogens: bool = False,
    ) -> Dict[Any, Any]:
        """
        Processes a dictionary of molecular data to generate conformers for each molecule,
        storing the results in a new key within the same dictionary.

        Parameters:
        - input_dict (Dict[Any, Any]): A dictionary where each key corresponds to a data identifier and
        the value is an RDKit molecule object.
        - mol_column (str): The key under which the RDKit molecule objects are stored in the dictionary.
        Defaults to 'mol'.
        - num_conformer_candidates (Optional[Union[str, int]]): Specifies the number of conformer
        candidates or 'auto' to use an automatic determination. Defaults to 'auto'.
        - embedding_method (str): Embedding method for generating the molecular geometry. Defaults to 'ETKDGv2'.
        - force_field_method (str): Method for force field energy minimization. Defaults to 'MMFF94'.
        - max_iter (Optional[Union[str, int]]): Maximum number of iterations for energy minimization.
        Defaults to 'auto'.
        - keep_hydrogens (bool): Whether to retain hydrogens in the final conformers. Defaults to False.

        Returns:
        - Dict[Any, Any]: A modified dictionary similar to the input but with an additional key ('mol_conf')
        for each entry containing the generated conformer.

        This method modifies the input dictionary in-place, adding a new key with the generated conformers.
        """
        result_dict = input_dict.copy()
        result_dict["mol_conf"] = ConformerGenerator.gen_conformers(
            input_dict[mol_column],
            num_conformer_candidates,
            embedding_method,
            force_field_method,
            max_iter,
            keep_hydrogens,
        )
        return result_dict

    def _conformers_parallel(
        self,
        data: Union[pd.DataFrame, List[Dict[Any, Any]]],
        mol_column: str = "mol",
        n_jobs: int = 4,
        verbose: int = 0,
    ) -> Union[pd.DataFrame, List[Dict[Any, Any]]]:
        """
        Generates conformers in parallel for molecules contained within a pandas DataFrame or
        a list of dictionaries.

        Parameters:
        - data (Union[pd.DataFrame, List[Dict[Any, Any]]]): Data structure containing molecules
        for which conformers are to be generated.
        - mol_column (str): Name of the column or key in the DataFrame/dictionary where
        RDKit molecules are stored. Default is 'mol'.
        - n_jobs (int): Number of parallel jobs to run. Default is 4.
        - verbose (int): Verbosity level for parallel processing. Default is 0.

        Returns:
        - Union[pd.DataFrame, List[Dict[Any, Any]]]: A new DataFrame or list of dictionaries,
        each with an additional key/column containing the generated conformers.
        """
        if isinstance(data, pd.DataFrame):
            result_data = data.copy()
            molecules = result_data[mol_column].tolist()
        elif isinstance(data, list):
            result_data = data
            molecules = [mol[mol_column] for mol in result_data]
        else:
            raise ValueError(
                "Input data must be a pandas DataFrame or a list of dictionaries."
            )

        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(self.gen_conformers)(mol) for mol in molecules
        )

        # Update the original data structure with results
        for i, mol in enumerate(result_data):
            mol["mol_conf"] = results[i]

        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(result_data)
        else:
            return result_data
