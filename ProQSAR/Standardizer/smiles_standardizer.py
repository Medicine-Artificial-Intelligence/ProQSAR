from typing import List, Optional, Tuple, Union
import pandas as pd
from rdkit import Chem
from joblib import Parallel, delayed
from ProQSAR.Standardizer.standardizer_wrapper import (
    normalize_molecule,
    canonicalize_tautomer,
    salts_remover,
    reionize_charges,
    uncharge_molecule,
    assign_stereochemistry,
    fragments_remover,
    remove_hydrogens_and_sanitize,
)
from ProQSAR.Utils.chem_utils import draw_mol_with_SVG


class SMILESStandardizer:
    """
    Class for comprehensive standardization of chemical structures represented in SMILES format.
    Utilizes RDKit to process and normalize chemical structures, ensuring consistency and comparability
    in cheminformatics applications.

    Methods:
        standardize_mol: Standardizes RDKit Mol objects with various chemical standardization steps.
        standardize_smiles: Converts SMILES strings to standardized RDKit Mol objects.
        standardize_dict_smiles: Standardizes SMILES strings within a pandas DataFrame or a list of dictionaries.
    """

    def standardize_mol(
        self,
        mol: Chem.Mol,
        verbose: bool = False,
        normalize: bool = True,
        tautomerize: bool = True,
        remove_salts: bool = False,
        handle_charges: bool = False,
        uncharge: bool = False,
        handle_stereo: bool = True,
        remove_fragments: bool = False,
        largest_fragment_only: bool = False,
    ) -> Optional[Chem.Mol]:
        """
        Applies a series of standardization procedures to an RDKit Mol object based on specified options.

        Parameters:
            mol (Chem.Mol): The molecule to be standardized.
            verbose (bool): If True, the molecule is visualized at each step.
            normalize (bool): Applies normalization corrections.
            tautomerize (bool): Canonicalizes tautomers.
            remove_salts (bool): Removes salt fragments.
            handle_charges (bool): Adjusts molecule to its most likely ionic state.
            uncharge (bool): Neutralizes the molecule by removing counter-ions.
            handle_stereo (bool): Handles stereochemistry.
            remove_fragments (bool): Removes small fragments.
            largest_fragment_only (bool): Keeps only the largest fragment.

        Returns:
            Chem.Mol: The standardized molecule or None if the molecule cannot be processed.

        Raises:
            ValueError: If the input molecule is None.
        """
        if mol is None:
            raise ValueError("Input molecule must not be None")

        # Ensure ring information is computed
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)

        # Apply standardization steps
        if normalize:
            mol = normalize_molecule(mol)
        if tautomerize:
            mol = canonicalize_tautomer(mol)
        if remove_salts:
            mol = salts_remover(mol)
        if handle_charges:
            mol = reionize_charges(mol)
        if uncharge:
            mol = uncharge_molecule(mol)
        if handle_stereo:
            assign_stereochemistry(mol, cleanIt=True, force=True)
        if remove_fragments or largest_fragment_only:
            mol = fragments_remover(mol)

        # Visualize the molecule if verbose is set
        if verbose:
            draw_mol_with_SVG(mol)

        # Finalize by removing explicit hydrogens and sanitizing
        return remove_hydrogens_and_sanitize(mol)

    def standardize_smiles(
        self, smiles: str, visualize: bool = False, **kwargs
    ) -> Tuple[Optional[str], Optional[Chem.Mol]]:
        """
        Converts a SMILES string to a standardized RDKit Mol object and returns both the SMILES and Mol.

        Parameters:
            smiles (str): The SMILES string to be standardized.
            visualize (bool): If set, visualizes the molecule during standardization.

        Returns:
            tuple: A tuple containing the standardized SMILES string and the Mol object, or (None, None)
            if unsuccessful.
        """
        original_mol = Chem.MolFromSmiles(smiles)
        if not original_mol:
            return None, None

        try:
            standardized_mol = self.standardize_mol(
                original_mol, verbose=visualize, **kwargs
            )
            standardized_smiles = Chem.MolToSmiles(standardized_mol)
            return standardized_smiles, standardized_mol
        except Chem.MolSanitizeException:
            return "Sanitization failed for SMILES: " + smiles, None

    def standardize_dict_smiles(
        self,
        data_input: Union[pd.DataFrame, List[dict]],
        key: str = "SMILES",
        visualize: bool = False,
        n_jobs: int = 4,
        **kwargs
    ) -> Union[pd.DataFrame, List[dict]]:
        """
        Standardizes SMILES strings within a pandas DataFrame or a list of dictionaries using parallel processing.

        Parameters:
            data_input (DataFrame or list of dicts): Data containing SMILES strings to be standardized.
            key (str): Key or column name for SMILES strings in the data.
            visualize (bool): If True, visualizes the molecules during standardization.
            n_jobs (int): Number of jobs to run in parallel.

        Returns:
            DataFrame or list of dicts: The input data with additional columns/keys for
            standardized SMILES and Mol objects.
        """
        data_type = type(data_input)
        print(data_type)
        if isinstance(data_input, pd.DataFrame):
            data_input = data_input.to_dict("records")

        if not isinstance(data_input, list) or not all(
            isinstance(item, dict) for item in data_input
        ):
            raise TypeError(
                "Input must be either a pandas DataFrame or a list of dictionaries."
            )

        standardized_results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(self.standardize_smiles)(
                reaction_data.get(key, ""), visualize=visualize, **kwargs
            )
            for reaction_data in data_input
        )

        for i, reaction_data in enumerate(data_input):
            reaction_data["standardized_" + key], reaction_data["standardized_mol"] = (
                standardized_results[i]
            )

        return data_input
