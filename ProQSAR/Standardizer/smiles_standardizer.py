import logging
import pandas as pd
from rdkit import Chem
from joblib import Parallel, delayed
from typing import List, Optional, Tuple, Union
from sklearn.base import BaseEstimator
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


class SMILESStandardizer(BaseEstimator):
    """
    Class for comprehensive standardization of chemical structures represented in SMILES format.
    Utilizes RDKit to process and normalize chemical structures, ensuring consistency and comparability
    in cheminformatics applications.

    Methods:
        standardize_mol: Standardizes RDKit Mol objects with various chemical standardization steps.
        standardize_smiles: Converts SMILES strings to standardized RDKit Mol objects.
        standardize_dict_smiles: Standardizes SMILES strings within a pandas DataFrame or a list of dictionaries.
    """

    def __init__(
        self,
        smiles_col: str = "SMILES",
        normalize: bool = True,
        tautomerize: bool = True,
        remove_salts: bool = False,
        handle_charges: bool = False,
        uncharge: bool = False,
        handle_stereo: bool = True,
        remove_fragments: bool = False,
        largest_fragment_only: bool = False,
        n_jobs: int = 1,
        deactivate: bool = False,
    ):
        self.smiles_col = smiles_col
        self.normalize = normalize
        self.tautomerize = tautomerize
        self.remove_salts = remove_salts
        self.handle_charges = handle_charges
        self.uncharge = uncharge
        self.handle_stereo = handle_stereo
        self.remove_fragments = remove_fragments
        self.largest_fragment_only = largest_fragment_only
        self.n_jobs = n_jobs
        self.deactivate = deactivate

    @staticmethod
    def smiles2mol(smiles: str) -> Optional[Chem.Mol]:
        """
        Convert SMILES string to RDKit Mol object.

        Parameters
        ----------
        smiles : str
            SMILES string to be converted.

        Returns
        -------
        Chem.Mol
            RDKit Mol object.
        """
        try:
            mol = Chem.rdmolfiles.MolFromSmiles(smiles)
            return mol
        except Exception as e:
            logging.error(f"Failed to convert SMILES to Mol: {e}")
            return None

    def standardize_mol(
        self,
        mol: Chem.Mol,
    ) -> Optional[Chem.Mol]:
        """
        Applies a series of standardization procedures to an RDKit Mol object based on specified options.

        Parameters:
            mol (Chem.Mol): The molecule to be standardized.
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
            logging.error("Input {mol} must not be None")

        # Ensure ring information is computed
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)

        # Apply standardization steps
        if self.normalize:
            mol = normalize_molecule(mol)
        if self.tautomerize:
            mol = canonicalize_tautomer(mol)
        if self.remove_salts:
            mol = salts_remover(mol)
        if self.handle_charges:
            mol = reionize_charges(mol)
        if self.uncharge:
            mol = uncharge_molecule(mol)
        if self.handle_stereo:
            assign_stereochemistry(mol, cleanIt=True, force=True)
        if self.remove_fragments or self.largest_fragment_only:
            mol = fragments_remover(mol)

        # Finalize by removing explicit hydrogens and sanitizing
        return remove_hydrogens_and_sanitize(mol)

    def standardize_smiles(
        self, smiles: str
    ) -> Tuple[Optional[str], Optional[Chem.Mol]]:
        """
        Converts a SMILES string to a standardized RDKit Mol object and returns both the SMILES and Mol.

        Parameters:
            smiles (str): The SMILES string to be standardized.

        Returns:
            tuple: A tuple containing the standardized SMILES string and the Mol object, or (None, None)
            if unsuccessful.
        """
        original_mol = SMILESStandardizer.smiles2mol(smiles)
        if not original_mol:
            return None, None

        try:
            standardized_mol = self.standardize_mol(original_mol)
            standardized_smiles = Chem.rdmolfiles.MolToSmiles(standardized_mol)
            return standardized_smiles, standardized_mol
        except Exception as e:
            logging.error(f"Failed to standardize {smiles}: {e}")
            return smiles, original_mol

    def standardize_dict_smiles(
        self,
        data_input: Union[pd.DataFrame, List[dict]],
    ) -> Union[pd.DataFrame, List[dict]]:
        """
        Standardizes SMILES strings within a pandas DataFrame or a list of dictionaries using parallel processing.

        Parameters:
            data_input (DataFrame or list of dicts): Data containing SMILES strings to be standardized.
            key (str): Key or column name for SMILES strings in the data.
            n_jobs (int): Number of jobs to run in parallel.

        Returns:
            DataFrame or list of dicts: The input data with additional columns/keys for
            standardized SMILES and Mol objects.
        """
        if self.deactivate:
            logging.info("SMILESStandardizer is deactivated. Skipping standardization.")
            return data_input

        if isinstance(data_input, pd.DataFrame):
            data_input = data_input.to_dict("records")

        if not isinstance(data_input, list) or not all(
            isinstance(item, dict) for item in data_input
        ):
            raise TypeError(
                "Input must be either a pandas DataFrame or a list of dictionaries."
            )

        standardized_results = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(self.standardize_smiles)(record.get(self.smiles_col, ""))
            for record in data_input
        )

        for i, record in enumerate(data_input):
            record["standardized_" + self.smiles_col], record["standardized_mol"] = (
                standardized_results[i]
            )

        return data_input
