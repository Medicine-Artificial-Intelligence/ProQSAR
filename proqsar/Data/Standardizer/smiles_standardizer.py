import logging
import pandas as pd
from rdkit import Chem
from joblib import Parallel, delayed
from typing import List, Optional, Tuple, Union
from sklearn.base import BaseEstimator
from .standardizer_wrapper import (
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

    This class provides a configurable pipeline of standardization steps using RDKit.
    It ensures molecules are normalized, tautomers canonicalized, salts removed, charges
    adjusted, stereochemistry assigned, fragments filtered, and hydrogen handling is
    consistent. The standardized molecules are output as both RDKit Mol objects and SMILES
    strings, suitable for downstream cheminformatics workflows.

    Methods
    -------
    smiles2mol(smiles: str) -> Optional[Chem.Mol]
        Convert a SMILES string to an RDKit Mol object.
    standardize_mol(mol: Chem.Mol) -> Optional[Chem.Mol]
        Apply all configured standardization steps to an RDKit Mol object.
    standardize_smiles(smiles: str) -> Tuple[Optional[str], Optional[Chem.Mol]]
        Convert and standardize a SMILES string, returning both the canonical SMILES
        and RDKit Mol object.
    standardize_dict_smiles(data_input: Union[pd.DataFrame, List[dict]])
        Apply standardization to all SMILES in a DataFrame or list of dicts.

    :param smiles_col: Column/key name containing SMILES strings in the input data.
    :type smiles_col: str, optional
    :param normalize: If True, normalize molecules (aromaticity, functional groups, etc.).
    :type normalize: bool, optional
    :param tautomerize: If True, canonicalize tautomers into a single representation.
    :type tautomerize: bool, optional
    :param remove_salts: If True, strip counter-ions and salt fragments.
    :type remove_salts: bool, optional
    :param handle_charges: If True, reionize charges to standard protonation states.
    :type handle_charges: bool, optional
    :param uncharge: If True, remove charges by neutralizing charged species.
    :type uncharge: bool, optional
    :param handle_stereo: If True, assign or clean stereochemistry information.
    :type handle_stereo: bool, optional
    :param remove_fragments: If True, discard extra fragments (e.g., keep only parent molecule).
    :type remove_fragments: bool, optional
    :param largest_fragment_only: If True, retain only the largest connected fragment.
    :type largest_fragment_only: bool, optional
    :param n_jobs: Number of parallel jobs to use when standardizing a batch of molecules.
    :type n_jobs: int, optional
    :param deactivate: If True, disable all standardization steps (useful for debugging).
    :type deactivate: bool, optional
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
        Convert a SMILES string to RDKit Mol object.

        :param smiles: SMILES string to be converted.
        :type smiles: str
        :return: RDKit Mol object or None if conversion fails.
        :rtype: Optional[Chem.Mol]
        """
        try:
            mol = Chem.rdmolfiles.MolFromSmiles(smiles)
            return mol
        except Exception as e:
            logging.error(f"Failed to convert SMILES to Mol: {e}")
            return None

    def standardize_mol(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """
        Standardize an RDKit Mol object using various chemical standardization steps.

        :param mol: The molecule to be standardized.
        :type mol: Chem.Mol
        :return: The standardized molecule or None if it cannot be processed.
        :rtype: Optional[Chem.Mol]
        :raises ValueError: If the input molecule is None.
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
        Convert a SMILES string to a standardized RDKit Mol object
        and return both standardized SMILES and Mol.

        :param smiles: The SMILES string to be standardized.
        :type smiles: str
        :return: Tuple containing the standardized SMILES string and Mol object,
                 or (None, None) if unsuccessful.
        :rtype: Tuple[Optional[str], Optional[Chem.Mol]]
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
        self, data_input: Union[pd.DataFrame, List[dict]]
    ) -> Union[pd.DataFrame, List[dict]]:
        """
        Standardize SMILES strings within a pandas DataFrame or a list of dictionaries
        using parallel processing.

        :param data_input: Data containing SMILES strings to be standardized.
                           Can be a pandas DataFrame or list of dicts.
        :type data_input: Union[pandas.DataFrame, List[dict]]
        :return: Input data with additional standardized SMILES and Mol columns/keys.
        :rtype: Union[pandas.DataFrame, List[dict]]
        :raises TypeError: If input is not a DataFrame or list of dictionaries.
        :raises Exception: Any unexpected exception encountered during processing.
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
