import pandas as pd
import numpy as np
import logging
from joblib import Parallel, delayed
from rdkit import Chem
from ProQSAR.Featurizer.pubchem import calcPubChemFingerAll
from ProQSAR.Featurizer.featurizer_wrapper import (
    RDKFp,
    ECFPs,
    MACCs,
    Avalon,
    RDKDes,
    mol2pharm2dgbfp,
)
from typing import Optional, Union, Dict, Any, List


class FeatureGenerator:
    def __init__(self, mol_col, activity_col, ID_col, save_dir, n_jobs=-1, verbose=1):
        self.mol_col = mol_col
        self.activity_col = activity_col
        self.ID_col = ID_col
        self.save_dir = save_dir
        self.n_jobs = n_jobs
        self.verbose = verbose

    @staticmethod
    def _mol_process(
        mol: Optional[Chem.Mol], feature_types: list = ["RDK5"]
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Compute various fingerprints based on the specified feature types for a given molecule object.

        Parameters:
        - mol (Optional[Chem.Mol]): Molecule object.
        - feature_types (list of str): List of fingerprint types to calculate.

        Returns:
        - Dict[str, Optional[np.ndarray]]: A dictionary with keys as fingerprint types
        and values as their respective numpy array representations.

        Raises:
        - ValueError: If the provided molecule object is None.
        """
        if mol is None:
            logging.error("Invalid molecule object provided.")
            return None

        result = {}
        for fp in feature_types:
            try:
                if fp.startswith("RDK"):
                    maxpath = int(fp[-1:])
                    logging.info("bug", maxpath)
                    fp_size = 2048 if maxpath <= 6 else 4096
                    result[fp] = RDKFp(mol, maxPath=maxpath, fpSize=fp_size)
                elif "ECFP" in fp:
                    d = int(fp[-1:]) * 2
                    nBits = 2048 if d < 6 else 4096
                    use_features = "feat" in fp.lower()
                    result[fp] = ECFPs(
                        mol, radius=d, nBits=nBits, useFeatures=use_features
                    )
                elif "FCFP" in fp:
                    d = int(fp[-1:]) * 2
                    nBits = 2048 if d < 6 else 4096
                    use_features = "fcfp" in fp.lower()
                    result[fp] = ECFPs(
                        mol, radius=d, nBits=nBits, useFeatures=use_features
                    )
                elif fp == "MACCS":
                    result[fp] = MACCs(mol)
                elif fp == "avalon":
                    result[fp] = Avalon(mol)
                elif fp == "rdkdes":
                    result[fp] = RDKDes(mol)
                elif fp == "pubchem":
                    result[fp] = calcPubChemFingerAll(mol)
                elif fp == "pharm2dgbfp":
                    result[fp] = mol2pharm2dgbfp(mol)
                else:
                    logging.error(f"Invalid fingerprint type: {fp}")
            except Exception as e:
                logging.error(f"Error processing {fp} for the molecule: {e}")

        return result

    @staticmethod
    def _single_process(
        record: Dict[str, Any],
        mol_col: str,
        activity_col: str,
        ID_col: str,
        feature_types: List[str] = ["RDK5"],
    ) -> Dict[str, Any]:
        """
        Processes a single record to extract features based on the molecule data.

        Parameters:
        - record (Dict[str, Any]): The dictionary containing the data.
        - mol_col (str): The key for the molecule column.
        - activity_col (str): The key for the activity column.
        - ID (str): The key for the identifier column.
        - feature_types (List[str]): List of feature types to process.

        Returns:
        - Dict[str, Any]: A dictionary with the processed features along with the original ID and activity.
        """
        try:
            mol = record[mol_col]
            act = record[activity_col]
            id = record[ID_col]
            result = FeatureGenerator._mol_process(mol, feature_types=feature_types)
            result[ID_col] = id
            result[activity_col] = act
            return result
        except KeyError as e:
            logging.error(f"Missing key in record: {e}")
            return None

    def generate_features(
        self,
        df: Union[pd.DataFrame, List[Dict[str, Any]]],
        feature_types: List[str] = ["RDK5"],
    ) -> pd.DataFrame:
        """
        Generates features for molecules contained in a DataFrame or a list of dictionaries using parallel processing.

        Parameters:
        - df (Union[pd.DataFrame, List[Dict[str, Any]]]): The input data as either
        a DataFrame or a list of dictionaries.
        - feature_types (List[str]): Types of features to generate.

        Returns:
        - pd.DataFrame: A DataFrame containing the original data augmented with new features.

        Raises:
        - ValueError: If the input data type is neither a pandas DataFrame nor a list of dictionaries.
        """
        if isinstance(df, pd.DataFrame):
            data = df.to_dict("records")

        elif isinstance(df, list):
            data = df

        else:
            logging.error("Invalid input data type", exc_info=True)
            return None

        # Parallel processing of records using joblib
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._single_process)(
                record, self.mol_col, self.activity_col, self.ID_col, feature_types
            )
            for record in data
        )

        return pd.DataFrame(results)
