import pandas as pd
import numpy as np
import logging
import os
from joblib import Parallel, delayed
from rdkit import Chem
from ProQSAR.Featurizer.PubChem import calcPubChemFingerAll
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
    def __init__(
        self,
        mol_col: str = "mol",
        activity_col: str = "activity",
        id_col: str = "id",
        feature_types: Union[list, str] = ["ECFP4", "RDK5", "FCFP4"],
        save_dir: Optional[str] = None,
        n_jobs=-1,
        verbose=1,
    ):

        self.mol_col = mol_col
        self.activity_col = activity_col
        self.id_col = id_col
        self.feature_types = feature_types
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
                    d = int(fp[-1:])
                    radius = d // 2 if d != 0 else 0
                    nBits = 2048 if d < 6 else 4096
                    use_features = False
                    result[fp] = ECFPs(
                        mol, radius=radius, nBits=nBits, useFeatures=use_features
                    )
                elif "FCFP" in fp:
                    d = int(fp[-1:])
                    radius = d // 2 if d != 0 else 0
                    nBits = 2048 if d < 6 else 4096
                    use_features = True
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
        id_col: str,
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
            id = record[id_col]
            result = FeatureGenerator._mol_process(mol, feature_types=feature_types)
            result[id_col] = id
            result[activity_col] = act
            return result
        except KeyError as e:
            logging.error(f"Missing key in record: {e}")
            return None

    @staticmethod
    def get_all_types():
        return [
            "ECFP2",
            "ECFP4",
            "ECFP6",
            "FCFP2",
            "FCFP4",
            "FCFP6",
            "RDK5",
            "RDK6",
            "RDK7",
            "MACCS",
            "avalon",
            "rdkdes",
            "pubchem",
            "pharm2dgbfp",
        ]

    def generate_features(
        self,
        df: Union[pd.DataFrame, List[Dict[str, Any]]],
    ) -> Dict[str, pd.DataFrame]:
        """
        Generates features for molecules contained in a DataFrame or a list of dictionaries
        using parallel processing.

        Parameters:
        - df (Union[pd.DataFrame, List[Dict[str, Any]]]): The input data as either
        a DataFrame or a list of dictionaries.

        Returns:
        - Dict[str, pd.DataFrame]: A dictionary where keys are feature types and values are DataFrames
        with expanded fingerprints.

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

        if self.feature_types == "all":
            self.feature_types = FeatureGenerator.get_all_types()

        elif isinstance(self.feature_types, str):
            self.feature_types = [self.feature_types]

        # Parallel processing of records using joblib
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._single_process)(
                record, self.mol_col, self.activity_col, self.id_col, self.feature_types
            )
            for record in data
        )
        results = pd.DataFrame(results)

        feature_dfs = {}

        for feature_type in self.feature_types:
            fp_df = pd.DataFrame(np.stack(results[feature_type]), index=results.index)
            feature_df = pd.concat(
                [results[[self.id_col, self.activity_col]], fp_df], axis=1
            )
            feature_df.columns = feature_df.columns.astype(str)

            if self.save_dir:
                os.makedirs(self.save_dir, exist_ok=True)
                save_path = os.path.join(self.save_dir, f"{feature_type}.csv")
                feature_df.to_csv(save_path, index=False)

            feature_dfs[feature_type] = feature_df

        return feature_dfs

    def setting(self, **kwargs):
        valid_keys = self.__dict__.keys()
        for key in kwargs:
            if key not in valid_keys:
                raise KeyError(f"'{key}' is not a valid attribute of FeatureGenerator.")
        self.__dict__.update(**kwargs)

        return self


    def get_params(self, deep=True) -> dict:
        """Return all hyperparameters as a dictionary."""
        out = {}
        for key in self.__dict__:
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                for sub_key, sub_value in deep_items:
                    out[f"{key}__{sub_key}"] = sub_value
            out[key] = value
            
        return out

    def __repr__(self):
        """Return a string representation of the estimator."""
        class_name = self.__class__.__name__
        params = self.get_params(deep=False)
        param_str = ", ".join(f"{key}={repr(value)}" for key, value in params.items())
        return f"{class_name}({param_str})"