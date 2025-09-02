import pandas as pd
import numpy as np
import logging
import os
from joblib import Parallel, delayed
from rdkit import Chem
from sklearn.base import BaseEstimator
from mordred import Calculator, descriptors
from ProQSAR.Featurizer.PubChem import calcPubChemFingerAll
from ProQSAR.Featurizer.featurizer_wrapper import (
    RDKFp,
    ECFPs,
    MACCs,
    Avalon,
    RDKDes,
    mol2pharm2dgbfp,
    MordredDes,
)
from typing import Optional, Union, Dict, Any, List


class FeatureGenerator(BaseEstimator):
    """
    Transformer that generates molecular feature DataFrames for a variety of
    fingerprint and descriptor types.

    Typical usage:
        fg = FeatureGenerator(n_jobs=4, save_dir="out")
        feature_dfs = fg.generate_features(df)  # dict: feature_type -> DataFrame

    Parameters
    ----------
    mol_col : str
        Column name in input records containing RDKit Mol objects (default "mol").
    activity_col : str
        Column name for target/activity values (default "activity").
    id_col : str
        Column name for a unique sample identifier (default "id").
    smiles_col : str
        Column name holding SMILES strings if present (default "SMILES").
    feature_types : list[str] | str
        Names of feature sets to compute. Defaults to commonly used types:
        ["ECFP4","FCFP4","RDK5","MACCS","avalon","rdkdes","pubchem","mordred"].
        Use FeatureGenerator.get_all_types() to list supported names.
    save_dir : Optional[str]
        If provided, generated feature CSVs will be saved under this directory.
    data_name : Optional[str]
        Base name used when saving files (appended with feature type).
    n_jobs : int
        Number of parallel jobs (joblib) to use for per-molecule processing.
    verbose : int
        Verbosity for Parallel.
    deactivate : bool
        If True, generation is skipped and input is returned unchanged.

    Attributes
    ----------
    All constructor args are stored as instance attributes.
    """

    def __init__(
        self,
        mol_col: str = "mol",
        activity_col: str = "activity",
        id_col: str = "id",
        smiles_col: str = "SMILES",
        feature_types: Union[list, str] = [
            "ECFP4",
            "FCFP4",
            "RDK5",
            "MACCS",
            "avalon",
            "rdkdes",
            "pubchem",
            "mordred",
        ],
        save_dir: Optional[str] = None,
        data_name: Optional[str] = None,
        n_jobs=1,
        verbose=0,
        deactivate: bool = False,
    ):

        self.mol_col = mol_col
        self.activity_col = activity_col
        self.id_col = id_col
        self.smiles_col = smiles_col
        self.save_dir = save_dir
        self.data_name = data_name
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.deactivate = deactivate
        self.feature_types = feature_types

    @staticmethod
    def _mol_process(
        mol: Optional[Chem.Mol], feature_types: list = ["RDK5"]
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Compute fingerprint/descriptor arrays for a single RDKit Mol object.

        Parameters
        ----------
        mol : rdkit.Chem.Mol or None
            Molecule to process. If None, an error is logged and None is returned.
        feature_types : list[str]
            List of fingerprint/descriptor type names to compute.

        Returns
        -------
        dict or None
            A mapping feature_type -> numpy array (or None for failed types), or
            None when `mol` is None.
        """
        if mol is None:
            logging.error("Invalid molecule object provided.")
            return None

        result = {}
        for fp in feature_types:
            try:
                if fp.startswith("RDK"):
                    maxpath = int(fp[-1:])
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
                elif fp == "mordred":
                    result[fp] = MordredDes(mol)
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
        smiles_col: str = "SMILES",
        feature_types: List[str] = ["RDK5"],
    ) -> Dict[str, Any]:
        """
        Joblib-compatible wrapper that extracts a molecule record and computes
        fingerprints/descriptors using `_mol_process`.

        Parameters
        ----------
        record : dict
            A mapping containing at least mol_col and id_col (and optionally
            activity_col and smiles_col).
        mol_col, activity_col, id_col, smiles_col : str
            Column keys within the record.
        feature_types : list[str]
            Feature types passed to `_mol_process`.

        Returns
        -------
        dict or None
            A dictionary containing fingerprint arrays and preserved metadata
            (id, mol, activity, smiles) or None if required keys are missing.
        """
        try:
            mol = record[mol_col]
            id = record[id_col]
            result = FeatureGenerator._mol_process(mol, feature_types=feature_types)
            result[id_col] = id
            result[mol_col] = record[mol_col]
            if activity_col in record.keys():
                result[activity_col] = record[activity_col]
            if smiles_col in record.keys():
                result[smiles_col] = record[smiles_col]
            return result
        except KeyError as e:
            logging.error(f"Missing key in record: {e}")
            return None

    @staticmethod
    def get_all_types():
        """
        Return the list of supported feature type names.
        """
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
            "mordred",
            # "pharm2dgbfp",
        ]

    def generate_features(
        self,
        df: Union[pd.DataFrame, List[Dict[str, Any]]],
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute feature DataFrames for each requested feature type.

        Parameters
        ----------
        df : pandas.DataFrame or list[dict]
            Input data. If a DataFrame is provided, it is converted to a list
            of records with `to_dict('records')`. Each record must contain the
            keys matching mol_col and id_col (and optionally activity_col/smiles_col).

        Returns
        -------
        dict | DataFrame | list | None
            - If `deactivate` is True, the original `df` is returned unchanged.
            - Otherwise, returns a dict mapping feature_type -> pandas.DataFrame
              (each DataFrame includes id/activity/SMILES/mol columns + fingerprint columns).
            - Returns None on invalid input types or unexpected failures.
        """
        if self.deactivate:
            logging.info("FeatureGenerator is deactivated. Skipping generate feature.")
            return df

        if isinstance(df, pd.DataFrame):
            data = df.to_dict("records")

        elif isinstance(df, list):
            data = df

        else:
            logging.error("Invalid input data type", exc_info=True)
            return None

        if isinstance(self.feature_types, str):
            self.feature_types = [self.feature_types]

        # Parallel processing of records using joblib
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._single_process)(
                record,
                self.mol_col,
                self.activity_col,
                self.id_col,
                self.smiles_col,
                self.feature_types,
            )
            for record in data
        )
        results = pd.DataFrame(results)

        feature_dfs = {}

        for feature_type in self.feature_types:
            # Stack fingerprint arrays into a 2D numpy array and create a DataFrame
            fp_df = pd.DataFrame(np.stack(results[feature_type]), index=results.index)

            if feature_type == "mordred":
                calc = Calculator(descriptors, ignore_3D=True)
                fp_df.columns = [str(des) for des in calc.descriptors]

            # Concat with ID & Activity columns
            feature_df = pd.concat(
                [
                    results.filter(
                        items=[
                            self.id_col,
                            self.activity_col,
                            self.smiles_col,
                            self.mol_col,
                        ]
                    ),
                    fp_df,
                ],
                axis=1,
            )

            feature_df.columns = feature_df.columns.astype(str)

            if self.save_dir:
                os.makedirs(self.save_dir, exist_ok=True)
                if self.data_name:
                    save_path = os.path.join(
                        self.save_dir, f"{self.data_name}_{feature_type}.csv"
                    )
                else:
                    save_path = os.path.join(self.save_dir, f"{feature_type}.csv")
                feature_df.to_csv(save_path, index=False)

            feature_dfs[feature_type] = feature_df

        return feature_dfs
