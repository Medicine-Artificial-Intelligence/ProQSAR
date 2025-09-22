import pandas as pd
import numpy as np
import logging
import os
from joblib import Parallel, delayed
from rdkit import Chem
from sklearn.base import BaseEstimator
from mordred import Calculator, descriptors
from .PubChem import calcPubChemFingerAll
from .featurizer_wrapper import (
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

    Typical usage
    -------------
    .. code-block:: python

        fg = FeatureGenerator(n_jobs=4, save_dir="out")
        feature_dfs = fg.generate_features(df)  # dict: feature_type -> DataFrame

    :param mol_col: Column name in input records containing RDKit Mol objects.
    :type mol_col: str, optional
    :param activity_col: Column name for target/activity values.
    :type activity_col: str, optional
    :param id_col: Column name for a unique sample identifier.
    :type id_col: str, optional
    :param smiles_col: Column name holding SMILES strings if present.
    :type smiles_col: str, optional
    :param feature_types: Names of feature sets to compute.
                         Defaults to commonly used types:
                         ["ECFP4","FCFP4","RDK5","MACCS","avalon","rdkdes","pubchem","mordred"].
    :type feature_types: list[str] or str, optional
    :param save_dir: Directory to save generated feature CSVs (if provided).
    :type save_dir: Optional[str], optional
    :param data_name: Base name used when saving files (appended with feature type).
    :type data_name: Optional[str], optional
    :param n_jobs: Number of parallel jobs (joblib) to use for per-molecule processing.
    :type n_jobs: int, optional
    :param verbose: Verbosity for Parallel.
    :type verbose: int, optional
    :param deactivate: If True, generation is skipped and input is returned unchanged.
    :type deactivate: bool, optional

    :ivar mol_col: Stores the molecule column name.
    :ivar activity_col: Stores the activity/target column name.
    :ivar id_col: Stores the ID column name.
    :ivar smiles_col: Stores the SMILES column name.
    :ivar feature_types: List of feature types requested.
    :ivar save_dir: Directory path for saving outputs.
    :ivar data_name: Dataset name prefix for saving.
    :ivar n_jobs: Number of parallel jobs.
    :ivar verbose: Verbosity level.
    :ivar deactivate: Skip feature generation if True.
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
        n_jobs: int = 1,
        verbose: int = 0,
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

        :param mol: RDKit molecule object to process. If None, an error is logged.
        :type mol: Optional[Chem.Mol]
        :param feature_types: List of fingerprint/descriptor type names to compute.
        :type feature_types: list[str]
        :return: Mapping of feature_type -> NumPy array (or None for failed types).
        :rtype: dict[str, Optional[numpy.ndarray]] or None
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
        Extract a molecule record and compute fingerprints/descriptors.

        :param record: Dictionary containing molecule data.
        :type record: dict
        :param mol_col: Column key for RDKit Mol.
        :type mol_col: str
        :param activity_col: Column key for activity values.
        :type activity_col: str
        :param id_col: Column key for unique identifier.
        :type id_col: str
        :param smiles_col: Column key for SMILES (optional).
        :type smiles_col: str, optional
        :param feature_types: Feature types to compute.
        :type feature_types: list[str]
        :return: Dictionary containing fingerprint arrays and metadata, or None on failure.
        :rtype: dict[str, Any] or None
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
    def get_all_types() -> List[str]:
        """
        Return the list of supported feature type names.

        :return: Supported feature types.
        :rtype: list[str]
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

        :param df: Input dataset. Either a pandas DataFrame or list of dict records.
        :type df: pandas.DataFrame or list[dict]
        :return: Mapping feature_type -> DataFrame. Each DataFrame includes metadata
                 (id, activity, SMILES, mol) plus fingerprint columns.
                 Returns `df` unchanged if deactivated, or None on error.
        :rtype: dict[str, pandas.DataFrame] or pandas.DataFrame or None
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

        # Parallel processing of records
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
            fp_df = pd.DataFrame(np.stack(results[feature_type]), index=results.index)

            if feature_type == "mordred":
                calc = Calculator(descriptors, ignore_3D=True)
                fp_df.columns = [str(des) for des in calc.descriptors]

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
