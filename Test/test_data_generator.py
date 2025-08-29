import os
import shutil
import tempfile
import unittest
import pandas as pd
from rdkit import Chem
from ProQSAR.Standardizer.smiles_standardizer import SMILESStandardizer
from ProQSAR.Featurizer.feature_generator import FeatureGenerator
from sklearn.base import BaseEstimator
from typing import Optional

class DataGenerator(BaseEstimator):
    def __init__(
        self,
        activity_col: str,
        id_col: str,
        smiles_col: str,
        mol_col: str = "mol",
        n_jobs: int = 1,
        save_dir: Optional[str] = "Project/DataGenerator",
        data_name: Optional[str] = None,
        config=None,
    ):
        from ProQSAR.Config.config import Config  # fallback only if none provided
        self.activity_col = activity_col
        self.id_col = id_col
        self.smiles_col = smiles_col
        self.mol_col = mol_col
        self.n_jobs = n_jobs
        self.save_dir = save_dir
        self.data_name = data_name
        self.config = config or Config()

        self.standardizer = self.config.standardizer.set_params(
            smiles_col=self.smiles_col, n_jobs=self.n_jobs
        )
        self.featurizer = self.config.featurizer.set_params(
            mol_col=(self.mol_col if self.standardizer.deactivate else "standardized_mol"),
            activity_col=self.activity_col,
            id_col=self.id_col,
            smiles_col=(self.smiles_col if self.standardizer.deactivate else ("standardized_" + self.smiles_col)),
            n_jobs=self.n_jobs,
            save_dir=self.save_dir,
        )

    def generate(self, data):
        standardized_data = pd.DataFrame(self.standardizer.standardize_dict_smiles(data))
        data_features = self.featurizer.set_params(data_name=self.data_name).generate_features(standardized_data)
        if len(data_features.keys()) == 1:
            return list(data_features.values())[0]
        else:
            return data_features

    def get_params(self, deep=True) -> dict:
        out = {}
        for key in self.__dict__:
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                for sub_key, sub_value in deep_items:
                    out[f"{key}__{sub_key}"] = sub_value
            out[key] = value
        return out


# --- minimal Config holder to avoid mocking ---
class LocalConfig:
    def __init__(self, standardizer: SMILESStandardizer, featurizer: FeatureGenerator):
        self.standardizer = standardizer
        self.featurizer = featurizer


class TestDataGenerator(unittest.TestCase):
    def setUp(self):
        # Small clean dataset
        self.records = [
            {"smiles": "CCO", "id": 1, "activity": 5.0},
            {"smiles": "CCN", "id": 2, "activity": 6.0},
        ]

    def test_generate_single_feature_active_standardizer(self):
        """
        Standardizer active (default): DataGenerator should use standardized_mol/standardized_smiles,
        and FeatureGenerator returns a dict with one key -> DataFrame.
        """
        std = SMILESStandardizer(smiles_col="smiles", n_jobs=1, deactivate=False)
        feat = FeatureGenerator(
            mol_col="standardized_mol",
            smiles_col="standardized_smiles",
            activity_col="activity",
            id_col="id",
            feature_types="RDK5",   # keep it light & fast
            n_jobs=1,
            verbose=0,
            deactivate=False,
            save_dir=None,
        )
        cfg = LocalConfig(std, feat)
        dg = DataGenerator("activity", "id", "smiles", config=cfg)

        out = dg.generate(self.records)
        # Should be a DataFrame (single feature type)
        self.assertIsInstance(out, pd.DataFrame)
        # Required columns present
        self.assertIn("id", out.columns)
        self.assertIn("activity", out.columns)
        self.assertIn("standardized_smiles", out.columns)
        self.assertIn("standardized_mol", out.columns)
        # at least one fingerprint column should exist (column count > meta cols)
        self.assertGreater(len(out.columns), 4)
        self.assertEqual(len(out), 2)

    def test_generate_multi_feature_active_standardizer(self):
        """
        Multiple feature types -> DataGenerator returns a dict of DataFrames.
        """
        std = SMILESStandardizer(smiles_col="smiles", n_jobs=1, deactivate=False)
        feat = FeatureGenerator(
            mol_col="standardized_mol",
            smiles_col="standardized_smiles",
            activity_col="activity",
            id_col="id",
            feature_types=["RDK5", "MACCS"],  # two light features
            n_jobs=1,
            verbose=0,
            deactivate=False,
            save_dir=None,
        )
        cfg = LocalConfig(std, feat)
        dg = DataGenerator("activity", "id", "smiles", config=cfg)

        out = dg.generate(self.records)
        self.assertIsInstance(out, dict)
        self.assertIn("RDK5", out)
        self.assertIn("MACCS", out)
        self.assertIsInstance(out["RDK5"], pd.DataFrame)
        self.assertIsInstance(out["MACCS"], pd.DataFrame)
        self.assertEqual(len(out["RDK5"]), 2)
        self.assertEqual(len(out["MACCS"]), 2)

    def test_generate_deactivated_standardizer_requires_mol_input(self):
        """
        When standardizer is deactivated, DataGenerator wires FeatureGenerator to read 'mol' and 'smiles'.
        Provide prebuilt mols to ensure pipeline runs and returns DataFrame.
        """
        # Build input with precomputed RDKit mols (since standardizer won't add standardized_mol)
        data_with_mol = []
        for r in self.records:
            mol = Chem.MolFromSmiles(r["smiles"])
            d = r.copy()
            d["mol"] = mol
            data_with_mol.append(d)

        std = SMILESStandardizer(smiles_col="smiles", n_jobs=1, deactivate=True)
        feat = FeatureGenerator(
            mol_col="mol",                 # note: not standardized_mol
            smiles_col="smiles",
            activity_col="activity",
            id_col="id",
            feature_types="RDK5",
            n_jobs=1,
            verbose=0,
            deactivate=False,
            save_dir=None,
        )
        cfg = LocalConfig(std, feat)
        dg = DataGenerator("activity", "id", "smiles", config=cfg)

        out = dg.generate(data_with_mol)
        self.assertIsInstance(out, pd.DataFrame)
        self.assertIn("mol", out.columns)
        self.assertIn("smiles", out.columns)
        self.assertEqual(len(out), 2)

    def test_get_params_deep_and_shallow(self):
        std = SMILESStandardizer(smiles_col="smiles", n_jobs=1, deactivate=False)
        feat = FeatureGenerator(
            mol_col="standardized_mol",
            smiles_col="standardized_smiles",
            activity_col="activity",
            id_col="id",
            feature_types="RDK5",
            n_jobs=1,
        )
        cfg = LocalConfig(std, feat)
        dg = DataGenerator("activity", "id", "smiles", config=cfg)

        deep = dg.get_params(deep=True)
        self.assertIn("activity_col", deep)
        # nested params from sklearn BaseEstimator
        self.assertIn("standardizer__smiles_col", deep)
        self.assertIn("featurizer__feature_types", deep)

        shallow = dg.get_params(deep=False)
        self.assertIn("featurizer", shallow)
        self.assertIn("standardizer", shallow)

    def test_generate_writes_csvs_when_save_dir_set(self):
        """
        Verify that FeatureGenerator writes CSVs to disk when save_dir + data_name are provided.
        """
        tmpdir = tempfile.mkdtemp(prefix="dg_save_")
        try:
            std = SMILESStandardizer(smiles_col="smiles", n_jobs=1, deactivate=False)
            feat = FeatureGenerator(
                mol_col="standardized_mol",
                smiles_col="standardized_smiles",
                activity_col="activity",
                id_col="id",
                feature_types=["RDK5", "MACCS"],
                n_jobs=1,
                save_dir=tmpdir,  # write here
            )
            cfg = LocalConfig(std, feat)
            dg = DataGenerator(
                "activity", "id", "smiles",
                config=cfg,
                save_dir=tmpdir,
                data_name="mini"
            )
            out = dg.generate(self.records)
            # files should exist
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "mini_RDK5.csv")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "mini_MACCS.csv")))
            # sanity on outputs
            self.assertIsInstance(out, dict)
            self.assertIn("RDK5", out)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_error_propagates_on_invalid_input_type(self):
        """
        SMILESStandardizer.standardize_dict_smiles raises TypeError on invalid input.
        Ensure DataGenerator propagates (we don't swallow exceptions).
        """
        std = SMILESStandardizer(smiles_col="smiles", n_jobs=1, deactivate=False)
        feat = FeatureGenerator(
            mol_col="standardized_mol",
            smiles_col="standardized_smiles",
            activity_col="activity",
            id_col="id",
            feature_types="RDK5",
            n_jobs=1,
        )
        cfg = LocalConfig(std, feat)
        dg = DataGenerator("activity", "id", "smiles", config=cfg)

        with self.assertRaises(TypeError):
            dg.generate({"not": "a list or dataframe"})  # invalid type


if __name__ == "__main__":
    unittest.main()
