import unittest
import pandas as pd
from ProQSAR.Standardizer.smiles_standardizer import SMILESStandardizer
from ProQSAR.Featurizer.features_generator import FeatureGenerator


class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.standardizer = SMILESStandardizer()
        self.example_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        self.example_dict = {"SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O"}
        self.example_smiles_list_dict = [
            {"SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O", "ID": "M001", "activity": 1},
            {"SMILES": "C1=CC=C(C=C1)C=O", "ID": "M002", "activity": 0},
        ]
        self.example_smiles_data = pd.DataFrame(self.example_smiles_list_dict)

        self.feature_gen = FeatureGenerator(
            mol_col="mol",
            activity_col="activity",
            ID_col="ID",
            save_dir=None,
            n_jobs=1,
            verbose=0,
        )

    def test_smiles_processing(self):
        _, standardized_mol = self.standardizer.standardize_smiles(self.example_smiles)
        results = self.feature_gen._mol_process(standardized_mol, ["RDK5"])
        print(results)
        self.assertIn("RDK5", results)

    def test_dict_processing(self):
        standardized_dict = self.standardizer.standardize_dict_smiles(
            self.example_smiles_list_dict
        )
        print(standardized_dict)
        self.feature_gen = FeatureGenerator(
            mol_col="standardized_mol",
            activity_col="activity",
            ID_col="ID",
            save_dir=None,
            n_jobs=1,
            verbose=0,
        )
        results = self.feature_gen.generate_features(
            standardized_dict, ["RDK5"]
        ).to_dict("records")
        print(results)
        self.assertIn("RDK5", results[0])
