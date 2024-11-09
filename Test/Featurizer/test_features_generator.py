import unittest
import pandas as pd
from rdkit import Chem
from ProQSAR.Featurizer.features_generator import FeatureGenerator


class TestFeatureGenerator(unittest.TestCase):

    def setUp(self):
        # Initialize the FeatureGenerator with example parameters
        self.feature_gen = FeatureGenerator(
            mol_col="mol",
            activity_col="activity",
            ID_col="ID",
            save_dir=None,
            n_jobs=1,
            verbose=0,
        )
        # Create a dummy molecule using RDKit
        self.mol = Chem.MolFromSmiles("C1=CC=CC=C1")  # Benzene

    def test_mol_process_with_valid_input(self):
        # Test processing with a known, valid molecule
        feature_types = [
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
        results = FeatureGenerator._mol_process(self.mol, feature_types)

        # Ensure results contain all requested fingerprint types
        for feature in feature_types:
            self.assertIn(feature, results)
        self.assertEqual(len(results["ECFP2"]), 2048)
        self.assertEqual(len(results["RDK7"]), 4096)
        self.assertEqual(len(results["MACCS"]), 167)
        self.assertEqual(len(results["avalon"]), 1024)
        self.assertEqual(len(results["pubchem"]), 881)
        self.assertEqual(len(results["pharm2dgbfp"]), 39972)

    def test_mol_process_with_none(self):
        results = FeatureGenerator._mol_process(None)
        self.assertEqual(results, None)

    def test_single_process(self):
        # This will use the dummy molecule created in setUp
        record = {"mol": self.mol, "activity": 1, "ID": "M001"}
        result = self.feature_gen._single_process(
            record,
            self.feature_gen.mol_col,
            self.feature_gen.activity_col,
            self.feature_gen.ID_col,
            ["RDK7"],
        )

        # Check if the result includes expected keys
        self.assertIn("ID", result)
        self.assertIn("activity", result)
        self.assertEqual(result["ID"], "M001")
        self.assertEqual(result["activity"], 1)
        self.assertEqual(len(result["RDK7"]), 4096)

    def test_generate_features_with_df(self):
        # Creating a DataFrame to test
        df = pd.DataFrame(
            {"mol": [self.mol, self.mol], "activity": [1, 0], "ID": ["M001", "M002"]}
        )
        feature_types = ["RDK5"]
        result_df = self.feature_gen.generate_features(df, feature_types)

        # Check the results
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertTrue("RDK5" in result_df.columns)

    def test_generate_features_with_list(self):
        # Test with list input
        data = [
            {"mol": self.mol, "activity": 1, "ID": "M001"},
            {"mol": self.mol, "activity": 0, "ID": "M002"},
        ]
        feature_types = ["RDK5"]
        result_df = self.feature_gen.generate_features(data, feature_types)

        # Check the results
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertTrue("RDK5" in result_df.columns)


if __name__ == "__main__":
    unittest.main()
