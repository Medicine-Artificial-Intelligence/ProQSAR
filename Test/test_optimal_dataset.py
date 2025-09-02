import os
import unittest
import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory
from ProQSAR.Config.config import Config
from ProQSAR.optimal_dataset import OptimalDataset


# Use the sample data function the user provided
def make_sample_data(n: int = 20, seed: int = 42) -> pd.DataFrame:
    """Create a sample dataset for testing OptimalDataset."""
    from rdkit import Chem

    rng = np.random.default_rng(seed)

    # Simple SMILES for testing
    smiles_list = ["CCO", "CCN", "CCC", "CCCO", "CCCN"] * (n // 5 + 1)
    smiles_list = smiles_list[:n]

    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "activity": rng.random(n) * 10.0,
            "smiles": smiles_list,
        }
    )

    # Add mol column
    df["mol"] = [Chem.MolFromSmiles(smi) for smi in df["smiles"]]

    return df


class TestOptimalDataset(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment.
        """
        self.df = make_sample_data(n=24, seed=1)
        self.temp_dir = TemporaryDirectory()
        self.cfg = Config()
        self.od = OptimalDataset(
            activity_col="activity",
            id_col="id",
            smiles_col="smiles",
            mol_col="mol",
            keep_all_train=False,
            save_dir=self.temp_dir.name,
            n_jobs=1,
            random_state=123,
            config=self.cfg,
            n_splits=2,
            n_repeats=1,
            visualize=None,
            save_cv_report=True,
            cv_report_name="cv_report",
            save_fig=False,
        )

    def tearDown(self):
        """
        Clean up the test directory after tests.
        """
        self.temp_dir.cleanup()

    def test_run(self):

        optimal = self.od.run(self.df)
        # Should produce something as optimal (tuple/index)
        self.assertIsNotNone(optimal)
        # Report saved
        self.assertTrue(
            os.path.isfile(os.path.join(self.temp_dir.name, "cv_report.csv"))
        )
        # shape summary DataFrame available
        summary = self.od.get_shape_summary_df()
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertTrue({"Feature Set", "Data"}.issubset(set(summary.columns)))
        # report present
        self.assertIsInstance(self.od.report, pd.DataFrame)
        self.assertIn("scoring", self.od.report.columns)
        self.assertIn("cv_cycle", self.od.report.columns)

    def test_get_params_deep_and_shallow(self):
        # shallow (no deep expansion) must contain top-level keys and component objects
        shallow = self.od.get_params(deep=False)
        self.assertIn("activity_col", shallow)
        self.assertIn("id_col", shallow)
        self.assertIn("datagenerator", shallow)
        self.assertIn("datapreprocessor", shallow)

        # deep should expand nested estimator params and also include top-level keys
        deep = self.od.get_params(deep=True)

        # basic sanity checks for preserved top-level values
        self.assertEqual(deep["activity_col"], "activity")
        self.assertEqual(deep["id_col"], "id")

        # ensure datagenerator/datapreprocessor objects are still present in deep output
        self.assertIn("datagenerator", deep)
        self.assertIn("datapreprocessor", deep)

        # ensure some nested prefixed keys exist for splitter (prefixed keys like 'splitter_...')
        self.assertTrue(
            any(k.startswith("splitter_") for k in deep.keys()),
            "expected at least one 'splitter_' prefixed key",
        )

        # ensure some nested double-underscore keys exist for preprocessor steps (e.g., 'duplicate__...')
        self.assertTrue(
            any(k.startswith("duplicate__") for k in deep.keys()),
            "expected 'duplicate__' prefixed keys from datapreprocessor",
        )

        # ensure standardizer and featurizer parameters were flattened with expected prefixes
        self.assertTrue(
            any(k.startswith("standardizer__") for k in deep.keys()),
            "expected standardizer__ keys",
        )
        self.assertTrue(
            any(k.startswith("featurizer__") for k in deep.keys()),
            "expected featurizer__ keys",
        )

        # sanity: shape_summary should be present (may be empty dict)
        self.assertIn("shape_summary", deep)
        self.assertIsInstance(deep["shape_summary"], dict)

    def test___repr___and_initial_attributes(self):

        rep = repr(self.od)
        self.assertIn("OptimalDataset(", rep)
        self.assertIn("activity_col", rep)
        self.assertIn("id_col", rep)
        self.assertIn("random_state", rep)

        # check initial attributes
        self.assertIsInstance(self.od.train, dict)
        self.assertIsInstance(self.od.test, dict)
        self.assertIsNone(self.od.data_features)
        self.assertIsNone(self.od.report)
        self.assertIsNone(self.od.optimal_set)


if __name__ == "__main__":
    unittest.main()
