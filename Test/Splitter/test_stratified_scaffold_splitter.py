from ProQSAR.Splitter.stratified_scaffold_splitter import StratifiedScaffoldSplitter
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
import numpy as np
import unittest
from ProQSAR.Splitter.scaffold_utils import get_scaffold_groups


class TestStratifiedScaffoldSplitter(unittest.TestCase):
    def setUp(self):
        """
        Set up the test datasets for the partitioning tests.
        """
        # Create a mock dataset for testing
        self.data = pd.DataFrame(
            {
                "smiles": [
                    "CNC(=O)c1c(C)oc2cc(Oc3ccnc4cc(C(=O)N5CCC(OC)C5)sc34)ccc12",
                    "CNC(=O)Nc1ccc(Oc2ncnc3cc(OCCCN4CCCCC4)c(OC)cc23)cc1Cl",
                    "CNC(=O)c1c(C)sc2cc(Oc3ccnc4cc(-c5nccn5C)sc34)ccc12",
                    "Cc1c(C(=O)NC2CC2)c2ccc(Oc3ccnc4cc(-c5nccn5C)sc34)cc2n1C",
                    "O=C1Nc2ccccc2C1=CNc1ccc(OCCCCN2CCOCC2)cc1",
                    "O=C1Nc2ccccc2C1=CNc1ccc(OCCCCN2CCCCC2)cc1",
                    "CNC(=O)c1ccccc1Sc1ccc2c(C=Cc3ccccn3)n[nH]c2c1",
                    "CN1CCN(CCCCOc2ccc(NC=C3C(=O)Nc4ccccc43)cc2)CC1",
                    "CCN(CC)CCCOc1ccc(NC=C2C(=O)Nc3ccccc32)cc1",
                    "O=C1Nc2ccccc2C1=CNc1ccc(OCCCN2CCCC2)cc1",
                    "O=C1Nc2ccccc2C1=CNc1ccc(OCCCN2CCOCC2)cc1",
                    "COc1ccccc1NC(=O)Nc1ccc(Oc2ncnc3cc(OC)c(OC)cc23)cc1F",
                    "CCCNC(=O)Nc1ccc(Oc2ccnc3cc(OCCCN(C)CCO)c(OC)cc23)cc1Cl",
                    "CCCNC(=O)c1c(C)n(C)c2cc(Oc3ccnc4cc(-c5nccn5C)sc34)ccc12",
                    "Nc1cccc(NC=C2C(=O)Nc3ccccc32)c1",
                    "CNC(=O)c1c(C)n(C)c2cc(Oc3ccnc4cc(C(=O)N5CCCC5CO)sc34)ccc12",
                    "CNC(=O)c1c(C)sc2cc(Oc3ccnc4cc(C(=O)N5CCC(OC)C5)sc34)ccc12",
                    "COc1cc2ncnc(Oc3ccc(NC(=O)Nc4ccc(F)cc4)c(Cl)c3)c2cc1OC",
                    "CCNC(=O)Nc1ccc(Oc2ncnc3cc(OCCCN4CCCCC4)c(OC)cc23)cc1Cl",
                    "CCCNC(=O)Nc1ccc(Oc2ccnc3cc(OCCCN4CCOCC4)c(OC)cc23)cc1Cl",
                ],
            }
        )
        self.data["pIC50"] = np.array([0] * 10 + [1] * 10)
        self.data["feature1"] = np.random.rand(self.data.shape[0])
        self.data["feature2"] = np.random.rand(self.data.shape[0])

        self.stratifiedscaffoldsplitter = StratifiedScaffoldSplitter(
            activity_col="pIC50", smiles_col="smiles", n_splits=5, random_state=42
        )

    def test_get_scaffold_groups_valid(self):
        # Test get_scaffold_groups function on DataFrame
        groups = get_scaffold_groups(self.data, "smiles", None)

        # Ensure correct output type and shape
        self.assertIsInstance(groups, np.ndarray)
        self.assertEqual(len(groups), len(self.data))
        self.assertGreater(len(np.unique(groups)), 1)

    def test_get_scaffold_groups_invalid(self):
        # Test get_scaffold_groups with invalid SMILES in DataFrame
        df = self.data.copy()
        df.loc[3:5, "smiles"] = "Invalid_SMILES"
        with self.assertRaises(AssertionError):
            _ = get_scaffold_groups(df, "smiles", None)

    def test_stratifiedscaffoldsplitter_size(self):
        data_train, data_test = self.stratifiedscaffoldsplitter.fit(self.data)

        self.assertEqual(data_train.shape[0], 16)
        self.assertEqual(data_test.shape[0], 4)

    def test_stratifiedscaffoldsplitter_df(self):
        data_train, data_test = self.stratifiedscaffoldsplitter.fit(self.data)

        self.assertIsInstance(data_train, pd.DataFrame)
        self.assertIsInstance(data_test, pd.DataFrame)

    def test_stratifiedscaffoldsplitter_scaffold(self):
        data_train, data_test = self.stratifiedscaffoldsplitter.fit(self.data)

        # Create scaffold sets for both train and test sets
        def get_scaffold_set(df):
            scaffolds = set()
            for smiles in df["smiles"]:
                mol = Chem.MolFromSmiles(smiles)
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=False
                )
                scaffolds.add(scaffold)
            return scaffolds

        train_scaffolds = get_scaffold_set(data_train)
        test_scaffolds = get_scaffold_set(data_test)

        # Ensure no overlap of scaffolds between training and test sets
        self.assertTrue(
            train_scaffolds.isdisjoint(test_scaffolds),
            "Scaffolds are not unique between train and test sets",
        )

    def test_stratifiedscaffoldsplitter_stratify(self):
        data_train, data_test = self.stratifiedscaffoldsplitter.fit(self.data)

        self.assertEqual((data_train["pIC50"] == 0).sum(), 8)
        self.assertEqual((data_train["pIC50"] == 1).sum(), 8)
        self.assertEqual((data_test["pIC50"] == 0).sum(), 2)
        self.assertEqual((data_test["pIC50"] == 1).sum(), 2)

    def test_init_params_are_stored(self):
        splitter = StratifiedScaffoldSplitter(
            activity_col="y",
            smiles_col="s",
            mol_col="molcol",
            random_state=123,
            n_splits=7,
            scaff_based="mean",
            shuffle=False,
        )
        self.assertEqual(splitter.activity_col, "y")
        self.assertEqual(splitter.smiles_col, "s")
        self.assertEqual(splitter.mol_col, "molcol")
        self.assertEqual(splitter.random_state, 123)
        self.assertEqual(splitter.n_splits, 7)
        self.assertEqual(splitter.scaff_based, "mean")
        self.assertFalse(splitter.shuffle)

    def test_fit_median_strategy_shuffle_true_sizes_and_disjoint(self):
        splitter = StratifiedScaffoldSplitter(
            activity_col="pIC50",
            smiles_col="smiles",
            mol_col="mol",
            random_state=99,
            n_splits=5,
            scaff_based="median",
            shuffle=True,
        )
        train, test = splitter.fit(self.data)
        self.assertEqual(train.shape[0] + test.shape[0], len(self.data))
        # Fold sizes can vary due to scaffold grouping; just ensure plausible bounds
        self.assertIn(test.shape[0], {3, 4, 5})
        self.assertEqual(len(set(train.index).intersection(set(test.index))), 0)

    def test_fit_mean_strategy_no_shuffle_sizes_and_disjoint(self):
        splitter = StratifiedScaffoldSplitter(
            activity_col="pIC50",
            smiles_col="smiles",
            random_state=None,
            n_splits=2,
            scaff_based="mean",
            shuffle=False,
        )
        train, test = splitter.fit(self.data)
        self.assertEqual(train.shape[0] + test.shape[0], len(self.data))
        self.assertEqual(len(set(train.index).intersection(set(test.index))), 0)

    def test_fit_works_when_columns_missing_due_to_errors_ignore(self):
        df = self.data.copy()
        # Drop a feature column; the splitter drops by name with errors="ignore"
        df = df.drop(columns=["feature2"], errors="ignore")
        splitter = StratifiedScaffoldSplitter(
            activity_col="pIC50",
            smiles_col="smiles",
        )
        train, test = splitter.fit(df)
        self.assertEqual(train.shape[0] + test.shape[0], len(df))
        self.assertEqual(len(set(train.index).intersection(set(test.index))), 0)


if __name__ == "__main__":
    unittest.main()
