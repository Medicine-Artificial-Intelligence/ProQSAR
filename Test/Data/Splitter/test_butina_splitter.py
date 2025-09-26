import unittest
import pandas as pd
from rdkit import Chem
from proqsar.Data.Splitter.butina_splitter import (
    ButinaSplitter,
)  # replace with actual module path


class TestButinaSplitter(unittest.TestCase):
    def setUp(self):
        # Simple dataset
        self.df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "SMILES": ["C", "CC", "CCC", "CCCC", "CCCCC"],
                "activity": [0.1, 0.5, 0.9, 1.2, 1.5],
            }
        )

    def test_init_defaults(self):
        splitter = ButinaSplitter(activity_col="activity", smiles_col="SMILES")
        self.assertEqual(splitter.test_size, 0.2)
        self.assertEqual(splitter.cutoff, 0.6)
        self.assertEqual(splitter.mol_col, "mol")

    def test_split_with_mol_col_absent(self):
        # no mol_col column provided
        splitter = ButinaSplitter(activity_col="activity", smiles_col="SMILES")
        train, test = splitter.fit(self.df)
        self.assertIsInstance(train, pd.DataFrame)
        self.assertIsInstance(test, pd.DataFrame)
        self.assertEqual(len(train) + len(test), len(self.df))
        self.assertTrue("mol" in train.columns or "mol" in test.columns)

    def test_split_with_existing_mol_col(self):
        df = self.df.copy()
        df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
        splitter = ButinaSplitter(
            activity_col="activity", smiles_col="SMILES", mol_col="mol"
        )
        train, test = splitter.fit(df)
        self.assertEqual(len(train) + len(test), len(df))

    def test_invalid_smiles_raises(self):
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "SMILES": ["INVALID", "###"],
                "activity": [0.1, 0.2],
            }
        )
        splitter = ButinaSplitter(activity_col="activity", smiles_col="SMILES")
        with self.assertRaises(ValueError):
            splitter.fit(df)

    def test_small_cutoff_creates_many_clusters(self):
        splitter = ButinaSplitter(
            activity_col="activity", smiles_col="SMILES", cutoff=0.1
        )
        train, test = splitter.fit(self.df)
        # With small cutoff, clusters are fine-grained, test set should not be empty
        self.assertGreater(len(test), 0)

    def test_large_cutoff_creates_few_clusters(self):
        splitter = ButinaSplitter(
            activity_col="activity", smiles_col="SMILES", cutoff=0.9
        )
        train, test = splitter.fit(self.df)
        # With large cutoff, clusters are coarser, test set should still exist
        self.assertGreaterEqual(len(train), 1)
        self.assertGreaterEqual(len(test), 1)

    def test_train_test_no_overlap(self):
        splitter = ButinaSplitter(activity_col="activity", smiles_col="SMILES")
        train, test = splitter.fit(self.df)
        overlap = set(train["id"]).intersection(set(test["id"]))
        self.assertEqual(len(overlap), 0)


if __name__ == "__main__":
    unittest.main()
