import unittest
import pandas as pd
from proqsar.Data.Splitter.scaffold_splitter import ScaffoldSplitter


class TestScaffoldSplitter(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "smiles": [
                    "c1ccccc1",  # benzene
                    "c1ccccc1",  # benzene dup
                    "c1ccncc1",  # pyridine
                    "C1CCCCC1",  # cyclohexane
                    "c1ccc2ccccc2c1",  # naphthalene (larger scaffold)
                ],
                "y": [0, 1, 0, 1, 0],
            }
        )
        self.splitter = ScaffoldSplitter(
            activity_col="y", smiles_col="smiles", test_size=0.4
        )

    def test_init_params(self):
        sp = ScaffoldSplitter(
            activity_col="y", smiles_col="s", mol_col="mol", test_size=0.3
        )
        self.assertEqual(sp.activity_col, "y")
        self.assertEqual(sp.smiles_col, "s")
        self.assertEqual(sp.mol_col, "mol")
        self.assertEqual(sp.test_size, 0.3)

    def test_fit_splits_by_scaffold_without_overlap_and_expected_indices(self):
        # Using the real utilities on the crafted dataset, the sorted scaffold
        # groups and greedy fill lead to deterministic indices:
        # Order by size desc then first index desc gives: [0,1], [4], [3], [2]
        # train_cutoff = (1 - 0.4) * 5 = 3 → train gets [0,1] and [4]; test gets [3] and [2]
        train, test = self.splitter.fit(self.df)
        self.assertEqual(len(train) + len(test), len(self.df))
        self.assertEqual(len(set(train.index).intersection(set(test.index))), 0)
        self.assertSetEqual(set(train.index.tolist()), {0, 1, 4})
        self.assertSetEqual(set(test.index.tolist()), {2, 3})

    def test_fit_real_utils_basic_behavior(self):
        train, test = self.splitter.fit(self.df)
        self.assertEqual(len(train) + len(test), len(self.df))
        self.assertEqual(len(set(train.index).intersection(set(test.index))), 0)
        # Respect test_size approximate (integer boundary handling)
        self.assertIn(len(test), {2})

    def test_fit_raises_on_invalid_smiles_via_check(self):
        df_bad = self.df.copy()
        df_bad.loc[0, "smiles"] = "Invalid_SMILES"
        with self.assertRaises(AssertionError):
            _ = self.splitter.fit(df_bad)


if __name__ == "__main__":
    unittest.main()
