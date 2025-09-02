import unittest
import pandas as pd
import numpy as np
from ProQSAR.Splitter import scaffold_utils as su


class TestScaffoldUtils(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "smiles": [
                    "c1ccccc1",  # benzene
                    "c1ccccc1",  # benzene duplicate
                    "c1ccncc1",  # pyridine
                    "C1CCCCC1",  # cyclohexane
                ],
            }
        )

    def test_generate_scaffold_dict_valid(self):
        scaff_dict = su.generate_scaffold_dict(self.df, "smiles")
        self.assertIsInstance(scaff_dict, dict)
        self.assertGreaterEqual(len(scaff_dict), 2)
        # All indices must be accounted for across groups
        all_indices = sorted([i for idxs in scaff_dict.values() for i in idxs])
        self.assertEqual(all_indices, list(range(len(self.df))))
        # check_scaffold_dict should pass
        su.check_scaffold_dict(self.df, scaff_dict)

    def test_generate_scaffold_list_and_groups_valid(self):
        scaff_list = su.generate_scaffold_list(self.df, "smiles")
        self.assertIsInstance(scaff_list, list)
        self.assertTrue(all(isinstance(g, list) for g in scaff_list))
        # Sum of group sizes equals dataset size
        self.assertEqual(sum(len(g) for g in scaff_list), len(self.df))
        # Groups array shape and coverage
        groups = su.get_scaffold_groups(self.df, "smiles")
        self.assertIsInstance(groups, np.ndarray)
        self.assertEqual(groups.shape[0], len(self.df))
        self.assertTrue(set(groups) == set(range(len(scaff_list))))

    def test_invalid_smiles_logs_and_skips_then_assertion_in_groups(self):
        df_bad = self.df.copy()
        df_bad.loc[1, "smiles"] = "Invalid_SMILES"
        # generate_scaffold_dict should skip the invalid row and not raise
        scaff_dict = su.generate_scaffold_dict(df_bad, "smiles")
        self.assertLess(sum(len(v) for v in scaff_dict.values()), len(df_bad))
        # get_scaffold_groups should assert because not all assigned
        with self.assertRaises(AssertionError):
            _ = su.get_scaffold_groups(df_bad, "smiles")

    def test_check_scaffold_dict_raises_on_mismatch(self):
        scaff_dict = {"X": [0, 1], "Y": [2]}  # missing index 3
        with self.assertRaises(AssertionError):
            su.check_scaffold_dict(self.df, scaff_dict)

    def test_check_scaffold_list_raises_on_mismatch(self):
        scaff_list = [[0], [1], [2]]  # missing index 3
        with self.assertRaises(AssertionError):
            su.check_scaffold_list(self.df, scaff_list)

    def test_generate_scaffold_dict_with_mol_col_precomputed(self):
        # Provide RDKit mols to exercise mol_col branch
        from rdkit import Chem

        df_mol = self.df.copy()
        df_mol["mol"] = [Chem.MolFromSmiles(s) for s in df_mol["smiles"]]
        scaff_dict = su.generate_scaffold_dict(df_mol, "smiles", mol_col="mol")
        self.assertIsInstance(scaff_dict, dict)
        # All indices present
        all_idx = sorted([i for g in scaff_dict.values() for i in g])
        self.assertEqual(all_idx, list(range(len(df_mol))))


if __name__ == "__main__":
    unittest.main()
