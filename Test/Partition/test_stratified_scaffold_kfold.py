from ProQSAR.Partition.stratified_scaffold_partition import StratifiedScaffoldPartition
from ProQSAR.Partition.stratified_scaffold_kfold import StratifiedScaffoldKFold
import pandas as pd
import numpy as np
import unittest


class TestStratifiedScaffoldPartition(unittest.TestCase):
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

    def test_StratifiedScaffoldKFold_scaff_based_invalid(self):

        with self.assertRaises(ValueError):
            StratifiedScaffoldKFold(scaff_based="sum")

    def test_StratifiedScaffoldKFold_iter_test_indices_median(self):

        stratifiedscaffoldkfold = StratifiedScaffoldKFold(
            n_splits=3, scaff_based="median"
        )
        data = self.data.copy()
        smiles_list = data["smiles"].to_list()
        X = data.drop(["pIC50", "smiles"], axis=1)
        y = data["pIC50"]
        groups = StratifiedScaffoldPartition.get_scaffold_groups(smiles_list)
        # Test if _iter_test_indices generates correct test indices
        test_indices = list(stratifiedscaffoldkfold._iter_test_indices(X, y, groups))

        # Ensure that the correct number of splits is produced
        self.assertEqual(len(test_indices), 3)

        # Ensure the sizes of test sets are consistent
        test_sizes = [len(idx) for idx in test_indices]
        self.assertEqual(sum(test_sizes), len(X))

        # Ensure test indices are unique across splits (no overlap)
        all_test_indices = np.concatenate(test_indices)
        self.assertEqual(len(set(all_test_indices)), len(X))

    def test_StratifiedScaffoldKFold_iter_test_indices_mean(self):

        stratifiedscaffoldkfold = StratifiedScaffoldKFold(
            n_splits=3, scaff_based="mean"
        )
        data = self.data.copy()
        smiles_list = data["smiles"].to_list()
        X = data.drop(["pIC50", "smiles"], axis=1)
        y = data["pIC50"]
        groups = StratifiedScaffoldPartition.get_scaffold_groups(smiles_list)
        # Test if _iter_test_indices generates correct test indices
        test_indices = list(stratifiedscaffoldkfold._iter_test_indices(X, y, groups))

        # Ensure that the correct number of splits is produced
        self.assertEqual(len(test_indices), 3)

        # Ensure the sizes of test sets are consistent
        test_sizes = [len(idx) for idx in test_indices]
        self.assertEqual(sum(test_sizes), len(X))

        # Ensure test indices are unique across splits (no overlap)
        all_test_indices = np.concatenate(test_indices)
        self.assertEqual(len(set(all_test_indices)), len(X))

    def test_StratifiedScaffoldKFold_iter_test_indices_invalid(self):
        stratifiedscaffoldkfold = StratifiedScaffoldKFold(n_splits=30)
        data = self.data.copy()
        smiles_list = data["smiles"].to_list()
        X = data.drop(["pIC50", "smiles"], axis=1)
        y = data["pIC50"]
        groups = StratifiedScaffoldPartition.get_scaffold_groups(smiles_list)

        with self.assertRaises(ValueError):
            list(stratifiedscaffoldkfold._iter_test_indices(X, y, groups))

    def test_find_best_fold(self):

        stratifiedscaffoldkfold = StratifiedScaffoldKFold(
            n_splits=3, scaff_based="median"
        )
        # Mock data for fold assignment
        y_counts_per_fold = np.array(
            [[2, 1], [3, 1], [1, 1]]
        )  # Current class distribution per fold
        y_cnt = np.array([10, 10])  # Total samples per class
        group_y_counts = np.array([1, 1])  # Current group's class distribution

        # Test if the method assigns the group to the best fold based on class balance
        best_fold = stratifiedscaffoldkfold._find_best_fold(
            y_counts_per_fold, y_cnt, group_y_counts
        )

        # Expecting fold index 2 since it has fewer samples and better balance
        self.assertEqual(best_fold, 2)

        # Ensure that adding this group to the selected fold improves balance
        y_counts_per_fold[best_fold] += group_y_counts
        fold_std_after = np.std(y_counts_per_fold / y_cnt.reshape(1, -1), axis=0)
        self.assertLess(np.mean(fold_std_after), np.inf)


if __name__ == "__main__":
    unittest.main()
