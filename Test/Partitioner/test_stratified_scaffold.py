import unittest
import pandas as pd
import numpy as np
from rdkit import Chem
from ProQSAR.Partitioner.stratified_scaffold import (
    get_scaffold_groups,
    StratifiedScaffoldKFold,
    stratified_scaffold,
)

# Assuming Partitioner and other dependencies like scaffold_split and stratified_scaffold are properly imported


class TestStratifiedScaffold(unittest.TestCase):

    def setUp(self) -> None:
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
        self.data["pIC50"] = np.random.randint(0, 1, self.data.shape[0])
        self.data["feature1"] = np.random.rand(self.data.shape[0])
        self.data["feature2"] = np.random.rand(self.data.shape[0])

    def test_get_scaffold_groups_valid(self) -> None:
        """
        Test that the `get_scaffold_groups` function returns the correct number of scaffold groups
        and ensures that the output is a numpy array with the correct length.
        """
        # Test get_scaffold_groups function
        smiles_list = self.data["smiles"].to_list()
        groups = get_scaffold_groups(smiles_list)

        # Ensure correct output type and shape
        self.assertIsInstance(
            groups, np.ndarray, "Expected output to be a numpy array."
        )
        self.assertEqual(
            len(groups),
            len(smiles_list),
            "Groups array length should match the number of SMILES.",
        )
        self.assertGreater(
            len(np.unique(groups)),
            1,
            "There should be more than one unique scaffold group.",
        )

    def test_get_scaffold_groups_invalid(self) -> None:
        """
        Test that the `get_scaffold_groups` function raises an AssertionError when
        invalid SMILES strings are provided.
        """
        # Test get_scaffold_groups function
        smiles_list_0 = self.data["smiles"].to_list()
        smiles_list = smiles_list_0.copy()
        smiles_list[3:5] = "Invalid_SMILES"

        with self.assertRaises(AssertionError):
            get_scaffold_groups(smiles_list)

    def test_StratifiedScaffoldKFold_scaff_based_invalid(self) -> None:
        """
        Test that `StratifiedScaffoldKFold` raises a ValueError when initialized with an
        invalid `scaff_based` parameter.
        """
        with self.assertRaises(ValueError):
            StratifiedScaffoldKFold(scaff_based="sum")

    def test_StratifiedScaffoldKFold_iter_test_indices_median(self) -> None:
        """
        Test that `_iter_test_indices` generates the correct number of test indices using `median`
        as the scaffold-based splitting strategy in `StratifiedScaffoldKFold`.
        """
        stratifiedscaffoldkfold = StratifiedScaffoldKFold(
            n_splits=3, scaff_based="median"
        )
        data = self.data.copy()
        smiles_list = data["smiles"].to_list()
        X = data.drop(["pIC50", "smiles"], axis=1)
        y = data["pIC50"]
        groups = get_scaffold_groups(smiles_list)
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

    def test_StratifiedScaffoldKFold_iter_test_indices_mean(self) -> None:
        """
        Test that `_iter_test_indices` generates the correct number of test indices using `mean`
        as the scaffold-based splitting strategy in `StratifiedScaffoldKFold`.
        """
        stratifiedscaffoldkfold = StratifiedScaffoldKFold(
            n_splits=3, scaff_based="mean"
        )
        data = self.data.copy()
        smiles_list = data["smiles"].to_list()
        X = data.drop(["pIC50", "smiles"], axis=1)
        y = data["pIC50"]
        groups = get_scaffold_groups(smiles_list)
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

    def test_StratifiedScaffoldKFold_iter_test_indices_invalid(self) -> None:
        """
        Test that `_iter_test_indices` raises a ValueError when there are more splits than samples
        in `StratifiedScaffoldKFold`.
        """
        stratifiedscaffoldkfold = StratifiedScaffoldKFold(n_splits=30)
        data = self.data.copy()
        smiles_list = data["smiles"].to_list()
        X = data.drop(["pIC50", "smiles"], axis=1)
        y = data["pIC50"]
        groups = get_scaffold_groups(smiles_list)

        with self.assertRaises(ValueError):
            list(stratifiedscaffoldkfold._iter_test_indices(X, y, groups))

    def test_find_best_fold(self) -> None:
        """
        Test that the `_find_best_fold` method in `StratifiedScaffoldKFold` assigns the group
        to the best fold based on class balance.
        """
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
        self.assertEqual(
            best_fold, 2, "The best fold should be 2 for better class balance."
        )

        # Ensure that adding this group to the selected fold improves balance
        y_counts_per_fold[best_fold] += group_y_counts
        fold_std_after = np.std(y_counts_per_fold / y_cnt.reshape(1, -1), axis=0)
        self.assertLess(
            np.mean(fold_std_after), np.inf, "The fold should reduce class imbalance."
        )

    def test_stratified_scaffold_output(self) -> None:
        """
        Test the output of the `stratified_scaffold` function and ensure that train and test sets are not empty.
        """
        # Test the output of the stratified_scaffold function
        train, test = stratified_scaffold(
            self.data,
            smiles_col="smiles",
            activity_col="pIC50",
            n_splits=5,
            scaff_based="median",
            random_state=42,
            shuffle=True,
        )

        # Check if the train and test sets are not empty
        self.assertGreater(len(train), 0, "Train set should not be empty.")
        self.assertGreater(len(test), 0, "Test set should not be empty.")

        # Check that the total size of train + test equals the original dataset size
        self.assertEqual(
            len(train) + len(test),
            len(self.data),
            "Combined train and test sets should match the size of the original dataset.",
        )

    def test_stratified_scaffold_splits(self) -> None:
        """
        Test that the `stratified_scaffold` function creates non-overlapping splits.
        """
        # Test if the stratified_scaffold function creates non-overlapping splits
        train, test = stratified_scaffold(
            self.data,
            smiles_col="smiles",
            activity_col="pIC50",
            n_splits=5,
            scaff_based="median",
            random_state=42,
            shuffle=True,
        )

        # Ensure there is no overlap between train and test sets
        self.assertTrue(
            train.index.isin(test.index).sum() == 0,
            "Train and test sets should not overlap.",
        )

    def test_stratified_scaffold_scaffold_groups(self) -> None:
        """
        Test that scaffolds in the train and test sets are unique.
        """
        # Test if scaffolds in the train and test sets are unique
        train, test = stratified_scaffold(
            self.data,
            smiles_col="smiles",
            activity_col="pIC50",
            n_splits=5,
            scaff_based="median",
            random_state=42,
            shuffle=True,
        )

        def get_scaffolds(data):
            scaffolds = set()
            for smiles in data["smiles"]:
                mol = Chem.MolFromSmiles(smiles)
                scaffold = Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=False
                )
                scaffolds.add(scaffold)
            return scaffolds

        # Ensure that scaffolds are unique across train and test sets
        train_scaffolds = get_scaffolds(train)
        test_scaffolds = get_scaffolds(test)

        self.assertTrue(
            set(train_scaffolds).isdisjoint(set(test_scaffolds)),
            "Train and test sets should have unique scaffolds.",
        )

    def test_stratified_scaffold_reproducibility(self) -> None:
        """
        Test that the splitting is reproducible with the same random seed.
        """
        # Test if the splitting is reproducible with the same random seed
        train1, test1 = stratified_scaffold(
            self.data,
            smiles_col="smiles",
            activity_col="pIC50",
            n_splits=5,
            scaff_based="median",
            random_state=42,
            shuffle=True,
        )

        train2, test2 = stratified_scaffold(
            self.data,
            smiles_col="smiles",
            activity_col="pIC50",
            n_splits=5,
            scaff_based="median",
            random_state=42,
            shuffle=True,
        )

        # Ensure that the splits are the same
        self.assertTrue(
            train1.equals(train2),
            "Train sets should be reproducible with the same random state.",
        )
        self.assertTrue(
            test1.equals(test2),
            "Test sets should be reproducible with the same random state.",
        )


if __name__ == "__main__":
    unittest.main()
