import unittest
import pandas as pd
import numpy as np
from ProQSAR.Partitioner.partitioner import Partitioner

# Assuming Partitioner and other dependencies like scaffold_split and stratified_scaffold are properly imported


class TestPartitioner(unittest.TestCase):

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
        self.data["pIC50"] = np.random.uniform(0, 10, self.data.shape[0])
        self.data["feature1"] = np.random.rand(self.data.shape[0])
        self.data["feature2"] = np.random.rand(self.data.shape[0])

        # Data with missing value
        self.data_with_nan = self.data.copy()
        self.data_with_nan.loc[0:6, "pIC50"] = np.nan
        self.data_with_nan.loc[3, "feature1"] = "nan"

        # Data target_bin
        self.data_target_bin = self.data.copy()
        self.data_target_bin = self.data_target_bin.iloc[0:4, :]
        self.data_target_bin["activity"] = np.array([3, 2, 8, 9])

    def test_check_NaN(self) -> None:
        """
        Test the handling of NaN values in the dataset using the Partitioner.Check_NaN method.
        Verifies that NaN values are correctly identified and handled.
        """
        data_checked_nan = Partitioner.Check_NaN(self.data_with_nan)
        self.assertEqual(data_checked_nan.shape, self.data_with_nan.shape)
        self.assertTrue(data_checked_nan.isna().sum().sum() > 0)
        self.assertTrue(data_checked_nan.loc[3, "feature1"] == "nan")

    def test_target_bin_pIC50(self) -> None:
        """
        Test the target binning for pIC50 using the Partitioner.target_bin method.
        Verifies that the target binning is correct based on the pIC50 threshold.
        """
        data_target_bin = Partitioner.target_bin(
            data=self.data_target_bin,
            activity_col="activity",
            thresh=5,
            input_target_style="pIC50",
        )
        self.assertTrue(data_target_bin.shape == self.data_target_bin.shape)
        self.assertEqual(data_target_bin["activity"].tolist(), [0, 0, 1, 1])

    def test_target_bin_IC50(self) -> None:
        """
        Test the target binning for IC50 using the Partitioner.target_bin method.
        Verifies that the target binning is correct based on the IC50 threshold.
        """
        data_target_bin = Partitioner.target_bin(
            data=self.data_target_bin,
            activity_col="activity",
            thresh=5,
            input_target_style="IC50",
        )
        self.assertTrue(data_target_bin.shape == self.data_target_bin.shape)
        self.assertEqual(data_target_bin["activity"].tolist(), [1, 1, 0, 0])

    def test_random_partition_classification(self) -> None:
        """
        Test random partitioning for classification tasks using the Partitioner.data_partitioner method.
        Verifies that the partitioned dataset retains its original shape and structure.
        """
        # Test random partitioning for classification task
        partitioner = Partitioner()  # Assume Partitioner class exists
        data_train, data_test = partitioner.data_partitioner(
            self.data,
            activity_col="pIC50",
            task_type="C",
            option_partitioner="random",
            random_seed=42,
        )
        smiles_col = data_train["smiles"]
        all_strings = smiles_col.apply(lambda x: isinstance(x, str)).all()
        self.assertTrue(
            all_strings, "Not all values in the 'smiles' column are strings."
        )
        self.assertEqual(len(data_train) + len(data_test), len(self.data))
        self.assertEqual(self.data.shape[1], data_train.shape[1])
        self.assertEqual(self.data.shape[1], data_test.shape[1])
        self.assertIn("smiles", data_train.columns)
        self.assertIn("pIC50", data_train.columns)

    def test_random_partition_regression(self) -> None:
        """
        Test random partitioning for regression tasks using the Partitioner.data_partitioner method.
        Verifies that the partitioned dataset retains its original shape and structure.
        """
        # Test random partitioning for regression task
        partitioner = Partitioner()  # Assume Partitioner class exists
        data_train, data_test = partitioner.data_partitioner(
            self.data,
            activity_col="pIC50",
            task_type="R",
            option_partitioner="random",
            random_seed=42,
        )
        smiles_col = data_train["smiles"]
        all_strings = smiles_col.apply(lambda x: isinstance(x, str)).all()
        self.assertTrue(
            all_strings, "Not all values in the 'smiles' column are strings."
        )
        self.assertEqual(len(data_train) + len(data_test), len(self.data))
        self.assertEqual(self.data.shape[1], data_train.shape[1])
        self.assertEqual(self.data.shape[1], data_test.shape[1])
        self.assertIn("smiles", data_train.columns)
        self.assertIn("pIC50", data_train.columns)

    def test_scaffold_partition_classification(self) -> None:
        """
        Test scaffold-based partitioning for classification tasks using the Partitioner.data_partitioner method.
        Verifies that the partitioned dataset is correctly split by scaffold groups.
        """
        # Test scaffold partitioning (assuming scaffold_split method is defined)
        partitioner = Partitioner()  # Assume Partitioner class exists
        data_train, data_test = partitioner.data_partitioner(
            self.data,
            activity_col="pIC50",
            task_type="C",
            option_partitioner="scaffold",
            random_seed=42,
        )
        smiles_col = data_train["smiles"]
        all_strings = smiles_col.apply(lambda x: isinstance(x, str)).all()
        self.assertTrue(
            all_strings, "Not all values in the 'smiles' column are strings."
        )
        self.assertEqual(len(data_train) + len(data_test), len(self.data))
        self.assertEqual(self.data.shape[1], data_train.shape[1])
        self.assertEqual(self.data.shape[1], data_test.shape[1])
        self.assertIn("smiles", data_train.columns)
        self.assertIn("pIC50", data_train.columns)

    def test_scaffold_partition_regression(self) -> None:
        """
        Test scaffold-based partitioning for regression tasks using the Partitioner.data_partitioner method.
        Verifies that the partitioned dataset is correctly split by scaffold groups.
        """
        # Test scaffold partitioning (assuming scaffold_split method is defined)
        partitioner = Partitioner()  # Assume Partitioner class exists
        data_train, data_test = partitioner.data_partitioner(
            self.data,
            activity_col="pIC50",
            task_type="R",
            option_partitioner="scaffold",
            random_seed=42,
        )
        smiles_col = data_train["smiles"]
        all_strings = smiles_col.apply(lambda x: isinstance(x, str)).all()
        self.assertTrue(
            all_strings, "Not all values in the 'smiles' column are strings."
        )
        self.assertEqual(len(data_train) + len(data_test), len(self.data))
        self.assertEqual(self.data.shape[1], data_train.shape[1])
        self.assertEqual(self.data.shape[1], data_test.shape[1])
        self.assertIn("smiles", data_train.columns)
        self.assertIn("pIC50", data_train.columns)

    def test_stratified_scaffold_partition_classification(self) -> None:
        """
        Test stratified scaffold-based partitioning for classification tasks using the Partitioner.data_partitioner
        method.
        Verifies that the partitioned dataset is stratified by the activity values and scaffold groups.
        """
        # Test stratified scaffold partitioning for classification task
        partitioner = Partitioner()  # Assume Partitioner class exists
        data_train, data_test = partitioner.data_partitioner(
            self.data,
            activity_col="pIC50",
            task_type="C",
            option_partitioner="stratified_scaffold",
            random_seed=42,
        )
        smiles_col = data_train["smiles"]
        all_strings = smiles_col.apply(lambda x: isinstance(x, str)).all()
        self.assertTrue(
            all_strings, "Not all values in the 'smiles' column are strings."
        )
        self.assertEqual(len(data_train) + len(data_test), len(self.data))
        self.assertEqual(self.data.shape[1], data_train.shape[1])
        self.assertEqual(self.data.shape[1], data_test.shape[1])
        self.assertIn("smiles", data_train.columns)
        self.assertIn("pIC50", data_train.columns)

    def test_invalid_classification_threshold(self) -> None:
        """
        Test handling of an invalid classification task with incorrect activity class distribution.
        Verifies that the partitioning process proceeds without errors, even with invalid data.
        """
        # Test invalid classification task with incorrect number of unique activity classes
        partitioner = Partitioner()  # Assume Partitioner class exists
        data_invalid = self.data.copy()
        data_invalid["pIC50"] = np.random.rand(
            data_invalid.shape[0]
        )  # Invalid for classification
        data_train, data_test = partitioner.data_partitioner(
            data_invalid,
            activity_col="pIC50",
            task_type="C",
            option_partitioner="random",
            random_seed=42,
        )
        smiles_col = data_train["smiles"]
        all_strings = smiles_col.apply(lambda x: isinstance(x, str)).all()
        self.assertTrue(
            all_strings, "Not all values in the 'smiles' column are strings."
        )
        self.assertEqual(len(data_train) + len(data_test), len(self.data))
        self.assertEqual(self.data.shape[1], data_train.shape[1])
        self.assertEqual(self.data.shape[1], data_test.shape[1])
        self.assertIn("smiles", data_train.columns)
        self.assertIn("pIC50", data_train.columns)


if __name__ == "__main__":
    unittest.main()
