from ProQSAR.Partition.data_partition import Partition
import pandas as pd
import numpy as np
import unittest


class TestPartition(unittest.TestCase):
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

        self.randompartition = Partition(
            self.data, "pIC50", "smiles", "random", test_size=0.2, random_state=42
        )
        self.stratifiedrandompartition = Partition(
            self.data,
            "pIC50",
            "smiles",
            "stratified_random",
            test_size=0.2,
            random_state=42,
        )
        self.scaffoldpartition = Partition(
            self.data, "pIC50", "smiles", "scaffold", test_size=0.2, random_state=42
        )
        self.stratifiedscaffoldpartition = Partition(
            self.data,
            "pIC50",
            "smiles",
            "stratified_scaffold",
            n_splits=5,
            random_state=42,
        )
        self.invalidpartition = Partition(self.data, "pIC50", "smiles", "invalid")

    def test_randompartition(self):
        data_train, data_test = self.randompartition.fit()

        self.assertEqual(data_train.shape[0], 16)
        self.assertEqual(data_test.shape[0], 4)
        self.assertIsInstance(data_train, pd.DataFrame)
        self.assertIsInstance(data_test, pd.DataFrame)

    def test_stratifiedrandompartition(self):
        data_train, data_test = self.stratifiedrandompartition.fit()

        self.assertEqual(data_train.shape[0], 16)
        self.assertEqual(data_test.shape[0], 4)
        self.assertIsInstance(data_train, pd.DataFrame)
        self.assertIsInstance(data_test, pd.DataFrame)

    def test_scaffoldpartition(self):
        data_train, data_test = self.scaffoldpartition.fit()

        self.assertEqual(data_train.shape[0], 16)
        self.assertEqual(data_test.shape[0], 4)
        self.assertIsInstance(data_train, pd.DataFrame)
        self.assertIsInstance(data_test, pd.DataFrame)

    def test_stratifiedscaffoldpartition(self):
        data_train, data_test = self.stratifiedscaffoldpartition.fit()

        self.assertEqual(data_train.shape[0], 16)
        self.assertEqual(data_test.shape[0], 4)
        self.assertIsInstance(data_train, pd.DataFrame)
        self.assertIsInstance(data_test, pd.DataFrame)

    def test_partition_invalid(self):
        with self.assertRaises(ValueError):
            data_train, data_test = self.invalidpartition.fit()


if __name__ == "__main__":
    unittest.main()
