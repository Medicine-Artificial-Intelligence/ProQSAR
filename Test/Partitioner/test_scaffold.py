import unittest
import pandas as pd
import numpy as np
from rdkit import Chem
from ProQSAR.Partitioner.scaffold import scaffold_split

# Assuming Partitioner and other dependencies like scaffold_split and stratified_scaffold are properly imported


class TestScaffoldSplit(unittest.TestCase):

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

    def test_scaffold_split_output_shapes(self) -> None:
        """
        Test if the scaffold_split function returns the correct sizes for train and test datasets
        based on the specified test size.
        """
        # Test if the split results in the expected train/test sizes
        test_size = 0.25
        data_train, data_test = scaffold_split(
            self.data, smiles_col="smiles", test_size=test_size, random_state=42
        )

        # Check if the sizes of train and test sets are correct
        num_test = int(np.floor(test_size * len(self.data)))
        self.assertEqual(len(data_test), num_test, "Test set size is incorrect")
        self.assertEqual(
            len(data_train), len(self.data) - num_test, "Training set size is incorrect"
        )

    def test_scaffold_split_unique_scaffolds(self) -> None:
        """
        Test that scaffolds are unique between training and test sets.
        Ensures that no scaffolds overlap between the training and test sets.
        """
        # Test that scaffolds in training and test sets are unique (no overlap)
        data_train, data_test = scaffold_split(
            self.data, smiles_col="smiles", test_size=0.2, random_state=42
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

        train_scaffolds = get_scaffolds(data_train)
        test_scaffolds = get_scaffolds(data_test)

        # Ensure no overlap of scaffolds between training and test sets
        self.assertTrue(
            train_scaffolds.isdisjoint(test_scaffolds),
            "Scaffolds are not unique between train and test sets",
        )

    def test_scaffold_split_reproducibility(self) -> None:
        """
        Test that the scaffold_split function is reproducible when using the same random seed.
        Ensures that running the function multiple times with the same seed produces the same results.
        """
        # Test that the split is reproducible with the same random seed
        data_train1, data_test1 = scaffold_split(
            self.data, smiles_col="smiles", test_size=0.2, random_state=42
        )
        data_train2, data_test2 = scaffold_split(
            self.data, smiles_col="smiles", test_size=0.2, random_state=42
        )

        # Check that the indices of train and test sets are the same
        self.assertTrue(
            data_train1.equals(data_train2), "Training sets are not reproducible"
        )
        self.assertTrue(data_test1.equals(data_test2), "Test sets are not reproducible")

    def test_scaffold_split_invalid(self) -> None:
        """
        Test that the scaffold_split function raises an error when provided with invalid SMILES strings.
        """
        data_valid = self.data.copy()
        data_valid.loc[0:3, "smiles"] = "Invalid_smiles"
        with self.assertRaises(ValueError):
            data_train, data_test = scaffold_split(
                data_valid, smiles_col="smiles", test_size=0.2, random_state=42
            )


if __name__ == "__main__":
    unittest.main()
