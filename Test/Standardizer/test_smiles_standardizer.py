import unittest
import pandas as pd
from rdkit import Chem
from ProQSAR.Standardizer.smiles_standardizer import SMILESStandardizer


class TestSMILESStandardizer(unittest.TestCase):

    def setUp(self):
        self.standardizer = SMILESStandardizer()
        self.example_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        self.example_smiles_list_dict = [
            {"SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O"},
            {"SMILES": "C1=CC=C(C=C1)C=O"},
        ]
        self.example_smiles_data = pd.DataFrame(self.example_smiles_list_dict)

    def test_standardize_mol(self):
        """Test standardizing a valid mol object."""
        mol = Chem.MolFromSmiles(self.example_smiles)
        standardized_mol = self.standardizer.standardize_mol(mol)
        self.assertIsNotNone(standardized_mol)
        # Additional checks can be added here based on expected behavior

    def test_standardize_smiles_valid(self):
        """Test standardizing a valid SMILES string."""
        smiles = "CCO"  # Ethanol
        standardized_smiles, standardized_mol = self.standardizer.standardize_smiles(
            smiles
        )
        self.assertIsNotNone(standardized_mol)
        self.assertIsNotNone(standardized_smiles)
        self.assertEqual(Chem.MolToSmiles(standardized_mol), "CCO")

    def test_standardize_smiles_invalid(self):
        """Test standardizing an invalid SMILES string."""
        smiles = "Invalid_smiles"  # A simple cyclohexanol to test the stereochemistry handling
        standardized_smiles, standardized_mol = self.standardizer.standardize_smiles(
            smiles
        )
        self.assertIsNone(standardized_mol)
        self.assertEqual(standardized_smiles, None)

    def test_standardize_dict_smiles(self):
        standardized_data = self.standardizer.standardize_dict_smiles(
            self.example_smiles_list_dict
        )
        self.assertIsInstance(standardized_data, list)
        for item in standardized_data:
            self.assertIn("standardized_SMILES", item)
            self.assertIn("standardized_mol", item)
            self.assertIsNotNone(item["standardized_SMILES"])
            self.assertIsInstance(item["standardized_mol"], Chem.Mol)

    def test_standardize_df_smiles(self):
        standardized_data = self.standardizer.standardize_dict_smiles(
            self.example_smiles_data
        )
        self.assertIsInstance(standardized_data, list)
        for item in standardized_data:
            self.assertIn("standardized_SMILES", item)
            self.assertIn("standardized_mol", item)
            self.assertIsNotNone(item["standardized_SMILES"])
            self.assertIsInstance(item["standardized_mol"], Chem.Mol)

    def test_handle_invalid_data_type(self):
        """Test standardization with invalid data type."""
        with self.assertRaises(TypeError):
            self.standardizer.standardize_dict_smiles("not a dataframe or list")

    def test_deactivate(self):
        self.standardizer.deactivate = True
        standardized_data = self.standardizer.standardize_dict_smiles(
            self.example_smiles_list_dict
        )
        self.assertEqual(standardized_data, self.example_smiles_list_dict)


if __name__ == "__main__":
    unittest.main()
