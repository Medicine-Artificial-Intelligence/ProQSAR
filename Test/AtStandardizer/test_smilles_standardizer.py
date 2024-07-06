import unittest
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))
from rdkit import Chem
from AtQSAR.AtStandardizer import SMILESStandardizer


class TestSMILESStandardizer(unittest.TestCase):

    def setUp(self):
        self.standardizer = SMILESStandardizer()
        self.example_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        self.example_smiles_data = [
            {"SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O"},  # Aspirin
            {"SMILES": "C1=CC=C(C=C1)C=O"},  # Benzaldehyde
        ]

    def test_standardize_mol(self):
        mol = Chem.MolFromSmiles(self.example_smiles)
        standardized_mol = self.standardizer.standardize_mol(mol)
        self.assertIsNotNone(standardized_mol)
        # Additional checks can be added here based on expected behavior

    def test_standardize_smiles(self):
        standardized_smiles, standardized_mol = self.standardizer.standardize_smiles(
            self.example_smiles
        )
        self.assertIsNotNone(standardized_smiles)
        self.assertIsNotNone(standardized_mol)
        # Additional checks can be added here based on expected behavior

    def test_standardize_dict_smiles(self):
        standardized_data = self.standardizer.standardize_dict_smiles(
            self.example_smiles_data
        )
        self.assertIsInstance(standardized_data, list)
        for item in standardized_data:
            self.assertIn("standardized_SMILES", item)
            self.assertIn("standardized_mol", item)
            self.assertIsNotNone(item["standardized_SMILES"])
            self.assertIsInstance(item["standardized_mol"], Chem.Mol)


if __name__ == "__main__":
    unittest.main()
