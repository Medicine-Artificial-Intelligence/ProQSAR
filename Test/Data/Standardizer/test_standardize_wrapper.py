import unittest
from rdkit import Chem
from proqsar.Data.Standardizer.standardizer_wrapper import (
    normalize_molecule,
    canonicalize_tautomer,
    salts_remover,
    reionize_charges,
    uncharge_molecule,
    assign_stereochemistry,
    fragments_remover,
    remove_hydrogens_and_sanitize,
)


class TestMoleculeFunctions(unittest.TestCase):

    def setUp(self):
        # Example molecule (Caffeine)
        self.mol = Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")

    def test_normalize_molecule(self):
        normalized_mol = normalize_molecule(self.mol)
        self.assertIsNotNone(normalized_mol)
        # Additional assertions based on expected behavior

    def test_canonicalize_tautomer(self):
        canonicalized_mol = canonicalize_tautomer(self.mol)
        self.assertIsNotNone(canonicalized_mol)
        # Additional assertions based on expected behavior

    def test_salts_remover(self):
        salt_free_mol = salts_remover(self.mol)
        self.assertIsNotNone(salt_free_mol)
        # Additional assertions based on expected behavior

    def test_reionize_charges(self):
        reionized_mol = reionize_charges(self.mol)
        self.assertIsNotNone(reionized_mol)
        # Additional assertions based on expected behavior

    def test_uncharge_molecule(self):
        uncharged_mol = uncharge_molecule(self.mol)
        self.assertIsNotNone(uncharged_mol)
        # Additional assertions based on expected behavior

    def test_assign_stereochemistry(self):
        assign_stereochemistry(self.mol)
        # Since the function does not return anything, test for expected side effects

    def test_fragmets_remover(self):
        largest_fragment = fragments_remover(self.mol)
        self.assertIsNotNone(largest_fragment)
        # Additional assertions based on expected behavior

    def test_remove_hydrogens_and_sanitize(self):
        sanitized_mol = remove_hydrogens_and_sanitize(self.mol)
        self.assertIsNotNone(sanitized_mol)
        # Additional assertions based on expected behavior


if __name__ == "__main__":
    unittest.main()
