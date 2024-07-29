import unittest
from rdkit import Chem
from rdkit.Chem.rdDistGeom import ETKDG
from ProQSAR.Conformer.conformer_utils import (
    _get_embedding_method,
    _get_num_conformers_from_molecule_size,
    _get_max_iter_from_molecule_size,
    _assert_correct_force_field,
    _assert_has_conformers,
)


class TestConformerUtils(unittest.TestCase):

    def test_get_embedding_method_valid(self):
        """Test getting a valid embedding method."""
        method = _get_embedding_method("ETKDG")
        self.assertIsNotNone(method)

    def test_get_num_conformers(self):
        """Test calculating the number of conformers based on molecule size."""
        molecule = Chem.MolFromSmiles("CCO")  # Ethanol
        num_conformers = _get_num_conformers_from_molecule_size(molecule)
        self.assertEqual(num_conformers, 10)

    def test_get_max_iter_from_molecule_size(self):
        """Test calculating max iterations from molecule size."""
        molecule = Chem.MolFromSmiles("CCCCCCCCCCCC")  # Dodecane
        max_iter = _get_max_iter_from_molecule_size(molecule)
        self.assertEqual(max_iter, 20 + 12 * 10)  # 140 iterations

    def test_assert_correct_force_field_valid(self):
        """Test validating a correct force field."""
        _assert_correct_force_field("MMFF94")  # No error should be raised

    def test_assert_correct_force_field_invalid(self):
        """Test validating an incorrect force field."""
        with self.assertRaises(ValueError):
            _assert_correct_force_field("nonexistent")

    def test_assert_has_conformers(self):
        """Test checking if a molecule has conformers."""
        molecule = Chem.MolFromSmiles("CCO")
        molecule = Chem.AddHs(molecule)
        _ = Chem.rdDistGeom.EmbedMultipleConfs(molecule, numConfs=10, params=ETKDG())
        _assert_has_conformers(molecule)

    def test_assert_has_conformers_fail(self):
        """Test molecule without conformers."""
        molecule = Chem.MolFromSmiles("CCO")
        with self.assertRaises(ValueError):
            _assert_has_conformers(molecule)


if __name__ == "__main__":
    unittest.main()
