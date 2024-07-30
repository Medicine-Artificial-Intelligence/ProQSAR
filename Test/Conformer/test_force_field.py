import unittest
from rdkit import Chem
from rdkit.Chem import AllChem
from ProQSAR.Conformer._force_field import ForceField


class TestForceField(unittest.TestCase):

    def setUp(self):
        # Create a simple molecule for testing
        self.molecule = Chem.MolFromSmiles("CCO")
        AllChem.EmbedMolecule(self.molecule)
        self.molecule = Chem.AddHs(self.molecule)

    def test_force_field_minimization_mmff94(self):
        result = ForceField.force_field_minimization(
            self.molecule, "MMFF94", max_iter=100
        )
        self.assertIsInstance(result, Chem.Mol)
        self.assertGreater(
            self.molecule.GetNumConformers(), 0
        )  # Ensure conformers exist

    def test_force_field_minimization_uff(self):
        result = ForceField.force_field_minimization(self.molecule, "UFF", max_iter=100)
        self.assertIsInstance(result, Chem.Mol)
        self.assertGreater(
            self.molecule.GetNumConformers(), 0
        )  # Ensure conformers exist

    def test_force_field_minimization_auto_iterations(self):
        result = ForceField.force_field_minimization(
            self.molecule, "MMFF94", max_iter="auto"
        )
        self.assertIsInstance(result, Chem.Mol)
        self.assertGreater(
            self.molecule.GetNumConformers(), 0
        )  # Ensure conformers exist

    def test_force_field_minimization_invalid_method(self):
        with self.assertRaises(ValueError):
            ForceField.force_field_minimization(self.molecule, "INVALID_METHOD")

    def test_force_field_minimization_no_conformers(self):
        molecule_no_conformers = Chem.MolFromSmiles("C")
        # No embedding or conformer creation
        with self.assertRaises(ValueError):
            ForceField.force_field_minimization(molecule_no_conformers, "MMFF94")

    def test_force_field_minimization_return_energies(self):
        result, energies = ForceField.force_field_minimization(
            self.molecule, "MMFF94", return_energies=True
        )
        self.assertIsInstance(result, Chem.Mol)
        self.assertIsInstance(energies, list)
        self.assertTrue(all(isinstance(e, float) for e in energies))

    # `test_compute_force_field_energy` Test Cases

    def test_compute_force_field_energy_mmff94(self):
        AllChem.MMFFOptimizeMoleculeConfs(self.molecule)
        energy = ForceField.compute_force_field_energy(self.molecule, 0, "MMFF94")
        self.assertIsInstance(energy, float)

    def test_compute_force_field_energy_uff(self):
        AllChem.UFFOptimizeMoleculeConfs(self.molecule)
        energy = ForceField.compute_force_field_energy(self.molecule, 0, "UFF")
        self.assertIsInstance(energy, float)

    def test_compute_force_field_energy_invalid_method(self):
        with self.assertRaises(ValueError):
            ForceField.compute_force_field_energy(self.molecule, 0, "INVALID_METHOD")

    def test_compute_force_field_energy_invalid_conformer_id(self):
        with self.assertRaises(ValueError):
            ForceField.compute_force_field_energy(self.molecule, 1, "MMFF94")

    # `test_get_lowest_energy_conformer` Test Cases

    def test_get_lowest_energy_conformer(self):
        AllChem.MMFFOptimizeMoleculeConfs(self.molecule)
        lowest_conformer = ForceField.get_lowest_energy_conformer(
            self.molecule, "MMFF94"
        )
        self.assertIsInstance(lowest_conformer, Chem.Mol)
        self.assertEqual(lowest_conformer.GetNumConformers(), 1)

    def test_get_lowest_energy_conformer_no_conformers(self):
        molecule_no_conformers = Chem.MolFromSmiles("C")
        with self.assertRaises(ValueError):
            ForceField.get_lowest_energy_conformer(molecule_no_conformers, "MMFF94")

    def test_get_lowest_energy_conformer_invalid_method(self):
        AllChem.MMFFOptimizeMoleculeConfs(self.molecule)
        with self.assertRaises(ValueError):
            ForceField.get_lowest_energy_conformer(self.molecule, "INVALID_METHOD")


if __name__ == "__main__":
    unittest.main()
