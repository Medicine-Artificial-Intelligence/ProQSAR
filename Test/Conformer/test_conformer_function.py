import unittest
from rdkit import Chem
from ProQSAR.Conformer.conformer_function import mol_embed, force_field_minimization, get_lowest_energy_conformer

class TestConformerFunction(unittest.TestCase):
    def setUp(self):
        # Create a test molecule with multiple conformers
        self.molecule = Chem.MolFromSmiles('C1CCCCC1')  # Cyclohexane
        # self.molecule.AddConformer(Chem.Conformer(), assignId=True)
        # self.molecule.AddConformer(Chem.Conformer(), assignId=True)
        # # Assuming energy setting for testing; real scenario would involve actual energy calculations
        # self.molecule.GetConformer(0).SetDoubleData("Energy", 10.0)
        # self.molecule.GetConformer(1).SetDoubleData("Energy", 5.0)

    def test_mol_embed_auto(self):
        molecule = Chem.MolFromSmiles('CCO')
        embedded_mol = mol_embed(molecule)
        self.assertIsNotNone(embedded_mol)
        self.assertTrue(embedded_mol.GetNumConformers() >= 1)

    def test_force_field_minimization(self):
        embedded_mol = mol_embed(self.molecule)
        minimized_mol = force_field_minimization(embedded_mol, return_energies=True)
        self.assertIsInstance(minimized_mol, tuple)
        self.assertIsNotNone(minimized_mol[0])
        self.assertGreater(len(minimized_mol[1]), 0)
    
    def test_get_lowest_energy_conformer(self):
        """Test that the lowest energy conformer is correctly identified."""
        embedded_mol = mol_embed(self.molecule, 10)
        minimized_mol, energy = force_field_minimization(embedded_mol, return_energies=True, force_field_method='MMFF94')
        print(minimized_mol)
        lowest_energy_mol = get_lowest_energy_conformer(minimized_mol, 'MMFF94')
        self.assertEqual(lowest_energy_mol.GetNumConformers(), 1)
        # Check if the lowest energy conformer is correctly kept
        lowest_energy = lowest_energy_mol.GetConformer().GetDoubleData("Energy")
        self.assertEqual(lowest_energy, 5.0)

if __name__ == '__main__':
    unittest.main()
