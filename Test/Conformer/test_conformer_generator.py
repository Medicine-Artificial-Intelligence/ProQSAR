import unittest
from rdkit import Chem
import pandas as pd

from ProQSAR.Conformer.conformer_generator import ConformerGenerator  
class TestConformerGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = ConformerGenerator(num_conformer_candidates=10)
        self.example_mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
        self.example_dict = {"mol": self.example_mol}
        self.example_smiles_list_dict = [
            {"SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O"},
            {"SMILES": "C1=CC=C(C=C1)C=O"},
        ]

    def test_gen_conformers_invalid_input(self):
        """ Test that non-Mol inputs raise an appropriate exception. """
        with self.assertRaises(ValueError):
            self.generator.gen_conformers("This is not a Mol object")

    def test_gen_conformers(self):
        """ Test that conformers are generated correctly. """
        result_mol = self.generator.gen_conformers(self.example_mol)
        self.assertIsInstance(result_mol, Chem.Mol)

    # def test_dict_process_with_simple_dict(self):
    #     """ Test the _dict_process method with a simple dictionary input. """
    #     # This will not work unless gen_conformers and related methods are designed to handle this simple case
    #     result_dict = self.generator._dict_process( self.example_dict)
    #     self.assertIn('mol_conf', result_dict)

    # def test_conformers_parallel_with_dataframe(self):
    #     """ Test the _conformers_parallel method using a DataFrame input. """
    #     test_molecule = Chem.MolFromSmiles('C')
    #     df = pd.DataFrame({'mol': [test_molecule, test_molecule]})

    #     # Similar to the previous test, this assumes _conformers_parallel can handle basic inputs directly
    #     result_df = self.generator._conformers_parallel(df)
    #     self.assertIn('mol_conf', result_df.columns)
    #     self.assertEqual(len(result_df), 2)  # Check that two rows exist

    # def test_conformers_parallel_with_invalid_data(self):
    #     """ Test error handling for invalid data types in _conformers_parallel method. """
    #     with self.assertRaises(ValueError):
    #         self.generator._conformers_parallel("This should fail")

if __name__ == '__main__':
    unittest.main()
