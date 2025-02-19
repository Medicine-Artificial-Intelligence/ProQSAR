import unittest
from rdkit import Chem
from ProQSAR.Conformer.conformation_generator import ConformationGenerator
import logging


logging.basicConfig(level=logging.INFO)


class TestConformationGenerator(unittest.TestCase):

    def setUp(self):
        self.molecule = Chem.MolFromSmiles("CCO")
        self.example_smiles_list_dict = [
            {"SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O"},
            {"SMILES": "C1=CC=C(C=C1)C=O"},
        ]

    def test_mol_process_valid(self):
        minimized_mol, minimized_energy = ConformationGenerator._mol_process(
            self.molecule,
            num_conformers="auto",
            embedding_method="ETKDGv3",
            num_threads=1,
            random_coords_threshold=100,
            random_seed=42,
            force_field_method="MMFF94",
            max_iter="auto",
            return_energies=False,
        )
        self.assertIsInstance(minimized_mol, Chem.Mol)
        self.assertIsInstance(minimized_energy, float)
        self.assertGreater(minimized_energy, -float("inf"))

    def test_smiles_process_valid(self):
        smiles = "CCO"
        minimized_mol, minimized_energy = ConformationGenerator._smiles_process(
            smiles,
            num_conformers="auto",
            embedding_method="ETKDGv3",
            num_threads=1,
            random_coords_threshold=100,
            random_seed=42,
            force_field_method="MMFF94",
            max_iter="auto",
            return_energies=False,
        )
        self.assertIsInstance(minimized_mol, Chem.Mol)
        self.assertIsInstance(minimized_energy, float)
        self.assertGreater(minimized_energy, -float("inf"))

    def test_smiles_process_invalid(self):
        smiles = "InvalidSmilesString"
        minimized_mol, minimized_energy = ConformationGenerator._smiles_process(
            smiles,
            num_conformers="auto",
            embedding_method="ETKDGv3",
            num_threads=1,
            random_coords_threshold=100,
            random_seed=42,
            force_field_method="MMFF94",
            max_iter="auto",
            return_energies=False,
        )
        self.assertIsNone(minimized_mol)
        self.assertIsNone(minimized_energy)

    def test_dict_process(self):
        # Process the dictionary containing the SMILES string
        result = ConformationGenerator._dict_process(self.example_smiles_list_dict[0])

        # Check that the keys 'conformer' and 'energy' are present in the results
        self.assertIn(
            "conformer", result, "The result should include the 'conformer' key."
        )
        self.assertIn("energy", result, "The result should include the 'energy' key.")

        # Ensure that neither 'conformer' nor 'energy' is None
        self.assertIsNotNone(result["conformer"], "Conformer should not be None.")
        self.assertIsNotNone(result["energy"], "Energy should not be None.")

        # Additional tests could check for expected types or ranges of values
        # Example: Check if energy is a float and within a reasonable range
        self.assertIsInstance(result["energy"], float, "Energy should be a float.")
        # This check assumes you know what a reasonable energy value might be
        self.assertGreater(
            result["energy"], -1000, "Energy should be within a reasonable range."
        )

    def test_parallel_process(self):
        # Processing the SMILES strings in parallel
        results = ConformationGenerator.parallel_process(self.example_smiles_list_dict)

        # Check the number of results
        self.assertEqual(
            len(results),
            len(self.example_smiles_list_dict),
            "The number of results should match the number of input items.",
        )

        # Check that results are returned for each molecule
        for result in results:
            self.assertIn(
                "conformer", result, "The result should include the 'conformer' key."
            )
            self.assertIn(
                "energy", result, "The result should include the 'energy' key."
            )
            self.assertIsNotNone(result["conformer"], "Conformer should not be None.")
            self.assertIsNotNone(result["energy"], "Energy should not be None.")


if __name__ == "__main__":
    unittest.main()
