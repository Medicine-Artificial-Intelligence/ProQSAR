import unittest
from rdkit import Chem
from rdkit.Chem.rdDistGeom import ETDG, ETKDG, ETKDGv2, ETKDGv3, srETKDGv3, KDG
from ProQSAR.Conformer._embedding import (
    Embeddings,
)


class TestEmbeddings(unittest.TestCase):

    def setUp(self) -> None:
        """
        Set up the test environment.
        """
        self.embeddings = Embeddings()
        self.mol = Chem.MolFromSmiles("C1CCCCC1")

    def test_get_embedding_method_valid(self):
        """
        Test _get_embedding_method with valid force field methods.
        """
        expected_types = {
            "ETDG": ETDG,
            "ETKDG": ETKDG,
            "ETKDGv2": ETKDGv2,
            "ETKDGv3": ETKDGv3,
            "srETKDGv3": srETKDGv3,
            "KDG": KDG,
        }

        for method, expected_type in expected_types.items():
            with self.subTest(method=method):
                result = self.embeddings._get_embedding_method(method)
                self.assertIsNotNone(result)

    def test_get_embedding_method_invalid(self):
        """
        Test _get_embedding_method with an invalid force field method.
        """
        with self.assertRaises(KeyError):
            self.embeddings._get_embedding_method("InvalidMethod")

    def test_get_num_conformers_from_molecule_size(self):
        """
        Test _get_num_conformers_from_molecule_size with various molecule sizes.
        """
        num_conformers = self.embeddings._get_num_conformers_from_molecule_size(
            self.mol,
            max_num_conformers=10,
            min_num_conformers=2,
            decr_num_conformers=0.04,
        )
        self.assertEqual(
            num_conformers, 10
        )  # As Cyclohexane has 6 atoms, we expect 10 conformers

    def test_mol_embed_auto_conformers(self):
        """
        Test mol_embed with automatic conformer generation.
        """
        embedded_mol = self.embeddings.mol_embed(
            self.mol,
            num_conformers="auto",
            embedding_method="ETKDGv3",
            num_threads=1,
            random_coords_threshold=100,
            random_seed=42,
        )
        self.assertIsInstance(embedded_mol, Chem.Mol)
        self.assertGreater(embedded_mol.GetNumConformers(), 0)

    def test_mol_embed_specific_conformers(self):
        """
        Test mol_embed with a specific number of conformers.
        """
        num_conformers = 5
        embedded_mol = self.embeddings.mol_embed(
            self.mol,
            num_conformers=num_conformers,
            embedding_method="ETKDGv3",
            num_threads=1,
            random_coords_threshold=100,
            random_seed=42,
        )
        self.assertIsInstance(embedded_mol, Chem.Mol)
        self.assertEqual(embedded_mol.GetNumConformers(), num_conformers)

    def test_mol_embed_invalid_num_conformers(self):
        """
        Test mol_embed with invalid num_conformers input.
        """
        with self.assertRaises(ValueError):
            self.embeddings.mol_embed(
                self.mol,
                num_conformers="invalid",
                embedding_method="ETKDGv3",
                num_threads=1,
                random_coords_threshold=100,
                random_seed=42,
            )


if __name__ == "__main__":
    unittest.main()
