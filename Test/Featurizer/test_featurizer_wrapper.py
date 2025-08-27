import unittest
import numpy as np
from rdkit import Chem
from ProQSAR.Featurizer.featurizer_wrapper import (
    RDKFp,
    ECFPs,
    MACCs,
    Avalon,
    RDKDes,
    mol2pharm2dgbfp,
)


class TestChemicalFingerprints(unittest.TestCase):
    def setUp(self):

        self.benzene = Chem.MolFromSmiles("C1=CC=CC=C1")
        self.invalid_mol = None

    def test_RDKFp_valid_input(self):
        result = RDKFp(self.benzene, maxPath=5, fpSize=2048)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.size, 2048)

    def test_RDKFp_invalid_input(self):
        result = RDKFp(self.invalid_mol)
        self.assertIsNone(result)

    def test_ECFPs_valid_input(self):
        result = ECFPs(self.benzene, radius=2, nBits=2048, useFeatures=False)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.size, 2048)

    def test_ECFPs_invalid_input(self):
        result = ECFPs(self.invalid_mol)
        self.assertIsNone(result)

    def test_MACCs_valid_input(self):
        result = MACCs(self.benzene)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.size, 167)

    def test_MACCs_invalid_input(self):
        result = MACCs(self.invalid_mol)
        self.assertIsNone(result)

    def test_Avalon_valid_input(self):
        result = Avalon(self.benzene)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.size, 1024)

    def test_Avalon_invalid_input(self):
        with self.assertRaises(ValueError):
            Avalon(self.invalid_mol)

    def test_RDKDes_valid_input(self):
        result = RDKDes(self.benzene)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float64)
        self.assertEqual(result.size, 217)

    def test_RDKDes_invalid_input(self):
        result = RDKDes(self.invalid_mol)
        self.assertIsNone(result)

    def test_mol2pharm2dgbfp_valid_input(self):
        result = mol2pharm2dgbfp(self.benzene)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.size, 39972)

    def test_mol2pharm2dgbfp_invalid_input(self):
        result = mol2pharm2dgbfp(self.invalid_mol)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
