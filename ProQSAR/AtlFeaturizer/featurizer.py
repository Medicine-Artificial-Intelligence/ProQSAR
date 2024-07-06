from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, rdMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from rdkit.Avalon import pyAvalonTools as fpAvalon
from cats2d.rd_cats2d import CATS2D


class MolecularFingerprint:

    @staticmethod
    def RDKFp(mol, maxPath=5, fpSize=2048, nBitsPerHash=2):
        """
        Calculate RDKit fingerprint of a molecule.

        Parameters:
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object.
        maxPath : int, optional
            Maximum path length (default is 5).
        fpSize : int, optional
            Size of the fingerprint (default is 2048).
        nBitsPerHash : int, optional
            Number of bits per hash (default is 2).

        Returns:
        numpy.ndarray
            An array representing the RDKit fingerprint.
        """
        fp = Chem.RDKFingerprint(
            mol, maxPath=maxPath, fpSize=fpSize, nBitsPerHash=nBitsPerHash
        )
        ar = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, ar)
        return ar

    @staticmethod
    def ECFPs(mol, radius=1, nBits=2048, useFeatures=False):
        """
        Calculate Extended-Connectivity Fingerprints (ECFP) for a molecule.

        Parameters:
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object.
        radius : int, optional
            The radius of the circular fingerprint (default is 1).
        nBits : int, optional
            Size of the fingerprint (default is 2048).
        useFeatures : bool, optional
            Whether to use feature invariants (default is False).

        Returns:
        numpy.ndarray
            An array representing the ECFP fingerprint.
        """
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=radius, nBits=nBits, useFeatures=useFeatures
        )
        ar = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, ar)
        return ar

    @staticmethod
    def MACCs(mol):
        """
        Generate MACCS keys fingerprint for a molecule.

        Parameters:
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object.

        Returns:
        numpy.ndarray
            An array representing the MACCS keys fingerprint.
        """
        fp = MACCSkeys.GenMACCSKeys(mol)
        ar = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, ar)
        return ar

    @staticmethod
    def Avalon(mol):
        """
        Calculate Avalon fingerprint for a molecule.

        Parameters:
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object.

        Returns:
        numpy.ndarray
            An array representing the Avalon fingerprint.
        """
        fp = fpAvalon.GetAvalonFP(mol, 1024)
        ar = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, ar)
        return ar

    @staticmethod
    def RDKDes(mol):
        """
        Calculate RDKit molecular descriptors for a molecule.

        Parameters:
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object.

        Returns:
        numpy.ndarray
            An array of calculated molecular descriptors.
        """
        des_list = [x[0] for x in Descriptors._descList]
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)
        return np.array(calculator.CalcDescriptors(mol), dtype=np.float64)

    @staticmethod
    def mol2pharm2dgbfp(mol):
        """
        Calculate 2D pharmacophore fingerprints (Gobbi) for a molecule.

        Parameters:
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object.

        Returns:
        numpy.ndarray
            An array representing the 2D pharmacophore fingerprint.
        """
        fp = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
        return np.frombuffer(fp.ToBitString().encode(), "u1") - ord("0")

    @staticmethod
    def mol2cats(mol):
        """
        Calculate CATS2D pharmacophore fingerprint for a molecule.

        Parameters:
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object.

        Returns:
        numpy.ndarray
            An array representing the CATS2D pharmacophore fingerprint.
        """
        cats = CATS2D(max_path_len=9)
        fp = cats.getCATs2D(mol)
        return np.array(fp, dtype=np.int8)
