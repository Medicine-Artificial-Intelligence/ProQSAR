import logging
import numpy as np
from typing import Optional
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Avalon import pyAvalonTools as fpAvalon
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from mordred import Calculator, descriptors


def RDKFp(
    mol: Chem.Mol, maxPath: int = 6, fpSize: int = 2048, numBitsPerFeature: int = 2
) -> Optional[np.ndarray]:
    """
    Calculate RDKit fingerprint of a molecule.

    :param mol: RDKit molecule object.
    :type mol: Chem.Mol
    :param maxPath: Maximum path length to consider. Default is 6.
    :type maxPath: int, optional
    :param fpSize: Size (number of bits) of the fingerprint. Default is 2048.
    :type fpSize: int, optional
    :param numBitsPerFeature: Number of bits set per feature. Default is 2.
    :type numBitsPerFeature: int, optional
    :return: RDKit fingerprint as a NumPy array, or None if calculation fails.
    :rtype: Optional[numpy.ndarray]

    .. code-block:: python

        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("c1ccccc1")
        >>> fp = RDKFp(mol)
        >>> fp.shape
        (2048,)
    """
    if mol is None:
        logging.error("Invalid molecule provided.")
        return None

    try:
        mfpgen = rdFingerprintGenerator.GetRDKitFPGenerator(
            maxPath=maxPath, fpSize=fpSize, numBitsPerFeature=numBitsPerFeature
        )
        fp = mfpgen.GetFingerprint(mol)
        ar = np.zeros((fpSize,), dtype=np.uint8)
        DataStructs.cDataStructs.ConvertToNumpyArray(fp, ar)
        return ar
    except Exception as e:
        logging.error(f"Failed to calculate RDKFp: {e}")
        return None


def ECFPs(
    mol: Chem.Mol, radius: int = 1, nBits: int = 2048, useFeatures: bool = False
) -> Optional[np.ndarray]:
    """
    Calculate Extended-Connectivity Fingerprints (ECFP) for a molecule.

    :param mol: RDKit molecule object.
    :type mol: Chem.Mol
    :param radius: Radius of circular fingerprint. Default is 1.
    :type radius: int, optional
    :param nBits: Size (number of bits) of the fingerprint. Default is 2048.
    :type nBits: int, optional
    :param useFeatures: Whether to use feature invariants instead of atom invariants. Default is False.
    :type useFeatures: bool, optional
    :return: ECFP fingerprint as a NumPy array, or None if calculation fails.
    :rtype: Optional[numpy.ndarray]

    .. code-block:: python

        >>> mol = Chem.MolFromSmiles("c1ccccc1")
        >>> ecfp = ECFPs(mol)
        >>> ecfp.shape
        (2048,)
    """
    if mol is None:
        logging.error("Invalid molecule provided.")
        return None

    try:
        atomInvariantsGenerator = (
            rdFingerprintGenerator.GetMorganFeatureAtomInvGen() if useFeatures else None
        )
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=nBits, atomInvariantsGenerator=atomInvariantsGenerator
        )
        fp = mfpgen.GetFingerprint(mol)
        ar = np.zeros((nBits,), dtype=np.uint8)
        DataStructs.cDataStructs.ConvertToNumpyArray(fp, ar)
        return ar
    except Exception as e:
        logging.error(f"Failed to calculate ECFPs/FCFPs: {e}")
        return None


def MACCs(mol: Chem.Mol) -> Optional[np.ndarray]:
    """
    Generate MACCS keys fingerprint for a molecule.

    :param mol: RDKit molecule object.
    :type mol: Chem.Mol
    :return: MACCS fingerprint (167 bits) as a NumPy array, or None if calculation fails.
    :rtype: Optional[numpy.ndarray]

    .. code-block:: python

        >>> mol = Chem.MolFromSmiles("c1ccccc1")
        >>> maccs = MACCs(mol)
        >>> maccs.shape
        (167,)
    """
    if mol is None:
        logging.error("Invalid molecule provided.")
        return None

    try:
        fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        ar = np.zeros((167,), dtype=np.uint8)
        DataStructs.cDataStructs.ConvertToNumpyArray(fp, ar)
        return ar
    except Exception as e:
        logging.error(f"Failed to calculate MACCS keys fingerprint: {e}")
        return None


def Avalon(mol: Chem.Mol) -> Optional[np.ndarray]:
    """
    Calculate the Avalon fingerprint for a molecule.

    :param mol: RDKit molecule object.
    :type mol: Chem.Mol
    :return: Avalon fingerprint (1024 bits) as a NumPy array, or None if calculation fails.
    :rtype: Optional[numpy.ndarray]
    :raises ValueError: If the molecule is None.

    .. code-block:: python

        >>> mol = Chem.MolFromSmiles("c1ccccc1")
        >>> avalon = Avalon(mol)
        >>> avalon.shape
        (1024,)
    """
    if mol is None:
        raise ValueError("Provided molecule is invalid.")

    try:
        fp = fpAvalon.GetAvalonFP(mol, 1024)
        ar = np.zeros((1024,), dtype=np.int8)
        DataStructs.cDataStructs.ConvertToNumpyArray(fp, ar)
        return ar
    except Exception as e:
        logging.error(f"Failed to calculate Avalon fingerprint: {e}")
        return None


def RDKDes(mol: Chem.Mol) -> Optional[np.ndarray]:
    """
    Calculate RDKit molecular descriptors for a given molecule.

    :param mol: RDKit molecule object.
    :type mol: Chem.Mol
    :return: Array of molecular descriptor values, or None if calculation fails.
    :rtype: Optional[numpy.ndarray]

    .. code-block:: python

        >>> mol = Chem.MolFromSmiles("c1ccccc1")
        >>> descs = RDKDes(mol)
        >>> len(descs)
        208
    """
    if mol is None:
        logging.error("Invalid molecule provided.")
        return None

    try:
        des_list = [x[0] for x in Descriptors._descList]
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)
        descriptors = calculator.CalcDescriptors(mol)
        return np.array(descriptors, dtype=np.float64)
    except Exception as e:
        logging.error(f"Failed to compute RDKit descriptors: {e}")
        return None


def mol2pharm2dgbfp(mol: Chem.Mol) -> Optional[np.ndarray]:
    """
    Calculate 2D pharmacophore fingerprints (Gobbi) for a molecule.

    :param mol: RDKit molecule object.
    :type mol: Chem.Mol
    :return: Pharmacophore fingerprint as a NumPy binary array, or None if calculation fails.
    :rtype: Optional[numpy.ndarray]

    .. code-block:: python

        >>> mol = Chem.MolFromSmiles("c1ccccc1")
        >>> fp = mol2pharm2dgbfp(mol)
        >>> fp.shape
        (1400,)
    """
    if mol is None:
        logging.error("Invalid molecule provided.")
        return None

    try:
        fp = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
        return np.frombuffer(fp.ToBitString().encode(), dtype=np.uint8) - ord("0")
    except Exception as e:
        logging.error(f"Failed to compute pharmacophore fingerprint: {e}")
        return None


def MordredDes(mol: Chem.Mol) -> Optional[np.ndarray]:
    """
    Calculate 2D Mordred molecular descriptors for a molecule.

    :param mol: RDKit molecule object.
    :type mol: Chem.Mol
    :return: Array of Mordred descriptor values (float), or None if calculation fails.
    :rtype: Optional[numpy.ndarray]

    .. code-block:: python

        >>> mol = Chem.MolFromSmiles("CCO")
        >>> descs = MordredDes(mol)
        >>> descs.shape
        (1613,)
    """
    if mol is None:
        logging.error("Invalid molecule provided.")
        return None

    try:
        calc = Calculator(descriptors, ignore_3D=True)
        desc = calc(mol)
        values = [float(val) if val is not None else np.nan for val in desc]
        return np.array(values, dtype=np.float64)
    except Exception as e:
        logging.error(f"Failed to compute Mordred descriptors: {e}")
        return None
