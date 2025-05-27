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

    Parameters:
    - mol (Chem.Mol): RDKit molecule object.
    - maxPath (int, optional): Maximum path length, default is 5.
    - fpSize (int, optional): Size of the fingerprint, default is 2048.
    - nBitsPerHash (int, optional): Number of bits per hash, default is 2.

    Returns:
    numpy.ndarray: A numpy array representing the RDKit fingerprint.

    Examples:
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('C1=CC=CC=C1')
    >>> RDKFp(mol)
    array([0, 1, 0, ..., 0])
    """
    if mol is None:
        logging.error("Invalid molecule provided.")
        return None

    mfpgen = rdFingerprintGenerator.GetRDKitFPGenerator(
        maxPath=maxPath, fpSize=fpSize, numBitsPerFeature=numBitsPerFeature)
    
    fp = mfpgen.GetFingerprint(mol)
    
    ar = np.zeros((fpSize,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, ar)
    return ar


def ECFPs(
    mol: Chem.Mol, radius: int = 1, nBits: int = 2048, useFeatures: bool = False
) -> Optional[np.ndarray]:
    """
    Calculate Extended-Connectivity Fingerprints (ECFP) for a molecule.

    Parameters:
    - mol (Chem.Mol): RDKit molecule object.
    - radius (int, optional): The radius of the circular fingerprint, default is 1.
    - nBits (int, optional): Size of the fingerprint, default is 2048.
    - useFeatures (bool, optional): Whether to use feature invariants, default is False.

    Returns:
    - numpy.ndarray: A numpy array representing the ECFP fingerprint.

    Examples:
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('C1=CC=CC=C1')
    >>> ECFPs(mol)
    array([0, 1, 0, ..., 0])
    """
    if mol is None:
        logging.error("Invalid molecule provided.")
        return None
    
    atomInvariantsGenerator = None
    if useFeatures:
        atomInvariantsGenerator = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=nBits, 
        atomInvariantsGenerator=atomInvariantsGenerator)
    
    fp = mfpgen.GetFingerprint(mol)

    ar = np.zeros((nBits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, ar)
    return ar


def MACCs(mol: Chem.Mol) -> Optional[np.ndarray]:
    """
    Generate MACCS keys fingerprint for a molecule.

    Parameters:
    mol (Chem.Mol): RDKit molecule object.

    Returns:
    numpy.ndarray: A numpy array representing the MACCS keys fingerprint.

    Raises:
    ValueError: If an invalid molecule is provided.

    Examples:
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('C1=CC=CC=C1')
    >>> MACCs(mol)
    array([0, 1, 0, ..., 0])
    """
    if mol is None:
        logging.error("Invalid molecule provided.")
        return None
    fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
    ar = np.zeros((167,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, ar)
    return ar


def Avalon(mol: Chem.Mol) -> Optional[np.ndarray]:
    """
    Calculate the Avalon fingerprint for a given RDKit molecule object. Avalon fingerprints
    are a type of chemical fingerprint used in cheminformatics for representing the presence
    or absence of particular substructures in a molecule.

    Parameters:
    - mol (Mol): RDKit molecule object from which the fingerprint is calculated.

    Returns:
    - Optional[numpy.ndarray]: A numpy array representing the Avalon fingerprint if the molecule is valid;
      None otherwise.

    Raises:
    - ValueError: If the input molecule is None, indicating an invalid molecule.

    Examples:
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('C1=CC=CC=C1')
    >>> fp_array = Avalon(mol)
    >>> fp_array.size
    1024

    Note:
    - The function initializes a zero array of size 1024 bits to store the fingerprint.
    """
    if mol is None:
        raise ValueError("Provided molecule is invalid.")

    try:
        fp = fpAvalon.GetAvalonFP(mol, 1024)
        ar = np.zeros((1024,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, ar)
        return ar
    except Exception as e:
        logging.error(f"Failed to calculate Avalon fingerprint: {e}")
        return None


def RDKDes(mol: Chem.Mol) -> Optional[np.ndarray]:
    """
    Calculate a comprehensive array of molecular descriptors using RDKit for a given molecule.

    Parameters:
    - mol (Mol): RDKit molecule object from which to calculate descriptors.

    Returns:
    - numpy.ndarray: An array of floating-point numbers representing the molecular descriptors.

    Raises:
    - ValueError: If an invalid molecule is provided.

    Examples:
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('C1=CC=CC=C1')
    >>> descriptors = RDKDes(mol)
    >>> len(descriptors)
    208  # Number of descriptors calculated by RDKit

    Note:
    - The number and types of descriptors returned depend on the RDKit version and the configured descriptors.
    """
    if mol is None:
        logging.error("Invalid molecule provided.")
        return None
    des_list = [x[0] for x in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)
    descriptors = calculator.CalcDescriptors(mol)
    return np.array(descriptors, dtype=np.float64)


def mol2pharm2dgbfp(mol: Chem.Mol) -> Optional[np.ndarray]:
    """
    Calculate 2D pharmacophore fingerprints (Gobbi) for a given molecule using RDKit's Pharm2D functionality.

    Parameters:
    - mol (Mol): RDKit molecule object to compute the 2D pharmacophore fingerprints.

    Returns:
    - numpy.ndarray: A binary array representing the 2D pharmacophore fingerprint.

    Raises:
    - ValueError: If an invalid molecule is provided.

    Examples:
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('C1=CC=CC=C1')
    >>> fp = mol2pharm2dgbfp(mol)
    >>> fp.size
    1400  # Size of the generated fingerprint array

    Note:
    - The generated fingerprint is sensitive to the configuration of the pharmacophore model.
    """
    if mol is None:
        logging.error("Invalid molecule provided.")
        return None
    
    fp = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
    return np.frombuffer(fp.ToBitString().encode(), dtype=np.uint8) - ord("0")

def MordredDes(mol: Chem.Mol) -> Optional[np.ndarray]:
    """
    Calculate 2D Mordred molecular descriptors for a given RDKit molecule.

    Parameters:
    - mol (Chem.Mol): RDKit molecule object.

    Returns:
    - Optional[np.ndarray]: A NumPy array of Mordred descriptors, or None if calculation fails.

    Examples:
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('CCO')
    >>> desc = MordredDes(mol)
    >>> desc.shape  # Depends on Mordred version and settings
    (1613,)  # Example output

    Notes:
    - Only 2D descriptors are calculated (ignore_3D=True).
    - Requires Mordred to be installed: pip install mordred
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
