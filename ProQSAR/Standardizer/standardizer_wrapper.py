import logging
from typing import Optional
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize


def normalize_molecule(mol: Chem.Mol) -> Chem.Mol:
    """
    Normalize a molecule using RDKit's Normalizer to correct functional groups and recharges.

    Parameters:
    - mol (Chem.Mol): The RDKit molecule object to be normalized.

    Returns:
    - Chem.Mol: The normalized RDKit molecule object.

    Example:
    >>> mol = Chem.MolFromSmiles("CC(=O)O")
    >>> normalized = normalize_molecule(mol)
    """
    return rdMolStandardize.Normalize(mol)


def canonicalize_tautomer(mol: Chem.Mol) -> Chem.Mol:
    """
    Canonicalize the tautomer of a molecule using RDKit's TautomerCanonicalizer.

    Parameters:
    - mol (Chem.Mol): The RDKit molecule object.

    Returns:
    - Chem.Mol: The molecule object with canonicalized tautomer.

    Example:
    >>> mol = Chem.MolFromSmiles("O=C1NC=CC1=O")
    >>> canonicalized = canonicalize_tautomer(mol)
    """
    return rdMolStandardize.CanonicalTautomer(mol)


def salts_remover(mol: Chem.Mol) -> Chem.Mol:
    """
    Remove salt fragments from a molecule using RDKit's SaltRemover.

    Parameters:
    - mol (Chem.Mol): The RDKit molecule object.

    Returns:
    - Chem.Mol: The molecule object with salts removed.

    Example:
    >>> mol = Chem.MolFromSmiles("CCO.Na")
    >>> desalted = salts_remover(mol)
    """
    return SaltRemover().StripMol(mol)


def reionize_charges(mol: Chem.Mol) -> Chem.Mol:
    """
    Adjust molecule to its most likely ionic state using RDKit's Reionizer.

    Parameters:
    - mol (Chem.Mol): The RDKit molecule object.

    Returns:
    - Chem.Mol: The molecule object with reionized charges.

    Example:
    >>> mol = Chem.MolFromSmiles("CC[NH3+]")
    >>> reionized = reionize_charges(mol)
    """
    reionizer = rdMolStandardize.Reionizer()
    return reionizer.reionize(mol)


def uncharge_molecule(mol: Chem.Mol) -> Chem.Mol:
    """
    Neutralize a molecule by removing counter-ions using RDKit's Uncharger.

    Parameters:
    - mol (Chem.Mol): The RDKit molecule object.

    Returns:
    - Chem.Mol: The neutralized molecule object.

    Example:
    >>> mol = Chem.MolFromSmiles("CC[NH3+].[Cl-]")
    >>> uncharged = uncharge_molecule(mol)
    """
    uncharger = rdMolStandardize.Uncharger()
    return uncharger.uncharge(mol)


def assign_stereochemistry(
    mol: Chem.Mol, cleanIt: bool = True, force: bool = True
) -> None:
    """
    Assign stereochemistry to a molecule using RDKit's AssignStereochemistry.

    Parameters:
    - mol (Chem.Mol): The RDKit molecule object.
    - cleanIt (bool): Clean the molecule; default is True.
    - force (bool): Force stereochemistry assignment; default is True.

    Returns:
    - None
    """
    Chem.AssignStereochemistry(mol, cleanIt=cleanIt, force=force)


def fragments_remover(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Remove small fragments from a molecule, keeping only the largest one.

    Parameters:
    - mol (Chem.Mol): The RDKit molecule object.

    Returns:
    - Chem.Mol or None: The molecule object with small fragments removed or None if an error occurs.

    Example:
    >>> mol = Chem.MolFromSmiles("CCC.CCCO")
    >>> largest = fragments_remover(mol)
    """
    try:
        largest_fragment = max(
            Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True),
            key=lambda m: m.GetNumAtoms(),
        )
        return largest_fragment
    except ValueError as e:
        logging.error(f"Failed to remove fragments: {e}")
        return None


def remove_hydrogens_and_sanitize(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Remove explicit hydrogens and sanitize a molecule.

    Parameters:
    - mol (Chem.Mol): The RDKit molecule object.

    Returns:
    - Chem.Mol or None: The molecule object with explicit hydrogens removed and sanitized, or None
    if sanitization fails.

    Example:
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> clean_mol = remove_hydrogens_and_sanitize(mol)
    """
    try:
        mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
        return mol
    except Exception as e:
        logging.error(f"Failed to sanitize molecule: {e}")
        return None
