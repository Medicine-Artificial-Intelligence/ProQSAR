import logging
from typing import Optional
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize


def normalize_molecule(mol: Chem.Mol) -> Chem.Mol:
    """
    Normalize a molecule using RDKit's Normalizer to correct functional groups and recharges.

    :param mol: The RDKit molecule object to be normalized.
    :type mol: Chem.Mol
    :return: The normalized RDKit molecule object.
    :rtype: Chem.Mol

    .. code-block:: python

        >>> mol = Chem.MolFromSmiles("CC(=O)O")
        >>> normalized = normalize_molecule(mol)
    """
    return rdMolStandardize.Normalize(mol)


def canonicalize_tautomer(mol: Chem.Mol) -> Chem.Mol:
    """
    Canonicalize the tautomer of a molecule using RDKit's TautomerCanonicalizer.

    :param mol: The RDKit molecule object.
    :type mol: Chem.Mol
    :return: The molecule object with canonicalized tautomer.
    :rtype: Chem.Mol

    .. code-block:: python

        >>> mol = Chem.MolFromSmiles("O=C1NC=CC1=O")
        >>> canonicalized = canonicalize_tautomer(mol)
    """
    return rdMolStandardize.CanonicalTautomer(mol)


def salts_remover(mol: Chem.Mol) -> Chem.Mol:
    """
    Remove salt fragments from a molecule using RDKit's SaltRemover.

    :param mol: The RDKit molecule object.
    :type mol: Chem.Mol
    :return: The molecule object with salts removed.
    :rtype: Chem.Mol

    .. code-block:: python

        >>> mol = Chem.MolFromSmiles("CCO.Na")
        >>> desalted = salts_remover(mol)
    """
    return SaltRemover().StripMol(mol)


def reionize_charges(mol: Chem.Mol) -> Chem.Mol:
    """
    Adjust a molecule to its most likely ionic state using RDKit's Reionizer.

    :param mol: The RDKit molecule object.
    :type mol: Chem.Mol
    :return: The molecule object with reionized charges.
    :rtype: Chem.Mol

    .. code-block:: python

        >>> mol = Chem.MolFromSmiles("CC[NH3+]")
        >>> reionized = reionize_charges(mol)
    """
    reionizer = rdMolStandardize.Reionizer()
    return reionizer.reionize(mol)


def uncharge_molecule(mol: Chem.Mol) -> Chem.Mol:
    """
    Neutralize a molecule by removing charges using RDKit's Uncharger.

    :param mol: The RDKit molecule object.
    :type mol: Chem.Mol
    :return: The neutralized molecule object.
    :rtype: Chem.Mol

    .. code-block:: python

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

    :param mol: The RDKit molecule object.
    :type mol: Chem.Mol
    :param cleanIt: Whether to clean the molecule before assignment. Default is True.
    :type cleanIt: bool, optional
    :param force: Whether to force stereochemistry assignment. Default is True.
    :type force: bool, optional
    :return: None
    :rtype: None
    """
    Chem.rdmolops.AssignStereochemistry(mol, cleanIt=cleanIt, force=force)


def fragments_remover(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Remove small fragments from a molecule, keeping only the largest one.

    :param mol: The RDKit molecule object.
    :type mol: Chem.Mol
    :return: The molecule object with only the largest fragment kept,
             or None if fragment removal fails.
    :rtype: Optional[Chem.Mol]

    .. code-block:: python

        >>> mol = Chem.MolFromSmiles("CCC.CCCO")
        >>> largest = fragments_remover(mol)
    """
    try:
        largest_fragment = max(
            Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True),
            key=lambda m: m.GetNumAtoms(),
        )
        return largest_fragment
    except ValueError as e:
        logging.error(f"Failed to remove fragments: {e}")
        return None


def remove_hydrogens_and_sanitize(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Remove explicit hydrogens and sanitize a molecule.

    :param mol: The RDKit molecule object.
    :type mol: Chem.Mol
    :return: The molecule object with explicit hydrogens removed and sanitized,
             or None if sanitization fails.
    :rtype: Optional[Chem.Mol]

    .. code-block:: python

        >>> mol = Chem.MolFromSmiles("CCO")
        >>> clean_mol = remove_hydrogens_and_sanitize(mol)
    """
    try:
        mol = Chem.rdmolops.RemoveHs(mol)
        Chem.rdmolops.SanitizeMol(mol)
        return mol
    except Exception as e:
        logging.error(f"Failed to sanitize molecule: {e}")
        return None
