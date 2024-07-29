from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers

def compute_force_field_energy(
    molecule: Chem.Mol, conformer_id: int, force_field_method: str = "UFF"
) -> float:
    """
    Computes the force field energy for a specified conformer of an RDKit molecule object.

    Parameters:
    - molecule (Chem.Mol): The molecule to calculate energy for.
    - conformer_id (int): The ID of the conformer whose energy needs to be computed.
    - force_field_method (str): The force field method to use, defaults to 'UFF'.

    Returns:
    - float: The energy of the specified conformer.

    Raises:
    - RuntimeError: If there is an issue initializing the force field.
    """
    if force_field_method == "MMFF":
        force_field_method = "MMFF94"

    if force_field_method.startswith("MMFF"):
        mmff_properties = rdForceFieldHelpers.MMFFGetMoleculeProperties(
            molecule, mmffVariant=force_field_method
        )
        if not mmff_properties:
            raise RuntimeError(
                f"Failed to initialize MMFF properties for {force_field_method}."
            )
        force_field = rdForceFieldHelpers.MMFFGetMoleculeForceField(
            molecule, mmff_properties, confId=conformer_id
        )
    else:
        force_field = rdForceFieldHelpers.UFFGetMoleculeForceField(
            molecule, confId=conformer_id
        )

    if not force_field:
        raise RuntimeError("Failed to initialize force field.")

    return force_field.CalcEnergy()

if __name__ == '__main__':
    molecule = Chem.MolFromSmiles('CCO')
    conformer_id = 0
    force_field_method = 'UFF'
    energy = compute_force_field_energy(molecule, conformer_id, force_field_method)
    print(energy)