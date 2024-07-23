from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG, display


def draw_mol_with_SVG(mol, molSize=(450, 150), drawOptions=None):
    """
    Visualize an RDKit molecule using SVG.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit Mol object to be visualized.
    molSize : tuple, optional
        Size of the image (width, height), default is (450, 150).
    drawOptions : Draw.MolDrawOptions, optional
        Custom drawing options to adjust how molecules are drawn.

    Examples
    --------
    >>> mol = Chem.MolFromSmiles('Cc1ccccc1')
    >>> draw_mol_with_SVG(mol, molSize=(500, 200))
    """
    try:
        # Prepare the molecule for drawing
        mc = Chem.Mol(mol.ToBinary())
        if not mc.GetNumConformers():
            Chem.rdDepictor.Compute2DCoords(mc)

        # Create a drawer for SVG output and set options if provided
        drawer = Draw.MolDraw2DSVG(molSize[0], molSize[1])
        if drawOptions:
            drawer.SetDrawOptions(drawOptions)

        # Draw the molecule
        drawer.DrawMolecule(mc)
        drawer.FinishDrawing()

        # Get the SVG text and clean the header
        svg = drawer.GetDrawingText().replace("svg:", "")
        display(SVG(svg))
    except Exception as e:
        print(f"Error drawing molecule: {e}")


def visualize_conformers(
    molecule: Chem.Mol,
    force_field_method: str = "MMFF94",
    subImgSize: tuple = (200, 200),
):
    """
    Visualize the conformers of a molecule.

    Parameters:
    molecule (Chem.Mol): The molecule whose conformers to visualize.
    force_field_method (str, optional): The force field method to use. Default is 'MMFF94'.
    subImgSize (tuple, optional): The size of the sub-images. Default is (200,200).

    Returns:
    Chem.Draw.IPythonConsole.display.Image: The image of the conformers.
    """
    conformers = []  # List to store the conformers of the molecule
    energies = []  # List to store the energies of the conformers

    if force_field_method == "MMFF":  # If the force field method is 'MMFF'
        force_field_method = "MMFF94"  # Change it to 'MMFF94'

    for conformer in molecule.GetConformers():  # For each conformer in the molecule
        new_molecule = Chem.Mol(molecule)  # Create a copy of the molecule
        new_molecule.RemoveAllConformers()  # Remove all conformers from the new molecule
        new_molecule.AddConformer(
            conformer, assignId=True
        )  # Add the current conformer to the new molecule
        conformers.append(
            new_molecule
        )  # Append the new molecule to the list of conformers
        if force_field_method.startswith(
            "MMFF"
        ):  # If the force field method starts with 'MMFF'
            mmff_properties = Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties(
                mol=molecule, mmffVariant=force_field_method
            )  # Get the MMFF properties of the molecule
            ff = Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField(
                molecule, mmff_properties, confId=0
            )  # Get the MMFF force field of the molecule
        else:  # If the force field method does not start with 'MMFF'
            ff = Chem.rdForceFieldHelpers.UFFGetMoleculeForceField(
                new_molecule, confId=0
            )  # Get the UFF force field of the new molecule
        energies.append(
            ff.CalcEnergy()
        )  # Calculate the energy of the conformer and append it to the list of energies
    legends = [
        f"{force_field_method} energy = {energy:.2f}" for energy in energies
    ]  # Create the legends for the conformers
    return IPythonConsole.ShowMols(
        conformers, legends=legends, subImgSize=subImgSize
    )  # Show the conformers
