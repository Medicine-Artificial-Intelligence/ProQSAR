import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from ProQSAR.Conformer import ConformerGenerator


def generate_conformers_parallel(molecule):
    """
    Generates conformers for a single molecule using ConformerGenerator.

    Parameters:
    - molecule: RDKit molecule object.

    Returns:
    - RDKit molecule object with generated conformers.

    Example:
    >>> from AtQSAR.AtConformer import ConformerGenerator
    >>> rdkit_mol = clean.loc[0, 'mol']
    >>> conf_gen = ConformerGenerator()
    >>> mol_with_conformers = generate_conformers_parallel(rdkit_mol)
    """
    # Create a ConformerGenerator instance
    conf_gen = ConformerGenerator()
    mol_with_conformers = conf_gen(molecule)
    return mol_with_conformers


def generate_conformers_parallel_dataframe(data, mol_column="mol", num_threads=8):
    """
    Generates conformers in parallel for molecules in a pandas DataFrame.

    Parameters:
    - data: pandas DataFrame with a column containing RDKit molecules.
    - mol_column: Name of the column in the DataFrame containing RDKit molecules. Default is 'mol'.
    - num_threads: Number of threads for parallel processing. Default is 8.

    Returns:
    - pandas DataFrame with an additional column containing RDKit molecules with generated conformers.

    Example:
    >>> import pandas as pd
    >>> from AtQSAR.AtConformer import ConformerGenerator
    >>> # Create a pandas DataFrame (replace this with your actual DataFrame)
    >>> data = pd.DataFrame({'mol': [rdkit_mol1, rdkit_mol2, rdkit_mol3, ...]})
    >>> # Generate conformers in parallel for the DataFrame
    >>> result_data = generate_conformers_parallel_dataframe(data)
    """
    # Create a copy of the input DataFrame
    result_data = data.copy()

    # Create a ThreadPoolExecutor to parallelize the function
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Apply the function to the DataFrame using concurrent.futures
        result_data["mol_conf"] = list(
            executor.map(generate_conformers_parallel, data[mol_column])
        )

    return result_data
