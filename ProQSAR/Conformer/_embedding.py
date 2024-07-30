import logging
from rdkit import Chem
from rdkit.Chem.rdDistGeom import ETDG, ETKDG, ETKDGv2, ETKDGv3, srETKDGv3, KDG
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


class Embeddings:
    """
    A class for handling molecular embeddings and conformer generation using RDKit.
    """

    def __init__(self) -> None:
        """
        Initializes the Embeddings class.
        """
        pass

    @staticmethod
    def _get_embedding_method(force_field_method: str = "ETKDGv3") -> Any:
        """
        Retrieves the appropriate RDKit embedding method object based on the specified
        force field method.

        Parameters:
        - force_field_method (str): The force field method identifier.
        Defaults to 'ETKDGv3'.
        Supported values are 'ETDG', 'ETKDG', 'ETKDGv2', 'ETKDGv3', 'srETKDGv3',
        and 'KDG'.

        Returns:
        - Any: An instance of the specified RDKit embedding method class.

        Raises:
        - KeyError: If the specified force field method is not supported.
        """
        _embedding_methods = {
            "ETDG": ETDG(),
            "ETKDG": ETKDG(),
            "ETKDGv2": ETKDGv2(),
            "ETKDGv3": ETKDGv3(),
            "srETKDGv3": srETKDGv3(),
            "KDG": KDG(),
        }

        if force_field_method not in _embedding_methods:
            raise KeyError(
                f"Unsupported force field method '{force_field_method}'. "
                f"Supported methods are: {list(_embedding_methods.keys())}."
            )

        return _embedding_methods[force_field_method]

    @staticmethod
    def _get_num_conformers_from_molecule_size(
        molecule: Chem.Mol,
        max_num_conformers: int = 10,
        min_num_conformers: int = 2,
        decr_num_conformers: float = 0.04,
    ) -> int:
        """
        Calculates the appropriate number of conformers to generate based on
        the molecule size.

        Parameters:
        - molecule (Chem.Mol): The RDKit molecule object for which to determine
        the number of conformers.
        - max_num_conformers (int): Maximum number of conformers to generate.
        Defaults to 10.
        - min_num_conformers (int): Minimum number of conformers to generate.
        Defaults to 2.
        - decr_num_conformers (float): Decrement factor for the number of conformers
        based on molecule size. Defaults to 0.04.

        Returns:
        - int: The calculated number of conformers to generate.
        """
        num_atoms = molecule.GetNumAtoms()
        suggested_num = max_num_conformers - int(num_atoms * decr_num_conformers)
        return max(min_num_conformers, min(max_num_conformers, suggested_num))

    @staticmethod
    def mol_embed(
        molecule: Chem.Mol,
        num_conformers: Optional[Union[int, str]] = "auto",
        embedding_method: str = "ETKDGv3",
        num_threads: int = 1,
        random_coords_threshold: int = 100,
        random_seed: int = 42,
    ) -> Chem.Mol:
        """
        Embeds conformers for an RDKit molecule object using specified embedding
        parameters.

        Parameters:
        - molecule (Chem.Mol): The RDKit molecule object to be embedded.
        - num_conformers (Optional[Union[int, str]]): The number of conformers to
        generate. If 'auto', the number is determined based on the molecule's size.
        Defaults to 'auto'.
        - embedding_method (str): The embedding method to use, corresponding to different
        RDKit embedding strategies. Defaults to 'ETKDGv3'.
        - num_threads (int): Number of threads to use for conformer generation.
        Defaults to 1.
        - random_coords_threshold (int): Atom count threshold above which random
        coordinates are used to initialize embeddings. Defaults to 100.
        - random_seed (int): Seed for the random number generator to ensure
        reproducibility. Defaults to 42.

        Returns:
        - Chem.Mol: The RDKit molecule with embedded conformers,
        or the original molecule if embedding fails.

        Raises:
        - ValueError: If an invalid number of conformers is provided.

        """
        # Validate num_conformers input
        if isinstance(num_conformers, str) and num_conformers != "auto":
            raise ValueError("num_conformers must be an integer or 'auto'.")

        # Copy the molecule to keep the original untouched
        mol_copy = Chem.Mol(molecule)
        mol_copy = Chem.AddHs(mol_copy)

        # Determine the number of conformers
        if num_conformers == "auto":
            num_conformers = Embeddings._get_num_conformers_from_molecule_size(mol_copy)

        # Retrieve embedding parameters
        params = Embeddings._get_embedding_method(embedding_method)
        params.numThreads = num_threads
        params.randomSeed = random_seed
        params.useRandomCoords = mol_copy.GetNumAtoms() > random_coords_threshold

        # Attempt to embed multiple conformers
        success = Chem.rdDistGeom.EmbedMultipleConfs(
            mol_copy, numConfs=num_conformers, params=params
        )

        if not success:
            logger.warning(
                "Failed to embed any conformers; attempting to compute 2D coordinates."
            )
            Chem.rdDepictor.Compute2DCoords(mol_copy)

        return mol_copy
