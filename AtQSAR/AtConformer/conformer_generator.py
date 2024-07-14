from rdkit import Chem
from dataclasses import dataclass
from typing import Optional, List, Union

from AtQSAR.AtConformer import conformer_function


@dataclass
class ConformerGenerator:
    """
    A class used to generate molecular conformers.

    ...

    Attributes
    ----------
    num_conformer_candidates : str or int
        Number of conformers to generate (default is 'auto')
    embedding_method : str
        The embedding method to use (default is 'ETKDGv2')
    force_field_method : str
        The force field method to use (default is 'MMFF94')
    max_iter : str or int
        Maximum number of iterations for generating a conformer (default is 'auto')
    keep_hydrogens : bool
        Whether to keep the hydrogens of the selected conformer (default is False)

    Methods
    -------
    __call__(molecule)
        Generates a conformer for the given molecule.
    available_embedding_methods()
        Returns a list of available embedding methods.
    available_force_field_methods()
        Returns a list of available force field methods.
    """

    num_conformer_candidates: Optional[Union[str, int]] = "auto"
    embedding_method: str = "ETKDGv2"
    force_field_method: Optional[str] = "MMFF94"
    max_iter: Optional[Union[str, int]] = "auto"
    keep_hydrogens: bool = False

    def __call__(self, molecule: Union[str, Chem.Mol]) -> Chem.Mol:
        """
        Generates a conformer for the given molecule.

        Parameters:
            molecule (str or Chem.Mol): The molecule for which to generate a conformer.

        Returns:
            Chem.Mol: The conformer of the molecule.
        """

        if not isinstance(molecule, Chem.Mol):
            return None

        molecule = Chem.AddHs(molecule)

        molecule = conformer_function.mol_embed(
            molecule, self.num_conformer_candidates, self.embedding_method
        )

        molecule = conformer_function.force_field_minimization(
            molecule, self.force_field_method, self.max_iter
        )

        molecule = conformer_function.get_lowest_energy_conformer(
            molecule, self.force_field_method
        )

        if not self.keep_hydrogens:
            molecule = Chem.RemoveHs(molecule)

        return molecule

    @property
    def available_embedding_methods(self) -> List[str]:
        """
        Returns a list of available embedding methods.

        Returns:
            List[str]: The list of available embedding methods.
        """
        return list(conformer_function._embedding_method.keys())

    @property
    def available_force_field_methods(self) -> List[str]:
        """
        Returns a list of available force field methods.

        Returns:
            List[str]: The list of available force field methods.
        """
        return conformer_function._available_ff_methods
