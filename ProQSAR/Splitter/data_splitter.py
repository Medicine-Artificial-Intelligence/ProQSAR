from ProQSAR.Splitter.random_splitter import RandomSplitter
from ProQSAR.Splitter.stratified_random_splitter import StratifiedRandomSplitter
from ProQSAR.Splitter.scaffold_splitter import ScaffoldSplitter
from ProQSAR.Splitter.stratified_scaffold_splitter import StratifiedScaffoldSplitter
from sklearn.base import BaseEstimator
from typing import Tuple, Optional
import os
import pandas as pd
import logging


class Splitter(BaseEstimator):
    """
    A class to handle various data partitioning strategies for training and testing sets.
    """

    def __init__(
        self,
        activity_col: str = "activity",
        smiles_col: str = "SMILES",
        mol_col: str = "mol",
        option: str = "random",
        test_size: float = 0.2,
        n_splits: int = 5,
        random_state: int = 42,
        save_dir: Optional[str] = "Project/Splitter",
        data_name: Optional[str] = None,
    ):
        """
        Constructs all the necessary attributes for the Splitter object.

        Parameters:
        -----------
        activity_col : str
            The name of the column representing the activity or target label.
        smiles_col : str
            The name of the column containing SMILES strings for molecular data.
        option : str
            The splitting method, either "random", "stratified_random", "scaffold", or "stratified_scaffold".
        test_size : float, optional
            The proportion of the dataset to include in the test split (default is 0.2).
        n_splits : int, optional
            Number of splits/folds to create for stratified partitions (default is 5).
        random_state : int, optional
            The random seed used by the random number generator (default is 42).
        """
        self.option = option
        self.test_size = test_size
        self.random_state = random_state
        self.activity_col = activity_col
        self.smiles_col = smiles_col
        self.mol_col = mol_col
        self.n_splits = n_splits
        self.save_dir = save_dir
        self.data_name = data_name

    def fit(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies the selected splitting strategy based on the 'option' attribute
        and returns the training and testing sets.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing the features and labels.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            The training and testing sets as pandas DataFrames.
        """
        try:
            if self.option == "random":
                splitter = RandomSplitter(
                    self.activity_col,
                    test_size=self.test_size,
                    random_state=self.random_state,
                )
            elif self.option == "stratified_random":
                splitter = StratifiedRandomSplitter(
                    self.activity_col,
                    test_size=self.test_size,
                    random_state=self.random_state,
                )
            elif self.option == "scaffold":
                splitter = ScaffoldSplitter(
                    self.activity_col,
                    self.smiles_col,
                    self.mol_col,
                    test_size=self.test_size,
                    random_state=self.random_state,
                )
            elif self.option == "stratified_scaffold":
                splitter = StratifiedScaffoldSplitter(
                    self.activity_col,
                    self.smiles_col,
                    self.mol_col,
                    n_splits=self.n_splits,
                    random_state=self.random_state,
                )
            else:
                raise ValueError(
                    f"Invalid splitting option: {self.option}."
                    "Choose from 'random', 'stratified_random', 'scaffold', or 'stratified_scaffold'."
                )

            data_train, data_test = splitter.fit(data)
            data_train = data_train.reset_index(drop=True).drop(
                columns=[self.smiles_col, self.mol_col], errors="ignore"
            )
            data_test = data_test.reset_index(drop=True).drop(
                columns=[self.smiles_col, self.mol_col], errors="ignore"
            )

            logging.info(
                f"Splitter: Data successfully partitioned using the '{self.option}' method."
            )

            if self.save_dir:
                os.makedirs(self.save_dir, exist_ok=True)
                name_suffix = f"_{self.data_name}" if self.data_name else ""
                data_train.to_csv(
                    f"{self.save_dir}/train{name_suffix}.csv", index=False
                )
                data_test.to_csv(f"{self.save_dir}/test{name_suffix}.csv", index=False)

            return data_train, data_test

        except ValueError as e:
            logging.error(f"Error: {e}")
            raise

        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise
