from ProQSAR.Partition.random_partition import RandomPartition
from ProQSAR.Partition.stratified_random_partition import StratifiedRandomPartition
from ProQSAR.Partition.scaffold_partition import ScaffoldPartition
from ProQSAR.Partition.stratified_scaffold_partition import StratifiedScaffoldPartition
from typing import Tuple
import pandas as pd
import logging


class Partition:
    """
    A class to handle various data partitioning strategies for training and testing sets.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        activity_col: str,
        smiles_col: str,
        option: str,
        test_size: float = 0.2,
        n_splits: int = 5,
        random_state: int = 42,
    ):
        """
        Constructs all the necessary attributes for the Partition object.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing the features and labels.
        activity_col : str
            The name of the column representing the activity or target label.
        smiles_col : str
            The name of the column containing SMILES strings for molecular data.
        option : str
            The partitioning method, either "random", "stratified_random", "scaffold", or "stratified_scaffold".
        test_size : float, optional
            The proportion of the dataset to include in the test split (default is 0.2).
        random_state : int, optional
            The random seed used by the random number generator (default is 42).
        n_splits : int, optional
            Number of splits/folds to create for stratified partitions (default is 5).
        """
        self.option = option
        self.data = data
        self.test_size = test_size
        self.random_state = random_state
        self.activity_col = activity_col
        self.smiles_col = smiles_col
        self.n_splits = n_splits

    def fit(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies the selected partitioning strategy based on the 'option' attribute
        and returns the training and testing sets.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            The training and testing sets as pandas DataFrames.
        """
        try:
            if self.option == "random":
                partition = RandomPartition(
                    self.data,
                    self.activity_col,
                    self.smiles_col,
                    test_size=self.test_size,
                    random_state=self.random_state,
                )
            elif self.option == "stratified_random":
                partition = StratifiedRandomPartition(
                    self.data,
                    self.activity_col,
                    self.smiles_col,
                    test_size=self.test_size,
                    random_state=self.random_state,
                )
            elif self.option == "scaffold":
                partition = ScaffoldPartition(
                    self.data,
                    self.activity_col,
                    self.smiles_col,
                    test_size=self.test_size,
                    random_state=self.random_state,
                )
            elif self.option == "stratified_scaffold":
                partition = StratifiedScaffoldPartition(
                    self.data,
                    self.activity_col,
                    self.smiles_col,
                    n_splits=self.n_splits,
                    random_state=self.random_state,
                )
            else:
                raise ValueError(
                    f"Invalid partition option: {self.option}."
                    "Choose from 'random', 'stratified_random', 'scaffold', or 'stratified_scaffold'."
                )

            data_train, data_test = partition.fit()

            logging.info(
                f"Data successfully partitioned using the '{self.option}' method."
            )

            return data_train, data_test

        except ValueError as e:
            logging.error(f"Error: {e}")
            raise

        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise
