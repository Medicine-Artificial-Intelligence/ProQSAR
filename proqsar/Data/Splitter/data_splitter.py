from .random_splitter import RandomSplitter
from .stratified_random_splitter import StratifiedRandomSplitter
from .random_scaffold_splitter import RandomScaffoldSplitter
from .scaffold_splitter import ScaffoldSplitter
from .stratified_scaffold_splitter import StratifiedScaffoldSplitter
from sklearn.base import BaseEstimator
from typing import Tuple, Optional
import os
import pandas as pd
import logging


class Splitter(BaseEstimator):
    """
    Unified interface for dataset partitioning into train/test subsets.

    This class provides a common interface to perform different dataset splitting
    strategies, such as random splitting, stratified random splitting,
    scaffold-based splitting, and scaffold-based stratified splitting.
    It also handles optional saving of train/test splits to disk.

    :param activity_col: Name of the column representing the activity or target label.
                         Default is ``"activity"``.
    :type activity_col: str
    :param smiles_col: Name of the column containing SMILES strings for molecular data.
                       Default is ``"SMILES"``.
    :type smiles_col: str
    :param mol_col: Name of the column containing RDKit Mol objects (if available).
                    Default is ``"mol"``.
    :type mol_col: str
    :param option: Splitting method, one of
                   ``"random"``, ``"stratified_random"``, ``"scaffold"``,
                   ``"random_scaffold"``, or ``"stratified_scaffold"``.
                   Default is ``"random"``.
    :type option: str
    :param test_size: Proportion of the dataset to include in the test split.
                      Default is ``0.2``.
    :type test_size: float
    :param n_splits: Number of folds for stratified splitting
                     (used only when option is ``"stratified_scaffold"``).
                     Default is ``5``.
    :type n_splits: int
    :param random_state: Random seed used by the random number generator.
                         Default is ``42``.
    :type random_state: int
    :param save_dir: Directory where train/test CSV files are saved.
                     If ``None``, no files are written.
                     Default is ``"Project/Splitter"``.
    :type save_dir: Optional[str]
    :param data_name: Optional name suffix for saved train/test files.
                      Default is ``None``.
    :type data_name: Optional[str]
    :param deactivate: If True, disables splitting and returns the full dataset as training set.
                       Default is ``False``.
    :type deactivate: bool

    **Example**

    .. code-block:: python

        import pandas as pd
        from proqsar.Data.Splitter.splitter import Splitter

        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "SMILES": ["CCO", "CCC", "CCN", "CCCl", "CCBr"],
            "activity": [1.2, 3.4, 2.1, 0.5, 4.7]
        })

        splitter = Splitter(option="random", test_size=0.4, random_state=0)
        train, test = splitter.fit(df)
        print(train.shape, test.shape)  # (3, 2) (2, 2)
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
        deactivate: bool = False,
    ):
        self.option = option
        self.test_size = test_size
        self.random_state = random_state
        self.activity_col = activity_col
        self.smiles_col = smiles_col
        self.mol_col = mol_col
        self.n_splits = n_splits
        self.save_dir = save_dir
        self.data_name = data_name
        self.deactivate = deactivate

    def fit(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Split the dataset into training and testing sets.

        :param data: Input dataset containing at least the activity column and SMILES column.
        :type data: pd.DataFrame

        :return: A tuple ``(train_df, test_df)`` where
                 - ``train_df`` is the training dataset (with SMILES and Mol columns dropped).
                 - ``test_df`` is the testing dataset (with SMILES and Mol columns dropped),
                   or ``None`` if ``deactivate=True``.
        :rtype: Tuple[pd.DataFrame, Optional[pd.DataFrame]]

        :raises ValueError: If an invalid splitting option is provided.
        :raises Exception: If an unexpected error occurs during splitting.
        """
        if self.deactivate:
            logging.info(
                "Splitter is deactivated. Returning original dataset as training set."
            )
            data_train = data.reset_index(drop=True).drop(
                columns=[self.smiles_col, self.mol_col], errors="ignore"
            )
            return data_train, None

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
                )
            elif self.option == "random_scaffold":
                splitter = RandomScaffoldSplitter(
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
                    f"Invalid splitting option: {self.option}. "
                    "Choose from 'random', 'stratified_random', "
                    "'scaffold', 'random_scaffold', 'stratified_scaffold'."
                )

            data_train, data_test = splitter.fit(data)
            data_train = data_train.reset_index(drop=True).drop(
                columns=[self.smiles_col, self.mol_col], errors="ignore"
            )
            data_test = data_test.reset_index(drop=True).drop(
                columns=[self.smiles_col, self.mol_col], errors="ignore"
            )

            logging.info(
                f"Splitter: Data successfully partitioned using '{self.option}' method."
            )

            if self.save_dir:
                os.makedirs(self.save_dir, exist_ok=True)
                suffix = f"_{self.data_name}" if self.data_name else ""
                data_train.to_csv(f"{self.save_dir}/train{suffix}.csv", index=False)
                data_test.to_csv(f"{self.save_dir}/test{suffix}.csv", index=False)

            return data_train, data_test

        except ValueError as e:
            logging.error(f"Error: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise
