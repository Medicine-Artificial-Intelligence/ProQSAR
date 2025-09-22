import pandas as pd
from typing import Tuple, Literal
from .stratified_scaffold_kfold import StratifiedScaffoldKFold
from .scaffold_utils import get_scaffold_groups


class StratifiedScaffoldSplitter:
    """
    Stratified scaffold-based data splitter.

    This splitter partitions a dataset into training and testing sets while:
      - grouping molecules by Bemis–Murcko scaffolds,
      - preserving the distribution of the activity (target) variable across splits.

    It uses :class:`~proqsar.Data.Splitter.stratified_scaffold_kfold.StratifiedScaffoldKFold`
    under the hood, ensuring scaffold integrity and stratification.

    :param activity_col: Name of the column representing the activity or target label.
    :type activity_col: str
    :param smiles_col: Name of the column containing SMILES strings for molecular data.
    :type smiles_col: str
    :param mol_col: Name of the column containing RDKit Mol objects (if available).
                    Default is ``"mol"``.
    :type mol_col: str
    :param random_state: Random seed for reproducibility of shuffling. Default is ``42``.
    :type random_state: int
    :param n_splits: Number of splits/folds to create. Default is ``5``.
    :type n_splits: int
    :param scaff_based: Strategy to compute scaffold-level activity values,
                        either ``"median"`` or ``"mean"``. Default is ``"median"``.
    :type scaff_based: Literal["median", "mean"]
    :param shuffle: Whether to shuffle data before splitting. Default is ``True``.
    :type shuffle: bool

    **Example**

    .. code-block:: python

        import pandas as pd
        from proqsar.Data.Splitter.stratified_scaffold_splitter import StratifiedScaffoldSplitter

        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "SMILES": ["CCO", "CCC", "CCN", "CCCl", "CCBr", "CCF"],
            "activity": [0, 1, 0, 1, 0, 1]
        })

        splitter = StratifiedScaffoldSplitter(
            activity_col="activity",
            smiles_col="SMILES",
            n_splits=3,
            scaff_based="median",
            random_state=0,
            shuffle=True
        )

        train, test = splitter.fit(df)
        print(train.shape, test.shape)
        # e.g., (4, 3) (2, 3)
    """

    def __init__(
        self,
        activity_col: str,
        smiles_col: str,
        mol_col: str = "mol",
        random_state: int = 42,
        n_splits: int = 5,
        scaff_based: Literal["median", "mean"] = "median",
        shuffle: bool = True,
    ):
        self.random_state = random_state
        self.activity_col = activity_col
        self.smiles_col = smiles_col
        self.mol_col = mol_col
        self.n_splits = n_splits
        self.scaff_based = scaff_based
        self.shuffle = shuffle

    def fit(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into training and testing sets using stratified scaffolds.

        :param data: Input dataset containing SMILES, activity, and optional Mol columns.
        :type data: pd.DataFrame

        :return: A tuple ``(train_df, test_df)`` where
                 - ``train_df`` is the scaffold-stratified training set,
                 - ``test_df`` is the scaffold-stratified testing set.
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """
        cv = StratifiedScaffoldKFold(
            n_splits=self.n_splits,
            random_state=self.random_state,
            shuffle=self.shuffle,
            scaff_based=self.scaff_based,
        )
        groups = get_scaffold_groups(data, self.smiles_col, self.mol_col)

        y = data[self.activity_col].to_numpy(dtype=float)
        X = data.drop(
            [self.activity_col, self.smiles_col, self.smiles_col],
            axis=1,
            errors="ignore",
        ).to_numpy()

        train_idx, test_idx = next(cv.split(X, y, groups))
        data_train = data.iloc[train_idx]
        data_test = data.iloc[test_idx]

        return data_train, data_test
