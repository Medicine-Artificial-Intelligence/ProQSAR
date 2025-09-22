import numpy as np
from sklearn.model_selection._split import (
    _BaseKFold,
    GroupsConsumerMixin,
)
from sklearn.utils import check_random_state
from sklearn.utils.validation import column_or_1d
from typing import List, Generator, Union, Literal, Optional
from sklearn.preprocessing import KBinsDiscretizer
from collections import defaultdict


class StratifiedScaffoldKFold(GroupsConsumerMixin, _BaseKFold):
    """
    Stratified K-Fold cross-validator with scaffold grouping.

    Ensures that each fold maintains a balanced distribution of classes while
    respecting scaffold groupings (e.g., Bemis–Murcko scaffolds).
    Useful for molecular datasets where scaffold leakage should be avoided.

    :param n_splits: Number of folds. Must be at least 2. Default is ``5``.
    :type n_splits: int
    :param shuffle: Whether to shuffle the data before splitting. Default is ``False``.
    :type shuffle: bool
    :param random_state: Random seed for reproducibility when ``shuffle=True``.
                         Default is ``None``.
    :type random_state: Optional[int]
    :param scaff_based: Strategy for scaffold aggregation when computing activity values.
                        Must be either ``"median"`` or ``"mean"``. Default is ``"median"``.
    :type scaff_based: Literal["median", "mean"]

    :raises ValueError: If ``scaff_based`` is not ``"median"`` or ``"mean"``.

    **Example**

    .. code-block:: python

        import numpy as np
        from proqsar.Data.Splitter.stratified_scaffold_kfold import StratifiedScaffoldKFold

        # Example data
        X = np.arange(20).reshape(-1, 1)
        y = np.array([0, 1] * 10)  # binary labels
        groups = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4,
                  5, 5, 6, 6, 7, 7, 8, 8, 9, 9]  # scaffold groups

        splitter = StratifiedScaffoldKFold(n_splits=5, scaff_based="median", shuffle=True, random_state=42)

        for fold_idx, test_idx in enumerate(splitter.split(X, y, groups)):
            print(f"Fold {fold_idx}: test indices {test_idx}")
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = False,
        random_state: Optional[int] = None,
        scaff_based: Literal["median", "mean"] = "median",
    ) -> None:
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.scaff_based = scaff_based

        if scaff_based not in ["median", "mean"]:
            raise ValueError(
                f'scaff_based is expected to be "median" or "mean". Got {repr(scaff_based)}'
            )

    def _iter_test_indices(
        self, X: np.ndarray, y: Union[np.ndarray, List], groups: List[int]
    ) -> Generator[List[int], None, None]:
        """
        Yield test indices for each fold.

        The splitting procedure ensures that:
          - class distributions are approximately preserved across folds,
          - molecules from the same scaffold group are kept together.

        :param X: Feature matrix of shape (n_samples, n_features).
        :type X: np.ndarray
        :param y: Target variable of shape (n_samples,).
        :type y: Union[np.ndarray, List]
        :param groups: Scaffold group identifiers of shape (n_samples,).
        :type groups: List[int]

        :yields: Test indices for the current fold.
        :rtype: Generator[List[int], None, None]

        :raises ValueError: If ``n_splits`` is greater than the number of members in any class.
        """
        rng = check_random_state(self.random_state)
        y = np.asarray(y)
        y = column_or_1d(y)

        # group molecules by scaffold
        scaffolds = defaultdict(list)
        for idx, scaff_idx in enumerate(groups):
            scaffolds[scaff_idx].append(idx)
        scaffold_lists = list(scaffolds.values())

        # discretize scaffold-level activity values
        n_bins = int(
            np.floor(
                len(scaffold_lists)
                / np.array([len(i) for i in scaffold_lists], dtype="i").mean()
            )
        )
        discretizer = KBinsDiscretizer(
            n_bins=n_bins, encode="ordinal", strategy="quantile"
        )

        scaff_act = [y[scaff] for scaff in scaffold_lists]
        if self.scaff_based == "median":
            scaff_act_val = [np.median(i) for i in scaff_act]
        else:  # "mean"
            scaff_act_val = [np.mean(i) for i in scaff_act]

        scaff_gr = discretizer.fit_transform(np.array(scaff_act_val).reshape(-1, 1))[
            :, 0
        ]

        # assign scaffold bins back to molecules
        bin_assign = np.full(len(X), -1, dtype="i")
        for i, _ in enumerate(scaff_gr):
            bin_assign[scaffold_lists[i]] = scaff_gr[i]
        y = bin_assign

        # validate split feasibility
        _, y_inv, y_cnt = np.unique(y, return_inverse=True, return_counts=True)
        if np.all(self.n_splits > y_cnt):
            raise ValueError(
                f"n_splits={self.n_splits} cannot be greater than "
                "the number of members in each class."
            )
        n_classes = len(y_cnt)

        # prepare per-group class distributions
        _, groups_inv, groups_cnt = np.unique(
            groups, return_inverse=True, return_counts=True
        )
        y_counts_per_group = np.zeros((len(groups_cnt), n_classes))
        for class_idx, group_idx in zip(y_inv, groups_inv):
            y_counts_per_group[group_idx, class_idx] += 1

        y_counts_per_fold = np.zeros((self.n_splits, n_classes))
        groups_per_fold = defaultdict(set)

        if self.shuffle:
            rng.shuffle(y_counts_per_group)

        # sort groups by class distribution variance
        sorted_groups_idx = np.argsort(
            -np.std(y_counts_per_group, axis=1), kind="mergesort"
        )

        # assign groups to folds
        for group_idx in sorted_groups_idx:
            group_y_counts = y_counts_per_group[group_idx]
            best_fold = self._find_best_fold(
                y_counts_per_fold=y_counts_per_fold,
                y_cnt=y_cnt,
                group_y_counts=group_y_counts,
            )
            y_counts_per_fold[best_fold] += group_y_counts
            groups_per_fold[best_fold].add(group_idx)

        # yield test indices for each fold
        for i in range(self.n_splits):
            test_indices = [
                idx
                for idx, group_idx in enumerate(groups_inv)
                if group_idx in groups_per_fold[i]
            ]
            yield test_indices

    def _find_best_fold(
        self,
        y_counts_per_fold: np.ndarray,
        y_cnt: np.ndarray,
        group_y_counts: np.ndarray,
    ) -> int:
        """
        Find the best fold to assign a group, minimizing class imbalance.

        :param y_counts_per_fold: Current per-fold class counts of shape (n_splits, n_classes).
        :type y_counts_per_fold: np.ndarray
        :param y_cnt: Total number of samples per class.
        :type y_cnt: np.ndarray
        :param group_y_counts: Class distribution of the current group.
        :type group_y_counts: np.ndarray

        :return: Index of the best fold for assignment.
        :rtype: int
        """
        best_fold = None
        min_eval = np.inf
        min_samples_in_fold = np.inf
        for i in range(self.n_splits):
            y_counts_per_fold[i] += group_y_counts
            std_per_class = np.std(y_counts_per_fold / y_cnt.reshape(1, -1), axis=0)
            y_counts_per_fold[i] -= group_y_counts
            fold_eval = np.mean(std_per_class)
            samples_in_fold = np.sum(y_counts_per_fold[i])
            is_current_fold_better = fold_eval < min_eval or (
                np.isclose(fold_eval, min_eval)
                and samples_in_fold < min_samples_in_fold
            )
            if is_current_fold_better:
                min_eval = fold_eval
                min_samples_in_fold = samples_in_fold
                best_fold = i
        return best_fold
