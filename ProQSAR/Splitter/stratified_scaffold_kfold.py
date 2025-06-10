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
    Stratified Scaffold K-Fold iterator that ensures that each fold has a similar
    distribution of classes, while respecting scaffold groupings. This class is particularly useful
    in scenarios where data points belong to predefined scaffold groups (e.g., molecular scaffolds)
    and you want to maintain a balanced distribution of classes across folds.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.

    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.

    random_state : Optional[int], default=None
        Random seed for shuffling the data.

    scaff_based : Literal['median', 'mean'], default='median'
        Strategy for scaffold-based splitting: 'median' or 'mean'.

    Raises
    ------
    ValueError
        If `scaff_based` is not 'median' or 'mean'.

    Examples
    --------
    stratified_kfold = StratifiedScaffoldKFold(n_splits=5, scaff_based='mean')
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
                'scaff_based is expected to be "median" or "mean". The assigned value was {val}'.format(
                    val=repr(scaff_based)
                )
            )

    def _iter_test_indices(
        self, X: np.ndarray, y: Union[np.ndarray, List], groups: List[int]
    ) -> Generator[List[int], None, None]:
        # Implementation is based on this kaggle kernel:
        # https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
        # and is a subject to Apache 2.0 License. You may obtain a copy of the
        # License at http://www.apache.org/licenses/LICENSE-2.0
        # Changelist:
        # - Refactored function to a class following scikit-learn KFold
        #   interface.
        # - Added heuristic for assigning group to the least populated fold in
        #   cases when all other criteria are equal
        # - Swtch from using python ``Counter`` to ``np.unique`` to get class
        #   distribution
        # - Added scikit-learn checks for input: checking that target is binary
        #   or multiclass, checking passed random state, checking that number
        #   of splits is less than number of members in each class, checking
        #   that least populated class has more members than there are splits.
        """
        Generates test indices for each fold in a stratified scaffold-based.

        Parameters
        ----------
        X : np.ndarray
            The data to split.

        y : Union[np.ndarray, List]
            The target variable.

        groups : List[int]
            Scaffold group identifiers for the samples.

        Yields
        ------
        test_indices : Generator[List[int], None, None]
            Indices for the test set in each fold.
        """
        rng = check_random_state(self.random_state)
        y = np.asarray(y)

        y = column_or_1d(y)

        scaffolds = defaultdict(list)
        for idx, scaff_idx in enumerate(groups):
            scaffolds[scaff_idx].append(idx)
        scaffold_lists = list(scaffolds.values())

        n_bins = int(
            np.floor(
                len(scaffold_lists)
                / np.array([len(i) for i in scaffold_lists], dtype="i").mean()
            )
        )
        discretizer = KBinsDiscretizer(
            n_bins=n_bins, encode="ordinal", strategy="quantile"
        )

        scaff_act = []
        for scaff in scaffold_lists:
            scaff_act.append(y[scaff])

        if self.scaff_based == "median":
            scaff_act_val = [np.median(i) for i in scaff_act]
        elif self.scaff_based == "mean":
            scaff_act_val = [np.mean(i) for i in scaff_act]

        scaff_gr = discretizer.fit_transform(np.array(scaff_act_val).reshape(-1, 1))[
            :, 0
        ]

        # assert len(scaff_gr) == len(scaffold_lists)
        # if len(scaff_gr) != len(scaffold_lists):
        #     raise ValueError('scaff_gr and scaffold_lists have different lengths')

        bin_assign = np.full(len(X), -1, dtype="i")
        for i, bin in enumerate(scaff_gr):
            bin_assign[scaffold_lists[i]] = scaff_gr[i]

        # assert -1 not in bin_assign
        # if -1 in bin_assign:
        #     raise ValueError('bin_assign contains -1')

        y = bin_assign

        _, y_inv, y_cnt = np.unique(y, return_inverse=True, return_counts=True)
        if np.all(self.n_splits > y_cnt):
            raise ValueError(
                "n_splits=%d cannot be greater than the"
                " number of members in each class." % (self.n_splits)
            )
        # n_smallest_class = np.min(y_cnt)
        # if self.n_splits > n_smallest_class:
        #     warnings.warn(
        #         "The least populated class in y has only %d"
        #         " members, which is less than n_splits=%d."
        #         % (n_smallest_class, self.n_splits),
        #         UserWarning,
        #     )
        n_classes = len(y_cnt)

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

        # Stable sort to keep shuffled order for groups with the same
        # class distribution variance
        sorted_groups_idx = np.argsort(
            -np.std(y_counts_per_group, axis=1), kind="mergesort"
        )

        for group_idx in sorted_groups_idx:
            group_y_counts = y_counts_per_group[group_idx]
            best_fold = self._find_best_fold(
                y_counts_per_fold=y_counts_per_fold,
                y_cnt=y_cnt,
                group_y_counts=group_y_counts,
            )
            y_counts_per_fold[best_fold] += group_y_counts
            groups_per_fold[best_fold].add(group_idx)

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
        Finds the best fold to assign the current group based on minimizing class imbalance.

        Parameters
        ----------
        y_counts_per_fold : np.ndarray
            The current count of classes in each fold.

        y_cnt : np.ndarray
            The count of samples in each class.

        group_y_counts : np.ndarray
            The class distribution in the current group.

        Returns
        -------
        best_fold : int
            The index of the best fold.
        """
        best_fold = None
        min_eval = np.inf
        min_samples_in_fold = np.inf
        for i in range(self.n_splits):
            y_counts_per_fold[i] += group_y_counts
            # Summarise the distribution over classes in each proposed fold
            std_per_class = np.std(y_counts_per_fold / y_cnt.reshape(1, -1), axis=0)
            y_counts_per_fold[i] -= group_y_counts
            fold_eval = np.mean(std_per_class)
            samples_in_fold = np.sum(y_counts_per_fold[i])
            is_current_fold_better = (
                fold_eval < min_eval
                or np.isclose(fold_eval, min_eval)
                and samples_in_fold < min_samples_in_fold
            )
            if is_current_fold_better:
                min_eval = fold_eval
                min_samples_in_fold = samples_in_fold
                best_fold = i
        return best_fold
