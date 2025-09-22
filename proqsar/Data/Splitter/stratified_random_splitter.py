import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split


class StratifiedRandomSplitter:
    """
    Split data into training and testing sets with stratification.

    This splitter ensures that the distribution of the target (activity column)
    is preserved across the train and test sets. It is useful when activity labels
    are imbalanced and you want to maintain similar proportions in both subsets.

    :param activity_col: Name of the column representing the activity or target label.
    :type activity_col: str
    :param test_size: Proportion of the dataset to include in the test split.
                      Default is ``0.2``.
    :type test_size: float
    :param random_state: Random seed for reproducibility of the split.
                         Default is ``42``.
    :type random_state: int

    **Example**

    .. code-block:: python

        import pandas as pd
        from proqsar.Data.Splitter.stratified_random_splitter import StratifiedRandomSplitter

        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "SMILES": ["CCO", "CCC", "CCN", "CCCl", "CCBr", "CCF"],
            "activity": [0, 1, 0, 1, 0, 1]
        })

        splitter = StratifiedRandomSplitter(activity_col="activity", test_size=0.33, random_state=0)
        train, test = splitter.fit(df)
        print(train["activity"].value_counts(normalize=True))
        print(test["activity"].value_counts(normalize=True))
        # Both train and test maintain class distribution
    """

    def __init__(
        self,
        activity_col: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.activity_col = activity_col

    def fit(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into training and testing sets with stratification.

        :param data: Input dataset containing at least the activity column.
        :type data: pd.DataFrame

        :return: A tuple ``(train_df, test_df)`` where
                 - ``train_df`` is the stratified training dataset,
                 - ``test_df`` is the stratified testing dataset.
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """
        data_train, data_test = train_test_split(
            data,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=data[self.activity_col],
        )
        return data_train, data_test
