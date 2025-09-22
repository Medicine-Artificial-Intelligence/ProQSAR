import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split


class RandomSplitter:
    """
    Randomly split a dataset into training and testing sets.

    This splitter partitions data into two subsets (train/test) using the
    standard ``train_test_split`` method from scikit-learn.

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
        from proqsar.Data.Splitter.random_splitter import RandomSplitter

        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "SMILES": ["CCO", "CCC", "CCN", "CCCl", "CCBr"],
            "activity": [1.2, 3.4, 2.1, 0.5, 4.7]
        })

        splitter = RandomSplitter(activity_col="activity", test_size=0.4, random_state=0)
        train, test = splitter.fit(df)
        print(train.shape, test.shape)
        # e.g., (3, 3) (2, 3)
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
        Split the dataset into training and testing sets randomly.

        :param data: Input dataset containing activity labels and features.
        :type data: pd.DataFrame

        :return: A tuple ``(train_df, test_df)`` where
                 - ``train_df`` is the training dataset,
                 - ``test_df`` is the testing dataset.
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """
        data_train, data_test = train_test_split(
            data, test_size=self.test_size, random_state=self.random_state
        )
        return data_train, data_test
