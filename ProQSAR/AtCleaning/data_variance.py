import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
import copy


class DataVarianceHandler:
    """
    Handles variance-related preprocessing of data, including:
    - Analyzing variance thresholds.
    - Removing features with low variance.

    Parameters:
    ----------
    data_train: pandas.DataFrame
        Data for training the model.
    data_test: pandas.DataFrame
        Data for external validation.
    activity_col: str
        Name of the activity column (e.g., pIC50, pChEMBL Value).
    id_col: str
        Column name of the identifier column.
    var_thresh: float, optional
        Variance threshold for feature removal (default is 0.05).
    visualize: bool, optional
        Whether to visualize the variance analysis (default is True).

    Attributes:
    ----------
    data_train: pandas.DataFrame
        Training data after preprocessing.
    data_test: pandas.DataFrame
        Test data after preprocessing.
    """

    def __init__(
        self,
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        activity_col: str,
        id_col: str,
        var_thresh: float = 0.05,
        visualize: bool = True,
    ) -> None:
        self.data_train = copy.deepcopy(data_train)
        self.data_test = copy.deepcopy(data_test)
        self.activity_col = activity_col
        self.id_col = id_col
        self.var_thresh = var_thresh
        self.visualize = visualize

    @staticmethod
    def variance_threshold_analysis(
        data: pd.DataFrame, id_col: str, activity_col: str, set_style="whitegrid"
    ) -> None:
        """
        Analyzes the impact of various variance thresholds on the number of features.
        Plots a graph to illustrate the relationship between the variance threshold and the number of features retained.

        Parameters:
        ----------
        data: pandas.DataFrame
            The DataFrame to analyze.
        id_col: str
            Name of the identifier column to exclude from the analysis.
        activity_col: str
            Name of the activity column to exclude from the analysis.
        """
        X = data.drop([id_col, activity_col], axis=1)
        thresholds = np.arange(0.0, 1, 0.05)
        results = []

        for t in thresholds:
            transform = VarianceThreshold(threshold=t)
            X_sel = transform.fit_transform(X)
            n_features = X_sel.shape[1]
            results.append(n_features)

        sns.set(style=set_style, rc={"lines.markeredgewidth": 2})
        plt.figure(figsize=(14, 8))
        plt.plot(thresholds, results, marker="o")  # Added marker
        plt.title("Variance Analysis", fontsize=24, weight="semibold")
        plt.xlabel("Variance Threshold", fontsize=16)
        plt.ylabel("Number of Features", fontsize=16)
        plt.grid(True)  # Added grid

        # Optional: Add annotations for key points
        for i, txt in enumerate(results):
            plt.annotate(
                txt,
                (thresholds[i], results[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=10,
            )

        plt.show()

    @staticmethod
    def select_features_by_variance(
        data: pd.DataFrame, activity_col: str, id_col: str, var_thresh: float
    ) -> list:
        """
        Selects features with variance above the specified threshold from a dataset.

        Parameters:
        ----------
        data: pandas.DataFrame
            The DataFrame to process for feature selection.
        activity_col: str
            Name of the activity column to exclude from feature selection.
        id_col: str
            Column name of the identifier to exclude from feature selection.
        var_thresh: float
            Variance threshold for feature selection.

        Returns:
        --------
        list
            List of column names that have variance above the threshold.
        """
        columns_to_exclude = [activity_col, id_col]
        selector = VarianceThreshold(var_thresh)
        temp = data.drop(columns_to_exclude, axis=1)
        selector.fit(temp)

        features = selector.get_support(indices=True)

        selected_columns = temp.columns[features].tolist()
        return columns_to_exclude + selected_columns

    def fit(self) -> None:
        """
        Execute the variance-related preprocessing steps on the training and test data.
        """
        if self.visualize:
            DataVarianceHandler.variance_threshold_analysis(
                self.data_train, self.id_col, self.activity_col
            )

        # Select features based on training data
        selected_features = DataVarianceHandler.select_features_by_variance(
            self.data_train, self.activity_col, self.id_col, self.var_thresh
        )

        # Apply the selected features to both training and test datasets
        self.data_train = self.data_train[selected_features]
        self.data_test = self.data_test[selected_features]
        return self.data_train, self.data_test
