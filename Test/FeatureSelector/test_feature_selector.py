import unittest
import pandas as pd
import numpy as np
import os
import shutil
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import NotFittedError
from ProQSAR.FeatureSelector.feature_selector import FeatureSelector


def create_classification_data(
    n_samples=40, n_features=40, n_informative=10, random_state=42
) -> pd.DataFrame:
    """
    Create a synthetic classification dataset.

    Parameters:
    ----------
    n_samples : int, optional
        The number of samples to generate, by default 40.
    n_features : int, optional
        The total number of features, by default 40.
    n_informative : int, optional
        The number of informative features, by default 10.
    random_state : int, optional
        Random seed for reproducibility, by default 42.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the generated features and target.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        random_state=random_state,
    )
    data = pd.DataFrame(X, columns=[f"Feature{i}" for i in range(1, n_features + 1)])
    data["ID"] = np.arange(n_samples)
    data["Activity"] = y
    return data


def create_regression_data(
    n_samples=40, n_features=40, n_informative=10, random_state=42
) -> pd.DataFrame:
    """
    Create a synthetic regression dataset.

    Parameters:
    ----------
    n_samples : int, optional
        The number of samples to generate, by default 40.
    n_features : int, optional
        The total number of features, by default 40.
    n_informative : int, optional
        The number of informative features, by default 10.
    random_state : int, optional
        Random seed for reproducibility, by default 42.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the generated features and target.
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        random_state=random_state,
    )
    data = pd.DataFrame(X, columns=[f"Feature{i}" for i in range(1, n_features + 1)])
    data["ID"] = np.arange(n_samples)
    data["Activity"] = y
    return data


class TestFeatureSelector(unittest.TestCase):
    """
    Unit test class for the FeatureSelector module.
    """

    def setUp(self):
        """
        Set up the test environment before each test method.
        """
        self.data_classification = create_classification_data()
        self.data_regression = create_regression_data()
        self.save_dir = "test_save"
        os.makedirs(self.save_dir, exist_ok=True)

    def tearDown(self):
        """
        Cleans up the test environment after each test method.
        """
        shutil.rmtree(self.save_dir)

    def test_fit_classification(self):
        """
        Test fitting the FeatureSelector on classification data.
        """
        fs = FeatureSelector(activity_col="Activity", id_col="ID", method="Anova")
        fs.fit(self.data_classification)
        self.assertIsNotNone(fs.feature_selector)
        self.assertEqual(fs.task_type, "C")

    def test_fit_regression(self):
        """
        Test fitting the FeatureSelector on regression data.
        """
        fs = FeatureSelector(activity_col="Activity", id_col="ID", method="Anova")
        fs.fit(self.data_regression)
        self.assertIsNotNone(fs.feature_selector)
        self.assertEqual(fs.task_type, "R")

    def test_transform_before_fit(self):
        """
        Test calling transform before fitting the FeatureSelector.
        """
        fs = FeatureSelector(activity_col="Activity", id_col="ID")
        with self.assertRaises(NotFittedError):
            fs.transform(self.data_classification)

    def test_fit_transform(self):
        """
        Test the fit_transform method of the FeatureSelector.
        """
        fs = FeatureSelector(activity_col="Activity", id_col="ID", select_method="Anova")
        transformed_data = fs.fit_transform(self.data_classification)
        self.assertNotEqual(
            transformed_data.shape[1], self.data_classification.shape[1]
        )

    def test_compare_feature_selectors_classification(self):
        """
        Test comparing feature selection methods on classification data.
        """
        fs = FeatureSelector(
            activity_col="Activity",
            id_col="ID",
            compare_visual="box",
        )
        result_df = fs.compare_feature_selectors(self.data_classification)
        self.assertIn("Mean", result_df.columns)
        self.assertIn("Std", result_df.columns)

    def test_compare_feature_selectors_regression(self):
        """
        Test comparing feature selection methods on regression data.
        """
        fs = FeatureSelector(
            activity_col="Activity", id_col="ID", compare_table="table"
        )
        result_df = fs.compare_feature_selectors(self.data_regression)
        self.assertIn("Mean", result_df.columns)
        self.assertIn("Std", result_df.columns)

    def test_static_transform(self):
        """
        Test the static_transform method of the FeatureSelector.
        """
        fs = FeatureSelector(
            activity_col="Activity", id_col="ID", method="Anova", save_dir="test_save"
        )
        fs.fit(self.data_classification)
        transformed_data = FeatureSelector.static_transform(
            self.data_classification, "test_save"
        )
        self.assertNotEqual(
            transformed_data.shape[1], self.data_classification.shape[1]
        )

    def test_static_transform_not_fitted(self):
        """
        Test static_transform method when the model is not fitted.
        """
        with self.assertRaises(NotFittedError):
            FeatureSelector.static_transform(
                self.data_classification, save_dir="non_existent_dir"
            )

    def test_select_best_method(self):
        """
        Test selecting the best feature selection method.
        """
        fs = FeatureSelector(activity_col="Activity", id_col="ID")
        best_method = fs._select_best_method(self.data_classification)
        self.assertIn(best_method, fs.method_map)


if __name__ == "__main__":
    unittest.main()
