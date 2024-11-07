import os
import shutil
import unittest
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.datasets import make_classification, make_regression
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

    def setUp(self):
        """
        Set up the test environment before each test method.
        """
        self.classification_data = create_classification_data()
        self.regression_data = create_regression_data()
        self.fs = FeatureSelector(
            activity_col="Activity", id_col="ID", save_method=True, save_trans_data=True
        )
        self.save_dir = "test_feature_selector"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def tearDown(self):
        # Cleanup any files created during testing
        shutil.rmtree(self.save_dir)

    def test_initialization(self):
        """Test proper initialization of the FeatureSelector instance"""
        self.assertEqual(self.fs.activity_col, "Activity")
        self.assertEqual(self.fs.id_col, "ID")
        self.assertEqual(self.fs.select_method, "best")

    def test_fit_classification(self):
        """Test the fit method on classification data"""
        selector = self.fs.fit(self.classification_data)
        self.assertIsNotNone(selector)

    def test_fit_regression(self):
        """Test the fit method on regression data"""
        fs_regression = FeatureSelector(
            activity_col="Activity", id_col="ID", scoring="r2"
        )
        selector = fs_regression.fit(self.regression_data)
        self.assertIsNotNone(selector)

    def test_transform_without_fit(self):
        """Test that transform raises an error if fit is not called first"""
        with self.assertRaises(NotFittedError):
            self.fs.transform(self.classification_data)

    def test_fit_transform(self):
        """Test the fit_transform method"""
        transformed_data = self.fs.fit_transform(self.classification_data)
        self.assertIn("Activity", transformed_data.columns)
        self.assertIn("ID", transformed_data.columns)
        self.assertIsInstance(transformed_data, pd.DataFrame)

    def test_unrecognized_select_method(self):
        """Test ValueError for unrecognized select method"""
        fs_invalid = FeatureSelector(
            activity_col="Activity", id_col="ID", select_method="unknown_method"
        )
        with self.assertRaises(ValueError):
            fs_invalid.fit(self.classification_data)

    def test_save_method(self):
        """Test saving of the feature selector and metadata after fitting"""
        self.fs.save_dir = "test_save"
        self.fs.fit(self.classification_data)
        self.assertTrue(os.path.exists(f"{self.fs.save_dir}/feature_selector.pkl"))
        shutil.rmtree(self.fs.save_dir)

    def test_save_transformed_data(self):
        """Test saving of transformed data after transform"""
        self.fs.save_dir = "test_save"
        self.fs.save_trans_data = True
        self.fs.fit(self.classification_data)
        transformed_data = self.fs.transform(self.classification_data)
        self.assertTrue(os.path.exists(f"{self.fs.save_dir}/fs_trans_data.csv"))
        self.assertIsInstance(transformed_data, pd.DataFrame)
        shutil.rmtree(self.fs.save_dir)


if __name__ == "__main__":
    unittest.main()
