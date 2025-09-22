import unittest
import pandas as pd
import numpy as np
import matplotlib
from tempfile import TemporaryDirectory
from sklearn.datasets import make_classification, make_regression
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
)
from proqsar.Model.FeatureSelector.feature_selector_utils import (
    _get_method_map,
    evaluate_feature_selectors,
)

matplotlib.use("Agg")


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


class TestFeatureSelectorUtils(unittest.TestCase):

    def setUp(self):
        # Generate classification and regression data
        self.temp_dir = TemporaryDirectory()
        self.classification_data = create_classification_data()
        self.regression_data = create_regression_data()
        self.activity_col = "Activity"
        self.id_col = "ID"
        self.add_method = {
            "DummyMethod": SelectKBest(score_func=mutual_info_classif, k=5)
        }
        self.select_methods_class = [
            "Anova",
            "RandomForestClassifier",
            "LogisticRegression",
        ]
        self.select_methods_reg = ["Anova", "RandomForestRegressor", "LassoCV"]

    def tearDown(self):
        # Automatically clean up the temporary directory
        self.temp_dir.cleanup()

    def test_get_method_map_classification(self):
        # Test method map for classification
        method_map = _get_method_map("C", add_method=self.add_method)
        self.assertIn("Anova", method_map)
        self.assertIn("MutualInformation", method_map)
        self.assertIn("RandomForestClassifier", method_map)
        self.assertIn("DummyMethod", method_map)
        self.assertIsInstance(method_map["DummyMethod"], SelectKBest)

    def test_get_method_map_regression(self):
        # Test method map for regression
        method_map = _get_method_map("R", add_method=self.add_method)
        self.assertIn("Anova", method_map)
        self.assertIn("MutualInformation", method_map)
        self.assertIn("RandomForestRegressor", method_map)
        self.assertIn("DummyMethod", method_map)
        self.assertIsInstance(method_map["DummyMethod"], SelectKBest)

    def test_get_method_map_invalid_task_type(self):
        # Test invalid task type handling
        with self.assertRaises(ValueError):
            _get_method_map("invalid_type")

    def test_evaluate_feature_selectors_classification(self):
        # Evaluate feature selectors for classification dataset
        result_df = evaluate_feature_selectors(
            data=self.classification_data,
            activity_col=self.activity_col,
            id_col=self.id_col,
            select_method=self.select_methods_class,
            scoring_list=["accuracy", "f1"],
            n_splits=3,
            n_repeats=2,
            visualize=None,
            save_fig=False,
            save_csv=False,
            n_jobs=3,
        )
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertIn("Anova", result_df.columns)

    def test_evaluate_feature_selectors_regression(self):
        # Evaluate feature selectors for regression dataset
        result_df = evaluate_feature_selectors(
            data=self.regression_data,
            activity_col=self.activity_col,
            id_col=self.id_col,
            select_method=self.select_methods_reg,
            scoring_list=["r2", "neg_mean_squared_error"],
            n_splits=3,
            n_repeats=2,
            visualize=None,
            save_fig=False,
            save_csv=False,
            n_jobs=3,
        )
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertIn("RandomForestRegressor", result_df.columns)

    def test_evaluate_feature_selectors_invalid_method(self):
        # Test for invalid method in evaluation
        with self.assertRaises(ValueError):
            evaluate_feature_selectors(
                data=self.classification_data,
                activity_col=self.activity_col,
                id_col=self.id_col,
                select_method=["InvalidMethod"],
                scoring_list=["accuracy"],
                n_splits=3,
                n_repeats=2,
                n_jobs=3,
            )

    def test_evaluate_feature_selectors_save_csv(self):
        # Test saving CSV with dummy path to validate no exceptions occur
        result_df = evaluate_feature_selectors(
            data=self.classification_data,
            activity_col=self.activity_col,
            id_col=self.id_col,
            select_method=self.select_methods_class,
            scoring_list=["accuracy"],
            n_splits=2,
            n_repeats=1,
            save_csv=True,
            save_dir=self.temp_dir.name,
            csv_name="test_csv",
        )
        self.assertIsInstance(result_df, pd.DataFrame)

    def test_evaluate_feature_selectors_visualization(self):
        # Test visualization option with dummy parameters
        result_df = evaluate_feature_selectors(
            data=self.classification_data,
            activity_col=self.activity_col,
            id_col=self.id_col,
            select_method=self.select_methods_class,
            scoring_list=["accuracy"],
            visualize="bar",
            save_fig=False,
        )
        self.assertIsInstance(result_df, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
