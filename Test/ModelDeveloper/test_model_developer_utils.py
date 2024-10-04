import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    RepeatedKFold,
)
from sklearn.linear_model import Ridge
from ProQSAR.ModelDeveloper.model_developer_utils import (
    _get_cv_strategy,
    _get_task_type,
    _get_method_map,
    _get_iv_scoring_list,
    _get_ev_scoring_dict,
)


def create_classification_data(
    n_samples=60, n_features=25, n_informative=10, random_state=42
) -> pd.DataFrame:
    """
    Generate a DataFrame containing synthetic classification data.

    Args:
        n_samples (int): The number of samples.
        n_features (int): The number of features.
        n_informative (int): The number of informative features.
        random_state (int): Seed for random number generation.

    Returns:
        pd.DataFrame: DataFrame with features, ID, and activity columns.
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
    n_samples=40, n_features=20, n_informative=10, random_state=42
) -> pd.DataFrame:
    """
    Generate a DataFrame containing synthetic regression data.

    Args:
        n_samples (int): The number of samples.
        n_features (int): The number of features.
        n_informative (int): The number of informative features.
        random_state (int): Seed for random number generation.

    Returns:
        pd.DataFrame: DataFrame with features, ID, and activity columns.
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


class TestModelMethods(unittest.TestCase):

    def setUp(self):
        # Set up classification and regression data
        self.class_data = create_classification_data()
        self.reg_data = create_regression_data()

    def test_get_task_type_classification(self):
        result = _get_task_type(self.class_data, "Activity")
        self.assertEqual(result, "C")

    def test_get_task_type_regression(self):
        result = _get_task_type(self.reg_data, "Activity")
        self.assertEqual(result, "R")

    def test_get_method_map_classification(self):
        method_map = _get_method_map("C")
        self.assertIn("Logistic", method_map)
        self.assertIn("KNN", method_map)

    def test_get_method_map_regression(self):
        method_map = _get_method_map("R")
        self.assertIn("Linear", method_map)
        self.assertIn("ElasticNet", method_map)

    def test_get_cv_strategy_classification(self):
        cv_strategy = _get_cv_strategy("C")
        self.assertIsInstance(cv_strategy, RepeatedStratifiedKFold)

    def test_get_cv_strategy_regression(self):
        cv_strategy = _get_cv_strategy("R")
        self.assertIsInstance(cv_strategy, RepeatedKFold)

    def test_get_iv_scoring_list_classification(self):
        scoring_list = _get_iv_scoring_list("C")
        self.assertIn("roc_auc", scoring_list)

    def test_get_iv_scoring_list_regression(self):
        scoring_list = _get_iv_scoring_list("R")
        self.assertIn("r2", scoring_list)

    def test_get_ev_scoring_dict_classification(self):
        y_test = np.array([0, 1, 1, 0])
        y_test_pred = np.array([0, 1, 1, 1])
        y_test_proba = np.array([0.2, 0.8, 0.9, 0.4])
        scoring_dict = _get_ev_scoring_dict("C", y_test, y_test_pred, y_test_proba)

        self.assertIn("roc_auc", scoring_dict)

    def test_get_ev_scoring_dict_regression(self):
        y_test = np.array([3.0, 2.5, 4.0, 5.0])
        y_test_pred = np.array([2.8, 2.6, 3.9, 4.9])
        scoring_dict = _get_ev_scoring_dict("R", y_test, y_test_pred)

        self.assertIn("r2", scoring_dict)
        self.assertIn("mean_squared_error", scoring_dict)
        self.assertIn("mean_absolute_error", scoring_dict)

    def test_additional_methods_in_method_map(self):
        add_method = {"CustomModel": Ridge(alpha=1.0)}
        method_map = _get_method_map("R", add_method=add_method)
        self.assertIn("CustomModel", method_map)

    def test_task_type_invalid_data(self):
        invalid_data = self.class_data.copy()
        invalid_data["Activity"] = 0
        with self.assertRaises(ValueError):
            _get_task_type(invalid_data, "Activity")

    def test_ev_scoring_dict_classification_missing_proba(self):
        y_test = np.array([0, 1, 1, 0])
        y_test_pred = np.array([0, 1, 1, 1])
        with self.assertRaises(TypeError):
            _get_ev_scoring_dict("C", y_test, y_test_pred)

    def test_iv_scoring_list_invalid_task(self):
        with self.assertRaises(ValueError):
            _get_iv_scoring_list("X")

    def test_ev_scoring_list_invalid_task(self):
        y_test = np.array([0, 1, 1, 0])
        y_test_pred = np.array([0, 1, 1, 1])
        with self.assertRaises(ValueError):
            _get_ev_scoring_dict("X", y_test, y_test_pred)


if __name__ == "__main__":
    unittest.main()
