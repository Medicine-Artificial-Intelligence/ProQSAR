import unittest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from proqsar.Model.Optimizer.optimizer import Optimizer
from sklearn.datasets import make_classification, make_regression


def create_classification_data(
    n_samples=40, n_features=40, n_informative=10, random_state=42
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        random_state=random_state,
    )
    data = pd.DataFrame(X, columns=[f"Feature{i}" for i in range(1, n_features + 1)])
    data["ID"] = range(n_samples)
    data["Activity"] = y
    return data


def create_regression_data(
    n_samples=40, n_features=40, n_informative=10, random_state=42
):
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        random_state=random_state,
    )
    data = pd.DataFrame(X, columns=[f"Feature{i}" for i in range(1, n_features + 1)])
    data["ID"] = range(n_samples)
    data["Activity"] = y
    return data


class TestOptimizer(unittest.TestCase):

    def setUp(self):
        self.classification_data = create_classification_data()
        self.regression_data = create_regression_data()

    def test_optimizer_classification(self):
        optimizer = Optimizer(
            activity_col="Activity",
            id_col="ID",
            scoring="accuracy",
            study_name="study_classification",
        )

        best_params, best_score = optimizer.optimize(self.classification_data)

        self.assertIsNotNone(best_params)
        self.assertIsInstance(best_params, dict)
        self.assertIsNotNone(best_score)
        self.assertIsInstance(best_score, float)

        # Check getter methods
        self.assertEqual(optimizer.get_best_params(), best_params)
        self.assertEqual(optimizer.get_best_score(), best_score)

        # Verify attributes after optimization
        self.assertIsNotNone(optimizer.task_type)
        self.assertEqual(optimizer.task_type, "C")  # Classification task
        self.assertIsNotNone(optimizer.cv)

    def test_optimizer_regression(self):
        optimizer = Optimizer(
            activity_col="Activity",
            id_col="ID",
            scoring="r2",
            study_name="study_regression",
        )

        best_params, best_score = optimizer.optimize(self.regression_data)

        self.assertIsNotNone(best_params)
        self.assertIsInstance(best_params, dict)
        self.assertIsNotNone(best_score)
        self.assertIsInstance(best_score, float)

        # Check getter methods
        self.assertEqual(optimizer.get_best_params(), best_params)
        self.assertEqual(optimizer.get_best_score(), best_score)

        # Verify attributes after optimization
        self.assertIsNotNone(optimizer.task_type)
        self.assertEqual(optimizer.task_type, "R")  # Regression task
        self.assertIsNotNone(optimizer.cv)

    def test_optimizer_with_custom_model(self):
        custom_model = {"CustomModel": (LogisticRegression(), {"C": [0.1, 5]})}
        optimizer = Optimizer(
            activity_col="Activity",
            id_col="ID",
            add_model=custom_model,
            study_name="study_custom_model",
        )

        best_params, best_score = optimizer.optimize(self.classification_data)

        self.assertIsNotNone(best_params)
        self.assertIn("model", best_params)
        self.assertIsNotNone(best_score)
        self.assertGreaterEqual(best_score, 0.0)

        # Verify that the custom model was used
        self.assertIn("CustomModel", optimizer.param_ranges)

    def test_optimizer_with_missing_columns(self):
        optimizer = Optimizer(
            activity_col="Activity",
            id_col="ID",
            select_model=["RandomForestClassifier"],
            scoring="accuracy",
        )

        data_missing_activity = self.classification_data.drop(columns=["Activity"])

        with self.assertRaises(KeyError):
            optimizer.optimize(data_missing_activity)

        data_missing_id = self.classification_data.drop(columns=["ID"])

        with self.assertRaises(KeyError):
            optimizer.optimize(data_missing_id)

    def test_get_best_params_before_optimization(self):
        optimizer = Optimizer(
            activity_col="Activity",
            id_col="ID",
            select_model=["RandomForestClassifier"],
            scoring="accuracy",
        )

        with self.assertRaises(AttributeError):
            optimizer.get_best_params()

    def test_get_best_score_before_optimization(self):
        optimizer = Optimizer(
            activity_col="Activity",
            id_col="ID",
            select_model=["RandomForestClassifier"],
            scoring="accuracy",
        )

        with self.assertRaises(AttributeError):
            optimizer.get_best_score()


if __name__ == "__main__":
    unittest.main()
