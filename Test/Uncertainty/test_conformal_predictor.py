import unittest
import numpy as np
import pandas as pd
import os
from tempfile import TemporaryDirectory
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier
from ProQSAR.ModelDeveloper.model_developer import ModelDeveloper
from ProQSAR.Uncertainty.conformal_predictor import ConformalPredictor


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
    data["ID"] = np.arange(n_samples)
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
    data["ID"] = np.arange(n_samples)
    data["Activity"] = y
    return data


class TestConformalPredictor(unittest.TestCase):

    def setUp(self):
        self.class_train_data = create_classification_data(random_state=42)
        self.class_cal_data = create_classification_data(random_state=41)
        self.class_test_data = create_classification_data(random_state=40)

        self.reg_train_data = create_regression_data(random_state=42)
        self.reg_cal_data = create_regression_data(random_state=41)
        self.reg_test_data = create_regression_data(random_state=40)

        self.classifier = ModelDeveloper(
            activity_col="Activity",
            id_col="ID",
            select_model="KNeighborsClassifier",
            n_jobs=-1,
        )
        self.regressor = ModelDeveloper(
            activity_col="Activity",
            id_col="ID",
            select_model="KNeighborsRegressor",
            n_jobs=-1,
        )

        self.classifier.fit(self.class_train_data)
        self.regressor.fit(self.reg_train_data)

        self.temp_dir = TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initialization(self):
        predictor = ConformalPredictor(
            model=self.classifier, activity_col="Activity", id_col="ID"
        )
        # Before fit, model is a ModelDeveloper; after fit, underlying estimator is set
        self.assertIsInstance(predictor.model, ModelDeveloper)
        predictor.fit(self.class_train_data)
        self.assertIsInstance(predictor.model, KNeighborsClassifier)

    def test_fit_classification(self):
        predictor = ConformalPredictor(
            model=self.classifier, activity_col="Activity", id_col="ID", save_dir=None
        )
        predictor.fit(self.class_cal_data)
        self.assertIsNotNone(predictor.cp)
        self.assertEqual(predictor.task_type, "C")

    def test_fit_regression(self):
        predictor = ConformalPredictor(
            model=self.regressor, activity_col="Activity", id_col="ID", save_dir=None
        )
        predictor.fit(self.reg_cal_data)
        self.assertIsNotNone(predictor.cp)
        self.assertEqual(predictor.task_type, "R")

    def test_predict_classification(self):
        predictor = ConformalPredictor(
            model=self.classifier, activity_col="Activity", id_col="ID", save_dir=None
        )
        predictor.fit(self.class_cal_data)
        predictions = predictor.predict(self.class_test_data)
        self.assertIn("Predicted value", predictions.columns)

    def test_predict_regression(self):
        predictor = ConformalPredictor(
            model=self.regressor, activity_col="Activity", id_col="ID", save_dir=None
        )
        predictor.fit(self.reg_cal_data)
        predictions = predictor.predict(self.reg_test_data)
        self.assertIn("Predicted value", predictions.columns)

    def test_predict_with_alpha_classification_sets(self):
        predictor = ConformalPredictor(
            model=self.classifier, activity_col="Activity", id_col="ID", save_dir=None
        )
        predictor.fit(self.class_cal_data)
        predictions = predictor.predict(self.class_test_data, alpha=[0.1, 0.2])
        self.assertIn("Predicted value", predictions.columns)
        self.assertTrue(any(col.startswith("Prediction Set (alpha=") for col in predictions.columns))

    def test_predict_with_alpha_regression_intervals(self):
        predictor = ConformalPredictor(
            model=self.regressor, activity_col="Activity", id_col="ID", save_dir=None
        )
        predictor.fit(self.reg_cal_data)
        predictions = predictor.predict(self.reg_test_data, alpha=[0.1])
        self.assertIn("Predicted value", predictions.columns)
        self.assertTrue(any(col.startswith("Prediction Interval (alpha=") for col in predictions.columns))

    def test_not_fitted_error(self):
        predictor = ConformalPredictor(
            model=self.classifier, activity_col="Activity", id_col="ID", save_dir=None
        )
        with self.assertRaises(NotFittedError):
            predictor.predict(self.class_test_data)

    def test_fit_missing_target_column_raises(self):
        predictor = ConformalPredictor(
            model=self.classifier, activity_col="Activity", id_col="ID", save_dir=None
        )
        data_missing_target = self.class_cal_data.drop(columns=["Activity"])
        with self.assertRaises(KeyError):
            predictor.fit(data_missing_target)

    def test_saving(self):
        predictor = ConformalPredictor(
            model=self.classifier,
            activity_col="Activity",
            id_col="ID",
            save_dir=self.temp_dir.name,
        )
        predictor.fit(self.class_cal_data)
        predictor.predict(self.class_test_data)

        self.assertTrue(os.path.exists(f"{self.temp_dir.name}/conformal_predictor.pkl"))
        self.assertTrue(
            os.path.exists(f"{self.temp_dir.name}/conformal_pred_result.csv")
        )


if __name__ == "__main__":
    unittest.main()
