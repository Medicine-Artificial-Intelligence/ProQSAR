import unittest
import pandas as pd
import numpy as np
from tempfile import TemporaryDirectory
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from proqsar.Model.ModelDeveloper.model_developer import ModelDeveloper
import os


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


class TestModelDeveloper(unittest.TestCase):

    def setUp(self):
        """Setup the test environment."""
        self.data = create_classification_data()
        self.train_data, self.test_data = train_test_split(
            self.data, test_size=0.2, random_state=42
        )

        self.model_dev = ModelDeveloper(
            activity_col="Activity",
            id_col="ID",
            add_model={"NewModel": RandomForestClassifier()},
        )
        self.temp_dir = TemporaryDirectory()
        self.model_dev.save_dir = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_fit_method(self):
        """Test fitting the model."""
        model = self.model_dev.fit(self.train_data)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(self.model_dev, "model"))

    def test_predict_method(self):
        """Test predicting using the fitted model."""
        self.model_dev.fit(self.train_data)
        predictions = self.model_dev.predict(self.test_data)
        self.assertIn("Predicted value", predictions.columns)
        self.assertEqual(predictions.shape[0], self.test_data.shape[0])

    def test_fit_invalid_model(self):
        """Test fitting with an invalid model raises ValueError."""
        self.model_dev.select_model = "invalid_model"
        with self.assertRaises(ValueError):
            self.model_dev.fit(self.train_data)

    def test_predict_not_fitted(self):
        """Test prediction raises NotFittedError if model is not fitted."""
        with self.assertRaises(NotFittedError):
            self.model_dev.predict(self.test_data)

    def test_string_model_with_cv_generates_report(self):
        md = ModelDeveloper(activity_col="Activity", id_col="ID", select_model="SVC")
        md.save_dir = self.temp_dir.name
        md.fit(self.train_data)
        self.assertIsNotNone(md.report)
        self.assertTrue(isinstance(md.report, pd.DataFrame))

    def test_list_model_without_cv_raises(self):
        md = ModelDeveloper(
            activity_col="Activity",
            id_col="ID",
            select_model=["SVC", "KNeighborsClassifier"],
            cross_validate=False,
        )
        with self.assertRaises(AttributeError):
            md.fit(self.train_data)

    def test_set_params_updates_and_invalid_key_raises(self):
        md = ModelDeveloper(activity_col="Activity", id_col="ID")
        md.set_params(save_pred_result=True, pred_result_name="out", n_jobs=2)
        self.assertTrue(md.save_pred_result)
        self.assertEqual(md.pred_result_name, "out")
        self.assertEqual(md.n_jobs, 2)
        with self.assertRaises(KeyError):
            md.set_params(bad_key=True)

    def test_save_model_and_prediction_artifacts(self):
        md = ModelDeveloper(
            activity_col="Activity",
            id_col="ID",
            save_model=True,
            save_pred_result=True,
        )
        md.save_dir = self.temp_dir.name
        md.fit(self.train_data)
        self.assertTrue(os.path.exists(f"{self.temp_dir.name}/model.pkl"))
        _ = md.predict(self.test_data)
        self.assertTrue(
            os.path.exists(f"{self.temp_dir.name}/{md.pred_result_name}.csv")
        )


if __name__ == "__main__":
    unittest.main()
