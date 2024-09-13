import os
import shutil
import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import NotFittedError
from ProQSAR.ModelDeveloper.model_developer import ModelDeveloper


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


class TestModelDeveloper(unittest.TestCase):

    def setUp(self):
        """Setup the test environment."""
        self.data_class = create_classification_data()
        self.data_reg = create_regression_data()
        self.model_dev_class = ModelDeveloper(
            activity_col="Activity",
            id_col="ID",
            method="best",
            scoring_target="accuracy",
            n_jobs=1,
            save_dir=None,
        )
        self.model_dev_reg = ModelDeveloper(
            activity_col="Activity",
            id_col="ID",
            method="KNN",
            scoring_target="r2",
            n_jobs=1,
            save_dir=None,
        )

    def test_classification_fit_predict(self):
        """Test fitting and predicting for classification."""
        self.model_dev_class.fit(self.data_class)
        result = self.model_dev_class.predict(self.data_class)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("Predicted values", result.columns)
        self.assertIn("Probability", result.columns)

    def test_regression_fit_predict(self):
        """Test fitting and predicting for regression."""
        self.model_dev_reg.fit(self.data_reg)
        result = self.model_dev_reg.predict(self.data_reg)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("Predicted values", result.columns)

    def test_invalid_method(self):
        """Test invalid method input."""
        with self.assertRaises(ValueError):
            model_dev_invalid = ModelDeveloper(
                activity_col="Activity", id_col="ID", method="invalid_method"
            )
            model_dev_invalid.fit(self.data_class)

    def test_not_fitted_error(self):
        """Test predict before fit, should raise NotFittedError."""
        with self.assertRaises(NotFittedError):
            self.model_dev_class.predict(self.data_class)

    def test_static_predict(self):
        """Test static prediction method with not fitted model."""
        with self.assertRaises(NotFittedError):
            ModelDeveloper.static_predict(self.data_class, "invalid_dir")

    def test_iv_model_comparison_report(self):
        """Test compare_models function."""
        result = self.model_dev_class.iv_model_comparison_report(self.data_class)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("Method", result.columns)
        self.assertIn("Mean", result.columns)

    def test_save(self):
        """Test saving of model files and figures."""
        model_dev_with_save = ModelDeveloper(
            activity_col="Activity",
            id_col="ID",
            method="best",
            save_dir="test_save",
            comparison_visual="bar",
            scoring_target="f1",
            save_fig=True,
        )
        model_dev_with_save.fit(self.data_class)
        model_dev_with_save.predict(self.data_class)

        self.assertTrue(os.path.exists("test_save/activity_col.pkl"))
        self.assertTrue(os.path.exists("test_save/pred_result.csv"))
        self.assertTrue(os.path.exists("test_save/iv_model_comparison_f1.csv"))
        self.assertTrue(os.path.exists("test_save/iv_model_comparison_f1_bar.png"))

    def test_internal_validation_report_classification(self):
        """Test internal validation report for classification."""
        result = self.model_dev_class.internal_validation_report(
            data=self.data_class,
            model="RF",
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("roc_auc", result.columns)

    def test_internal_validation_report_regression(self):
        """Test internal validation report for regression."""
        result = self.model_dev_reg.internal_validation_report(
            data=self.data_reg,
            model="RF",
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("r2", result.columns)

    def test_external_validation_report_classification(self):
        """Test external validation report for classification."""
        train_data = self.data_class.sample(frac=0.7, random_state=42)
        test_data = self.data_class.drop(train_data.index)

        result = self.model_dev_class.external_validation_report(
            data_train=train_data,
            data_test=test_data,
            scoring_list=["roc_auc", "f1", "accuracy", "brier_score"],
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("roc_auc", result.columns)

    def test_external_validation_report_regression(self):
        """Test external validation report for regression."""
        train_data = self.data_reg.sample(frac=0.7, random_state=42)
        test_data = self.data_reg.drop(train_data.index)

        result = self.model_dev_reg.external_validation_report(
            data_train=train_data,
            data_test=test_data,
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("r2", result.columns)

    def test_internal_validation_report_invalid_scoring(self):
        """Test handling of invalid scoring metrics in internal validation."""
        with self.assertRaises(ValueError):
            self.model_dev_class.internal_validation_report(
                data=self.data_class, scoring_list=["invalid_metric"]
            )

    def test_external_validation_report_invalid_model(self):
        """Test handling of invalid models in external validation."""
        with self.assertRaises(ValueError):
            self.model_dev_class.external_validation_report(
                data_train=self.data_class,
                data_test=self.data_class,
                select_model=["InvalidModel"],
            )

    def tearDown(self):
        """Clean up test artifacts."""
        if os.path.exists("test_save"):
            shutil.rmtree("test_save")


if __name__ == "__main__":
    unittest.main()
