import os
import unittest
import numpy as np
import pandas as pd
import matplotlib
from tempfile import TemporaryDirectory
from sklearn.datasets import make_classification, make_regression
from ProQSAR.ModelDeveloper.model_validation import ModelValidation

matplotlib.use("Agg")


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


class TestModelReports(unittest.TestCase):

    def setUp(self):
        self.class_data = create_classification_data()
        self.reg_data = create_regression_data()
        self.temp_dir = TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cv_report_classification(self):
        # Test cv_report for classification data
        cv_result = ModelValidation.cross_validation_report(
            self.class_data, activity_col="Activity", id_col="ID"
        )
        self.assertIsInstance(cv_result, pd.DataFrame)
        self.assertGreater(len(cv_result), 0)

    def test_cv_report_regression(self):
        # Test cv_report for regression data
        cv_result = ModelValidation.cross_validation_report(
            self.reg_data, activity_col="Activity", id_col="ID"
        )
        self.assertIsInstance(cv_result, pd.DataFrame)
        self.assertGreater(len(cv_result), 0)

    def test_ev_report_classification(self):
        # Test ev_report for classification data (with train/test split)
        data_train = self.class_data.sample(frac=0.8, random_state=42)
        data_test = self.class_data.drop(data_train.index)
        ev_result = ModelValidation.external_validation_report(
            data_train, data_test, activity_col="Activity", id_col="ID"
        )
        self.assertIsInstance(ev_result, pd.DataFrame)
        self.assertGreater(len(ev_result), 0)

    def test_ev_report_regression(self):
        # Test ev_report for regression data (with train/test split)
        data_train = self.reg_data.sample(frac=0.8, random_state=42)
        data_test = self.reg_data.drop(data_train.index)
        ev_result = ModelValidation.external_validation_report(
            data_train, data_test, activity_col="Activity", id_col="ID"
        )
        self.assertIsInstance(ev_result, pd.DataFrame)
        self.assertGreater(len(ev_result), 0)

    def test_ev_report_save_csv(self):
        data_train = self.class_data.sample(frac=0.8, random_state=42)
        data_test = self.class_data.drop(data_train.index)
        ModelValidation.external_validation_report(
            data_train,
            data_test,
            activity_col="Activity",
            id_col="ID",
            select_model=["KNeighborsClassifier", "SVC", "ExtraTreesClassifier"],
            scoring_list=["roc_auc", "f1", "recall"],
            save_csv=True,
            csv_name="test_ev_report",
            save_dir=self.temp_dir.name,
        )
        # Ensure the csv file is saved
        self.assertTrue(os.path.exists(f"{self.temp_dir.name}/test_ev_report.csv"))

    def test_invalid_graph_type(self):
        # Test invalid graph type in _plot_cv_report
        cv_result = ModelValidation.cross_validation_report(
            self.class_data,
            activity_col="Activity",
            id_col="ID",
            scoring_list=["accuracy"],
        )
        with self.assertRaises(ValueError):
            ModelValidation._plot_cv_report(
                report_df=cv_result, scoring_list=["accuracy"], graph_type="invalid"
            )

    def test_invalid_select_model(self):
        # Test cv_report with an invalid model
        with self.assertRaises(ValueError):
            ModelValidation.cross_validation_report(
                self.class_data,
                activity_col="Activity",
                id_col="ID",
                select_model=["InvalidModel"],
            )

    def test_plot_cv_report_bar(self):
        cv_result = ModelValidation.cross_validation_report(
            self.class_data,
            activity_col="Activity",
            id_col="ID",
            scoring_list=["accuracy"],
        )
        ModelValidation._plot_cv_report(
            report_df=cv_result, scoring_list=["accuracy"], graph_type="bar"
        )
        # Ensure no exception occurs when plotting

    def test_plot_cv_report_save_fig(self):
        # Test _plot_cv_report with save_fig=True
        cv_result = ModelValidation.cross_validation_report(
            self.class_data,
            activity_col="Activity",
            id_col="ID",
            scoring_list=["accuracy"],
            visualize="box",
            save_fig=True,
            fig_prefix="test_cv_graph",
            save_dir=self.temp_dir.name,
        )
        # Ensure the figure file is saved
        self.assertTrue(
            os.path.exists(f"{self.temp_dir.name}/test_cv_graph_box.png")
        )


if __name__ == "__main__":
    unittest.main()
