import os
import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from ProQSAR.ModelDeveloper.model_validation import (
    _plot_iv_report,
    iv_report,
    ev_report,
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


class TestModelReports(unittest.TestCase):

    def setUp(self):
        # Set up classification and regression data
        self.class_data = create_classification_data()
        self.reg_data = create_regression_data()

    def test_iv_report_classification(self):
        # Test iv_report for classification data
        iv_result = iv_report(self.class_data, activity_col="Activity", id_col="ID")
        self.assertIsInstance(iv_result, pd.DataFrame)
        self.assertGreater(len(iv_result), 0)

    def test_iv_report_regression(self):
        # Test iv_report for regression data
        iv_result = iv_report(self.reg_data, activity_col="Activity", id_col="ID")
        self.assertIsInstance(iv_result, pd.DataFrame)
        self.assertGreater(len(iv_result), 0)

    def test_ev_report_classification(self):
        # Test ev_report for classification data (with train/test split)
        data_train = self.class_data.sample(frac=0.8, random_state=42)
        data_test = self.class_data.drop(data_train.index)
        ev_result = ev_report(
            data_train, data_test, activity_col="Activity", id_col="ID"
        )
        self.assertIsInstance(ev_result, pd.DataFrame)
        self.assertGreater(len(ev_result), 0)

    def test_ev_report_regression(self):
        # Test ev_report for regression data (with train/test split)
        data_train = self.reg_data.sample(frac=0.8, random_state=42)
        data_test = self.reg_data.drop(data_train.index)
        ev_result = ev_report(
            data_train, data_test, activity_col="Activity", id_col="ID"
        )
        self.assertIsInstance(ev_result, pd.DataFrame)
        self.assertGreater(len(ev_result), 0)

    def test_ev_report_save_csv(self):
        data_train = self.class_data.sample(frac=0.8, random_state=42)
        data_test = self.class_data.drop(data_train.index)
        ev_report(
            data_train,
            data_test,
            activity_col="Activity",
            id_col="ID",
            select_method=["KNN", "SVM", "ExT"],
            scoring_list=["roc_auc", "f1", "recall"],
            save_csv=True,
            csv_name="test_ev_report",
            save_dir="test_dir",
        )
        # Ensure the csv file is saved
        self.assertTrue(os.path.exists("test_dir/test_ev_report.csv"))
        # Cleanup created file
        os.remove("test_dir/test_ev_report.csv")
        os.rmdir("test_dir")

    def test_invalid_graph_type(self):
        # Test invalid graph type in _plot_iv_report
        iv_result = iv_report(
            self.class_data,
            activity_col="Activity",
            id_col="ID",
            scoring_list=["accuracy"],
        )
        with self.assertRaises(ValueError):
            _plot_iv_report(
                report_df=iv_result, scoring_list=["accuracy"], graph_type="invalid"
            )

    def test_invalid_select_method(self):
        # Test iv_report with an invalid method
        with self.assertRaises(ValueError):
            iv_report(
                self.class_data,
                activity_col="Activity",
                id_col="ID",
                select_method=["InvalidMethod"],
            )

    def test_plot_iv_report_bar(self):
        iv_result = iv_report(
            self.class_data,
            activity_col="Activity",
            id_col="ID",
            scoring_list=["accuracy"],
        )
        _plot_iv_report(
            report_df=iv_result, scoring_list=["accuracy"], graph_type="bar"
        )
        # Ensure no exception occurs when plotting

    def test_plot_iv_report_save_fig(self):
        # Test _plot_iv_report with save_fig=True
        iv_result = iv_report(
            self.class_data,
            activity_col="Activity",
            id_col="ID",
            scoring_list=["accuracy"],
        )
        _plot_iv_report(
            report_df=iv_result,
            scoring_list=["accuracy"],
            save_fig=True,
            fig_name="test_iv_graph",
            save_dir="test_dir",
        )
        # Ensure the figure file is saved
        self.assertTrue(os.path.exists("test_dir/test_iv_graph_accuracy_box.png"))
        # Cleanup created file
        os.remove("test_dir/test_iv_graph_accuracy_box.png")
        os.rmdir("test_dir")


if __name__ == "__main__":
    unittest.main()
