import os
import unittest
import pandas as pd
import numpy as np
import matplotlib
from tempfile import TemporaryDirectory
from sklearn.datasets import make_classification, make_regression
from ProQSAR.Analysis.statistical_analysis import StatisticalAnalysis
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


class TestStatisticalAnalysis(unittest.TestCase):

    def setUp(self):
        self.class_data = create_classification_data()
        self.reg_data = create_regression_data()
        self.cv_class = ModelValidation.cross_validation_report(
            self.class_data, activity_col="Activity", id_col="ID"
        )

        self.cv_reg = ModelValidation.cross_validation_report(
            self.reg_data, activity_col="Activity", id_col="ID"
        )
        self.temp_dir = TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_extract_scoring_dfs(self):
        scoring_dfs, scoring_list, method_list = (
            StatisticalAnalysis.extract_scoring_dfs(
                self.cv_class,
                scoring_list="accuracy",
                method_list=["KNeighborsClassifier", "SVC", "ExtraTreesClassifier"],
            )
        )
        self.assertEqual(scoring_list, ["accuracy"])
        self.assertListEqual(
            method_list, ["KNeighborsClassifier", "SVC", "ExtraTreesClassifier"]
        )
        self.assertFalse(scoring_dfs.empty)

    def test_check_variance_homogeneity(self):
        result_df = StatisticalAnalysis.check_variance_homogeneity(
            self.cv_class,
            scoring_list="accuracy",
            save_csv=True,
            save_dir=self.temp_dir.name,
            csv_name="check_variance_homogeneity",
        )
        self.assertIn("variance_fold_difference", result_df.columns)
        self.assertIn("p_value", result_df.columns)
        self.assertFalse(result_df.empty)
        self.assertTrue(
            os.path.exists(f"{self.temp_dir.name}/check_variance_homogeneity.csv")
        )

    def test_check_normality(self):
        with self.assertRaises(ValueError):
            StatisticalAnalysis.check_normality(self.cv_class, scoring_list="invalid")

        StatisticalAnalysis.check_normality(
            self.cv_class,
            scoring_list=["accuracy", "f1"],
            save_fig=True,
            save_dir=self.temp_dir.name,
            fig_name="check_normality",
        )

        self.assertTrue(os.path.exists(f"{self.temp_dir.name}/check_normality.pdf"))

    def test_anova_test(self):
        with self.assertRaises(ValueError) as context:
            StatisticalAnalysis.test(
                self.cv_reg, scoring_list="r2", select_test="invalid_test"
            )
        self.assertIn("Unsupported test", str(context.exception))

        StatisticalAnalysis.test(
            self.cv_reg,
            scoring_list="r2",
            select_test="AnovaRM",
            save_fig=True,
            save_dir=self.temp_dir.name,
            fig_name="AnovaRM",
        )
        self.assertTrue(os.path.exists(f"{self.temp_dir.name}/AnovaRM.pdf"))

    def test_friedman_test(self):
        StatisticalAnalysis.test(
            self.cv_reg,
            scoring_list="r2",
            select_test="friedman",
            save_fig=True,
            save_dir=self.temp_dir.name,
            fig_name="friedman",
        )
        self.assertTrue(os.path.exists(f"{self.temp_dir.name}/friedman.pdf"))

    def test_posthoc_conover_friedman(self):
        pc_results, rank_results = StatisticalAnalysis.posthoc_conover_friedman(
            self.cv_reg,
            scoring_list="r2",
            save_fig=True,
            save_result=True,
            save_dir=self.temp_dir.name,
        )

        self.assertIn("r2", pc_results)
        self.assertIn("r2", rank_results)
        self.assertTrue(os.path.exists(f"{self.temp_dir.name}/cofried_sign_plot.pdf"))
        self.assertTrue(os.path.exists(f"{self.temp_dir.name}/cofried_ccd.pdf"))
        self.assertTrue(os.path.exists(f"{self.temp_dir.name}/cofried_pc_r2.csv"))

    def test_posthoc_tukeyhsd(self):
        tukey_results = StatisticalAnalysis.posthoc_tukeyhsd(
            self.cv_class,
            scoring_list=["f1", "accuracy"],
            save_fig=True,
            save_result=True,
            save_dir=self.temp_dir.name,
        )

        self.assertIn("f1", tukey_results)
        self.assertIn("df_means", tukey_results["f1"])
        self.assertIn("result_tab", tukey_results["f1"])
        self.assertTrue(os.path.exists(f"{self.temp_dir.name}/tukey_result_tab_f1.csv"))
        self.assertTrue(
            os.path.exists(f"{self.temp_dir.name}/tukey_result_tab_accuracy.csv")
        )
        self.assertTrue(os.path.exists(f"{self.temp_dir.name}/tukey_mcs.pdf"))
        self.assertTrue(os.path.exists(f"{self.temp_dir.name}/tukey_ci.pdf"))


if __name__ == "__main__":
    unittest.main()
