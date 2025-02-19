import os
import unittest
import pandas as pd
import matplotlib
from tempfile import TemporaryDirectory
from ProQSAR.Analysis.statistical_analysis import StatisticalAnalysis

matplotlib.use("Agg")


class TestStatisticalAnalysis(unittest.TestCase):

    def setUp(self):
        self.cv_class = pd.read_csv("Data/cv_class.csv")
        self.cv_reg = pd.read_csv("Data/cv_reg.csv")
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
