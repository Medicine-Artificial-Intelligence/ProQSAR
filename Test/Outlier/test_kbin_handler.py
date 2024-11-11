import unittest
import pandas as pd
import numpy as np
import os
import shutil
from tempfile import TemporaryDirectory
from ProQSAR.Outlier.kbin_handler import KBinHandler


class TestKBinHandler(unittest.TestCase):
    """
    Unit tests for the KBinHandler class from the ProQSAR.Outlier.kbin_handler module.

    Tests the functionality of the KBinHandler, including:
    - Fitting the model and checking for bad features
    - Transforming data with the fitted model
    - Static transformation without fitting
    - Fit and transform in a single step
    - Handling cases where static transformation is attempted without a model
    """

    def setUp(self):
        """
        Set up the test environment by creating a temporary directory
        and initializing a DataFrame with some outliers.

        Creates a directory `test_outlier_handler` and populates it with a DataFrame.
        """
        self.temp_dir = TemporaryDirectory()

        np.random.seed(42)
        self.data = pd.DataFrame(
            {
                "ID": range(1, 11),
                "Activity": np.random.rand(10),
                "Binary1": np.random.choice([0, 1], size=10),
                "Binary2": np.random.choice([0, 1], size=10),
                "Feature1": np.random.normal(0, 1, 10),  # No outliers
                "Feature2": np.random.normal(0, 1, 10),  # Outliers
                "Feature3": np.random.normal(0, 1, 10),  # Outliers
                "Feature4": np.random.normal(0, 1, 10),  # Outliers
                "Feature5": np.random.normal(0, 1, 10),  # No outliers
            }
        )

        # Introduce some outliers
        self.data.loc[0, "Feature2"] = 10
        self.data.loc[1, "Feature3"] = -20
        self.data.loc[2, "Feature4"] = 10

    def tearDown(self):
        """
        Clean up after all tests are run by deleting the temporary directory.
        """
        self.temp_dir.cleanup()

    def test_fit(self):
        """
        Test the `fit` method of KBinHandler.

        Verifies that the handler correctly identifies bad features and
        that the necessary files are saved to disk.
        """
        handler = KBinHandler(
            id_col="ID",
            activity_col="Activity",
            n_bins=3,
            save_dir=self.temp_dir.name,
            save_method=True,
        )
        handler.fit(self.data)

        self.assertEqual(handler.bad, ["Feature2", "Feature3", "Feature4"])

        self.assertTrue(os.path.exists(f"{self.temp_dir.name}/kbin_handler.pkl"))

    def test_transform(self):
        """
        Test the `transform` method of KBinHandler.

        Verifies that the handler correctly transforms the data and the shape of
        the transformed data is as expected.
        """
        handler = KBinHandler(
            id_col="ID",
            activity_col="Activity",
            n_bins=3,
            encode="ordinal",
            strategy="uniform",
            save_method=False,
        )
        handler.fit(self.data)
        transformed_data = handler.transform(self.data)

        self.assertEqual(transformed_data.shape[1], 9)

    def test_fit_transform(self):
        """
        Test the `fit_transform` method of KBinHandler.

        Verifies that the handler can fit the model and transform the data in one step.
        Checks the shape of the transformed data.
        """
        handler = KBinHandler(
            id_col="ID", activity_col="Activity", n_bins=3, save_method=False
        )
        transformed_data = handler.fit_transform(self.data)

        self.assertEqual(transformed_data.shape[1], 9)

    def test_no_bad_features(self):
        """
        Test the fit method when there are no bad features in the data.
        """
        data_no_outliers = self.data.copy()
        data_no_outliers.drop(
            columns=["Feature2", "Feature3", "Feature4"], inplace=True
        )

        handler = KBinHandler(
            id_col="ID", activity_col="Activity", n_bins=3, save_method=False
        )
        handler.fit(data_no_outliers)
        self.assertEqual(handler.bad, [])
        self.assertIsNone(handler.kbin)

    def test_save_transformed_data(self):
        """
        Test saving transformed data when `save_trans_data` is enabled.
        """
        handler = KBinHandler(
            id_col="ID",
            activity_col="Activity",
            n_bins=3,
            save_dir=self.temp_dir.name,
            save_trans_data=True,
            trans_data_name="test_kbin_trans_data",
            save_method=True,
        )
        handler.fit_transform(self.data)
        self.assertTrue(
            os.path.exists(f"{self.temp_dir.name}/test_kbin_trans_data.csv")
        )

    def test_save_directory_creation(self):
        """
        Test that the `save_dir` is created if it doesn't exist.
        """
        non_existent_dir = "non_existent_dir"
        handler = KBinHandler(
            id_col="ID",
            activity_col="Activity",
            n_bins=3,
            save_dir=non_existent_dir,
            save_method=True,
        )
        handler.fit(self.data)
        self.assertTrue(os.path.exists(non_existent_dir))
        shutil.rmtree(non_existent_dir)

    def test_save_trans_data_name_no_file_exists(self):
        """
        Tests the save method with no file exists.
        """
        handler = KBinHandler(
            id_col="ID",
            activity_col="Activity",
            save_dir=self.temp_dir.name,
            save_trans_data=True,
        )
        handler.fit_transform(self.data)

        expected_filename = os.path.join(
            self.temp_dir.name, f"{handler.trans_data_name}.csv"
        )
        self.assertTrue(
            os.path.exists(expected_filename),
            f"Expected file not found: {expected_filename}",
        )

        self.assertTrue(expected_filename.endswith(".csv"))

    def test_save_trans_data_name_with_existing_file(self):
        """
        Tests the save method with existing file.
        """
        handler = KBinHandler(
            id_col="ID",
            activity_col="Activity",
            save_dir=self.temp_dir.name,
            save_trans_data=True,
        )
        existing_file = os.path.join(
            self.temp_dir.name, f"{handler.trans_data_name}.csv"
        )
        transformed_data = pd.DataFrame(
            {"id": [1, 2], "activity": ["A", "B"], "feature1": [1, 2]}
        )
        transformed_data.to_csv(existing_file, index=False)

        handler.fit_transform(self.data)

        expected_filename = os.path.join(
            self.temp_dir.name, f"{handler.trans_data_name} (1).csv"
        )
        self.assertTrue(
            os.path.exists(expected_filename),
            f"Expected file not found: {expected_filename}",
        )

        self.assertTrue(expected_filename.endswith(".csv"))


if __name__ == "__main__":
    unittest.main()
