import unittest
import pandas as pd
import numpy as np
import os
import shutil
from sklearn.exceptions import NotFittedError
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
        self.save_dir = "test_outlier_handler"
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)

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
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)

    def test_fit(self):
        """
        Test the `fit` method of KBinHandler.

        Verifies that the handler correctly identifies bad features and
        that the necessary files are saved to disk.
        """
        handler = KBinHandler(
            id_col="ID", activity_col="Activity", n_bins=3, save_dir="test_dir"
        )
        handler.fit(self.data)

        self.assertEqual(handler.bad, ["Feature2", "Feature3", "Feature4"])

        self.assertTrue(os.path.exists("test_dir/bad_features.pkl"))
        self.assertTrue(os.path.exists("test_dir/kbin.pkl"))

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
        )
        handler.fit(self.data)
        transformed_data = handler.transform(self.data)

        self.assertEqual(transformed_data.shape[1], 9)

    def test_static_transform(self):
        """
        Test the `static_transform` method of KBinHandler.

        Verifies that static transformation works as expected when a model has been fitted
        and saved. Checks the shape of the transformed data.
        """
        handler = KBinHandler(id_col="ID", activity_col="Activity", save_dir="test_dir")
        transformed_data = handler.fit(self.data)
        transformed_data = KBinHandler.static_transform(self.data, "test_dir")

        self.assertEqual(transformed_data.shape[1], 9)

    def test_fit_transform(self):
        """
        Test the `fit_transform` method of KBinHandler.

        Verifies that the handler can fit the model and transform the data in one step.
        Checks the shape of the transformed data.
        """
        handler = KBinHandler(id_col="ID", activity_col="Activity", n_bins=3)
        transformed_data = handler.fit_transform(self.data)

        self.assertEqual(transformed_data.shape[1], 9)

    def test_static_transform_no_model(self):
        """
        Test static transformation when no model has been fitted.

        Verifies that a NotFittedError is raised when attempting static transformation
        without a saved model.
        """

        with self.assertRaises(NotFittedError):
            KBinHandler.static_transform(self.data, "test_dir_no_model")


if __name__ == "__main__":
    unittest.main()
