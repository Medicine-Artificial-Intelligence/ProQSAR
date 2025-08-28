import unittest
import pandas as pd
import numpy as np
import os
from tempfile import TemporaryDirectory
from sklearn.datasets import make_blobs
from sklearn.exceptions import NotFittedError
from ProQSAR.Outlier.multivariate_outliers import MultivariateOutliersHandler


class TestMultivariateOutliersHandler(unittest.TestCase):
    """
    Test suite for the MultivariateOutliersHandler class.
    """

    def setUp(self):
        """
        Set up the test data and the MultivariateOutliersHandler instance for testing.
        """
        # Create a dataset without outliers
        self.data_no_outlier, _ = make_blobs(
            n_samples=95, centers=1, n_features=5, random_state=42
        )
        self.data_no_outlier = pd.DataFrame(
            self.data_no_outlier, columns=[f"feature_{i}" for i in range(5)]
        )

        # Introduce multivariate outliers by modifying several feature combinations
        np.random.seed(42)
        outliers = self.data_no_outlier.sample(n=5)
        outliers.iloc[:, 0] += np.random.uniform(
            100, 200, size=5
        )  # Add large values to the first feature
        outliers.iloc[:, 1] += np.random.uniform(
            100, 200, size=5
        )  # Add large values to the second feature
        self.data = pd.concat([self.data_no_outlier, outliers], ignore_index=True)

        # Add binary and ID columns
        self.data["ID"] = range(1, len(self.data) + 1)
        self.data["Activity"] = np.random.choice([0, 1], size=len(self.data))

        self.handler = MultivariateOutliersHandler(id_col="ID", activity_col="Activity")
        self.temp_dir = TemporaryDirectory()
        self.handler.save_dir = self.temp_dir.name

    def tearDown(self):
        """
        Clean up any files created during testing.
        """
        self.temp_dir.cleanup()

    def test_fit(self):
        """
        Test the fit method of the MultivariateOutliersHandler.
        """
        self.handler.fit(self.data)
        self.assertIsNotNone(self.handler.multi_outlier_handler)

    def test_transform_without_fit(self):
        """
        Test the transform method without fitting the model.
        """
        with self.assertRaises(NotFittedError):
            self.handler.transform(self.data)

    def test_fit_transform(self):
        """
        Test the fit_transform method of the MultivariateOutliersHandler.
        """
        transformed_data = self.handler.fit_transform(self.data)
        self.assertNotEqual(transformed_data.shape[0], self.data.shape[0])

    def test_compare_multivariate_methods(self):
        """
        Test the compare_multivariate_methods method of the MultivariateOutliersHandler.
        """
        comparison_table1 = MultivariateOutliersHandler.compare_multivariate_methods(
            self.data, activity_col="Activity", id_col="ID"
        )
        comparison_table2 = MultivariateOutliersHandler.compare_multivariate_methods(
            data1=self.data, data2=self.data, activity_col="Activity", id_col="ID"
        )
        self.assertEqual(comparison_table1.shape[0], 5)
        self.assertEqual(comparison_table2.shape[0], 10)

    def test_save_method(self):
        """
        Test the save method functionality by ensuring the model is saved to disk.
        """
        self.handler.save_method = True
        self.handler.fit(self.data)
        model_path = os.path.join(self.temp_dir.name, "multi_outlier_handler.pkl")
        self.assertTrue(os.path.exists(model_path))

    def test_transform_data_save(self):
        """
        Test the save_trans_data option to ensure transformed data is saved.
        """
        self.handler.save_trans_data = True
        self.handler.fit(self.data)
        transformed_data = self.handler.transform(self.data)
        transformed_data_path = os.path.join(
            self.temp_dir.name, f"{self.handler.trans_data_name}.csv"
        )
        self.assertTrue(os.path.exists(transformed_data_path))
        self.assertEqual(
            pd.read_csv(transformed_data_path).shape[0], transformed_data.shape[0]
        )

    def test_invalid_method(self):
        """
        Test that an unsupported method raises a ValueError.
        """
        self.handler.select_method = "UnsupportedMethod"
        with self.assertRaises(ValueError):
            self.handler.fit(self.data)

    def test_local_outlier_factor(self):
        """
        Test LocalOutlierFactor.
        """
        self.handler.select_method = "LocalOutlierFactor"
        self.handler.fit(self.data)
        transformed_data = self.handler.transform(self.data)
        self.assertIsNotNone(transformed_data)

    def test_missing_columns_handling(self):
        """
        Test that missing ID and activity columns do not cause an error.
        """
        handler_no_cols = MultivariateOutliersHandler(select_method="IsolationForest")
        handler_no_cols.fit(self.data.drop(columns=["ID", "Activity"]))
        self.assertIsNotNone(handler_no_cols.multi_outlier_handler)


if __name__ == "__main__":
    unittest.main()
