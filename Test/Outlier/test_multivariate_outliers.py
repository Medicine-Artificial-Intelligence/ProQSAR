import unittest
import pandas as pd
import numpy as np
import os
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

        self.save_dir = "test_model_dir"
        self.handler = MultivariateOutliersHandler(
            id_col="ID", activity_col="Activity", save_dir=self.save_dir
        )

    def tearDown(self):
        """
        Clean up any files created during testing.
        """
        if os.path.exists(self.save_dir):
            for file in os.listdir(self.save_dir):
                os.remove(os.path.join(self.save_dir, file))
            os.rmdir(self.save_dir)

    def test_fit(self):
        """
        Test the fit method of the MultivariateOutliersHandler.
        """
        self.handler.fit(self.data)
        self.assertIsNotNone(self.handler.model)

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

    def test_static_transform_without_model(self):
        """
        Test the static_transform method without a saved model.
        """
        with self.assertRaises(NotFittedError):
            MultivariateOutliersHandler.static_transform(self.data, "no_model")

    def test_static_transform_with_model(self):
        """
        Test the static_transform method with a saved model.
        """
        self.handler.fit(self.data)
        transformed_data = MultivariateOutliersHandler.static_transform(
            self.data, self.save_dir
        )
        self.assertNotEqual(transformed_data.shape[0], self.data.shape[0])

    def test_compare_multivariate_methods(self):
        """
        Test the compare_multivariate_methods method of the MultivariateOutliersHandler.
        """
        comparison_table = MultivariateOutliersHandler.compare_multivariate_methods(
            self.data
        )
        self.assertEqual(comparison_table.shape[0], 5)


if __name__ == "__main__":
    unittest.main()
