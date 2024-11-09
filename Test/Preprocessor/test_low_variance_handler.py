import unittest
import pandas as pd
import numpy as np
import os
from ProQSAR.Preprocessor.low_variance_handler import LowVarianceHandler


class TestLowVarianceHandler(unittest.TestCase):

    def setUp(self):
        """
        Create sample data for testing.
        """
        np.random.seed(42)
        self.data = pd.DataFrame(
            {
                "ID": np.arange(1, 21),
                "Activity": np.random.rand(20) * 10,
                "Feature1": np.random.choice([0, 1], 20),
                "Feature2": np.random.choice([0, 1], 20),
                "Feature3": np.random.normal(0, np.sqrt(0.01), 20),
                "Feature4": np.random.normal(0, np.sqrt(0.01), 20),
                "Feature5": np.random.normal(0, np.sqrt(0.5), 20),
                "Feature6": np.random.normal(0, np.sqrt(0.8), 20),
                "Feature7": np.random.normal(0, np.sqrt(1.0), 20),
            }
        )
        self.handler = LowVarianceHandler(
            activity_col="Activity",
            id_col="ID",
            var_thresh=0.05,
            visualize=False,
            save_image=False,
            save_dir="test_dir",
        )

    def tearDown(self):
        """
        Clean up the test directory after tests.
        """
        if os.path.exists("test_dir"):
            for file in os.listdir("test_dir"):
                os.remove(os.path.join("test_dir", file))
            os.rmdir("test_dir")

    def test_variance_threshold_analysis(self):
        """
        Test the variance threshold analysis method.
        """
        self.handler.variance_threshold_analysis(
            self.data, "ID", "Activity", save_image=False
        )
        # No assertions needed, just ensure no exceptions are raised

    def test_select_features_by_variance(self):
        """
        Test the feature selection by variance threshold.
        """
        selected_features = self.handler.select_features_by_variance(
            self.data, "Activity", "ID", 0.05
        )
        expected_features = [
            "Activity",
            "ID",
            "Feature1",
            "Feature2",
            "Feature5",
            "Feature6",
            "Feature7",
        ]
        self.assertEqual(selected_features, expected_features)

    def test_fit(self):
        """
        Test the fit method.
        """
        self.handler.fit(self.data)
        self.assertTrue(os.path.exists("test_dir/low_variance_handler.pkl"))

    def test_transform(self):
        """
        Test the transform method.
        """
        self.handler.fit(self.data)
        transformed_data = self.handler.transform(self.data)
        expected_columns = [
            "Activity",
            "ID",
            "Feature1",
            "Feature2",
            "Feature5",
            "Feature6",
            "Feature7",
        ]
        self.assertListEqual(list(transformed_data.columns), expected_columns)

    def test_fit_transform(self):
        """
        Test the fit_transform method.
        """
        transformed_data = self.handler.fit_transform(self.data)
        expected_columns = [
            "Activity",
            "ID",
            "Feature1",
            "Feature2",
            "Feature5",
            "Feature6",
            "Feature7",
        ]
        self.assertListEqual(list(transformed_data.columns), expected_columns)

    def test_fit_transform_binary_only(self):
        """
        Test the feature selection by variance threshold with only binary features.
        """
        binary_columns = ["Activity", "ID", "Feature1", "Feature2"]
        binary_data = self.data[binary_columns]
        transformed_data = self.handler.fit_transform(binary_data)
        self.assertListEqual(list(transformed_data.columns), binary_columns)

    def test_fit_transform_no_non_binary_meeting_threshold(self):
        """
        Test the fit_transform method with no non-binary features meeting the threshold.
        """
        handler_high_thresh = LowVarianceHandler(
            activity_col="Activity",
            id_col="ID",
            var_thresh=5,  # High threshold to ensure no non-binary features meet it
            visualize=False,
            save_image=False,
            save_dir="test_dir",
        )
        transformed_data = handler_high_thresh.fit_transform(self.data)
        expected_columns = ["Activity", "ID", "Feature1", "Feature2"]
        self.assertListEqual(list(transformed_data.columns), expected_columns)


if __name__ == "__main__":
    unittest.main()
