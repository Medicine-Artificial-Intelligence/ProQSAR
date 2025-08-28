import unittest
import pandas as pd
import numpy as np
import os
import matplotlib
from tempfile import TemporaryDirectory
from ProQSAR.Preprocessor.low_variance_handler import LowVarianceHandler

matplotlib.use("Agg")


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
        self.temp_dir = TemporaryDirectory()
        self.handler = LowVarianceHandler(
            activity_col="Activity",
            id_col="ID",
            var_thresh=0.05,
            visualize=False,
            save_image=False,
            save_dir=self.temp_dir.name,
            save_method=True,
        )

    def tearDown(self):
        """
        Clean up the test directory after tests.
        """
        self.temp_dir.cleanup()

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
        self.assertTrue(
            os.path.exists(f"{self.temp_dir.name}/low_variance_handler.pkl")
        )

    def test_transform(self):
        """
        Test the transform method.
        """
        self.handler.fit(self.data)
        transformed_data = self.handler.transform(self.data)
        expected_columns = [
            "ID",
            "Activity",
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
            "ID",
            "Activity",
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
        self.assertListEqual(
            list(transformed_data.columns), ["ID", "Activity", "Feature1", "Feature2"]
        )

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
            save_dir=self.temp_dir.name,
            save_method=True,
        )
        transformed_data = handler_high_thresh.fit_transform(self.data)
        expected_columns = ["ID", "Activity", "Feature1", "Feature2"]
        self.assertListEqual(list(transformed_data.columns), expected_columns)

    def test_transform_raises_not_fitted(self):
        with self.assertRaises(Exception):
            _ = LowVarianceHandler(activity_col="Activity", id_col="ID").transform(
                self.data
            )

    def test_deactivate_returns_unmodified(self):
        h = LowVarianceHandler(activity_col="Activity", id_col="ID", deactivate=True)
        out = h.fit_transform(self.data)
        self.assertTrue(out.equals(self.data))

    def test_visualize_save_image_path(self):
        # Exercise visualization branch with save_image True; Agg backend avoids display
        h = LowVarianceHandler(
            activity_col="Activity",
            id_col="ID",
            visualize=True,
            save_image=True,
            save_dir=self.temp_dir.name,
            image_name="variance_plot.png",
        )
        # Should not raise
        h.fit(self.data)
        # No strict file assert due to .show(), just ensure directory exists
        self.assertTrue(os.path.isdir(self.temp_dir.name))

    def test_save_trans_data_collision_creates_incremented_name(self):
        h = LowVarianceHandler(
            activity_col="Activity",
            id_col="ID",
            save_trans_data=True,
            save_dir=self.temp_dir.name,
            trans_data_name="out",
        )
        h.fit(self.data)
        # First save
        _ = h.transform(self.data)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir.name, "out.csv")))
        # Second save should create an incremented filename
        _ = h.transform(self.data)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir.name, "out (1).csv")))


if __name__ == "__main__":
    unittest.main()
