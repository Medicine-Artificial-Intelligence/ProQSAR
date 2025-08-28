import unittest
import pandas as pd
import numpy as np
import os
from tempfile import TemporaryDirectory
from ProQSAR.Preprocessor.duplicate_handler import DuplicateHandler


def create_sample_data() -> pd.DataFrame:
    """
    Creates a sample DataFrame with duplicate rows and columns.

    Returns:
    - pd.DataFrame: The sample data with duplicates.
    """
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "ID": range(1, 6),
            "Activity": np.random.rand(5),
            "Feature1": np.random.choice([0, 1], 5),
            "Feature2": np.random.choice([0, 1], 5),
            "Feature3": np.random.choice([0, 1], 5),
            "Feature4": np.random.rand(5),
            "Feature5": np.random.rand(5),
            "Feature6": np.random.rand(5),
        }
    )
    # Add duplicate rows
    data = pd.concat([data, data.iloc[0:2]], ignore_index=True)
    # Add duplicate columns
    data["Feature1"] = data["Feature2"]
    data["Feature5"] = data["Feature6"]

    return data


class TestDuplicateHandler(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment, creating sample train and test data,
        and initializing the DuplicateHandler instance.
        """
        self.train_data = create_sample_data()
        self.test_data = create_sample_data()
        self.temp_dir = TemporaryDirectory()
        self.handler = DuplicateHandler(
            id_col="ID", activity_col="Activity", save_dir=self.temp_dir.name
        )

    def tearDown(self):
        """
        Clean up the test environment by removing the temporary save directory.
        """
        self.temp_dir.cleanup()

    def test_fit(self):
        """
        Test the fit method of DuplicateHandler.
        """
        self.handler.fit(self.train_data)
        self.assertEqual(self.handler.dup_cols, ["Feature2", "Feature6"])

    def test_transform(self):
        """
        Test the transform method of DuplicateHandler.
        """
        self.handler.fit(self.train_data)
        transformed_test_data = self.handler.transform(self.test_data)

        self.assertNotIn("Feature2", transformed_test_data.columns)
        self.assertNotIn("Feature6", transformed_test_data.columns)
        self.assertEqual(len(transformed_test_data), 5)

    def test_fit_transform(self):
        """
        Test the fit_transform method of DuplicateHandler.
        """
        transformed_train_data = self.handler.fit_transform(self.train_data)

        self.assertNotIn("Feature2", transformed_train_data.columns)
        self.assertNotIn("Feature6", transformed_train_data.columns)
        self.assertEqual(len(transformed_train_data), 5)

    def test_no_duplicates(self):
        """
        Test the DuplicateHandler with data that has no duplicates.
        """
        data_no_duplicates = self.train_data.drop(
            index=[5, 6], columns=["Feature2", "Feature6"]
        )
        transformed_data = self.handler.fit_transform(data_no_duplicates)

        self.assertEqual(transformed_data.shape, data_no_duplicates.shape)

    def test_save_trans_data_name_no_file_exists(self):
        """
        Tests the save method with no file exists.
        """
        handler = DuplicateHandler(
            id_col="ID",
            activity_col="Activity",
            save_dir=self.temp_dir.name,
            save_trans_data=True,
        )
        handler.fit_transform(self.train_data)

        expected_filename = os.path.join(
            self.temp_dir.name, f"{handler.trans_data_name}.csv"
        )
        self.assertTrue(
            os.path.exists(expected_filename),
            f"Expected file not found: {expected_filename}",
        )

        # Check that the file exists with the correct name and that it's a CSV
        self.assertTrue(expected_filename.endswith(".csv"))

    def test_save_trans_data_name_with_existing_file(self):
        """
        Tests the save method with existing file.
        """
        handler = DuplicateHandler(
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

        handler.fit_transform(self.train_data)

        # Check that the file is saved with the updated name (e.g., test_trans_data (1).csv)
        expected_filename = os.path.join(
            self.temp_dir.name, f"{handler.trans_data_name} (1).csv"
        )
        self.assertTrue(
            os.path.exists(expected_filename),
            f"Expected file not found: {expected_filename}",
        )

        # Check that the file exists with the correct name and that it's a CSV
        self.assertTrue(expected_filename.endswith(".csv"))

    def test_deactivate_returns_unmodified(self):
        h = DuplicateHandler(id_col="ID", activity_col="Activity", deactivate=True)
        out = h.fit_transform(self.train_data)
        self.assertTrue(out.equals(self.train_data))

    def test_transform_before_fit_handles_missing_dup_cols(self):
        # If transform is called before fit, dup_cols is None; ensure it doesn't crash
        h = DuplicateHandler(id_col="ID", activity_col="Activity")
        out = h.transform(self.train_data)
        self.assertIsInstance(out, pd.DataFrame)
        self.assertEqual(len(out), 5)

    def test_disable_cols_or_rows_flags(self):
        h = DuplicateHandler(id_col="ID", activity_col="Activity", cols=False, rows=True)
        h.fit(self.train_data)
        # Columns should not be removed, only rows
        out = h.transform(self.train_data)
        self.assertIn("Feature2", out.columns)
        self.assertIn("Feature6", out.columns)
        self.assertEqual(len(out), 5)  # duplicate rows removed
        # Now disable rows removal
        h2 = DuplicateHandler(id_col="ID", activity_col="Activity", cols=True, rows=False)
        h2.fit(self.train_data)
        out2 = h2.transform(self.train_data)
        self.assertNotIn("Feature2", out2.columns)
        self.assertNotIn("Feature6", out2.columns)
        self.assertEqual(len(out2), 7)  # rows kept

    def test_save_method_persists_model(self):
        h = DuplicateHandler(id_col="ID", activity_col="Activity", save_method=True, save_dir=self.temp_dir.name)
        h.fit(self.train_data)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir.name, "duplicate_handler.pkl")))


if __name__ == "__main__":
    unittest.main()
