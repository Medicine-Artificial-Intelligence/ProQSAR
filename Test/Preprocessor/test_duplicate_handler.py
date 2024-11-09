import unittest
import pandas as pd
import numpy as np
import shutil
import os
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
        self.save_dir = "temp_save_dir"
        os.makedirs(self.save_dir, exist_ok=True)
        self.handler = DuplicateHandler(
            id_col="ID", activity_col="Activity", save_dir=self.save_dir
        )

    def tearDown(self):
        """
        Clean up the test environment by removing the temporary save directory.
        """
        shutil.rmtree(self.save_dir)

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
            save_dir=self.save_dir,
            save_trans_data=True,
        )
        handler.fit_transform(self.train_data)

        expected_filename = os.path.join(
            self.save_dir, f"{handler.trans_data_name}.csv"
        )
        self.assertTrue(
            os.path.exists(expected_filename),
            f"Expected file not found: {expected_filename}",
        )

        # Check that the file exists with the correct name and that it's a CSV
        self.assertTrue(expected_filename.endswith(".csv"))

        # Clean up: Remove the file after the test
        os.remove(expected_filename)

    def test_save_trans_data_name_with_existing_file(self):
        """
        Tests the save method with existing file.
        """
        handler = DuplicateHandler(
            id_col="ID",
            activity_col="Activity",
            save_dir=self.save_dir,
            save_trans_data=True,
        )
        existing_file = os.path.join(self.save_dir, f"{handler.trans_data_name}.csv")
        transformed_data = pd.DataFrame(
            {"id": [1, 2], "activity": ["A", "B"], "feature1": [1, 2]}
        )
        transformed_data.to_csv(existing_file, index=False)

        handler.fit_transform(self.train_data)

        # Check that the file is saved with the updated name (e.g., test_trans_data (1).csv)
        expected_filename = os.path.join(
            self.save_dir, f"{handler.trans_data_name} (1).csv"
        )
        self.assertTrue(
            os.path.exists(expected_filename),
            f"Expected file not found: {expected_filename}",
        )

        # Check that the file exists with the correct name and that it's a CSV
        self.assertTrue(expected_filename.endswith(".csv"))

        # Clean up: Remove the files after the test
        os.remove(existing_file)
        os.remove(expected_filename)


if __name__ == "__main__":
    unittest.main()
