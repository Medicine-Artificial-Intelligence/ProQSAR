import unittest
import pandas as pd
import numpy as np
import pickle
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

        with open(f"{self.save_dir}/cols_to_exclude.pkl", "rb") as file:
            cols_to_exclude = pickle.load(file)
        with open(f"{self.save_dir}/dup_cols.pkl", "rb") as file:
            dup_cols = pickle.load(file)

        self.assertEqual(cols_to_exclude, ["ID", "Activity"])
        self.assertEqual(dup_cols, ["Feature2", "Feature6"])

    def test_transform(self):
        """
        Test the transform method of DuplicateHandler.
        """
        self.handler.fit(self.train_data)
        transformed_test_data = self.handler.transform(self.test_data, self.save_dir)

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

    def test_transform_without_fit(self):
        """
        Tests the transform method without fitting first.
        """
        with self.assertRaises(FileNotFoundError):
            self.handler.transform(self.test_data, self.save_dir)


if __name__ == "__main__":
    unittest.main()
