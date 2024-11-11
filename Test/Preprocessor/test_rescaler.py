import unittest
import pandas as pd
import numpy as np
import os
from tempfile import TemporaryDirectory
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    FunctionTransformer,
)
from ProQSAR.Preprocessor.rescaler import Rescaler


def create_sample_data() -> pd.DataFrame:
    """
    Creates a sample DataFrame for testing.

    Returns
    -------
    pd.DataFrame
        A sample DataFrame with random and sequential data.
    """
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "ID": range(1, 21),
            "Activity": np.random.rand(20),
            "Feature1": range(20),
            "Feature2": range(20, 40),
            "Feature3": np.random.choice([1, 2], 20),
            "Feature4": np.random.choice([0, 1], 20),
            "Feature5": np.random.choice([0, 1], 20),
        }
    )
    return data


class TestRescaler(unittest.TestCase):

    def setUp(self):
        """
        Sets up the test environment before each test.
        """
        self.train_data = create_sample_data()
        self.test_data = create_sample_data()
        self.temp_dir = TemporaryDirectory()
        self.rescaler = Rescaler(
            id_col="ID",
            activity_col="Activity",
            save_dir=self.temp_dir.name,
            save_method=True,
        )

    def tearDown(self):
        """
        Cleans up the test environment after each test.
        """
        self.temp_dir.cleanup()

    def test_get_scaler(self):
        """
        Tests the _get_scaler method for various scaler methods.
        """
        self.assertIsInstance(self.rescaler._get_scaler("MinMaxScaler"), MinMaxScaler)
        self.assertIsInstance(
            self.rescaler._get_scaler("StandardScaler"), StandardScaler
        )
        self.assertIsInstance(self.rescaler._get_scaler("RobustScaler"), RobustScaler)
        self.assertIsInstance(self.rescaler._get_scaler("None"), FunctionTransformer)
        with self.assertRaises(ValueError):
            self.rescaler._get_scaler("UnsupportedScaler")

    def test_fit(self):
        """
        Tests the fit method to ensure the scaler is fitted and saved correctly.
        """
        self.rescaler.fit(self.train_data)

        self.assertTrue(os.path.exists(f"{self.temp_dir.name}/rescaler.pkl"))
        self.assertIn("Feature1", self.rescaler.non_binary_cols)
        self.assertIn("Feature2", self.rescaler.non_binary_cols)
        self.assertIn("Feature3", self.rescaler.non_binary_cols)

    def test_transform(self):
        """
        Tests the transform method to ensure data is transformed correctly.
        """
        self.rescaler.fit(self.train_data)
        transformed_test_data = self.rescaler.transform(self.test_data)
        self.assertFalse(transformed_test_data.equals(self.test_data))

    def test_fit_transform(self):
        """
        Tests the fit_transform method to ensure data is fitted and transformed correctly.
        """
        transformed_train_data = self.rescaler.fit_transform(self.train_data)
        self.assertFalse(transformed_train_data.equals(self.train_data))

    def test_only_binary_columns(self):
        """
        Tests the fit_transform method with only binary columns in the data.
        """
        data = self.train_data.drop(columns=["Feature1", "Feature2", "Feature3"])
        transformed_data = self.rescaler.fit_transform(data)
        self.assertTrue(transformed_data.equals(data))

    def test_no_binary_columns(self):
        """
        Tests the fit_transform method with no binary columns in the data.
        """
        data = self.train_data.drop(columns=["Feature4", "Feature5"])
        transformed_data = self.rescaler.fit_transform(data)
        self.assertFalse(transformed_data.equals(self.train_data))

    def test_save_trans_data_name_no_file_exists(self):
        """
        Tests the save method with no file exists.
        """
        rescaler = Rescaler(
            id_col="ID",
            activity_col="Activity",
            save_dir=self.temp_dir.name,
            save_trans_data=True,
        )
        rescaler.fit_transform(self.train_data)

        expected_filename = os.path.join(
            self.temp_dir.name, f"{rescaler.trans_data_name}.csv"
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
        rescaler = Rescaler(
            id_col="ID",
            activity_col="Activity",
            save_dir=self.temp_dir.name,
            save_trans_data=True,
        )
        existing_file = os.path.join(
            self.temp_dir.name, f"{rescaler.trans_data_name}.csv"
        )
        transformed_data = pd.DataFrame(
            {"id": [1, 2], "activity": ["A", "B"], "feature1": [1, 2]}
        )
        transformed_data.to_csv(existing_file, index=False)

        rescaler.fit_transform(self.train_data)

        # Check that the file is saved with the updated name (e.g., test_trans_data (1).csv)
        expected_filename = os.path.join(
            self.temp_dir.name, f"{rescaler.trans_data_name} (1).csv"
        )
        self.assertTrue(
            os.path.exists(expected_filename),
            f"Expected file not found: {expected_filename}",
        )

        # Check that the file exists with the correct name and that it's a CSV
        self.assertTrue(expected_filename.endswith(".csv"))


if __name__ == "__main__":
    unittest.main()
