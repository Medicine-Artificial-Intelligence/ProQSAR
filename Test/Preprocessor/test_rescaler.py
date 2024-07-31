import unittest
import pandas as pd
import numpy as np
import os
import shutil
import pickle
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
        self.save_dir = "test_scaler_dir"
        self.rescaler = Rescaler(
            id_col="ID", activity_col="Activity", save_dir=self.save_dir
        )

    def tearDown(self):
        """
        Cleans up the test environment after each test.
        """
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)

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

        self.assertTrue(os.path.exists(f"{self.save_dir}/scaler.pkl"))
        self.assertTrue(os.path.exists(f"{self.save_dir}/non_binary_cols.pkl"))

        with open(f"{self.save_dir}/non_binary_cols.pkl", "rb") as file:
            non_binary_cols = pickle.load(file)
            self.assertIn("Feature1", non_binary_cols)
            self.assertIn("Feature2", non_binary_cols)
            self.assertIn("Feature3", non_binary_cols)

    def test_transform(self):
        """
        Tests the transform method to ensure data is transformed correctly.
        """
        self.rescaler.fit(self.train_data)
        transformed_test_data = self.rescaler.transform(self.test_data, self.save_dir)
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

    def test_transform_without_fit(self):
        """
        Tests the transform method without fitting the scaler first.
        """
        with self.assertRaises(FileNotFoundError):
            self.rescaler.transform(self.test_data, self.save_dir)


if __name__ == "__main__":
    unittest.main()
