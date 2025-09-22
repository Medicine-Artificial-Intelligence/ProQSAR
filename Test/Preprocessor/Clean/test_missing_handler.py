import unittest
import os
import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory
from sklearn.exceptions import NotFittedError
from proqsar.Preprocessor.Clean.missing_handler import MissingHandler


def create_sample_data() -> pd.DataFrame:
    """
    Creates a sample DataFrame for testing purposes with missing values introduced.

    Returns:
    - pd.DataFrame: The generated DataFrame with missing values.
    """
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "ID": range(1, 21),
            "Activity": np.random.rand(20),
            "Feature1": np.random.choice([0, 1], 20),
            "Feature2": np.random.choice([0, 1], 20),
            "Feature3": np.random.choice([0, 1], 20),
            "Feature4": np.random.choice([0, 1], 20),
            "Feature5": np.random.choice([0, 1], 20),
            "Feature6": np.random.rand(20),
            "Feature7": np.random.rand(20),
            "Feature8": np.random.rand(20),
            "Feature9": np.random.rand(20),
            "Feature10": np.random.rand(20),
        }
    )

    # Introduce missing values
    missing_rates = {
        "Feature1": 0.10,
        "Feature2": 0.20,
        "Feature3": 0.30,
        "Feature4": 0.40,
        "Feature5": 0.50,
        "Feature6": 0.10,
        "Feature7": 0.20,
        "Feature8": 0.30,
        "Feature9": 0.40,
        "Feature10": 0.50,
    }

    for feature, rate in missing_rates.items():
        n_missing = int(rate * len(data))
        missing_indices = np.random.choice(data.index, n_missing, replace=False)
        data.loc[missing_indices, feature] = np.nan

    return data


class TestMissingHandler(unittest.TestCase):

    def setUp(self):
        """
        Sets up the test environment before each test method.
        """
        self.train_data = create_sample_data()
        self.test_data = create_sample_data()
        self.temp_dir = TemporaryDirectory()

    def tearDown(self):
        """
        Cleans up the test environment after each test method.
        """
        self.temp_dir.cleanup()

    def test_fit(self):
        """
        Tests the fit method of MissingHandler.
        """
        handler = MissingHandler(
            id_col="ID",
            activity_col="Activity",
        ).fit(self.train_data)
        self.assertIsNotNone(handler)

    def test_transform(self):
        """
        Tests the transform method of MissingHandler.
        """
        handler = MissingHandler(id_col="ID", activity_col="Activity")
        handler.fit(self.train_data)
        imputed_test_data = handler.transform(self.test_data)
        self.assertFalse(imputed_test_data.isnull().any().any())

    def test_fit_transform(self):
        """
        Tests the fit_transform method with default (mean) imputation strategy.
        """
        handler = MissingHandler(id_col="ID", activity_col="Activity")
        imputed_train_data = handler.fit_transform(self.train_data)
        self.assertFalse(imputed_train_data.isnull().any().any())

    def test_knn_imputation_strategy(self):
        """
        Tests the fit_transform method with KNN imputation strategy.
        """
        handler = MissingHandler(
            id_col="ID",
            activity_col="Activity",
            imputation_strategy="knn",
            n_neighbors=3,
        )
        imputed_train_data = handler.fit_transform(self.train_data)
        self.assertFalse(imputed_train_data.isnull().any().any())

    def test_median_imputation_strategy(self):
        """
        Tests the fit_transform method with median imputation strategy.
        """
        handler = MissingHandler(
            id_col="ID",
            activity_col="Activity",
            imputation_strategy="median",
        )
        imputed_train_data = handler.fit_transform(self.train_data)
        self.assertFalse(imputed_train_data.isnull().any().any())

    def test_mode_imputation_strategy(self):
        """
        Tests the fit_transform method with mode imputation strategy.
        """
        handler = MissingHandler(
            id_col="ID",
            activity_col="Activity",
            imputation_strategy="mode",
        )
        imputed_train_data = handler.fit_transform(self.train_data)
        self.assertFalse(imputed_train_data.isnull().any().any())

    def test_mice_imputation_strategy(self):
        """
        Tests the fit_transform method with MICE imputation strategy.
        """
        handler = MissingHandler(
            id_col="ID",
            activity_col="Activity",
            imputation_strategy="mice",
        )
        imputed_train_data = handler.fit_transform(self.train_data)
        self.assertFalse(imputed_train_data.isnull().any().any())

    def test_dropping_high_missing_columns(self):
        """
        Tests that columns with a high percentage of missing values are dropped.
        """
        handler = MissingHandler(
            id_col="ID",
            activity_col="Activity",
            missing_thresh=40,
        )
        imputed_train_data = handler.fit_transform(self.train_data)
        imputed_test_data = handler.transform(self.test_data)

        self.assertEqual(len(imputed_train_data.columns), 10)
        self.assertEqual(len(imputed_test_data.columns), 10)
        self.assertNotIn("Feature5", imputed_train_data.columns)
        self.assertNotIn("Feature10", imputed_train_data.columns)

    def test_no_binary_columns(self):
        """
        Tests the fit_transform method when there are no binary columns.
        """
        binary_cols = [f"Feature{i}" for i in range(1, 6)]
        train_no_binary = self.train_data.drop(columns=binary_cols)

        handler = MissingHandler(id_col="ID", activity_col="Activity")

        imputed_train_data = handler.fit_transform(train_no_binary)

        self.assertFalse(imputed_train_data.isnull().any().any())

    def test_only_binary_columns(self):
        """
        Tests the fit_transform method when there are only binary columns.
        """
        non_binary_cols = [f"Feature{i}" for i in range(6, 11)]
        train_only_binary = self.train_data.drop(columns=non_binary_cols)

        handler = MissingHandler(id_col="ID", activity_col="Activity")

        imputed_train_data = handler.fit_transform(train_only_binary)

        self.assertFalse(imputed_train_data.isnull().any().any())

    def test_fit_unsupported_imputer(self):
        """
        Tests the fit method with an unsupported imputation strategy.
        """
        handler = MissingHandler(
            id_col="ID",
            activity_col="Activity",
            imputation_strategy="unsupported_imputer",
        )
        with self.assertRaises(ValueError):
            handler.fit(self.train_data)

    def test_transform_without_fit(self):
        """
        Tests the transform method without fitting the imputation models first.
        """
        handler = MissingHandler(id_col="ID", activity_col="Activity")
        with self.assertRaises(NotFittedError):
            handler.transform(self.test_data)

    def test_save_trans_data_name_no_file_exists(self):
        """
        Tests the save method with no file exists.
        """
        handler = MissingHandler(
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
        handler = MissingHandler(
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


if __name__ == "__main__":
    unittest.main()
