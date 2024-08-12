import unittest
import os
import shutil
import numpy as np
import pandas as pd
from ProQSAR.Preprocessor.missing_handler import MissingHandler


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
        self.save_dir = "temp_save_dir"
        os.makedirs(self.save_dir, exist_ok=True)

    def tearDown(self):
        """
        Cleans up the test environment after each test method.
        """
        shutil.rmtree(self.save_dir)

    def test_fit(self):
        """
        Tests the fit method of MissingHandler.
        """
        handler = MissingHandler(
            id_col="ID", activity_col="Activity", save_dir=self.save_dir
        )
        handler.fit(self.train_data)
        self.assertTrue(os.path.exists(f"{self.save_dir}/binary_imputer.pkl"))
        self.assertTrue(os.path.exists(f"{self.save_dir}/binary_cols.pkl"))
        self.assertTrue(os.path.exists(f"{self.save_dir}/non_binary_imputer.pkl"))
        self.assertTrue(os.path.exists(f"{self.save_dir}/columns_to_exclude.pkl"))
        self.assertTrue(os.path.exists(f"{self.save_dir}/drop_cols.pkl"))

    def test_transform(self):
        """
        Tests the transform method of MissingHandler.
        """
        handler = MissingHandler(
            id_col="ID", activity_col="Activity", save_dir=self.save_dir
        )
        handler.fit(self.train_data)
        imputed_test_data = handler.transform(self.test_data)
        self.assertFalse(imputed_test_data.isnull().any().any())

    def test_fit_transform(self):
        """
        Tests the fit_transform method with default (mean) imputation strategy.
        """
        handler = MissingHandler(
            id_col="ID", activity_col="Activity", save_dir=self.save_dir
        )
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
            save_dir=self.save_dir,
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
            save_dir=self.save_dir,
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
            save_dir=self.save_dir,
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
            save_dir=self.save_dir,
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
            save_dir=self.save_dir,
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

        handler = MissingHandler(
            id_col="ID", activity_col="Activity", save_dir=self.save_dir
        )

        imputed_train_data = handler.fit_transform(train_no_binary)

        self.assertFalse(imputed_train_data.isnull().any().any())

    def test_only_binary_columns(self):
        """
        Tests the fit_transform method when there are only binary columns.
        """
        non_binary_cols = [f"Feature{i}" for i in range(6, 11)]
        train_only_binary = self.train_data.drop(columns=non_binary_cols)

        handler = MissingHandler(
            id_col="ID", activity_col="Activity", save_dir=self.save_dir
        )

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
            save_dir=self.save_dir,
        )
        with self.assertRaises(ValueError):
            handler.fit(self.train_data)

    def test_transform_without_fit(self):
        """
        Tests the transform method without fitting the imputation models first.
        """
        handler = MissingHandler(
            id_col="ID", activity_col="Activity", save_dir=self.save_dir
        )
        with self.assertRaises(FileNotFoundError):
            handler.transform(self.test_data)

    def test_static_transform(self):
        """
        Tests the static_transform method using saved imputers.
        """
        handler = MissingHandler(
            id_col="ID", activity_col="Activity", save_dir=self.save_dir
        )
        handler.fit(self.train_data)
        imputed_test_data = MissingHandler.static_transform(
            self.test_data, self.save_dir
        )
        self.assertFalse(imputed_test_data.isnull().any().any())


if __name__ == "__main__":
    unittest.main()
