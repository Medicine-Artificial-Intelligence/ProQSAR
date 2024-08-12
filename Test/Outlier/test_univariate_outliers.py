import unittest
import pandas as pd
import numpy as np
import os
import shutil

from ProQSAR.Outlier.univariate_outliers import UnivariateOutliersHandler


class TestUnivariateOutliersHandler(unittest.TestCase):
    """
    Unit test case for the UnivariateOutliersHandler class.

    Tests various methods for handling outliers and transforming data using different strategies.
    """

    def setUp(self):
        """
        Set up the test environment before any test is run.
        Creates a sample dataframe and initializes necessary directories.
        """
        self.save_dir = "test_outlier_handler"
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)

        np.random.seed(0)

        self.data = pd.DataFrame(
            {
                "ID": range(1, 11),
                "Activity": np.random.rand(10),
                "Binary1": np.random.choice([0, 1], size=10),
                "Binary2": np.random.choice([0, 1], size=10),
                "Feature1": np.random.normal(0, 1, 10),  # No outliers
                "Feature2": np.random.normal(0, 1, 10),  # Outliers
                "Feature3": np.random.normal(0, 1, 10),  # Outliers
                "Feature4": np.random.normal(0, 1, 10),  # Outliers
                "Feature5": np.random.normal(0, 1, 10),  # No outliers
            }
        )

        # Introduce some outliers
        self.data.loc[0, "Feature2"] = 10
        self.data.loc[1, "Feature3"] = -20
        self.data.loc[2, "Feature4"] = 10

    def tearDown(self):
        """
        Clean up after all tests are run.
        Deletes the test directory created during setup.
        """
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)

    def test_iqr_method(self):
        """
        Test the 'iqr' method of outlier handling.
        """
        handler = UnivariateOutliersHandler(
            id_col="ID",
            activity_col="Activity",
            handling_method="iqr",
            save_dir=self.save_dir,
        )
        handler.fit(self.data)
        transformed_data = handler.transform(self.data)
        self.assertTrue(transformed_data["Feature2"].max() < 100)
        self.assertTrue(transformed_data["Feature3"].min() > -20)
        self.assertTrue(transformed_data["Feature4"].max() < 100)

    def test_winsorization_method(self):
        """
        Test the 'winsorization' method of outlier handling.
        """
        handler = UnivariateOutliersHandler(
            id_col="ID",
            activity_col="Activity",
            handling_method="winsorization",
            save_dir=self.save_dir,
        )
        handler.fit(self.data)
        transformed_data = handler.transform(self.data)
        self.assertTrue(transformed_data["Feature2"].max() < 10)
        self.assertTrue(transformed_data["Feature3"].min() > -10)
        self.assertTrue(transformed_data["Feature4"].max() < 10)

    def test_imputation_method(self) -> None:
        """
        Test the 'imputation' method of outlier handling, ensuring NaNs are imputed correctly.
        """
        handler = UnivariateOutliersHandler(
            id_col="ID",
            activity_col="Activity",
            handling_method="imputation",
            save_dir=self.save_dir,
        )

        # Use _impute_nan method directly
        _, bad = handler._feature_quality(self.data)
        iqr_thresholds = handler._iqr_threshold(self.data[bad])
        imputed_data = handler._impute_nan(self.data, iqr_thresholds)

        # Check if NaNs were correctly introduced
        self.assertTrue(imputed_data["Feature2"].isna().sum() > 0)
        self.assertTrue(imputed_data["Feature3"].isna().sum() > 0)
        self.assertTrue(imputed_data["Feature4"].isna().sum() > 0)

        # Verify the imputation (transform) method handles NaNs
        transformed_data = handler.fit_transform(self.data)

        # Check if NaNs are handled by MissingHandler
        self.assertFalse(transformed_data["Feature2"].isna().any())
        self.assertFalse(transformed_data["Feature3"].isna().any())
        self.assertFalse(transformed_data["Feature4"].isna().any())

    def test_power_method(self) -> None:
        """
        Test the 'power' method of outlier handling.
        """
        handler = UnivariateOutliersHandler(
            id_col="ID",
            activity_col="Activity",
            handling_method="power",
            save_dir=self.save_dir,
        )
        handler.fit(self.data)
        transformed_data = handler.transform(self.data)
        self.assertEqual(transformed_data.shape, self.data.shape)

    def test_normal_method(self) -> None:
        """
        Test the 'normal' method of outlier handling.
        """
        handler = UnivariateOutliersHandler(
            id_col="ID",
            activity_col="Activity",
            handling_method="normal",
            save_dir=self.save_dir,
        )
        handler.fit(self.data)
        transformed_data = handler.transform(self.data)
        self.assertEqual(transformed_data.shape, self.data.shape)

    def test_uniform_method(self) -> None:
        """
        Test the 'uniform' method of outlier handling.
        """
        handler = UnivariateOutliersHandler(
            id_col="ID",
            activity_col="Activity",
            handling_method="uniform",
            save_dir=self.save_dir,
        )
        handler.fit(self.data)
        transformed_data = handler.transform(self.data)
        self.assertEqual(transformed_data.shape, self.data.shape)

    def test_static_transform(self) -> None:
        """
        Test the 'static_transform' method for handling outliers using saved parameters.
        """

        handler = UnivariateOutliersHandler(
            id_col="ID",
            activity_col="Activity",
            handling_method="normal",
            save_dir=self.save_dir,
        )
        handler.fit(self.data)
        transformed_data_static = UnivariateOutliersHandler.static_transform(
            self.data, self.save_dir
        )
        self.assertEqual(transformed_data_static.shape, self.data.shape)

    def test_compare_outlier_methods(self) -> None:
        """
        Test the 'compare_outlier_methods' method to ensure different handling methods are compared correctly.
        """
        comparison_table1 = UnivariateOutliersHandler.compare_outlier_methods(
            data1=self.data, activity_col="Activity", id_col="ID"
        )
        comparison_table2 = UnivariateOutliersHandler.compare_outlier_methods(
            data1=self.data, data2=self.data, activity_col="Activity", id_col="ID"
        )
        self.assertEqual(
            comparison_table1.shape[0], 6
        )  # Should have 6 rows, one for each method
        self.assertEqual(
            comparison_table2.shape[0], 6
        )  # Should have 6 rows, one for each method
        print(comparison_table2)


if __name__ == "__main__":
    unittest.main()
