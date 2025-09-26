import unittest
import pandas as pd
import numpy as np
import os
from typing import Optional
from tempfile import TemporaryDirectory
from proqsar.Preprocessor.Clean.duplicate_handler import DuplicateHandler


def create_sample_data(
    n_unique: int = 5,
    dup_sets: int = 2,
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Creates a sample DataFrame with duplicate rows and columns.

    Returns:
    - pd.DataFrame: The sample data with duplicates.
    """
    rng = np.random.RandomState(random_state)

    # Base unique rows
    base = pd.DataFrame(
        {
            "ID": np.arange(1, n_unique + 1),
            "Activity": rng.rand(n_unique),
            "Feature1": rng.choice([0, 1], size=n_unique),
            "Feature2": rng.choice([0, 1], size=n_unique),
            "Feature3": rng.choice([0, 1], size=n_unique),
            "Feature4": rng.rand(n_unique),
            "Feature5": rng.rand(n_unique),
            "Feature6": rng.rand(n_unique),
        }
    )

    # Build full DataFrame by appending dup_sets copies of the first two rows
    df = base.copy()
    for _ in range(dup_sets):
        df = pd.concat([df, base.iloc[:2]], ignore_index=True)

    # Make deterministic, intentional edits similar to the original function:
    # - set Activity of first row to 0.5
    df.loc[df.index[0], "Activity"] = 0.5

    # - set Activities for the last three rows to [0.8, 0.8, 0.5]
    #   (this mirrors original indices 6,7,8 when n_unique=5 and dup_sets=2)
    if len(df) >= 3:
        last_idx = df.index[-3:]
        df.loc[last_idx[0], "Activity"] = 0.8
        df.loc[last_idx[1], "Activity"] = 0.8
        df.loc[last_idx[2], "Activity"] = 0.5

    # - assign ID values to appended rows so they continue the sequence
    appended_count = dup_sets * 2
    if appended_count > 0:
        start_id = n_unique + 1
        new_ids = list(range(start_id, start_id + appended_count))
        df_ids_slice = df.index[-appended_count:]
        df.loc[df_ids_slice, "ID"] = new_ids

    # Create duplicate columns (Feature1 <- Feature2, Feature5 <- Feature6)
    # Use .copy() to avoid unexpected view/reference issues
    df["Feature1"] = df["Feature2"].copy()
    df["Feature5"] = df["Feature6"].copy()

    # Ensure types are preserved (no unintended upcasts)
    # (we made row selections with .loc + copy, so dtypes should be preserved)

    return df


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
            index=[5, 6, 7, 8], columns=["Feature2", "Feature6"]
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
        h = DuplicateHandler(
            id_col="ID", activity_col="Activity", cols=False, rows=True
        )
        h.fit(self.train_data)
        # Columns should not be removed, only rows
        out = h.transform(self.train_data)
        self.assertIn("Feature2", out.columns)
        self.assertIn("Feature6", out.columns)
        self.assertEqual(len(out), 5)  # duplicate rows removed
        # Now disable rows removal
        h2 = DuplicateHandler(
            id_col="ID", activity_col="Activity", cols=True, rows=False
        )
        h2.fit(self.train_data)
        out2 = h2.transform(self.train_data)
        self.assertNotIn("Feature2", out2.columns)
        self.assertNotIn("Feature6", out2.columns)
        self.assertEqual(len(out2), 9)  # rows kept

    def test_save_method_persists_model(self):
        h = DuplicateHandler(
            id_col="ID",
            activity_col="Activity",
            save_method=True,
            save_dir=self.temp_dir.name,
        )
        h.fit(self.train_data)
        self.assertTrue(
            os.path.exists(os.path.join(self.temp_dir.name, "duplicate_handler.pkl"))
        )

    def test_keep_first_and_last_behaviour(self):
        # use 'first' (default behavior similar to pandas.drop_duplicates keep='first')
        dh_first = DuplicateHandler(activity_col="Activity", id_col="ID", keep="first")
        dh_first.fit(self.train_data)
        out_first = dh_first.transform(self.train_data)

        # compute expected by dropping duplicates on feature columns (after removing dup_cols)
        df_no_dup = self.train_data.drop(columns=dh_first.dup_cols, errors="ignore")
        feature_cols = [c for c in df_no_dup.columns if c not in {"Activity", "ID"}]
        expected_first = df_no_dup.drop_duplicates(
            subset=feature_cols, keep="first"
        ).reset_index(drop=True)
        self.assertEqual(out_first.shape[0], expected_first.shape[0])

        # 'last'
        dh_last = DuplicateHandler(activity_col="Activity", id_col="ID", keep="last")
        dh_last.fit(self.train_data)
        out_last = dh_last.transform(self.train_data)
        expected_last = df_no_dup.drop_duplicates(
            subset=feature_cols, keep="last"
        ).reset_index(drop=True)
        self.assertEqual(out_last.shape[0], expected_last.shape[0])

    def test_keep_random_is_deterministic_with_seed(self):
        # same seed => same output order & rows
        dh1 = DuplicateHandler(
            activity_col="Activity", id_col="ID", keep="random", random_state=123
        )
        dh1.fit(self.train_data)
        out1 = dh1.transform(self.train_data)

        dh2 = DuplicateHandler(
            activity_col="Activity", id_col="ID", keep="random", random_state=123
        )
        dh2.fit(self.train_data)
        out2 = dh2.transform(self.train_data)

        pd.testing.assert_frame_equal(
            out1.reset_index(drop=True), out2.reset_index(drop=True)
        )

        # number of rows should equal number of unique feature groups
        df_no_dup = self.train_data.drop(columns=dh1.dup_cols, errors="ignore")
        feature_cols = [c for c in df_no_dup.columns if c not in {"Activity", "ID"}]
        unique_groups = df_no_dup.drop_duplicates(subset=feature_cols).shape[0]
        self.assertEqual(out1.shape[0], unique_groups)

    def test_keep_min_max_selects_correct_activity(self):
        # MIN
        dh_min = DuplicateHandler(activity_col="Activity", id_col="ID", keep="min")
        dh_min.fit(self.train_data)
        out_min = dh_min.transform(self.train_data)

        # For each retained row check that its activity equals the min activity in its group
        df_no_dup = self.train_data.drop(columns=dh_min.dup_cols, errors="ignore")
        feature_cols = [c for c in df_no_dup.columns if c not in {"ID", "Activity"}]
        groups = df_no_dup.groupby(feature_cols, dropna=False, sort=False)
        # Build mapping from feature-tuple to min activity
        min_map = {k: g["Activity"].min() for k, g in groups}

        for _, row in out_min.iterrows():
            key = tuple(row[c] for c in feature_cols)
            self.assertAlmostEqual(row["Activity"], min_map[key])

        # MAX
        dh_max = DuplicateHandler(activity_col="Activity", id_col="ID", keep="max")
        dh_max.fit(self.train_data)
        out_max = dh_max.transform(self.train_data)
        max_map = {
            k: g["Activity"].max()
            for k, g in df_no_dup.groupby(feature_cols, dropna=False, sort=False)
        }
        for _, row in out_max.iterrows():
            key = tuple(row[c] for c in feature_cols)
            self.assertAlmostEqual(row["Activity"], max_map[key])

    def test_keep_mean_median_preserve_other_column_dtypes(self):
        dh_mean = DuplicateHandler(activity_col="Activity", id_col="ID", keep="mean")
        dh_mean.fit(self.train_data)
        out_mean = dh_mean.transform(self.train_data)
        # ID should remain integer dtype (or at least same dtype as original)
        self.assertEqual(out_mean["ID"].dtype, self.train_data["ID"].dtype)
        # Feature3 was integer-like (0/1) in original -> dtype preserved
        self.assertEqual(out_mean["Feature3"].dtype, self.train_data["Feature3"].dtype)
        # Activity must be numeric (float after mean)
        self.assertTrue(pd.api.types.is_float_dtype(out_mean["Activity"].dtype))

        dh_median = DuplicateHandler(
            activity_col="Activity", id_col="ID", keep="median"
        )
        dh_median.fit(self.train_data)
        out_median = dh_median.transform(self.train_data)
        self.assertTrue(pd.api.types.is_float_dtype(out_median["Activity"].dtype))


if __name__ == "__main__":
    unittest.main()
