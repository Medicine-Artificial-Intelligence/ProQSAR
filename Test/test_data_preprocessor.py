import os
import unittest
import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory
from proqsar.Config.config import Config
from proqsar.data_preprocessor import DataPreprocessor


def make_preproc_df(
    n: int = 20,
    seed: int = 42,
    id_col: str = "id",
    activity_col: str = "activity",
    add_row_duplicates: bool = True,
    add_column_duplicates: bool = True,
    missing_rates: dict | None = None,
) -> pd.DataFrame:
    """
    Create a synthetic dataset consistent with the provided examples:
      - Binary + low-variance normal + mixed-variance continuous features
      - Optional per-column missing values (by rate)
      - Optional duplicate rows (append first two rows)
      - Optional duplicate columns (Feature1=Feature2 and Feature5=Feature6)
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            id_col: np.arange(1, n + 1),
            activity_col: rng.random(n) * 10.0,  # ~U(0,10)
            "Feature1": rng.integers(0, 2, n),  # binary
            "Feature2": rng.integers(0, 2, n),  # binary
            "Feature3": rng.normal(0.0, 0.1, n),  # low-variance normal (std ~0.1)
            "Feature4": rng.normal(0.0, 0.1, n),  # low-variance normal
            "Feature5": rng.normal(0.0, np.sqrt(0.5), n),  # mixed variance
            "Feature6": rng.normal(0.0, np.sqrt(0.8), n),
            "Feature7": rng.normal(0.0, 1.0, n),
            "Feature8": rng.random(n),
            "Feature9": rng.random(n),
            "Feature10": rng.random(n),
        }
    )

    if missing_rates is None:
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

    for col, rate in missing_rates.items():
        if col in df.columns and 0.0 < rate <= 1.0:
            mask = rng.random(n) < rate
            df.loc[mask, col] = np.nan

    if add_row_duplicates and n >= 2:
        df = pd.concat([df, df.iloc[0:2]], ignore_index=True)

    if add_column_duplicates:
        if "Feature2" in df and "Feature1" in df:
            df["Feature1"] = df["Feature2"]
        if "Feature6" in df and "Feature5" in df:
            df["Feature5"] = df["Feature6"]

    return df


class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        # Use lowercase schema for DataPreprocessor tests
        self.df = make_preproc_df(n=40, id_col="id", activity_col="activity", seed=7)

    def _fast_config(self, **overrides) -> Config:

        cfg = Config()
        for key, params in overrides.items():
            getattr(cfg, key).set_params(**params)
        return cfg

    def test_pipeline_order_and_param_wiring(self):
        cfg = self._fast_config()
        dp = DataPreprocessor(
            activity_col="activity", id_col="id", config=cfg, save_dir=None
        )

        self.assertEqual(
            list(dp.pipeline.named_steps.keys()),
            [
                "duplicate",
                "missing",
                "lowvar",
                "univ_outlier",
                "kbin",
                "multiv_outlier",
                "rescaler",
            ],
        )
        for step in dp.pipeline.named_steps.values():
            self.assertTrue(hasattr(step, "activity_col"))
            self.assertTrue(hasattr(step, "id_col"))
            self.assertEqual(step.activity_col, "activity")
            self.assertEqual(step.id_col, "id")

    def test_fit_returns_self(self):
        cfg = self._fast_config()
        dp = DataPreprocessor(
            activity_col="activity", id_col="id", config=cfg, save_dir=None
        )
        out = dp.fit(self.df)
        self.assertIs(out, dp)

    def test_transform_without_save_dir(self):
        cfg = self._fast_config()
        dp = DataPreprocessor(
            activity_col="activity", id_col="id", config=cfg, save_dir=None
        )
        dp.fit(self.df)
        tr = dp.transform(self.df)

        self.assertIsInstance(tr, pd.DataFrame)
        self.assertIn("id", tr.columns)
        self.assertIn("activity", tr.columns)
        self.assertLessEqual(len(tr), len(self.df))
        self.assertGreater(len(tr), 0)

    def test_transform_with_save_dir_and_data_name(self):
        cfg = self._fast_config()
        with TemporaryDirectory(prefix="prep_") as tmp:
            dp = DataPreprocessor(
                activity_col="activity",
                id_col="id",
                config=cfg,
                save_dir=tmp,
                data_name="toy",
            )
            dp.fit(self.df)
            tr = dp.transform(self.df)
            out_path = os.path.join(tmp, "toy_preprocessed.csv")
            self.assertTrue(os.path.isfile(out_path))
            loaded = pd.read_csv(out_path)
            self.assertEqual(len(loaded), len(tr))
            self.assertIn("id", loaded.columns)
            self.assertIn("activity", loaded.columns)

    def test_get_params_deep_and_shallow(self):
        cfg = self._fast_config()
        dp = DataPreprocessor(
            activity_col="activity", id_col="id", config=cfg, save_dir=None
        )
        shallow = dp.get_params(deep=False)
        self.assertIn("activity_col", shallow)
        self.assertIn("duplicate", shallow)
        self.assertNotIn("pipeline", shallow)

        deep = dp.get_params(deep=True)
        self.assertIn("duplicate__activity_col", deep)
        self.assertIn("missing__id_col", deep)
        self.assertIn("multiv_outlier__select_method", deep)

    def test_deactivate_some_steps_passthrough(self):
        cfg = self._fast_config(
            missing={"deactivate": True},
            multiv_outlier={"select_method": "IsolationForest", "deactivate": True},
            kbin={"deactivate": True},
            rescaler={"deactivate": True},
        )
        dp = DataPreprocessor(
            activity_col="activity", id_col="id", config=cfg, save_dir=None
        )
        dp.fit(self.df)
        tr = dp.transform(self.df)
        self.assertIsInstance(tr, pd.DataFrame)
        self.assertIn("id", tr.columns)
        self.assertIn("activity", tr.columns)

    def test_invalid_multivariate_method_raises(self):
        bad_cfg = self._fast_config()
        bad_cfg.multiv_outlier.set_params(select_method="__INVALID__")
        dp = DataPreprocessor(
            activity_col="activity", id_col="id", config=bad_cfg, save_dir=None
        )
        with self.assertRaises(ValueError):
            dp.fit(self.df)


if __name__ == "__main__":
    unittest.main()
