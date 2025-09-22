import unittest
from proqsar.Config.validation_config import CrossValidationConfig


class TestValidationConfig(unittest.TestCase):
    def test_cross_validation_config_defaults_and_overrides(self):
        cfg = CrossValidationConfig()
        self.assertIsNone(cfg.scoring_target)
        self.assertIsNone(cfg.scoring_list)
        self.assertEqual(cfg.n_splits, 5)
        self.assertEqual(cfg.n_repeats, 5)
        self.assertFalse(cfg.save_cv_report)
        self.assertEqual(cfg.cv_report_name, "cv_report")
        self.assertIsNone(cfg.visualize)
        self.assertFalse(cfg.save_fig)
        self.assertEqual(cfg.fig_prefix, "cv_graph")

        cfg2 = CrossValidationConfig(
            scoring_target="f1",
            scoring_list=["f1", "accuracy"],
            n_splits=3,
            n_repeats=2,
            save_cv_report=True,
            cv_report_name="report",
            visualize="box",
            save_fig=True,
            fig_prefix="pref",
        )
        self.assertEqual(cfg2.scoring_target, "f1")
        self.assertEqual(cfg2.scoring_list, ["f1", "accuracy"])
        self.assertEqual(cfg2.n_splits, 3)
        self.assertEqual(cfg2.n_repeats, 2)
        self.assertTrue(cfg2.save_cv_report)
        self.assertEqual(cfg2.cv_report_name, "report")
        self.assertEqual(cfg2.visualize, "box")
        self.assertTrue(cfg2.save_fig)
        self.assertEqual(cfg2.fig_prefix, "pref")


if __name__ == "__main__":
    unittest.main()
