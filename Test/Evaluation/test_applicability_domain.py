import unittest
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError
from tempfile import TemporaryDirectory
from sklearn.datasets import make_regression
from proqsar.Evaluation.applicability_domain import ApplicabilityDomain


def create_regression_data(
    n_samples=40, n_features=40, n_informative=10, random_state=42
):
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        random_state=random_state,
    )
    data = pd.DataFrame(X, columns=[f"Feature{i}" for i in range(1, n_features + 1)])
    data["ID"] = np.arange(n_samples)
    data["Activity"] = y
    return data


class TestApplicabilityDomain(unittest.TestCase):
    def setUp(self):
        self.train_data = create_regression_data(random_state=42)
        self.test_data = create_regression_data(random_state=41)
        self.temp_dir = TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            ApplicabilityDomain(method="invalid")

    def test_fit_ocsvm(self):
        ad = ApplicabilityDomain(
            method="ocsvm",
            activity_col="Activity",
            id_col="ID",
            save_dir=self.temp_dir.name,
        )
        ad.fit(self.train_data)
        self.assertIsNotNone(ad.ad)
        self.assertTrue(
            os.path.exists(f"{self.temp_dir.name}/applicability_domain.pkl")
        )

    def test_fit_knn(self):
        ad = ApplicabilityDomain(
            method="knn", activity_col="Activity", id_col="ID", save_dir=None
        )
        ad.fit(self.train_data)
        self.assertIsNotNone(ad.ad)

    def test_fit_lof(self):
        ad = ApplicabilityDomain(
            method="lof", activity_col="Activity", id_col="ID", save_dir=None
        )
        ad.fit(self.train_data)
        self.assertIsNotNone(ad.ad)

    def test_predict_ocsvm(self):
        ad = ApplicabilityDomain(
            method="ocsvm", activity_col="Activity", id_col="ID", save_dir=None
        )
        ad.fit(self.train_data)
        result = ad.predict(self.test_data)
        self.assertEqual(len(result), len(self.test_data))
        self.assertIn("Applicability domain", result.columns)

    def test_predict_knn(self):
        ad = ApplicabilityDomain(
            method="knn", activity_col="Activity", id_col="ID", save_dir=None
        )
        ad.fit(self.train_data)
        result = ad.predict(self.test_data)
        self.assertEqual(len(result), len(self.test_data))

    def test_predict_lof(self):
        ad = ApplicabilityDomain(
            method="lof", activity_col="Activity", id_col="ID", save_dir=None
        )
        ad.fit(self.train_data)
        result = ad.predict(self.test_data)
        self.assertEqual(len(result), len(self.test_data))

    def test_predict_without_fit(self):
        ad = ApplicabilityDomain(
            method="knn", activity_col="Activity", id_col="ID", save_dir=None
        )
        with self.assertRaises(NotFittedError):
            ad.predict(self.test_data)

    def test_saving(self):
        ad = ApplicabilityDomain(
            method="ocsvm",
            activity_col="Activity",
            id_col="ID",
            save_dir=self.temp_dir.name,
        )
        ad.fit(self.train_data)
        ad.predict(self.test_data)
        with open(f"{self.temp_dir.name}/applicability_domain.pkl", "rb") as file:
            loaded_ad = pickle.load(file)
        self.assertIsNotNone(loaded_ad)
        self.assertEqual(loaded_ad.method, "ocsvm")
        self.assertTrue(os.path.exists(f"{self.temp_dir.name}/ad_pred_result.csv"))

    def test_deactivate_returns_without_fitting(self):
        ad = ApplicabilityDomain(
            method="ocsvm",
            activity_col="Activity",
            id_col="ID",
            deactivate=True,
            save_dir=self.temp_dir.name,
        )
        res = ad.fit(self.train_data)
        self.assertIsNone(res)
        # predict should raise NotFittedError since not fitted
        with self.assertRaises(NotFittedError):
            ad.predict(self.test_data)

    def test_ocsvm_manual_gamma_path(self):
        ad = ApplicabilityDomain(
            method="ocsvm",
            activity_col="Activity",
            id_col="ID",
            gamma=0.1,
            save_dir=None,
        )
        ad.fit(self.train_data)
        self.assertEqual(ad.optimal_gamma, 0.1)
        self.assertIsNotNone(ad.ad)

    def test_no_save_dir_does_not_write_files(self):
        ad = ApplicabilityDomain(
            method="ocsvm", activity_col="Activity", id_col="ID", save_dir=None
        )
        ad.fit(self.train_data)
        ad.predict(self.test_data)
        self.assertFalse(
            os.path.exists(os.path.join(self.temp_dir.name, "applicability_domain.pkl"))
        )
        self.assertFalse(
            os.path.exists(os.path.join(self.temp_dir.name, "ad_pred_result.csv"))
        )

    def test_predict_includes_id_column_when_present(self):
        ad = ApplicabilityDomain(
            method="knn", activity_col="Activity", id_col="ID", save_dir=None
        )
        ad.fit(self.train_data)
        res = ad.predict(self.test_data)
        self.assertIn("ID", res.columns)


if __name__ == "__main__":
    unittest.main()
