import unittest
import optuna
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    ElasticNetCV,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from proqsar.Model.Optimizer.optimizer_utils import (
    _get_model_list,
    _get_model_and_params,
)


class TestModelAndParams(unittest.TestCase):

    def setUp(self):
        self.study = optuna.create_study(direction="maximize")
        self.trial = self.study.ask()

    def test_get_model_list_classification(self):
        expected_models = [
            "LogisticRegression",
            "KNeighborsClassifier",
            "SVC",
            "RandomForestClassifier",
            "ExtraTreesClassifier",
            "AdaBoostClassifier",
            "GradientBoostingClassifier",
            "XGBClassifier",
            "CatBoostClassifier",
            "MLPClassifier",
        ]
        result = _get_model_list("C")
        self.assertListEqual(result, expected_models)

    def test_get_model_list_regression(self):
        expected_models = [
            "LinearRegression",
            "KNeighborsRegressor",
            "SVR",
            "RandomForestRegressor",
            "ExtraTreesRegressor",
            "AdaBoostRegressor",
            "GradientBoostingRegressor",
            "XGBRegressor",
            "CatBoostRegressor",
            "MLPRegressor",
            "Ridge",
            "ElasticNetCV",
        ]
        result = _get_model_list("R")
        self.assertListEqual(result, expected_models)

    def test_get_model_list_with_custom_model(self):
        add_model = {"CustomClassifier": (LogisticRegression(), {"C": (0.1, 1.0)})}
        result = _get_model_list("C", add_model=add_model)
        self.assertIn("CustomClassifier", result)
        self.assertEqual(len(result), 11)  # 10 default models + 1 custom

    def test_get_logistic_regression_params(self):
        param_ranges = {
            "LogisticRegression": {"C": (0.01, 10.0), "max_iter": (100, 300)}
        }
        model, params = _get_model_and_params(
            self.trial, "LogisticRegression", param_ranges
        )
        self.assertIsInstance(model, LogisticRegression)
        self.assertIn("C", params)
        self.assertIn("max_iter", params)

    def test_get_kneighbors_classifier_params(self):
        param_ranges = {
            "KNeighborsClassifier": {"n_neighbors": (1, 20), "leaf_size": (10, 50)}
        }
        model, params = _get_model_and_params(
            self.trial, "KNeighborsClassifier", param_ranges
        )
        self.assertIsInstance(model, KNeighborsClassifier)
        self.assertIn("n_neighbors", params)
        self.assertIn("leaf_size", params)

    def test_get_svc_params(self):
        param_ranges = {"SVC": {"C": (0.1, 10.0), "kernel": ("linear", "rbf")}}
        model, params = _get_model_and_params(self.trial, "SVC", param_ranges)
        self.assertIsInstance(model, SVC)
        self.assertIn("C", params)
        self.assertIn("kernel", params)

    def test_get_randomforest_classifier_params(self):
        param_ranges = {
            "RandomForestClassifier": {"n_estimators": (10, 100), "max_depth": (3, 10)}
        }
        model, params = _get_model_and_params(
            self.trial, "RandomForestClassifier", param_ranges
        )
        self.assertIsInstance(model, RandomForestClassifier)
        self.assertIn("n_estimators", params)
        self.assertIn("max_depth", params)

    def test_get_extratrees_classifier_params(self):
        param_ranges = {
            "ExtraTreesClassifier": {"n_estimators": (10, 100), "max_depth": (3, 10)}
        }
        model, params = _get_model_and_params(
            self.trial, "ExtraTreesClassifier", param_ranges
        )
        self.assertIsInstance(model, ExtraTreesClassifier)
        self.assertIn("n_estimators", params)
        self.assertIn("max_depth", params)

    def test_get_ada_classifier_params(self):
        param_ranges = {
            "AdaBoostClassifier": {
                "n_estimators": (10, 100),
                "learning_rate": (0.01, 2.0),
            }
        }
        model, params = _get_model_and_params(
            self.trial, "AdaBoostClassifier", param_ranges
        )
        self.assertIsInstance(model, AdaBoostClassifier)
        self.assertIn("n_estimators", params)
        self.assertIn("learning_rate", params)

    def test_get_gradient_classifier_params(self):
        param_ranges = {
            "GradientBoostingClassifier": {
                "n_estimators": (10, 100),
                "max_depth": (3, 10),
            }
        }
        model, params = _get_model_and_params(
            self.trial, "GradientBoostingClassifier", param_ranges
        )
        self.assertIsInstance(model, GradientBoostingClassifier)
        self.assertIn("n_estimators", params)
        self.assertIn("max_depth", params)

    def test_get_xgb_classifier_params(self):
        param_ranges = {
            "XGBClassifier": {"n_estimators": (10, 100), "max_depth": (3, 10)}
        }
        model, params = _get_model_and_params(self.trial, "XGBClassifier", param_ranges)
        self.assertIsInstance(model, XGBClassifier)
        self.assertIn("n_estimators", params)
        self.assertIn("max_depth", params)

    def test_get_cat_classifier_params(self):
        param_ranges = {
            "CatBoostClassifier": {"iterations": (50, 100), "depth": (3, 10)}
        }
        model, params = _get_model_and_params(
            self.trial, "CatBoostClassifier", param_ranges
        )
        self.assertIsInstance(model, CatBoostClassifier)
        self.assertIn("iterations", params)
        self.assertIn("depth", params)

    def test_get_mlp_classifier_params(self):
        param_ranges = {
            "MLPClassifier": {"hidden_layer_sizes": (50, 100), "alpha": (0.001, 0.1)}
        }
        model, params = _get_model_and_params(self.trial, "MLPClassifier", param_ranges)
        self.assertIsInstance(model, MLPClassifier)
        self.assertIn("hidden_layer_sizes", params)
        self.assertIn("alpha", params)

    def test_get_linear_regression_no_params(self):
        param_ranges = {}
        model, params = _get_model_and_params(
            self.trial, "LinearRegression", param_ranges
        )
        self.assertIsInstance(model, LinearRegression)
        self.assertEqual(params, {})  # LinearRegression has no hyperparameters

    def test_get_kneighbors_regressor_params(self):
        param_ranges = {
            "KNeighborsRegressor": {"n_neighbors": (1, 20), "leaf_size": (10, 50)}
        }
        model, params = _get_model_and_params(
            self.trial, "KNeighborsRegressor", param_ranges
        )
        self.assertIsInstance(model, KNeighborsRegressor)
        self.assertIn("n_neighbors", params)
        self.assertIn("leaf_size", params)

    def test_get_svr_params(self):
        param_ranges = {"SVR": {"C": (0.1, 10.0), "kernel": ("linear", "rbf")}}
        model, params = _get_model_and_params(self.trial, "SVR", param_ranges)
        self.assertIsInstance(model, SVR)
        self.assertIn("C", params)
        self.assertIn("kernel", params)

    def test_get_randomforest_regressor_params(self):
        param_ranges = {
            "RandomForestRegressor": {"n_estimators": (10, 100), "max_depth": (3, 10)}
        }
        model, params = _get_model_and_params(
            self.trial, "RandomForestRegressor", param_ranges
        )
        self.assertIsInstance(model, RandomForestRegressor)
        self.assertIn("n_estimators", params)
        self.assertIn("max_depth", params)

    def test_get_extratrees_regressor_params(self):
        param_ranges = {
            "ExtraTreesRegressor": {"n_estimators": (10, 100), "max_depth": (3, 10)}
        }
        model, params = _get_model_and_params(
            self.trial, "ExtraTreesRegressor", param_ranges
        )
        self.assertIsInstance(model, ExtraTreesRegressor)
        self.assertIn("n_estimators", params)
        self.assertIn("max_depth", params)

    def test_get_ada_regressor_params(self):
        param_ranges = {
            "AdaBoostRegressor": {
                "n_estimators": (10, 100),
                "learning_rate": (0.01, 2.0),
            }
        }
        model, params = _get_model_and_params(
            self.trial, "AdaBoostRegressor", param_ranges
        )
        self.assertIsInstance(model, AdaBoostRegressor)
        self.assertIn("n_estimators", params)
        self.assertIn("learning_rate", params)

    def test_get_gradient_regressor_params(self):
        param_ranges = {
            "GradientBoostingRegressor": {
                "n_estimators": (10, 100),
                "max_depth": (3, 10),
            }
        }
        model, params = _get_model_and_params(
            self.trial, "GradientBoostingRegressor", param_ranges
        )
        self.assertIsInstance(model, GradientBoostingRegressor)
        self.assertIn("n_estimators", params)
        self.assertIn("max_depth", params)

    def test_get_xgb_regressor_params(self):
        param_ranges = {
            "XGBRegressor": {"n_estimators": (10, 100), "max_depth": (3, 10)}
        }
        model, params = _get_model_and_params(self.trial, "XGBRegressor", param_ranges)
        self.assertIsInstance(model, XGBRegressor)
        self.assertIn("n_estimators", params)
        self.assertIn("max_depth", params)

    def test_get_cat_regressor_params(self):
        param_ranges = {
            "CatBoostRegressor": {"iterations": (50, 100), "depth": (3, 10)}
        }
        model, params = _get_model_and_params(
            self.trial, "CatBoostRegressor", param_ranges
        )
        self.assertIsInstance(model, CatBoostRegressor)
        self.assertIn("iterations", params)
        self.assertIn("depth", params)

    def test_get_mlp_regressor_params(self):
        param_ranges = {
            "MLPRegressor": {"hidden_layer_sizes": (50, 100), "alpha": (0.001, 0.1)}
        }
        model, params = _get_model_and_params(self.trial, "MLPRegressor", param_ranges)
        self.assertIsInstance(model, MLPRegressor)
        self.assertIn("hidden_layer_sizes", params)
        self.assertIn("alpha", params)

    def test_get_ridge_params(self):
        param_ranges = {"Ridge": {"alpha": (0.1, 1.0)}}
        model, params = _get_model_and_params(self.trial, "Ridge", param_ranges)
        self.assertIsInstance(model, Ridge)
        self.assertIn("alpha", params)

    def test_get_elasticnetcv_params(self):
        param_ranges = {"ElasticNetCV": {"l1_ratio": (0.0, 1.0)}}
        model, params = _get_model_and_params(self.trial, "ElasticNetCV", param_ranges)
        self.assertIsInstance(model, ElasticNetCV)
        self.assertIn("l1_ratio", params)

    def test_invalid_model_name(self):
        param_ranges = {}
        with self.assertRaises(ValueError):
            _get_model_and_params(self.trial, "UnknownModel", param_ranges)

    def test_get_model_and_params_with_custom_model(self):
        add_model = {"CustomClassifier": (LogisticRegression(), {"C": (0.1, 1.0)})}
        model, params = _get_model_and_params(
            self.trial, "CustomClassifier", param_ranges={}, add_model=add_model
        )
        self.assertIsInstance(model, LogisticRegression)
        self.assertIn("C", params)
        self.assertTrue(0.1 <= params["C"] <= 1.0)

    def test_get_model_and_params_with_categorical_params(self):
        add_model = {
            "CustomClassifier": (SVC(), {"kernel": ["linear", "rbf", "sigmoid"]})
        }
        model, params = _get_model_and_params(
            self.trial, "CustomClassifier", param_ranges={}, add_model=add_model
        )
        self.assertIsInstance(model, SVC)
        self.assertIn("kernel", params)
        self.assertIn(params["kernel"], ["linear", "rbf", "sigmoid"])

    def test_get_model_and_params_with_custom_model_no_params(self):
        add_model = {"CustomClassifier": (LogisticRegression(), {})}
        model, params = _get_model_and_params(
            self.trial, "CustomClassifier", param_ranges={}, add_model=add_model
        )
        self.assertIsInstance(model, LogisticRegression)
        self.assertEqual(params, {})


if __name__ == "__main__":
    unittest.main()
