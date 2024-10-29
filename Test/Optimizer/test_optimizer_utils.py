import unittest
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, ElasticNetCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from optuna.trial import create_trial
from ProQSAR.Optimizer.optimizer_utils import _get_model_list, _get_model_and_params

dummy_trial = create_trial()

class TestModelAndParams(unittest.TestCase):
    
    # Test classification model list
    def test_get_model_list_classification(self):
        expected_models = [
            'LogisticRegression', 'KNeighborsClassifier', 'SVC', 
            'RandomForestClassifier', 'ExtraTreesClassifier', 'AdaBoostClassifier', 
            'GradientBoostingClassifier', 'XGBClassifier', 'CatBoostClassifier', 
            'MLPClassifier'
        ]
        result = _get_model_list('C')
        self.assertListEqual(result, expected_models)

    # Test regression model list
    def test_get_model_list_regression(self):
        expected_models = [
            'LinearRegression', 'KNeighborsRegressor', 'SVR', 
            'RandomForestRegressor', 'ExtraTreesRegressor', 'AdaBoostRegressor', 
            'GradientBoostingRegressor', 'XGBRegressor', 'CatBoostRegressor', 
            'MLPRegressor', 'Ridge', 'ElasticNetCV'
        ]
        result = _get_model_list('R')
        self.assertListEqual(result, expected_models)

    # Test adding custom model to classification model list
    def test_get_model_list_with_custom_model(self):
        add_model = {'CustomClassifier': (LogisticRegression(), {'C': (0.1, 1.0)})}
        result = _get_model_list('C', add_model=add_model)
        self.assertIn('CustomClassifier', result)
        self.assertEqual(len(result), 11)  # 10 default models + 1 custom

    # Test logistic regression parameters
    def test_get_logistic_regression_params(self):
        param_ranges = {'LogisticRegression': {'C': (0.01, 10.0), 'max_iter': (100, 300)}}
        model, params = _get_model_and_params(dummy_trial, 'LogisticRegression', param_ranges)
        self.assertIsInstance(model, LogisticRegression)
        self.assertIn('C', params)
        self.assertIn('max_iter', params)

    # Test KNeighborsClassifier parameters
    def test_get_kneighbors_classifier_params(self):
        param_ranges = {'KNeighborsClassifier': {'n_neighbors': (1, 20), 'leaf_size': (10, 50)}}
        model, params = _get_model_and_params(dummy_trial, 'KNeighborsClassifier', param_ranges)
        self.assertIsInstance(model, KNeighborsClassifier)
        self.assertIn('n_neighbors', params)
        self.assertIn('leaf_size', params)

    # Test linear regression with no parameters
    def test_get_linear_regression_no_params(self):
        param_ranges = {}
        model, params = _get_model_and_params(dummy_trial, 'LinearRegression', param_ranges)
        self.assertIsInstance(model, LinearRegression)
        self.assertEqual(params, {})  # LinearRegression has no hyperparameters

    # Test KNeighborsRegressor parameters
    def test_get_kneighbors_regressor_params(self):
        param_ranges = {'KNeighborsRegressor': {'n_neighbors': (1, 20), 'leaf_size': (10, 50)}}
        model, params = _get_model_and_params(dummy_trial, 'KNeighborsRegressor', param_ranges)
        self.assertIsInstance(model, KNeighborsRegressor)
        self.assertIn('n_neighbors', params)
        self.assertIn('leaf_size', params)

    # Test Ridge regression parameters
    def test_get_ridge_params(self):
        param_ranges = {'Ridge': {'alpha': (0.1, 1.0)}}
        model, params = _get_model_and_params(dummy_trial, 'Ridge', param_ranges)
        self.assertIsInstance(model, Ridge)
        self.assertIn('alpha', params)

    # Test ElasticNetCV parameters
    def test_get_elasticnetcv_params(self):
        param_ranges = {'ElasticNetCV': {'l1_ratio': (0.0, 1.0)}}
        model, params = _get_model_and_params(dummy_trial, 'ElasticNetCV', param_ranges)
        self.assertIsInstance(model, ElasticNetCV)
        self.assertIn('l1_ratio', params)

    # Test invalid model name
    def test_invalid_model_name(self):
        param_ranges = {}
        with self.assertRaises(ValueError) as context:
            _get_model_and_params(dummy_trial, 'UnknownModel', param_ranges)
        self.assertEqual(str(context.exception), "Model UnknownModel not found in available models")

    # Additional tests for other classifiers
    def test_get_svc_params(self):
        param_ranges = {'SVC': {'C': (0.1, 10.0), 'kernel': ('linear', 'rbf')}}
        model, params = _get_model_and_params(dummy_trial, 'SVC', param_ranges)
        self.assertIsInstance(model, SVC)
        self.assertIn('C', params)
        self.assertIn('kernel', params)

    def test_get_randomforest_classifier_params(self):
        param_ranges = {'RandomForestClassifier': {'n_estimators': (10, 100), 'max_depth': (3, 10)}}
        model, params = _get_model_and_params(dummy_trial, 'RandomForestClassifier', param_ranges)
        self.assertIsInstance(model, RandomForestClassifier)
        self.assertIn('n_estimators', params)
        self.assertIn('max_depth', params)

    # Additional tests for other regressors
    def test_get_svr_params(self):
        param_ranges = {'SVR': {'C': (0.1, 10.0), 'kernel': ('linear', 'rbf')}}
        model, params = _get_model_and_params(dummy_trial, 'SVR', param_ranges)
        self.assertIsInstance(model, SVR)
        self.assertIn('C', params)
        self.assertIn('kernel', params)

    def test_get_randomforest_regressor_params(self):
        param_ranges = {'RandomForestRegressor': {'n_estimators': (10, 100), 'max_depth': (3, 10)}}
        model, params = _get_model_and_params(dummy_trial, 'RandomForestRegressor', param_ranges)
        self.assertIsInstance(model, RandomForestRegressor)
        self.assertIn('n_estimators', params)
        self.assertIn('max_depth', params)

if __name__ == '__main__':
    unittest.main()
