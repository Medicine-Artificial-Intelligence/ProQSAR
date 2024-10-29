from typing import Optional
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    ElasticNetCV,
    Ridge,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
)
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

def _get_model_list(task_type: str, add_model: Optional[dict] = None):
    if task_type == 'C':
        model_list = [
            'LogisticRegression',
            'KNeighborsClassifier',
            'SVC',
            'RandomForestClassifier',
            'ExtraTreesClassifier',
            'AdaBoostClassifier',
            'GradientBoostingClassifier',
            'XGBClassifier',
            'CatBoostClassifier',
            'MLPClassifier'
        ]
    elif task_type == 'R':
        model_list = [
            'LinearRegression',
            'KNeighborsRegressor',
            'SVR',
            'RandomForestRegressor',
            'ExtraTreesRegressor',
            'AdaBoostRegressor',
            'GradientBoostingRegressor',
            'XGBRegressor',
            'CatBoostRegressor',
            'MLPRegressor',
            'Ridge',
            'ElasticNetCV'
        ]
    add_model = add_model if add_model else {}

    return model_list + list(add_model.keys())

def _get_model_and_params(
        trial,
        model_name: str,
        param_ranges: Optional[dict] = None,
        add_model: Optional[dict] = None
        ):
    """
    Get model instance and parameter configuration based on model name.

    Parameters:
        trial (optuna.trial.Trial): An Optuna trial object.
        model_name (str): Name of the model to configure.

    Returns:
        tuple: A model instance and a dictionary of parameters.
    """
    if model_name == 'LogisticRegression':
        model = LogisticRegression()
        params = {
            'C': trial.suggest_float('C', *param_ranges[model_name].get('C', (0.01, 10.0))),
            'max_iter': trial.suggest_int('max_iter', *param_ranges[model_name].get('max_iter', (100, 300)))
        }
    elif model_name == 'KNeighborsClassifier':
        model = KNeighborsClassifier()
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', *param_ranges[model_name].get('n_neighbors', (1, 20))),
            'leaf_size': trial.suggest_int('leaf_size', *param_ranges[model_name].get('leaf_size', (10, 50)))
        }
    elif model_name == 'SVC':
        model = SVC()
        params = {
            'C': trial.suggest_float('C', *param_ranges[model_name].get('C', (0.1, 10.0))),
            'kernel': trial.suggest_categorical('kernel', param_ranges[model_name].get('kernel', ['linear', 'rbf'])),
            'gamma': trial.suggest_float('gamma', *param_ranges[model_name].get('gamma', (0.001, 1.0)))
        }
    elif model_name == 'RandomForestClassifier':
        model = RandomForestClassifier()
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *param_ranges[model_name].get('n_estimators', (10, 250))),
            'max_depth': trial.suggest_int('max_depth', *param_ranges[model_name].get('max_depth', (2, 32))),
            'min_samples_split': trial.suggest_int('min_samples_split', *param_ranges[model_name].get('min_samples_split', (2, 10)))
        }
    elif model_name == 'ExtraTreesClassifier':
        model = ExtraTreesClassifier()
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *param_ranges[model_name].get('n_estimators', (10, 250))),
            'max_depth': trial.suggest_int('max_depth', *param_ranges[model_name].get('max_depth', (2, 32))),
        }
    elif model_name == 'AdaBoostClassifier':
        model = AdaBoostClassifier()
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *param_ranges[model_name].get('n_estimators', (10, 250))),
            'learning_rate': trial.suggest_float('learning_rate', *param_ranges[model_name].get('learning_rate', (0.01, 2.0)))
        }
    elif model_name == 'GradientBoostingClassifier':
        model = GradientBoostingClassifier()
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *param_ranges[model_name].get('n_estimators', (10, 250))),
            'learning_rate': trial.suggest_float('learning_rate', *param_ranges[model_name].get('learning_rate', (0.01, 2.0))),
            'max_depth': trial.suggest_int('max_depth', *param_ranges[model_name].get('max_depth', (2, 32)))
        }
    elif model_name == 'XGBClassifier':
        model = XGBClassifier()
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *param_ranges[model_name].get('n_estimators', (10, 250))),
            'learning_rate': trial.suggest_float('learning_rate', *param_ranges[model_name].get('learning_rate', (0.01, 1.0))),
            'max_depth': trial.suggest_int("max_depth", *param_ranges[model_name].get("max_depth", (2, 32)))
        }
    elif model_name == 'CatBoostClassifier':
        model = CatBoostClassifier()
        params = {
            'iterations': trial.suggest_int('iterations', *param_ranges[model_name].get('iterations', (50, 200))),
            'learning_rate': trial.suggest_float('learning_rate', *param_ranges[model_name].get('learning_rate', (0.01, 1.0))),
            'depth': trial.suggest_int('depth', *param_ranges[model_name].get('depth', (2, 10)))
        }
    elif model_name == 'MLPClassifier':
        model = MLPClassifier()
        params = {
            'hidden_layer_sizes': trial.suggest_int('hidden_layer_sizes', *param_ranges[model_name].get('hidden_layer_sizes', (50, 200))),
            'alpha': trial.suggest_float('alpha', *param_ranges[model_name].get('alpha', (0.0001, 0.01))),
            'learning_rate': trial.suggest_categorical('learning_rate', param_ranges[model_name].get('learning_rate', ['constant', 'invscaling', 'adaptive']))
        }
    elif model_name == 'LinearRegression':
        model = LinearRegression()
        params = {}
    elif model_name == 'KNeighborsRegressor':
        model = KNeighborsRegressor()
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', *param_ranges[model_name].get('n_neighbors', (1, 20))),
            'leaf_size': trial.suggest_int('leaf_size', *param_ranges[model_name].get('leaf_size', (10, 50)))
        }
    elif model_name == 'SVR':
        model = SVR()
        params = {
            'C': trial.suggest_float('C', *param_ranges[model_name].get('C', (0.1, 10.0))),
            'kernel': trial.suggest_categorical('kernel', param_ranges[model_name].get('kernel', ['linear', 'poly', 'rbf', 'sigmoid']))
        }
    elif model_name == 'RandomForestRegressor':
        model = RandomForestRegressor()
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *param_ranges[model_name].get('n_estimators', (10, 250))),
            'max_depth': trial.suggest_int('max_depth', *param_ranges[model_name].get('max_depth', (2, 32)))
        }
    elif model_name == 'ExtraTreesRegressor':
        model = ExtraTreesRegressor()
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *param_ranges[model_name].get("n_estimators", (10, 250))),
            'max_depth': trial.suggest_int('max_depth', *param_ranges[model_name].get("max_depth", (2, 32)))
        }
    elif model_name == 'AdaBoostRegressor':
        model = AdaBoostRegressor()
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *param_ranges[model_name].get('n_estimators', (10, 250))),
            'learning_rate': trial.suggest_float('learning_rate', *param_ranges[model_name].get('learning_rate', (0.01, 2.0)))
        }
    elif model_name == 'GradientBoostingRegressor':
        model = GradientBoostingRegressor()
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *param_ranges[model_name].get('n_estimators', (10, 250))),
            'learning_rate': trial.suggest_float('learning_rate', *param_ranges[model_name].get('learning_rate', (0.01, 1.0))),
            'max_depth': trial.suggest_int('max_depth', *param_ranges[model_name].get('max_depth', (2, 32)))
        }
    elif model_name == 'XGBRegressor':
        model = XGBRegressor()
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *param_ranges[model_name].get('n_estimators', (10, 250))),
            'learning_rate': trial.suggest_float('learning_rate', *param_ranges[model_name].get('learning_rate', (0.01, 1.0))),
            'max_depth': trial.suggest_int("max_depth", *param_ranges[model_name].get("max_depth", (2, 32)))
        }
    elif model_name == 'CatBoostRegressor':
        model = CatBoostRegressor()
        params = {
            'iterations': trial.suggest_int('iterations', *param_ranges[model_name].get('iterations', (50, 200))),
            'learning_rate': trial.suggest_float('learning_rate', *param_ranges[model_name].get('learning_rate', (0.01, 1.0))),
            'depth': trial.suggest_int('depth', *param_ranges[model_name].get('depth', (2, 10)))
        }
    elif model_name == 'MLPRegressor':
        model = MLPRegressor()
        params = {
            'hidden_layer_sizes': trial.suggest_int('hidden_layer_sizes', *param_ranges[model_name].get('hidden_layer_sizes', (50, 200))),
            'alpha': trial.suggest_float('alpha', *param_ranges[model_name].get('alpha', (0.0001, 0.01))),
            'learning_rate': trial.suggest_categorical('learning_rate', param_ranges[model_name].get('learning_rate', ['constant', 'invscaling', 'adaptive']))
        }
    elif model_name == 'Ridge':
        model = Ridge()
        params = {
            'alpha': trial.suggest_float('alpha', *param_ranges[model_name].get('alpha', (0.1, 10.0)))
        }
    elif model_name == 'ElasticNetCV':
        model = ElasticNetCV()
        params = {
            'l1_ratio': trial.suggest_float('l1_ratio', *param_ranges[model_name].get('l1_ratio', (0.1, 1.0)))
        }
    elif model_name in add_model:
        model, custom_params = add_model[model_name]
        params = {}
        for param, range_ in custom_params.items():
            if isinstance(range_[0], int) and isinstance(range_[1], int):
                params[param] = trial.suggest_int(param, *range_)
            elif isinstance(range_[0], float) or isinstance(range_[1], float):
                params[param] = trial.suggest_float(param, *range_)
            else:
                params[param] = trial.suggest_categorical(param, range_)
    else:
        raise ValueError(f"Model {model_name} is not supported.")

    return model, params