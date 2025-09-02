import logging
from typing import Dict, Tuple, Union
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


def _get_model_list(task_type: str, add_model: dict = {}) -> list[str]:
    """
    Return a list of default model names appropriate for the task.

    Parameters
    ----------
    task_type : str
        'C' for classification or 'R' for regression.
    add_model : dict, optional
        Additional user-supplied models (mapping name -> model spec). The keys
        of this dict will be appended to the returned list so callers can see
        custom model names.

    Returns
    -------
    list[str]
        A list of model names (strings). If an unexpected error occurs an
        empty list is returned (and the error is logged).
    """
    try:
        if task_type == "C":
            model_list = [
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
        elif task_type == "R":
            model_list = [
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

        return model_list + list(add_model.keys())

    except Exception as e:
        logging.error(f"Error in _get_model_list: {e}")
        return []


def _get_model_and_params(
    trial, model_name: str, param_ranges: dict = {}, add_model: dict = {}
) -> Tuple[object, Dict[str, Union[int, float, str]]]:
    """
    Construct a model instance and a dict of hyperparameters sampled from a
    trial object (e.g., an Optuna trial).

    The function supports many common sklearn / xgboost / catboost models and
    returns a tuple `(model, params)` where `model` is an instantiated estimator
    and `params` is a dictionary containing sampled hyperparameter values for
    that estimator. If `model_name` appears in `add_model`, a user-specified
    model and parameter ranges can be used.

    Parameters
    ----------
    trial : Any
        Optimization trial object exposing suggest_float / suggest_int /
        suggest_categorical methods (e.g., optuna.trial.Trial). The code
        calls trial.suggest_* directly to sample parameter values.
    model_name : str
        Name of the model to instantiate (one of the supported names).
    param_ranges : dict, optional
        Mapping model_name -> dict of parameter ranges. Each parameter range is
        a tuple of either ints or floats or a list for categorical choices.
        Example:
            param_ranges = {
                "RandomForestClassifier": {"n_estimators": (50, 200)}
            }
    add_model : dict, optional
        Mapping for custom models. If `model_name in add_model`, the function
        will expect add_model[model_name] to be a tuple like `(model_instance, custom_param_ranges)`
        where `custom_param_ranges` is a dict mapping parameter name -> range.

    Returns
    -------
    (model, params) : tuple
        model : object
            Instantiated model object (not yet fitted).
        params : dict
            Sampled hyperparameter values for the model. Keys are parameter names.

    Raises
    ------
    ValueError
        If `model_name` is not supported or not found in `add_model`.
    Exception
        Any unexpected exception is logged and re-raised.
    """
    try:
        if model_name == "LogisticRegression":
            model = LogisticRegression()
            params = {
                "C": trial.suggest_float(
                    "C", *param_ranges.get(model_name, {}).get("C", (0.01, 10.0))
                ),
                "max_iter": trial.suggest_int(
                    "max_iter",
                    *param_ranges.get(model_name, {}).get("max_iter", (100, 300)),
                ),
            }
        elif model_name == "KNeighborsClassifier":
            model = KNeighborsClassifier()
            params = {
                "n_neighbors": trial.suggest_int(
                    "n_neighbors",
                    *param_ranges.get(model_name, {}).get("n_neighbors", (1, 20)),
                ),
                "leaf_size": trial.suggest_int(
                    "leaf_size",
                    *param_ranges.get(model_name, {}).get("leaf_size", (10, 50)),
                ),
            }
        elif model_name == "SVC":
            model = SVC()
            params = {
                "C": trial.suggest_float(
                    "C", *param_ranges.get(model_name, {}).get("C", (0.1, 10.0))
                ),
                "kernel": trial.suggest_categorical(
                    "kernel",
                    param_ranges.get(model_name, {}).get("kernel", ["linear", "rbf"]),
                ),
                "gamma": trial.suggest_float(
                    "gamma",
                    *param_ranges.get(model_name, {}).get("gamma", (0.001, 1.0)),
                ),
            }
        elif model_name == "RandomForestClassifier":
            model = RandomForestClassifier()
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    *param_ranges.get(model_name, {}).get("n_estimators", (10, 250)),
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    *param_ranges.get(model_name, {}).get("max_depth", (2, 32)),
                ),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split",
                    *param_ranges.get(model_name, {}).get("min_samples_split", (2, 10)),
                ),
            }
        elif model_name == "ExtraTreesClassifier":
            model = ExtraTreesClassifier()
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    *param_ranges.get(model_name, {}).get("n_estimators", (10, 250)),
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    *param_ranges.get(model_name, {}).get("max_depth", (2, 32)),
                ),
            }
        elif model_name == "AdaBoostClassifier":
            model = AdaBoostClassifier()
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    *param_ranges.get(model_name, {}).get("n_estimators", (10, 250)),
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    *param_ranges.get(model_name, {}).get("learning_rate", (0.01, 2.0)),
                ),
            }
        elif model_name == "GradientBoostingClassifier":
            model = GradientBoostingClassifier()
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    *param_ranges.get(model_name, {}).get("n_estimators", (10, 250)),
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    *param_ranges.get(model_name, {}).get("learning_rate", (0.01, 2.0)),
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    *param_ranges.get(model_name, {}).get("max_depth", (2, 32)),
                ),
            }
        elif model_name == "XGBClassifier":
            model = XGBClassifier()
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    *param_ranges.get(model_name, {}).get("n_estimators", (10, 250)),
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    *param_ranges.get(model_name, {}).get("learning_rate", (0.01, 1.0)),
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    *param_ranges.get(model_name, {}).get("max_depth", (2, 32)),
                ),
            }
        elif model_name == "CatBoostClassifier":
            model = CatBoostClassifier()
            params = {
                "iterations": trial.suggest_int(
                    "iterations",
                    *param_ranges.get(model_name, {}).get("iterations", (50, 200)),
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    *param_ranges.get(model_name, {}).get("learning_rate", (0.01, 1.0)),
                ),
                "depth": trial.suggest_int(
                    "depth", *param_ranges.get(model_name, {}).get("depth", (2, 10))
                ),
            }
        elif model_name == "MLPClassifier":
            model = MLPClassifier(max_iter=10000)
            params = {
                "hidden_layer_sizes": trial.suggest_int(
                    "hidden_layer_sizes",
                    *param_ranges.get(model_name, {}).get(
                        "hidden_layer_sizes", (50, 200)
                    ),
                ),
                "alpha": trial.suggest_float(
                    "alpha",
                    *param_ranges.get(model_name, {}).get("alpha", (0.0001, 0.01)),
                ),
            }
        elif model_name == "LinearRegression":
            model = LinearRegression()
            params = {}
        elif model_name == "KNeighborsRegressor":
            model = KNeighborsRegressor()
            params = {
                "n_neighbors": trial.suggest_int(
                    "n_neighbors",
                    *param_ranges.get(model_name, {}).get("n_neighbors", (1, 20)),
                ),
                "leaf_size": trial.suggest_int(
                    "leaf_size",
                    *param_ranges.get(model_name, {}).get("leaf_size", (10, 50)),
                ),
            }
        elif model_name == "SVR":
            model = SVR()
            params = {
                "C": trial.suggest_float(
                    "C", *param_ranges.get(model_name, {}).get("C", (0.1, 10.0))
                ),
                "kernel": trial.suggest_categorical(
                    "kernel",
                    param_ranges.get(model_name, {}).get(
                        "kernel", ["linear", "poly", "rbf", "sigmoid"]
                    ),
                ),
            }
        elif model_name == "RandomForestRegressor":
            model = RandomForestRegressor()
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    *param_ranges.get(model_name, {}).get("n_estimators", (10, 250)),
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    *param_ranges.get(model_name, {}).get("max_depth", (2, 32)),
                ),
            }
        elif model_name == "ExtraTreesRegressor":
            model = ExtraTreesRegressor()
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    *param_ranges.get(model_name, {}).get("n_estimators", (10, 250)),
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    *param_ranges.get(model_name, {}).get("max_depth", (2, 32)),
                ),
            }
        elif model_name == "AdaBoostRegressor":
            model = AdaBoostRegressor()
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    *param_ranges.get(model_name, {}).get("n_estimators", (10, 250)),
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    *param_ranges.get(model_name, {}).get("learning_rate", (0.01, 2.0)),
                ),
            }
        elif model_name == "GradientBoostingRegressor":
            model = GradientBoostingRegressor()
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    *param_ranges.get(model_name, {}).get("n_estimators", (10, 250)),
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    *param_ranges.get(model_name, {}).get("learning_rate", (0.01, 1.0)),
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    *param_ranges.get(model_name, {}).get("max_depth", (2, 32)),
                ),
            }
        elif model_name == "XGBRegressor":
            model = XGBRegressor()
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    *param_ranges.get(model_name, {}).get("n_estimators", (10, 250)),
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    *param_ranges.get(model_name, {}).get("learning_rate", (0.01, 1.0)),
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    *param_ranges.get(model_name, {}).get("max_depth", (2, 32)),
                ),
            }
        elif model_name == "CatBoostRegressor":
            model = CatBoostRegressor()
            params = {
                "iterations": trial.suggest_int(
                    "iterations",
                    *param_ranges.get(model_name, {}).get("iterations", (50, 200)),
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    *param_ranges.get(model_name, {}).get("learning_rate", (0.01, 1.0)),
                ),
                "depth": trial.suggest_int(
                    "depth", *param_ranges.get(model_name, {}).get("depth", (2, 10))
                ),
            }
        elif model_name == "MLPRegressor":
            model = MLPRegressor(max_iter=10000)
            params = {
                "hidden_layer_sizes": trial.suggest_int(
                    "hidden_layer_sizes",
                    *param_ranges.get(model_name, {}).get(
                        "hidden_layer_sizes", (50, 200)
                    ),
                ),
                "alpha": trial.suggest_float(
                    "alpha",
                    *param_ranges.get(model_name, {}).get("alpha", (0.0001, 0.01)),
                ),
            }
        elif model_name == "Ridge":
            model = Ridge()
            params = {
                "alpha": trial.suggest_float(
                    "alpha", *param_ranges.get(model_name, {}).get("alpha", (0.1, 10.0))
                )
            }
        elif model_name == "ElasticNetCV":
            model = ElasticNetCV()
            params = {
                "l1_ratio": trial.suggest_float(
                    "l1_ratio",
                    *param_ranges.get(model_name, {}).get("l1_ratio", (0.1, 1.0)),
                )
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

    except Exception as e:
        logging.error(f"Error in _get_model_and_params for model {model_name}: {e}")
        raise
