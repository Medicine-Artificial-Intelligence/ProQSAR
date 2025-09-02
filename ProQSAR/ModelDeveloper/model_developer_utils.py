import numpy as np
import pandas as pd
from typing import Union, Optional, List, Dict
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    RepeatedKFold,
)
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
from sklearn.dummy import DummyClassifier, DummyRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    log_loss,
    brier_score_loss,
    r2_score,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    mean_absolute_percentage_error,
    max_error,
    matthews_corrcoef,
)


def _get_task_type(data: pd.DataFrame, activity_col: str) -> str:
    """
    Infer whether the modelling task is classification or regression.

    Logic:
      - If the target column contains exactly 2 unique values -> binary classification ('C')
      - If the target column contains >2 unique values -> regression ('R')
      - Otherwise raise ValueError.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing features and the target column.
    activity_col : str
        Column name containing the target values.

    Returns
    -------
    str
        'C' for classification or 'R' for regression.

    Raises
    ------
    ValueError
        If there are fewer than 2 unique values in the target column.
    """
    y_data = data[activity_col]
    unique_targets = len(np.unique(y_data))
    if unique_targets == 2:
        return "C"
    elif unique_targets > 2:
        return "R"
    else:
        raise ValueError("Insufficient number of categories to determine model type.")


def _get_model_map(
    task_type: Optional[str] = None,
    add_model: dict = {},
    n_jobs: int = 1,
    random_state: Optional[int] = 42,
) -> Dict[str, object]:
    """
    Build a default dictionary mapping model names -> instantiated estimators.

    Parameters
    ----------
    task_type : Optional[str]
        'C' for classification, 'R' for regression, or None to include both sets.
    add_model : dict, optional
        Additional model entries to merge into the default map. Values may be:
        - an estimator instance, or
        - a tuple whose first element is an estimator instance (legacy support).
        If the supplied estimator is not fitted, and it accepts 'random_state'
        in get_params(), this function will set that parameter for reproducibility.
    n_jobs : int, default=1
        Number of parallel jobs passed to applicable estimators.
    random_state : Optional[int], default=42
        Seed used to set 'random_state' on added estimators if applicable.

    Returns
    -------
    dict[str, object]
        Mapping from human-readable model name to sklearn-like estimator instance.

    Raises
    ------
    ValueError
        If `task_type` is not 'C', 'R', or None.
    """
    model_map_c = {
        "DummyClassifier": DummyClassifier(random_state=random_state),
        "LogisticRegression": LogisticRegression(
            max_iter=10000, solver="liblinear", random_state=random_state, n_jobs=n_jobs
        ),
        "KNeighborsClassifier": KNeighborsClassifier(n_jobs=n_jobs),
        "SVC": SVC(probability=True, max_iter=10000, random_state=random_state),
        "RandomForestClassifier": RandomForestClassifier(
            random_state=random_state, n_jobs=n_jobs
        ),
        "ExtraTreesClassifier": ExtraTreesClassifier(
            random_state=random_state, n_jobs=n_jobs
        ),
        "AdaBoostClassifier": AdaBoostClassifier(
            n_estimators=100, random_state=random_state
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            random_state=random_state
        ),
        "XGBClassifier": XGBClassifier(
            random_state=random_state, verbosity=0, eval_metric="logloss", n_jobs=n_jobs
        ),
        "CatBoostClassifier": CatBoostClassifier(
            random_state=random_state, verbose=0, thread_count=n_jobs
        ),
        "MLPClassifier": MLPClassifier(
            alpha=0.01,
            max_iter=10000,
            hidden_layer_sizes=(150,),
            random_state=random_state,
        ),
    }

    model_map_r = {
        "DummyRegressor": DummyRegressor(),
        "LinearRegression": LinearRegression(n_jobs=n_jobs),
        "KNeighborsRegressor": KNeighborsRegressor(n_jobs=n_jobs),
        "SVR": SVR(),
        "RandomForestRegressor": RandomForestRegressor(
            random_state=random_state, n_jobs=n_jobs
        ),
        "ExtraTreesRegressor": ExtraTreesRegressor(
            random_state=random_state, n_jobs=n_jobs
        ),
        "AdaBoostRegressor": AdaBoostRegressor(random_state=random_state),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            random_state=random_state
        ),
        "XGBRegressor": XGBRegressor(
            random_state=random_state,
            verbosity=0,
            objective="reg:squarederror",
            n_jobs=n_jobs,
        ),
        "CatBoostRegressor": CatBoostRegressor(
            random_state=random_state, verbose=0, thread_count=n_jobs
        ),
        "MLPRegressor": MLPRegressor(
            alpha=0.01,
            max_iter=10000,
            hidden_layer_sizes=(150,),
            random_state=random_state,
        ),
        "Ridge": Ridge(random_state=random_state),
        "ElasticNetCV": ElasticNetCV(cv=5, n_jobs=n_jobs, random_state=random_state),
    }

    if task_type == "C":
        model_map = model_map_c
    elif task_type == "R":
        model_map = model_map_r
    elif task_type is None:
        model_map = {**model_map_c, **model_map_r}
    else:
        raise ValueError(
            "Invalid task_type. Please choose 'C' for classification or 'R' for regression."
        )

    if add_model:
        for name, val in add_model.items():
            if isinstance(val, tuple) and isinstance(val[0], BaseEstimator):
                model = val[0]
            else:
                model = val

            # Check if the model has not been fitted yet.
            try:
                check_is_fitted(model)
            except NotFittedError:
                if "random_state" in model.get_params():
                    model.set_params(random_state=random_state)

            model_map[name] = model

    return model_map


def _get_cv_strategy(
    task_type: str,
    n_splits: int = 10,
    n_repeats: int = 3,
    random_state: Optional[int] = 42,
) -> Union[RepeatedStratifiedKFold, RepeatedKFold]:
    """
    Return a repeated CV splitting strategy appropriate for the task.

    Parameters
    ----------
    task_type : str
        'C' for classification or 'R' for regression.
    n_splits : int
        Number of folds per repetition.
    n_repeats : int
        Number of repetitions.
    random_state : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    RepeatedStratifiedKFold | RepeatedKFold
        CV splitter to be used in cross-validation.

    Raises
    ------
    ValueError
        If task_type is not 'C' or 'R'.
    """
    if task_type == "C":
        return RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
    elif task_type == "R":
        return RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
    else:
        raise ValueError(
            "Invalid task_type. Please choose 'C' for classification or 'R' for regression."
        )


def _get_cv_scoring(task_type: str) -> list:
    """
    Return a list of scoring metric keys appropriate for sklearn's cross_validate.

    Parameters
    ----------
    task_type : str
        'C' or 'R'

    Returns
    -------
    list[str]
        Metric names (strings) that sklearn understands for scoring.

    Raises
    ------
    ValueError
        If task_type is not 'C' or 'R'.
    """
    if task_type == "C":
        return [
            "roc_auc",
            "average_precision",
            "accuracy",
            "recall",
            "precision",
            "f1",
            "neg_log_loss",
            "neg_brier_score",
            "matthews_corrcoef",
        ]
    elif task_type == "R":
        return [
            "r2",
            "neg_mean_squared_error",
            "neg_root_mean_squared_error",
            "neg_mean_absolute_error",
            "neg_median_absolute_error",
            "neg_mean_absolute_percentage_error",
            "max_error",
        ]
    else:
        raise ValueError(
            "Invalid task_type. Please choose 'C' for classification or 'R' for regression."
        )


def _get_ev_scoring(
    task_type: str,
    y_test: pd.Series,
    y_test_pred: pd.Series,
    y_test_proba: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """
    Compute evaluation metrics for a single test fold.

    Parameters
    ----------
    task_type : str
        'C' for classification, 'R' for regression.
    y_test : pd.Series
        True labels/values for the test split.
    y_test_pred : pd.Series
        Predicted labels/values.
    y_test_proba : pd.Series, optional
        Predicted probabilities for the positive class (classification only).

    Returns
    -------
    dict[str, float]
        Mapping metric_name -> computed value for the fold.

    Raises
    ------
    ValueError
        If task_type is not supported or required inputs for a metric are missing.
    """
    if task_type == "C":
        scoring_dict = {
            "roc_auc": roc_auc_score(y_test, y_test_proba),
            "average_precision": average_precision_score(y_test, y_test_proba),
            "accuracy": accuracy_score(y_test, y_test_pred),
            "recall": recall_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred),
            "f1": f1_score(y_test, y_test_pred, average="binary"),
            "log_loss": log_loss(y_test, y_test_proba),
            "brier_score": brier_score_loss(y_test, y_test_proba),
            "matthews_corrcoef": matthews_corrcoef(y_test, y_test_pred),
        }

    elif task_type == "R":
        scoring_dict = {
            "r2": r2_score(y_test, y_test_pred),
            "mean_squared_error": mean_squared_error(y_test, y_test_pred),
            "root_mean_squared_error": root_mean_squared_error(y_test, y_test_pred),
            "mean_absolute_error": mean_absolute_error(y_test, y_test_pred),
            "median_absolute_error": median_absolute_error(y_test, y_test_pred),
            "mean_absolute_percentage_error": mean_absolute_percentage_error(
                y_test, y_test_pred
            ),
            "max_error": max_error(y_test, y_test_pred),
        }
    else:
        raise ValueError(
            "Invalid task type. Please choose 'C' for classification or 'R' for regression."
        )

    return scoring_dict


def _match_cv_ev_metrics(cv_scoring: Union[str, List[str]]) -> List:
    """
    Convert cross-validation scoring metric keys (possibly prefixed with 'neg_')
    into corresponding evaluation metric names used by _get_ev_scoring.

    Example:
      - 'neg_mean_squared_error' -> 'mean_squared_error'
      - 'accuracy' -> 'accuracy'

    Parameters
    ----------
    cv_scoring : str | list[str]
        CV scoring key(s) as used by sklearn's cross_validate.

    Returns
    -------
    list[str]
        Corresponding evaluation metric names (without 'neg_' prefix).
    """
    if isinstance(cv_scoring, str):
        cv_scoring = [cv_scoring]

    ev_scoring = []
    for cv_name in cv_scoring:
        # Determine the corresponding evaluation metric name by removing 'neg_' if present.
        ev_name = cv_name[4:] if cv_name.startswith("neg_") else cv_name
        ev_scoring.append(ev_name)

    return ev_scoring
