import numpy as np
import pandas as pd
from typing import Union, Optional
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
    mean_absolute_error,
    median_absolute_error,
    mean_absolute_percentage_error,
    max_error,
)


def _get_task_type(data: pd.DataFrame, activity_col: str) -> str:
    """
    Determines the type of task based on the number of unique target values.

    Args:
        data (pd.DataFrame): Data containing the features and target.
        activity_col (str): Column name for the target variable.

    Returns:
        str: 'C' for classification (binary), 'R' for regression (continuous).

    Raises:
        ValueError: If insufficient categories to determine model type.
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
    task_type: str,
    add_model: Optional[dict] = None,
    n_jobs: int = -1,
) -> dict:
    """
    Retrieves a dictionary mapping model names to their corresponding estimators.

    Args:
        task_type (str): 'C' for classification, 'R' for regression.
        add_model (Optional[dict]): Additional models to add to the map.
        n_jobs (Optional[int]): Number of jobs for parallelization.

    Returns:
        dict: A dictionary of model names and estimators.
    """
    if task_type == "C":
        model_map = {
            "LogisticRegression": LogisticRegression(
                max_iter=10000, solver="liblinear", random_state=42, n_jobs=n_jobs
            ),
            "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=20, n_jobs=n_jobs),
            "SVC": SVC(probability=True, max_iter=10000),
            "RandomForestClassifier": RandomForestClassifier(
                random_state=42, n_jobs=n_jobs
            ),
            "ExtraTreesClassifier": ExtraTreesClassifier(
                random_state=42, n_jobs=n_jobs
            ),
            "AdaBoostClassifier": AdaBoostClassifier(n_estimators=100, random_state=42),
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
            "XGBClassifier": XGBClassifier(
                random_state=42, verbosity=0, eval_metric="logloss"
            ),
            "CatBoostClassifier": CatBoostClassifier(random_state=42, verbose=0),
            "MLPClassifier": MLPClassifier(
                alpha=0.01, max_iter=10000, hidden_layer_sizes=(150,), random_state=42
            ),
        }
    elif task_type == "R":
        model_map = {
            "LinearRegression": LinearRegression(n_jobs=n_jobs),
            "KNeighborsRegressor": KNeighborsRegressor(n_jobs=n_jobs),
            "SVR": SVR(),
            "RandomForestRegressor": RandomForestRegressor(
                random_state=42, n_jobs=n_jobs
            ),
            "ExtraTreesRegressor": ExtraTreesRegressor(random_state=42, n_jobs=n_jobs),
            "AdaBoostRegressor": AdaBoostRegressor(random_state=42),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
            "XGBRegressor": XGBRegressor(
                random_state=42,
                verbosity=0,
                objective="reg:squarederror",
            ),
            "CatBoostRegressor": CatBoostRegressor(random_state=42, verbose=0),
            "MLPRegressor": MLPRegressor(
                alpha=0.01, max_iter=10000, hidden_layer_sizes=(150,), random_state=42
            ),
            "Ridge": Ridge(),
            "ElasticNetCV": ElasticNetCV(cv=5, n_jobs=n_jobs),
        }
    else:
        raise ValueError(
            "Invalid task_type. Please choose 'C' for classification or 'R' for regression."
        )

    if add_model:
        model_map.update(add_model)

    return model_map


def _get_cv_strategy(
    task_type: str,
    n_splits: int = 10,
    n_repeats: int = 3,
) -> Union[RepeatedStratifiedKFold, RepeatedKFold]:
    """
    Defines the cross-validation strategy based on task type.

    Args:
        task_type (str): 'C' for classification, 'R' for regression.
        n_splits (int): Number of splits for cross-validation.
        n_repeats (int): Number of repetitions for cross-validation.

    Returns:
        RepeatedStratifiedKFold or RepeatedKFold: Cross-validation strategy.
    """
    if task_type == "C":
        return RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=42
        )
    elif task_type == "R":
        return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    else:
        raise ValueError(
            "Invalid task_type. Please choose 'C' for classification or 'R' for regression."
        )


def _get_cv_scoring_list(task_type: str) -> list:
    """
    Returns a list of scoring metrics based on the task type.

    Args:
        task_type (str): 'C' for classification, 'R' for regression.

    Returns:
        list: List of scoring metrics.
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


def _get_ev_scoring_dict(
    task_type: str,
    y_test: pd.Series,
    y_test_pred: pd.Series,
    y_test_proba: Optional[pd.Series] = None,
) -> dict:
    """
    Returns a dictionary of evaluation metrics based on the task type.

    Args:
        task_type (str): 'C' for classification, 'R' for regression.
        y_test (pd.Series): Ground truth values.
        y_test_pred (pd.Series): Predicted values.
        y_test_proba (Optional[pd.Series]): Predicted probabilities (for classification).

    Returns:
        dict: A dictionary of evaluation metrics.
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
        }

    elif task_type == "R":
        scoring_dict = {
            "r2": r2_score(y_test, y_test_pred),
            "mean_squared_error": mean_squared_error(y_test, y_test_pred),
            "root_mean_squared_error": mean_squared_error(
                y_test, y_test_pred, squared=False
            ),
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
