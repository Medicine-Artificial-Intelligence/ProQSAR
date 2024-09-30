import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.base import BaseEstimator
from typing import Union, Optional, List
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
from sklearn.model_selection import cross_validate
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
    Determines the task type (classification or regression) based on the target variable.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset containing the target variable.

    Returns
    -------
    str
        The type of task: "C" for classification or "R" for regression.

    Raises
    ------
    ValueError
        If the number of unique categories in the target variable is insufficient.
    """   
    y_data = data[activity_col]
    unique_targets = len(np.unique(y_data))
    if unique_targets == 2:
        return "C"
    elif unique_targets > 2:
        return "R"
    else:
        raise ValueError(
            "Insufficient number of categories to determine model type."
            )
        
def _get_method_map(
    task_type: str, 
    add_method: Optional[dict] = None
    ) -> dict[str, BaseEstimator]:
    """
    Gets the method map based on the task type.

    Parameters
    ----------
    task_type : str
        The type of task ("C" or "R").

    Returns
    -------
    dict[str, BaseEstimator]
        The method map for the task type.
    """
    if task_type == "C":
        method_map = {
            "Logistic": LogisticRegression(
                max_iter=10000, solver="liblinear", random_state=42
            ),
            "KNN": KNeighborsClassifier(n_neighbors=20),
            "SVM": SVC(probability=True, max_iter=10000),
            "RF": RandomForestClassifier(random_state=42),
            "ExT": ExtraTreesClassifier(random_state=42),
            "Ada": AdaBoostClassifier(n_estimators=100, random_state=42),
            "Grad": GradientBoostingClassifier(random_state=42),
            "XGB": XGBClassifier(random_state=42, verbosity=0, eval_metric="logloss"),
            "CatB": CatBoostClassifier(random_state=42, verbose=0),
            "MLP": MLPClassifier(
                alpha=0.01, max_iter=10000, hidden_layer_sizes=(150,), random_state=42
            ),
        }
    else:
        method_map = {
            "Linear": LinearRegression(),
            "KNN": KNeighborsRegressor(),
            "SVM": SVR(),
            "RF": RandomForestRegressor(random_state=42),
            "ExT": ExtraTreesRegressor(random_state=42),
            "Ada": AdaBoostRegressor(random_state=42),
            "Grad": GradientBoostingRegressor(random_state=42),
            "XGB": XGBRegressor(
                random_state=42, verbosity=0, objective="reg:squarederror"
            ),
            "CatB": CatBoostRegressor(random_state=42, verbose=0),
            "MLP": MLPRegressor(
                alpha=0.01, max_iter=10000, hidden_layer_sizes=(150,), random_state=42
            ),
            "Ridge": Ridge(),
            "ElasticNet": ElasticNetCV(cv=5),
        }

    if add_method:
        method_map.update(add_method)

    return method_map

def _get_cv_strategy(
    task_type: str
    ) -> Union[RepeatedStratifiedKFold, RepeatedKFold]:
        """
        Determines the cross-validation strategy based on the task type.

        Parameters
        ----------
        task_type : str
            The type of task ("C" or "R").

        Returns
        -------
        Union[RepeatedStratifiedKFold, RepeatedKFold]
            The cross-validation strategy.
        """
        if task_type == "C":
            return RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
        else:
            return RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
        
def _get_iv_scoring_list(task_type: str):
    """
    Determines the scoring target for model selection based on the task type.

    Parameters:
        task_type (str): The type of task ('C' for classification or 'R' for regression).

    Returns:
        str: The scoring target metric.
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
    else:
        return [
            "r2",
            "neg_mean_squared_error",
            "neg_root_mean_squared_error",
            "neg_mean_absolute_error",
            "neg_median_absolute_error",
            "neg_mean_absolute_percentage_error",
            "max_error",
        ]


def _get_ev_scoring_(
    task_type: str, y_test, y_test_pred, y_test_proba
) -> str:
    """
    Compute external validation scoring metrics.

    Args:
        task_type (str): The type of task ('C' for classification, 'R' for regression).
        y_test (np.ndarray): True labels.
        y_test_pred (np.ndarray): Predicted labels.
        y_test_proba (Optional[np.ndarray]): Predicted probabilities (for classification).

    Returns:
        dict: A dictionary with scoring metrics.
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

    else:
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

    return scoring_dict

