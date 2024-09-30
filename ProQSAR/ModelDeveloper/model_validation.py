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
from ProQSAR.ModelDeveloper.model_developer_utils import (
    _get_task_type,
    _get_method_map,
    _get_cv_strategy,
     _get_scoring_target,
    _get_iv_scoring_list,
    _get_ev_scoring_dict,
    _plot_compare_models,
)

def _get_best_method(data: pd.DataFrame, scoring_target: str) -> str:
   
    result_df = iv_report(data, [scoring_target])
    best_method = result_df.loc[result_df[f"{scoring_target}_mean"].idxmax(), "Method"]
    return best_method
        
def iv_report(
    data: pd.DataFrame,
    activity_col: str,
    id_col: str,
    add_method: Optional[dict] = None,
    select_method: Optional[List[str]] = None,
    scoring_list: Optional[List[str]] = None,
    visualize: Optional[str] = "box",
    save_fig: Optional[str] = None,
    save_csv: Optional[str] = None,
    save_dir: Optional[str] = None,
    n_jobs: int = -1
) -> pd.DataFrame:

    X_data = data.drop([activity_col, id_col], axis=1)
    y_data = data[activity_col]

    task_type = _get_task_type(data, activity_col)
    method_map = _get_method_map(task_type, add_method)
    cv = _get_cv_strategy(task_type)

    if scoring_list is None:
        scoring_list = _get_iv_scoring_list(task_type)
    else:
        scoring_list = scoring_list

    comparison_result = []
    models_to_compare = {}

    if select_method is None:
        models_to_compare = method_map
    else:
        for name in select_method:
            if name in method_map:
                models_to_compare.update({name: method_map[name]})
            else:
                raise ValueError(f"Method '{name}' is not recognized.")

    for name, model in models_to_compare.items():
        scores = cross_validate(
            model,
            X_data,
            y_data,
            cv=cv,
            scoring=scoring_list,
            n_jobs=n_jobs,
        )
        for metric in scoring_list:
            metric_scores = scores[f"test_{metric}"]
            method_result = {
                "Method": name,
                f"{metric}_mean": round(np.mean(metric_scores), 3),
                f"{metric}_std": round(np.std(metric_scores, 3)),
                f"{metric}_median": round(np.median(metric_scores), 3),
            }
            for i, score in enumerate(metric_scores):
                method_result[f"{metric}_{i+1}"] = score
        comparison_result.append(method_result)

    iv_df = pd.DataFrame(comparison_result)

########### chua sua ###########
    display(iv_df)

    if visualize:
        _plot_compare_models()

    if save_dir:
        save_name = "iv_report" if save_name is None else save_name
        self.iv_df.to_csv(f"{self.save_dir}/{save_name}.csv")
########### chua sua ###########

    return iv_df


def _plot_compare_models(self) -> None:
    """
    Plots the comparison of model performance based on internal validation scores.

    Raises:
        ValueError: If an invalid comparison visualization type is specified.
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(20, 10))

    score_columns = [
        col for col in self.iv_comparison_report.columns if col.startswith("Score_")
    ]
    melted_result = self.iv_comparison_report.melt(
        id_vars=["Method"],
        value_vars=score_columns,
        var_name="Score",
        value_name="Value",
    )

    if self.comparison_visual == "box":
        plot = sns.boxplot(
            x="Method",
            y="Value",
            data=melted_result,
            showmeans=True,
            width=0.5,
            palette="plasma",
            meanprops={
                "marker": "o",
                "markerfacecolor": "red",
                "markeredgecolor": "black",
            },
            medianprops={"color": "red", "linewidth": 2},
            boxprops={"edgecolor": "w"},
        )
    elif self.comparison_visual == "bar":
        plot = sns.barplot(
            x="Method",
            y="Value",
            data=melted_result,
            errorbar="sd",
            palette="plasma",
            width=0.5,
            color="black",
        )
    elif self.comparison_visual == "violin":
        plot = sns.violinplot(
            x="Method", y="Value", data=melted_result, inner=None, palette="plasma"
        )
        sns.stripplot(
            x="Method",
            y="Value",
            data=melted_result,
            color="white",
            size=5,
            jitter=True,
        )
    else:
        raise ValueError(
            f"Invalid comparison_visual '{self.comparison_visual}'. Choose 'box', 'bar' or 'violin'."
        )

    plot.set_title("Compare performance of different models", fontsize=16)
    plot.set_xlabel("Model", fontsize=14)
    plot.set_ylabel(f"{self.scoring_target.capitalize()} Score", fontsize=14)

    # Adding the mean values to the plot
    for i, row in self.iv_comparison_report.iterrows():
        position = 0.05 if self.comparison_visual == "bar" else (row["Mean"] + 0.015)
        plot.text(
            i,
            position,
            str(row["Mean"]),
            horizontalalignment="center",
            size="x-large",
            color="w",
            weight="semibold",
        )

    if self.save_fig and self.save_dir:
        plt.savefig(
            f"{self.save_dir}/iv_model_comparison_{self.scoring_target}_{self.comparison_visual}.png",
            dpi=300,
        )

    plt.show()


def ev_report(
    self,
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    select_model: Optional[List[str]] = None,
    scoring_list: Optional[List[str]] = None,
    save_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate an external validation report for selected models.

    Args:
        data_train (pd.DataFrame): Training dataset.
        data_test (pd.DataFrame): Testing dataset.
        select_model (Optional[List[str]]): List of model names to evaluate. If None, evaluate all models.
        scoring_list (Optional[List[str]]): List of metrics to include in the report.
        save_name (Optional[str]): The name for saving the report.

    Returns:
        pd.DataFrame: DataFrame containing the external validation report.
    """
    X_train = data_train.drop([self.activity_col, self.id_col], axis=1)
    y_train = data_train[self.activity_col]
    X_test = data_test.drop([self.activity_col, self.id_col], axis=1)
    y_test = data_test[self.activity_col]

    self.task_type = _get_task_type(data_train, self.activity_col)
    self.method_map = _get_method_map(self.task_type)

    models_to_compare = {}
    if select_model is None:
        models_to_compare = self.method_map
    else:
        for name in select_model:
            if name in self.method_map:
                models_to_compare.update({name: self.method_map[name]})
            else:
                raise ValueError(f"Method '{name}' is not recognized.")

    ev_score = {}
    for name, model in models_to_compare.items():
        model.fit(X=X_train, y=y_train)
        y_test_pred = model.predict(X_test)
        y_test_proba = (
            model.predict_proba(X_test)[:, 1] if self.task_type == "C" else None
        )

        scoring_dict = self._get_ev_scoring_dict(
            self.task_type, y_test, y_test_pred, y_test_proba
        )

        if scoring_list is None:
            ev_score[name] = scoring_dict
        else:
            ev_score[name] = {}
            for metric in scoring_list:
                if metric in scoring_dict:
                    ev_score[name].update({metric: scoring_dict[metric]})
                else:
                    raise ValueError(f"'{metric}' is not recognized.")

    self.ev_df = pd.DataFrame(ev_score).T

    if self.save_dir:
        save_name = "ev_report" if save_name is None else save_name
        self.ev_df.to_csv(f"{self.save_dir}/{save_name}.csv")

    return self.ev_df
