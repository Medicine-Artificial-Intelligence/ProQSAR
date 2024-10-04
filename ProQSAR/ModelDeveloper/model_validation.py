import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
from sklearn.model_selection import cross_validate
from ProQSAR.ModelDeveloper.model_developer_utils import (
    _get_task_type,
    _get_method_map,
    _get_cv_strategy,
    _get_iv_scoring_list,
    _get_ev_scoring_dict,
)


def _plot_iv_report(
    report_df: pd.DataFrame,
    scoring_list: List[str],
    graph_type: Optional[str] = "box",
    save_fig: bool = False,
    fig_name: str = "iv_graph",
    save_dir: str = "Project/Model_Development",
) -> None:
    """
    Plots internal validation report for model comparison based on specified scoring metrics.

    Parameters:
    -----------
    report_df : pd.DataFrame
        DataFrame containing the model comparison scores.
    scoring_list : List[str]
        List of metrics used to evaluate model performance.
    graph_type : Optional[str]
        Type of graph to plot ('box', 'bar', 'violin'). Default is 'box'.
    save_fig : bool
        Whether to save the figure. Default is False.
    fig_name : str
        Name of the figure to save. Default is 'iv_graph'.
    save_dir : str
        Directory where the figure will be saved. Default is 'Project/Model_Development'.

    Returns:
    --------
    None
    """

    sns.set_style("whitegrid")

    for metric in scoring_list:
        plt.figure(figsize=(20, 10))

        score_columns = [
            col for col in report_df.columns if col.startswith(f"{metric}_fold")
        ]
        melted_result = report_df.melt(
            id_vars=["Method"],
            value_vars=score_columns,
            var_name="Score",
            value_name="Value",
        )

        if graph_type == "box":
            plot = sns.boxplot(
                x="Method",
                y="Value",
                data=melted_result,
                showmeans=True,
                width=0.5,
                palette="plasma",
                hue="Method",
                meanprops={
                    "marker": "o",
                    "markerfacecolor": "red",
                    "markeredgecolor": "black",
                },
                medianprops={"color": "red", "linewidth": 2},
                boxprops={"edgecolor": "w"},
            )
        elif graph_type == "bar":
            plot = sns.barplot(
                x="Method",
                y="Value",
                data=melted_result,
                errorbar="sd",
                palette="plasma",
                hue="Method",
                width=0.5,
                color="black",
            )
        elif graph_type == "violin":
            plot = sns.violinplot(
                x="Method",
                y="Value",
                data=melted_result,
                inner=None,
                palette="plasma",
                hue="Method",
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
                f"Invalid graph type '{graph_type}'. Choose 'box', 'bar' or 'violin'."
            )

        plot.set_title(
            f"Compare the performance of different models on {metric}", fontsize=16
        )
        plot.set_xlabel("Model", fontsize=14)
        plot.set_ylabel(f"{metric.capitalize()}", fontsize=14)

        # Adding the mean values to the plot
        for i, row in report_df.iterrows():
            position = 0.05 if graph_type == "bar" else (row[f"{metric}_mean"] + 0.015)
            plot.text(
                i,
                position,
                str(row[f"{metric}_mean"]),
                horizontalalignment="center",
                size="x-large",
                color="w",
                weight="semibold",
            )

        if save_fig:
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                f"{save_dir}/{fig_name}_{metric}_{graph_type}.png",
                dpi=300,
            )

    plt.show()


def iv_report(
    data: pd.DataFrame,
    activity_col: str,
    id_col: str,
    add_method: Optional[dict] = None,
    select_method: Optional[List[str]] = None,
    scoring_list: Optional[List[str]] = None,
    n_splits: int = 10,
    n_repeats: int = 3,
    visualize: Optional[str] = None,
    save_fig: bool = False,
    save_csv: bool = False,
    fig_name: str = "iv_graph",
    csv_name: str = "iv_report",
    save_dir: str = "Project/Model_Development",
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Performs internal validation (cross-validation) for multiple models and generates a report.

    Parameters:
    -----------
    data : pd.DataFrame
        The full dataset used for validation.
    activity_col : str
        The target column in the dataset.
    id_col : str
        The identifier column in the dataset.
    add_method : Optional[dict]
        Dictionary of additional models to include.
    select_method : Optional[List[str]]
        List of models to be selected for validation.
    scoring_list : Optional[List[str]]
        List of scoring metrics for the validation. If None, default metrics are used.
    n_splits : int
        Number of splits for cross-validation. Default is 10.
    n_repeats : int
        Number of repeats for cross-validation. Default is 3.
    visualize : Optional[str]
        Visualization type ('box', 'bar', 'violin'). Default is None.
    save_fig : bool
        Whether to save the figures generated. Default is False.
    save_csv : bool
        Whether to save the report to a CSV file. Default is False.
    fig_name : str
        File name for saving the figure.
    csv_name : str
        File name for saving the report.
    save_dir : str
        Directory where the figure/report will be saved.
    n_jobs : int
        Number of parallel jobs to run.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the validation results for each model.
    """

    X_data = data.drop([activity_col, id_col], axis=1)
    y_data = data[activity_col]

    task_type = _get_task_type(data, activity_col)
    method_map = _get_method_map(task_type, add_method, n_jobs)
    cv = _get_cv_strategy(task_type, n_splits=n_splits, n_repeats=n_repeats)

    if scoring_list is None:
        scoring_list = _get_iv_scoring_list(task_type)

    result = []
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

        method_result = {"Method": name}
        for metric in scoring_list:
            metric_scores = scores[f"test_{metric}"]
            method_result[f"{metric}_mean"] = round(np.mean(metric_scores), 3)
            method_result[f"{metric}_std"] = round(np.std(metric_scores), 3)
            method_result[f"{metric}_median"] = round(np.median(metric_scores), 3)
            for i, score in enumerate(metric_scores):
                method_result[f"{metric}_fold{i+1}"] = score

        result.append(method_result)

    iv_df = pd.DataFrame(result)

    if visualize:
        _plot_iv_report(
            report_df=iv_df,
            scoring_list=scoring_list,
            graph_type=visualize,
            save_fig=save_fig,
            fig_name=fig_name,
            save_dir=save_dir,
        )

    if save_csv:
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        iv_df.to_csv(f"{save_dir}/{csv_name}.csv")

    return iv_df


def ev_report(
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    activity_col: str,
    id_col: str,
    add_method: Optional[dict] = None,
    select_method: Optional[List[str]] = None,
    scoring_list: Optional[List[str]] = None,
    save_csv: bool = False,
    csv_name: str = "ev_report",
    save_dir: str = "Project/Model_Development",
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Performs external validation (on test data) for multiple models and generates a report.

    Parameters:
    -----------
    data_train : pd.DataFrame
        Training data used to fit the models.
    data_test : pd.DataFrame
        Test data used for evaluating the models.
    activity_col : str
        The target column in the dataset.
    id_col : str
        The identifier column in the dataset.
    add_method : Optional[dict]
        Dictionary of additional models to include.
    select_method : Optional[List[str]]
        List of models to be selected for validation.
    scoring_list : Optional[List[str]]
        List of scoring metrics for the validation. If None, default metrics are used.
    save_csv : bool
        Whether to save the report to a CSV file. Default is False.
    csv_name : str
        File name for saving the report.
    save_dir : str
        Directory where the figure/report will be saved.
    n_jobs : int
        Number of parallel jobs to run.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the evaluation results for each model.
    """

    X_train = data_train.drop([activity_col, id_col], axis=1)
    y_train = data_train[activity_col]
    X_test = data_test.drop([activity_col, id_col], axis=1)
    y_test = data_test[activity_col]

    task_type = _get_task_type(data_train, activity_col)
    method_map = _get_method_map(task_type, add_method, n_jobs)

    models_to_compare = {}
    if select_method is None:
        models_to_compare = method_map
    else:
        for name in select_method:
            if name in method_map:
                models_to_compare.update({name: method_map[name]})
            else:
                raise ValueError(f"Method '{name}' is not recognized.")

    ev_score = {}
    for name, model in models_to_compare.items():
        model.fit(X=X_train, y=y_train)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1] if task_type == "C" else None

        scoring_dict = _get_ev_scoring_dict(
            task_type, y_test, y_test_pred, y_test_proba
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

    ev_df = pd.DataFrame(ev_score).T

    if save_csv:
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        ev_df.to_csv(f"{save_dir}/{csv_name}.csv")

    return ev_df
