import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
from sklearn.model_selection import cross_validate
from ProQSAR.ModelDeveloper.model_developer_utils import (
    _get_task_type,
    _get_model_map,
    _get_cv_strategy,
    _get_cv_scoring_list,
    _get_ev_scoring_dict,
)


def _plot_cv_report(
    report_df: pd.DataFrame,
    scoring_list: List[str],
    graph_type: Optional[str] = "box",
    save_fig: bool = False,
    fig_prefix: str = "cv_graph",
    save_dir: str = "Project/ModelDevelopment",
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
    fig_prefix : str
        Name of the figure to save. Default is 'cv_graph'.
    save_dir : str
        Directory where the figure will be saved. Default is 'Project/Model_Development'.

    Returns:
    --------
    None
    """
    
    sns.set_context("notebook")
    sns.set_style("whitegrid")

    nrow = math.ceil(len(scoring_list) / 2)
    
    nmethod = len(report_df.columns.unique())

    figure, axes = plt.subplots(
            nrow, 2, sharex=False, sharey=False, figsize=(3 * nmethod, 7 * nrow)
        )
    axes = axes.flatten()  # Turn 2D array to 1D array

    for i, metric in enumerate(scoring_list):

        # Select only rows that correspond to the current metric
        metric_rows = report_df.xs(metric, level="scoring")

        # Melt the DataFrame to long format for plotting
        melted_result = metric_rows.reset_index().melt(
            id_vars=["cv_cycle"],
            var_name="Model",
            value_name="Value",
        )
        
        if graph_type == "box":
            plot = sns.boxplot(
                x="Model",
                y="Value",
                data=melted_result,
                ax=axes[i],
                showmeans=True,
                width=0.5,
                palette="plasma",
                hue="Model",
                meanprops={
                    "marker": "o",
                    "markerfacecolor": "red",
                    "markeredgecolor": "black",
                },
                medianprops={"color": "red", "linewidth": 1.5},
                boxprops={"edgecolor": "w"},
            )
        elif graph_type == "bar":
            plot = sns.barplot(
                x="Model",
                y="Value",
                data=melted_result,
                ax=axes[i],
                errorbar="sd",
                palette="plasma",
                hue="Model",
                width=0.5,
                color="black",
            )
        elif graph_type == "violin":
            plot = sns.violinplot(
                x="Model",
                y="Value",
                data=melted_result,
                ax=axes[i],
                inner=None,
                palette="plasma",
                hue="Model",
            )
            sns.stripplot(
                x="Model",
                y="Value",
                data=melted_result,
                ax=axes[i],
                color="white",
                size=5,
                jitter=True,
            )
        else:
            raise ValueError(
                f"Invalid graph type '{graph_type}'. Choose 'box', 'bar' or 'violin'."
            )

        plot.set_xlabel("")
        plot.set_ylabel(f"{metric.upper()}", fontsize=12)
        
        # Wrap labels
        labels = [item.get_text() for item in plot.get_xticklabels()]
        new_labels = []
        for label in labels:
            if "Regression" in label:
                new_label = label.replace("Regression", "\nRegression")
            elif "Regressor" in label:
                new_label = label.replace("Regressor", "\nRegressor")
            elif "Classifier" in label:
                new_label = label.replace("Classifier", "\nClassifier")
            else:
                new_label = label
            new_labels.append(new_label)
        plot.set_xticks(list(range(0, len(labels))))
        plot.set_xticklabels(new_labels)
        plot.tick_params(axis="both", labelsize=12)
            
    # If there are less plots than cells in the grid, hide the remaining cells
    if (len(scoring_list) % 2) != 0:
        for i in range(len(scoring_list), nrow * 2):
            axes[i].set_visible(False)
                
        if save_fig:
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                f"{save_dir}/{fig_prefix}_{metric}_{graph_type}.png",
                dpi=300, bbox_inches="tight"
            )
            
    plt.show()


def cross_validation_report(
    data: pd.DataFrame,
    activity_col: str,
    id_col: str,
    add_model: Optional[dict] = None,
    select_model: Optional[List[str]] = None,
    scoring_list: Optional[List[str]] = None,
    n_splits: int = 5,
    n_repeats: int = 5,
    visualize: Optional[str] = None,
    include_stats: bool = True,
    save_fig: bool = False,
    save_csv: bool = False,
    fig_prefix: str = "cv_graph",
    csv_name: str = "cv_report",
    save_dir: str = "Project/ModelDevelopment",
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
    add_model : Optional[dict]
        Dictionary of additional models to include.
    select_model : Optional[List[str]]
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
    fig_prefix : str
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
    model_map = _get_model_map(task_type, add_model, n_jobs)
    cv = _get_cv_strategy(task_type, n_splits=n_splits, n_repeats=n_repeats)

    scoring_list = scoring_list or _get_cv_scoring_list(task_type)

    result = []
    models_to_compare = {}

    if select_model is None:
        models_to_compare = model_map
    else:
        for name in select_model:
            if name in model_map:
                models_to_compare.update({name: model_map[name]})
            else:
                raise ValueError(f"Model '{name}' is not recognized.")

    for name, model in models_to_compare.items():
        scores = cross_validate(
            model,
            X_data,
            y_data,
            cv=cv,
            scoring=scoring_list,
            n_jobs=n_jobs,
        )
                
        # Collect fold scores for each cycle
        for cycle in range(n_splits * n_repeats):
            for metric in scoring_list:
                model_result = {
                    "scoring": metric,
                    "cv_cycle": cycle + 1,
                    name: scores[f"test_{metric}"][cycle],
                }
                result.append(model_result)

        # Optionally add mean, std, and median for each model and scoring metric
        if include_stats:
            for metric in scoring_list:
                metric_scores = scores[f"test_{metric}"]
                result.append({
                    "scoring": metric,
                    "cv_cycle": "mean",
                    name: round(np.mean(metric_scores), 3),
                })
                result.append({
                    "scoring": metric,
                    "cv_cycle": "std",
                    name: round(np.std(metric_scores), 3),
                })
                result.append({
                    "scoring": metric,
                    "cv_cycle": "median",
                    name: round(np.median(metric_scores), 3),
                })

    # Create a DataFrame in wide format
    cv_df = pd.DataFrame(result)
    cv_df = cv_df.set_index(["scoring", "cv_cycle"]).sort_index()


    if visualize:
        _plot_cv_report(
            report_df=cv_df,
            scoring_list=scoring_list,
            graph_type=visualize,
            save_fig=save_fig,
            fig_prefix=fig_prefix,
            save_dir=save_dir,
        )

    if save_csv:
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        cv_df.to_csv(f"{save_dir}/{csv_name}.csv")

    return cv_df


def external_validation_report(
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    activity_col: str,
    id_col: str,
    add_model: Optional[dict] = None,
    select_model: Optional[List[str]] = None,
    scoring_list: Optional[List[str]] = None,
    save_csv: bool = False,
    csv_name: str = "ev_report",
    save_dir: str = "Project/ModelDevelopment",
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
    add_model : Optional[dict]
        Dictionary of additional models to include.
    select_model : Optional[List[str]]
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
    model_map = _get_model_map(task_type, add_model, n_jobs)

    models_to_compare = {}
    if select_model is None:
        models_to_compare = model_map
    else:
        for name in select_model:
            if name in model_map:
                models_to_compare.update({name: model_map[name]})
            else:
                raise ValueError(f"Model '{name}' is not recognized.")

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

    ev_df = pd.DataFrame(ev_score)

    if save_csv:
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        ev_df.to_csv(f"{save_dir}/{csv_name}.csv")

    return ev_df
