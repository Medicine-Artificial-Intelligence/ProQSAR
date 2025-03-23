import os
import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Union
from sklearn.model_selection import (
    cross_validate,
)
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    mutual_info_classif,
    f_regression,
    mutual_info_regression,
    f_classif,
)
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
from sklearn.linear_model import LogisticRegression, LassoCV
from xgboost import XGBClassifier, XGBRegressor
from ProQSAR.ModelDeveloper.model_validation import ModelValidation
from ProQSAR.ModelDeveloper.model_developer_utils import (
    _get_task_type,
    _get_cv_strategy,
    _get_cv_scoring,
)


def _get_method_map(
    task_type: str,
    add_method: dict = {},
    n_jobs: int = -1,
) -> dict[str, object]:
    """
    Creates a dictionary of feature selection methods based on the task type.

    Parameters:
    -----------
    task_type : str
        Specifies the type of task: 'C' for classification or 'R' for regression.
    add_method :
        Additional feature selection methods to include, with method names as keys and selectors as values.
    n_jobs : int
        Number of parallel jobs for methods that support parallel processing. Default is -1 (use all processors).

    Returns:
    --------
    Dict[str, object]
        Dictionary of feature selection methods keyed by method name.

    Raises:
    -------
    ValueError
        If task_type is not 'C' or 'R'.
    """
    try:
        if task_type == "C":
            method_map = {
                "Anova": SelectKBest(score_func=f_classif, k=20),
                "MutualInformation": SelectKBest(score_func=mutual_info_classif, k=20),
                "RandomForestClassifier": SelectFromModel(
                    RandomForestClassifier(random_state=42, n_jobs=n_jobs)
                ),
                "ExtraTreesClassifier": SelectFromModel(
                    ExtraTreesClassifier(random_state=42, n_jobs=n_jobs)
                ),
                "AdaBoostClassifier": SelectFromModel(
                    AdaBoostClassifier(random_state=42)
                ),
                "GradientBoostingClassifier": SelectFromModel(
                    GradientBoostingClassifier(random_state=42)
                ),
                "XGBClassifier": SelectFromModel(
                    XGBClassifier(random_state=42, verbosity=0, eval_metric="logloss")
                ),
                "LogisticRegression": SelectFromModel(
                    LogisticRegression(
                        random_state=42,
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratio=0.5,
                        max_iter=1000,
                        n_jobs=n_jobs,
                    )
                ),
            }
        elif task_type == "R":
            method_map = {
                "Anova": SelectKBest(score_func=f_regression, k=20),
                "MutualInformation": SelectKBest(
                    score_func=mutual_info_regression, k=20
                ),
                "RandomForestRegressor": SelectFromModel(
                    RandomForestRegressor(random_state=42, n_jobs=n_jobs)
                ),
                "ExtraTreesRegressor": SelectFromModel(
                    ExtraTreesRegressor(random_state=42, n_jobs=n_jobs)
                ),
                "AdaBoostRegressor": SelectFromModel(
                    AdaBoostRegressor(random_state=42)
                ),
                "GradientBoostingRegressor": SelectFromModel(
                    GradientBoostingRegressor(random_state=42)
                ),
                "XGBRegressor": SelectFromModel(
                    XGBRegressor(random_state=42, verbosity=0, eval_metric="rmse")
                ),
                "LassoCV": SelectFromModel(LassoCV(random_state=42, n_jobs=n_jobs)),
            }

        else:
            raise ValueError(
                "Invalid task_type. Please choose 'C' for classification or 'R' for regression."
            )

        if add_method:
            method_map.update(add_method)

        return method_map

    except Exception as e:
        logging.error(f"Error in _get_method_map: {e}")
        raise


def evaluate_feature_selectors(
    data: pd.DataFrame,
    activity_col: str,
    id_col: str,
    add_method: dict = {},
    select_method: Optional[Union[list, str]] = None,
    scoring_list: Optional[Union[list, str]] = None,
    n_splits: int = 5,
    n_repeats: int = 5,
    include_stats: bool = True,
    visualize: Optional[Union[str, List[str]]] = None,
    save_fig: bool = False,
    save_csv: bool = False,
    fig_prefix: str = "fs_graph",
    csv_name: str = "fs_report",
    save_dir: str = "Project/FeatureSelector",
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Evaluates various feature selection methods using cross-validation on the given dataset.

    Parameters:
    -----------
    data : pd.DataFrame
        The full dataset including features and target for feature selection evaluation.
    activity_col : str
        The target column in the dataset.
    id_col : str
        The identifier column in the dataset.
    add_method : Optional[Dict[str, object]]
        Additional feature selection methods to include.
    select_method : Optional[List[str]]
        List of specific methods to use for feature selection. If None, all available methods are used.
    scoring_list : Optional[List[str]]
        List of scoring metrics for model evaluation. If None, default metrics are used based on task type.
    n_splits : int
        Number of splits for cross-validation. Default is 10.
    n_repeats : int
        Number of repeats for cross-validation. Default is 3.
    visualize : Optional[str]
        Type of visualization ('box', 'bar', 'violin') for displaying results. Default is None.
    save_fig : bool
        Whether to save the generated figures. Default is False.
    save_csv : bool
        Whether to save the report as a CSV file. Default is False.
    fig_prefix : str
        Prefix for the saved figure file name. Default is 'fs_graph'.
    csv_name : str
        File name for saving the report as CSV. Default is 'fs_report'.
    save_dir : str
        Directory where the report and figures will be saved. Default is 'Project/FeatureSelector'.
    n_jobs : int
        Number of parallel jobs for processing. Default is -1 (use all processors).

    Returns:
    --------
    pd.DataFrame
        DataFrame containing cross-validation results for each feature selection method.

    Raises:
    -------
    ValueError
        If a selected method is not recognized in the method map.
    """
    try:
        logging.info("Starting feature selection evaluation.")

        if isinstance(scoring_list, str):
            scoring_list = [scoring_list]

        if isinstance(select_method, str):
            select_method = [select_method]

        X_data = data.drop([activity_col, id_col], axis=1)
        y_data = data[activity_col]

        task_type = _get_task_type(data, activity_col)
        method_map = _get_method_map(task_type, add_method, n_jobs)
        method_map.update({"NoFS": None})
        cv = _get_cv_strategy(task_type, n_splits=n_splits, n_repeats=n_repeats)
        scoring_list = scoring_list or _get_cv_scoring(task_type)

        methods_to_compare = {}

        if select_method is None:
            methods_to_compare = method_map
        else:
            for name in select_method:
                if name in method_map:
                    methods_to_compare.update({name: method_map[name]})
                else:
                    raise ValueError(f"Method '{name}' is not recognized.")

        result = []

        for name, method in methods_to_compare.items():
            if name == "NoFS":
                selected_X = X_data  # No feature selection
            else:
                selector = method.fit(X_data, y_data)
                selected_X = selector.transform(X_data)

            model = (
                RandomForestClassifier(random_state=42)
                if task_type == "C"
                else RandomForestRegressor(random_state=42)
            )
            result.append(
                ModelValidation._perform_cross_validation(
                    {name: model},
                    selected_X,
                    y_data,
                    cv,
                    scoring_list,
                    include_stats,
                    n_splits,
                    n_repeats,
                    n_jobs,
                )
            )

        # Pivot the DataFrame so that each model becomes a separate column
        result_df = pd.concat(result).pivot_table(
            index=["scoring", "cv_cycle"],
            columns="method",
            values="value",
            aggfunc="first",
        )
        # Sort index and columns to maintain a consistent order
        result_df = result_df.sort_index(axis=0).sort_index(axis=1)

        # Reset index
        result_df = result_df.reset_index().rename_axis(None, axis="columns")

        # Visualization if requested
        if visualize is not None:
            if isinstance(visualize, str):
                visualize = [visualize]

            for graph_type in visualize:
                ModelValidation._plot_cv_report(
                    report_df=result_df,
                    scoring_list=scoring_list,
                    graph_type=graph_type,
                    save_fig=save_fig,
                    fig_prefix=fig_prefix,
                    save_dir=save_dir,
                )

        if save_csv:
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            result_df.to_csv(f"{save_dir}/{csv_name}.csv", index=False)
            logging.info(
                f"Feature selection evaluation data saved at: {save_dir}/{csv_name}.csv"
            )

        logging.info("Feature selection evaluation completed successfully.")
        return result_df

    except Exception as e:
        logging.error(f"Error in evaluate_feature_selectors: {e}")
        raise
