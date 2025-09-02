import os
import logging
import pandas as pd
from typing import Optional, List, Union
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
    n_jobs: int = 1,
    random_state: Optional[int] = 42,
) -> dict[str, object]:
    """
    Build a dictionary mapping human-readable method names to feature-selector
    objects appropriate for the given task type.

    The returned selectors are configured with reasonable defaults:
      - SelectKBest with ANOVA / mutual information depending on task
      - SelectFromModel wrapping tree-based models, linear models, or XGBoost

    Parameters
    ----------
    task_type : str
        Either 'C' for classification or 'R' for regression. Determines which
        selectors and scoring functions are used.
    add_method : dict, optional
        Optional additional method mappings to merge into the default map.
        The keys should be method names and the values should be selector
        objects or callables compatible with the interface used below.
    n_jobs : int, default=1
        Number of parallel jobs passed to underlying estimators where supported.
    random_state : Optional[int], default=42
        Random seed used to instantiate stochastic estimators.

    Returns
    -------
    dict[str, object]
        Mapping from method name to an instantiated feature-selection object.

    Raises
    ------
    ValueError
        If `task_type` is not one of the supported options ('C' or 'R').

    Notes
    -----
    This function intentionally returns instances (not classes). Callers will
    call .fit(...)/.transform(...) on the returned selectors.
    """
    try:
        if task_type == "C":
            method_map = {
                "Anova": SelectKBest(score_func=f_classif, k=20),
                "MutualInformation": SelectKBest(score_func=mutual_info_classif, k=20),
                "RandomForestClassifier": SelectFromModel(
                    RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
                ),
                "ExtraTreesClassifier": SelectFromModel(
                    ExtraTreesClassifier(random_state=random_state, n_jobs=n_jobs)
                ),
                "AdaBoostClassifier": SelectFromModel(
                    AdaBoostClassifier(random_state=random_state)
                ),
                "GradientBoostingClassifier": SelectFromModel(
                    GradientBoostingClassifier(random_state=random_state)
                ),
                "XGBClassifier": SelectFromModel(
                    XGBClassifier(
                        random_state=random_state, verbosity=0, eval_metric="logloss"
                    )
                ),
                "LogisticRegression": SelectFromModel(
                    LogisticRegression(
                        random_state=random_state,
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
                    RandomForestRegressor(random_state=random_state, n_jobs=n_jobs)
                ),
                "ExtraTreesRegressor": SelectFromModel(
                    ExtraTreesRegressor(random_state=random_state, n_jobs=n_jobs)
                ),
                "AdaBoostRegressor": SelectFromModel(
                    AdaBoostRegressor(random_state=random_state)
                ),
                "GradientBoostingRegressor": SelectFromModel(
                    GradientBoostingRegressor(random_state=random_state)
                ),
                "XGBRegressor": SelectFromModel(
                    XGBRegressor(
                        random_state=random_state, verbosity=0, eval_metric="rmse"
                    )
                ),
                "LassoCV": SelectFromModel(
                    LassoCV(random_state=random_state, n_jobs=n_jobs)
                ),
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
    n_jobs: int = 1,
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Evaluate multiple feature-selection strategies by applying each selector to
    the input data, training a default model on the selected features, and
    performing repeated cross-validation.

    The function:
      - infers task type (classification or regression) from `data` and `activity_col`
      - constructs a method map of selectors (merged with `add_method`)
      - for each selector:
          - fits the selector, transforms X to the selected feature set
          - trains a default RandomForest model (classifier/regressor depending on task)
          - performs repeated cross-validation using ModelValidation utilities
      - aggregates and returns a CV report DataFrame with columns ['scoring','cv_cycle', 'method', 'value']

    Parameters
    ----------
    data : pd.DataFrame
        Input data containing feature columns and the activity & id columns.
    activity_col : str
        Column name containing the target variable.
    id_col : str
        Column name containing a unique identifier for each sample (will be dropped).
    add_method : dict, optional
        Additional selectors to include in the method map (key -> selector instance).
    select_method : list or str or None, optional
        If provided, evaluate only the named selectors (must exist in the method map).
    scoring_list : list or str or None, optional
        Metrics to evaluate. If None, defaults to the project's CV scoring for the task.
    n_splits : int, default=5
        Number of folds for each repetition in repeated CV.
    n_repeats : int, default=5
        Number of repeated cross-validation runs.
    include_stats : bool, default=True
        Whether to include summary statistics in the reported CV outputs.
    visualize : str | list[str] | None, optional
        If provided, one or more visualization types to produce via ModelValidation.
    save_fig : bool, default=False
        If True, save generated figures to disk.
    save_csv : bool, default=False
        If True, save the aggregated CV report CSV to `save_dir/csv_name.csv`.
    fig_prefix : str, default="fs_graph"
        Filename prefix used for saved figures.
    csv_name : str, default="fs_report"
        Base filename for the saved CSV report.
    save_dir : str, default="Project/FeatureSelector"
        Directory used to store saved figures and CSVs.
    n_jobs : int, default=1
        Number of parallel jobs passed to underlying estimators where supported.
    random_state : Optional[int], default=42
        Random seed used to instantiate stochastic estimators.

    Returns
    -------
    pd.DataFrame
        A flattened CV report DataFrame with columns ['scoring','cv_cycle','method','value']
        (and possibly other CV-statistics depending on ModelValidation._perform_cross_validation).

    Raises
    ------
    ValueError
        If a user-specified method in `select_method` is not found in the method map.
    Exception
        Unexpected exceptions are logged and re-raised.
    """
    try:

        if isinstance(scoring_list, str):
            scoring_list = [scoring_list]

        if isinstance(select_method, str):
            select_method = [select_method]

        X_data = data.drop([activity_col, id_col], axis=1)
        y_data = data[activity_col]

        task_type = _get_task_type(data, activity_col)
        method_map = _get_method_map(
            task_type, add_method, n_jobs, random_state=random_state
        )
        method_map.update({"NoFS": None})
        cv = _get_cv_strategy(
            task_type, n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
        scoring_list = scoring_list or _get_cv_scoring(task_type)

        methods_to_compare = {}

        if select_method is None:
            methods_to_compare = method_map
        else:
            for name in select_method:
                if name in method_map:
                    methods_to_compare.update({name: method_map[name]})
                else:
                    raise ValueError(
                        f"FeatureSelector: Method '{name}' is not recognized."
                    )

        result = []

        for name, method in methods_to_compare.items():
            if name == "NoFS":
                selected_X = X_data  # No feature selection
            else:
                selector = method.fit(X_data, y_data)
                selected_X = selector.transform(X_data)

            model = (
                RandomForestClassifier(random_state=random_state)
                if task_type == "C"
                else RandomForestRegressor(random_state=random_state)
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
                f"FeatureSelector evaluation data saved at: {save_dir}/{csv_name}.csv"
            )

        return result_df

    except Exception as e:
        logging.error(f"Error in evaluate_feature_selectors: {e}")
        raise
