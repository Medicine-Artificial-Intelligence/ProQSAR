import os
import numpy as np
import pandas as pd
from typing import Union, Any, Optional, List
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
from ProQSAR.ModelDeveloper.model_validation import _plot_cv_report
from ProQSAR.ModelDeveloper.model_developer_utils import (
    _get_task_type,
    _get_cv_strategy,
    _get_cv_scoring_list
)

def _get_method_map(
    task_type: str,
    add_method: Optional[dict] = None,
    n_jobs: int = -1,
) -> dict:

    if task_type == "C":
        method_map = {
            "Anova": SelectKBest(score_func=f_classif, k=20),
            "MutualInformation": SelectKBest(score_func=mutual_info_classif, k=20),
            "RandomForestClassifier": SelectFromModel(RandomForestClassifier(random_state=42)),
            "ExtraTreesClassifier": SelectFromModel(ExtraTreesClassifier(random_state=42, n_jobs=n_jobs)),
            "AdaBoostClassifier": SelectFromModel(AdaBoostClassifier(random_state=42)),
            "GradientBoostingClassifier": SelectFromModel(GradientBoostingClassifier(random_state=42)),
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
                )
            ),
        }
    elif task_type == "R":
        method_map = {
            "Anova": SelectKBest(score_func=f_regression, k=20),
            "MutualInformation": SelectKBest(score_func=mutual_info_regression, k=20),
            "RandomForestRegressor": SelectFromModel(RandomForestRegressor(random_state=42)),
            "ExtraTreesRegressor": SelectFromModel(ExtraTreesRegressor(random_state=42,  n_jobs=n_jobs)),
            "AdaBoostRegressor": SelectFromModel(AdaBoostRegressor(random_state=42)),
            "GradientBoostingRegressor": SelectFromModel(GradientBoostingRegressor(random_state=42)),
            "XGBRegressor": SelectFromModel(
                XGBRegressor(random_state=42, verbosity=0, eval_metric="rmse")
            ),
            "LassoCV": SelectFromModel(LassoCV(random_state=42))
        }
            
    else:
        raise ValueError(
            "Invalid task_type. Please choose 'C' for classification or 'R' for regression."
        )

    if add_method:
        method_map.update(add_method)

    return method_map

def evaluate_feature_selectors(
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
    fig_prefix: str = "fs_graph",
    csv_name: str = "fs_report",
    save_dir: str = "Project/FeatureSelector",
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Compare different feature selection methods and evaluate their performance.

    Parameters:
        ----------
    data : pd.DataFrame
        The dataset containing features and the target variable.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the comparison results for each method.
    """

    X_data = data.drop([activity_col, id_col], axis=1)
    y_data = data[activity_col]

    task_type = _get_task_type(data, activity_col)
    method_map = _get_method_map(task_type, add_method, n_jobs)
    cv = _get_cv_strategy(
        task_type, n_splits=n_splits, n_repeats=n_repeats
    )
    scoring_list = scoring_list or _get_cv_scoring_list(task_type)

    result = []
    methods_to_compare = {}

    if select_method is None:
        methods_to_compare = method_map
    else:
        for name in select_method:
            if name in method_map:
                methods_to_compare.update({name: method_map[name]})
            else:
                raise ValueError(f"Method '{name}' is not recognized.")

    for name, method in method_map.items():
        selector = method.fit(X_data, y_data)
        selected_X = selector.transform(X_data)
        model = (
            RandomForestClassifier(random_state=42)
            if task_type == "C"
            else RandomForestRegressor(random_state=42)
        )
        scores = cross_validate(
            model, selected_X, y_data, cv=cv, scoring=scoring_list, n_jobs=n_jobs
        )
        
        method_result = {"FeatureSelector": name}
        for metric in scoring_list:
            metric_scores = scores[f"test_{metric}"]
            method_result[f"{metric}_mean"] = round(np.mean(metric_scores), 3)
            method_result[f"{metric}_std"] = round(np.std(metric_scores), 3)
            method_result[f"{metric}_median"] = round(np.median(metric_scores), 3)
            for i, score in enumerate(metric_scores):
                method_result[f"{metric}_fold{i+1}"] = score
                
        result.append(method_result)

    result_df = pd.DataFrame(result)

    if visualize:
        _plot_cv_report(
            report_df=result_df,
            scoring_list=scoring_list,
            graph_type=visualize,
            save_fig=save_fig,
            fig_prefix=fig_prefix,
            save_dir=save_dir,
            xlabel="FeatureSelector",
        )

    if save_csv:
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        result_df.to_csv(f"{save_dir}/{csv_name}.csv")

    return result_df
 