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
from ..ModelDeveloper.model_validation import ModelValidation
from ..ModelDeveloper.model_developer_utils import (
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
    Build a mapping from method names to feature-selection objects.

    Depending on the task type (classification or regression), this function
    prepares a default set of feature-selection strategies:
      - Filter methods (ANOVA, mutual information).
      - Embedded methods (RandomForest, ExtraTrees, AdaBoost, GradientBoosting, XGBoost).
      - Linear models (LogisticRegression, LassoCV).

    :param task_type: Type of learning task. ``"C"`` for classification, ``"R"`` for regression.
    :type task_type: str
    :param add_method: Additional custom methods to add (name → selector instance). Default is ``{}``.
    :type add_method: dict
    :param n_jobs: Number of parallel jobs for estimators that support it. Default is ``1``.
    :type n_jobs: int
    :param random_state: Random seed for reproducibility. Default is ``42``.
    :type random_state: Optional[int]

    :return: Dictionary mapping method names to instantiated feature selectors.
    :rtype: dict[str, object]

    :raises ValueError: If ``task_type`` is not ``"C"`` or ``"R"``.

    **Example**

    .. code-block:: python

        from proqsar.FeatureSelector.feature_selector import _get_method_map

        methods = _get_method_map("C", n_jobs=2, random_state=0)
        print(methods.keys())
        # dict_keys(['Anova', 'MutualInformation', 'RandomForestClassifier', ...])
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
    Evaluate multiple feature-selection strategies.

    Each selector is applied to the dataset, followed by training a default
    RandomForest model and evaluating performance via repeated cross-validation.

    :param data: Input dataset with feature columns and activity & ID columns.
    :type data: pd.DataFrame
    :param activity_col: Name of the target variable column.
    :type activity_col: str
    :param id_col: Name of the identifier column (will be excluded from training).
    :type id_col: str
    :param add_method: Extra feature selectors to include (name → selector instance). Default is ``{}``.
    :type add_method: dict
    :param select_method: Subset of methods to evaluate. Can be a string or list of names. Default is ``None`` (all).
    :type select_method: Optional[Union[list, str]]
    :param scoring_list: Evaluation metrics. If ``None``, uses default scoring per task. Default is ``None``.
    :type scoring_list: Optional[Union[list, str]]
    :param n_splits: Number of CV folds. Default is ``5``.
    :type n_splits: int
    :param n_repeats: Number of repeated CV cycles. Default is ``5``.
    :type n_repeats: int
    :param include_stats: Whether to include summary statistics in the report. Default is ``True``.
    :type include_stats: bool
    :param visualize: Types of visualizations to generate (e.g., ``"boxplot"``). Default is ``None``.
    :type visualize: Optional[Union[str, List[str]]]
    :param save_fig: Whether to save generated figures. Default is ``False``.
    :type save_fig: bool
    :param save_csv: Whether to save the CV report as a CSV file. Default is ``False``.
    :type save_csv: bool
    :param fig_prefix: Prefix for saved figure filenames. Default is ``"fs_graph"``.
    :type fig_prefix: str
    :param csv_name: Base name for the saved CSV file. Default is ``"fs_report"``.
    :type csv_name: str
    :param save_dir: Directory for saving outputs. Default is ``"Project/FeatureSelector"``.
    :type save_dir: str
    :param n_jobs: Number of parallel jobs for estimators. Default is ``1``.
    :type n_jobs: int
    :param random_state: Random seed for reproducibility. Default is ``42``.
    :type random_state: Optional[int]

    :return: Cross-validation report with performance metrics for each selector.
    :rtype: pd.DataFrame

    :raises ValueError: If a method in ``select_method`` is not recognized.
    :raises Exception: For unexpected runtime errors.

    **Example**

    .. code-block:: python

        import pandas as pd
        from proqsar.FeatureSelector.feature_selector import evaluate_feature_selectors

        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "feature2": [5, 4, 3, 2, 1],
            "activity": [0, 1, 0, 1, 0]
        })

        report = evaluate_feature_selectors(
            df,
            activity_col="activity",
            id_col="id",
            select_method=["Anova", "MutualInformation"],
            n_splits=3,
            n_repeats=2,
            scoring_list="accuracy"
        )

        print(report.head())
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

        if select_method is None:
            methods_to_compare = method_map
        else:
            methods_to_compare = {}
            for name in select_method:
                if name in method_map:
                    methods_to_compare[name] = method_map[name]
                else:
                    raise ValueError(
                        f"FeatureSelector: Method '{name}' is not recognized."
                    )

        result = []
        for name, method in methods_to_compare.items():
            if name == "NoFS":
                selected_X = X_data
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

        result_df = pd.concat(result).pivot_table(
            index=["scoring", "cv_cycle"],
            columns="method",
            values="value",
            aggfunc="first",
        )
        result_df = result_df.sort_index(axis=0).sort_index(axis=1)
        result_df = result_df.reset_index().rename_axis(None, axis="columns")

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
