import os
import gc
import math
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Union, Tuple, Dict, Any
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from .model_developer_utils import (
    _get_task_type,
    _get_model_map,
    _get_cv_strategy,
    _get_cv_scoring,
    _get_ev_scoring,
)


class ModelValidation:
    """
    Collection of static utilities for model evaluation, cross-validation reporting,
    external validation and plotting of common model diagnostics (ROC/PR curves,
    scatter plots for regression, CV summary plots).

    Methods are implemented as ``@staticmethod`` so they can be used without
    instantiating the class.
    """

    @staticmethod
    def _plot_cv_report(
        report_df: pd.DataFrame,
        scoring_list: Union[list, str],
        graph_type: Optional[str] = "box",
        save_fig: bool = False,
        fig_prefix: str = "cv_graph",
        save_dir: str = "Project/ModelDevelopment",
    ) -> None:
        """
        Create summary plots from a cross-validation report DataFrame.

        The function supports 'box', 'bar' and 'violin' plot types. The
        ``report_df`` is expected to be in a wide format where rows have
        columns 'scoring' and 'cv_cycle' and other columns correspond to
        evaluated methods.

        :param report_df: CV report in wide format (contains 'scoring' and 'cv_cycle').
        :type report_df: pd.DataFrame
        :param scoring_list: Metric name or list of metric names to plot.
        :type scoring_list: Union[list, str]
        :param graph_type: One of ``'box'``, ``'bar'`` or ``'violin'`` (default: ``'box'``).
        :type graph_type: Optional[str]
        :param save_fig: If ``True``, save the resulting figure to disk (default: ``False``).
        :type save_fig: bool
        :param fig_prefix: Filename prefix for the saved figure (default: ``'cv_graph'``).
        :type fig_prefix: str
        :param save_dir: Directory to save plots if ``save_fig=True`` (default: ``'Project/ModelDevelopment'``).
        :type save_dir: str

        :raises ValueError: If ``graph_type`` is not one of the supported options.
        :raises Exception: Unexpected exceptions are logged and re-raised.

        :return: None
        :rtype: None
        """
        try:
            if isinstance(scoring_list, str):
                scoring_list = [scoring_list]

            scoring_list.sort()

            sns.set_context("notebook")
            sns.set_style("whitegrid")

            nrow = math.ceil(len(scoring_list) / 2)

            nmethod = len(
                report_df.drop(columns=["scoring", "cv_cycle"]).columns.unique()
            )

            figure, axes = plt.subplots(
                nrow, 2, sharex=False, sharey=False, figsize=(3 * nmethod, 7 * nrow)
            )
            axes = axes.flatten()  # Turn 2D array to 1D array

            for i, metric in enumerate(scoring_list):

                # Select only rows that correspond to the current metric
                metric_rows = report_df[report_df["scoring"] == metric]

                # Melt the DataFrame to long format for plotting
                melted_result = metric_rows.drop(columns="scoring").melt(
                    id_vars=["cv_cycle"],
                    var_name="method",
                    value_name="value",
                )
                # Remove rows where cv_cycle is 'mean', 'std', or 'median'
                melted_result = melted_result[
                    ~melted_result["cv_cycle"].isin(["mean", "std", "median"])
                ]

                if graph_type == "box":
                    plot = sns.boxplot(
                        x="method",
                        y="value",
                        data=melted_result,
                        ax=axes[i],
                        showmeans=True,
                        width=0.5,
                        palette="plasma",
                        hue="method",
                    )
                elif graph_type == "bar":
                    plot = sns.barplot(
                        x="method",
                        y="value",
                        data=melted_result,
                        ax=axes[i],
                        errorbar="sd",
                        capsize=0.25,
                        palette="plasma",
                        hue="method",
                        width=0.5,
                        color="black",
                        err_kws={"linewidth": 1.2},
                    )
                elif graph_type == "violin":
                    plot = sns.violinplot(
                        x="method",
                        y="value",
                        data=melted_result,
                        ax=axes[i],
                        width=0.5,
                        inner=None,
                    )
                    for violin in plot.collections:
                        violin.set_facecolor("#ADD3ED")
                        violin.set_edgecolor("#ADD3ED")

                    sns.stripplot(
                        x="method",
                        y="value",
                        data=melted_result,
                        ax=axes[i],
                        palette="plasma",
                        hue="method",
                        size=5,
                        jitter=True,
                    )
                else:
                    raise ValueError(
                        f"Invalid graph type '{graph_type}'. Choose 'box', 'bar' or 'violin'."
                    )

                plot.set_xlabel("")
                plot.set_ylabel(f"{metric.upper()}")

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
                    f"{save_dir}/{fig_prefix}_{graph_type}.pdf",
                    dpi=300,
                    bbox_inches="tight",
                )

            plt.tight_layout()

        except Exception as e:
            logging.error(f"Error while plotting CV report: {e}")
            raise

    @staticmethod
    def _perform_cross_validation(
        models: Dict[str, Any],
        X_data: Union[pd.DataFrame, np.ndarray],
        y_data: Union[pd.Series, np.ndarray],
        cv,
        scoring_list: List[str],
        include_stats: bool,
        n_splits: int,
        n_repeats: int,
        n_jobs: int,
    ) -> pd.DataFrame:
        """
        Internal helper that runs sklearn.cross_validate for multiple models and
        returns a flattened DataFrame with per-cycle scores and optional stats.

        :param models: Mapping from model name to estimator (must support sklearn API).
        :type models: Dict[str, Any]
        :param X_data: Feature matrix (DataFrame or ndarray).
        :type X_data: Union[pd.DataFrame, np.ndarray]
        :param y_data: Target vector (Series or ndarray).
        :type y_data: Union[pd.Series, np.ndarray]
        :param cv: Cross-validation splitter (e.g., RepeatedKFold).
        :type cv: object
        :param scoring_list: List of scoring metric names compatible with sklearn.
        :type scoring_list: List[str]
        :param include_stats: If True, add mean, std and median rows per method/metric.
        :type include_stats: bool
        :param n_splits: Number of folds per repeat.
        :type n_splits: int
        :param n_repeats: Number of repeats.
        :type n_repeats: int
        :param n_jobs: Number of parallel jobs for cross_validate.
        :type n_jobs: int

        :return: DataFrame with columns ['scoring', 'cv_cycle', 'method', 'value']
                 and optional aggregated rows for 'mean', 'std', 'median'.
        :rtype: pd.DataFrame
        """
        result = []

        for name, model in models.items():
            logging.info(f"Cross-validating model: {name}")
            # Perform cross-validation
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
                    result.append(
                        {
                            "scoring": metric,
                            "cv_cycle": cycle + 1,
                            "method": name,
                            "value": scores[f"test_{metric}"][cycle],
                        }
                    )

            # Optionally add mean, std, and median for each model and scoring metric
            if include_stats:
                for metric in scoring_list:
                    metric_scores = scores[f"test_{metric}"]
                    result.append(
                        {
                            "scoring": metric,
                            "cv_cycle": "mean",
                            "method": name,
                            "value": np.mean(metric_scores),
                        }
                    )
                    result.append(
                        {
                            "scoring": metric,
                            "cv_cycle": "std",
                            "method": name,
                            "value": np.std(metric_scores),
                        }
                    )
                    result.append(
                        {
                            "scoring": metric,
                            "cv_cycle": "median",
                            "method": name,
                            "value": np.median(metric_scores),
                        }
                    )
            del scores
            gc.collect()

        return pd.DataFrame(result)

    @staticmethod
    def cross_validation_report(
        data: pd.DataFrame,
        activity_col: str,
        id_col: str,
        add_model: dict = {},
        select_model: Optional[Union[list, str]] = None,
        scoring_list: Optional[Union[list, str]] = None,
        n_splits: int = 5,
        n_repeats: int = 5,
        include_stats: bool = True,
        visualize: Optional[Union[str, List[str]]] = None,
        save_fig: bool = False,
        save_csv: bool = False,
        fig_prefix: str = "cv_graph",
        csv_name: str = "cv_report",
        save_dir: str = "Project/ModelDevelopment",
        n_jobs: int = 1,
        random_state: Optional[int] = 42,
    ) -> pd.DataFrame:
        """
        Run cross-validation for supplied models and return a structured report.

        :param data: DataFrame including feature columns, activity_col and id_col.
        :type data: pd.DataFrame
        :param activity_col: Name of the target column.
        :type activity_col: str
        :param id_col: Name of the id column to be dropped before training.
        :type id_col: str
        :param add_model: Extra model entries to include when building the model map.
        :type add_model: dict
        :param select_model: If provided, restrict evaluation to named models.
        :type select_model: Optional[Union[list, str]]
        :param scoring_list: If None, default scoring is selected by task via _get_cv_scoring.
        :type scoring_list: Optional[Union[list, str]]
        :param n_splits: Number of folds per repeat (default 5).
        :type n_splits: int
        :param n_repeats: Number of CV repeats (default 5).
        :type n_repeats: int
        :param include_stats: Whether to include aggregated statistics rows (mean/std/median).
        :type include_stats: bool
        :param visualize: Visualizations to produce via _plot_cv_report.
        :type visualize: Optional[Union[str, List[str]]]
        :param save_fig: If True, save visualization figures.
        :type save_fig: bool
        :param save_csv: If True, save the resulting report CSV.
        :type save_csv: bool
        :param fig_prefix: Prefix for plot filenames.
        :type fig_prefix: str
        :param csv_name: Filename for saved CSV.
        :type csv_name: str
        :param save_dir: Directory for saving figures and CSVs.
        :type save_dir: str
        :param n_jobs: Number of parallel jobs for model operations.
        :type n_jobs: int
        :param random_state: Random seed used for model instantiation.
        :type random_state: Optional[int]

        :return: A wide-format CV report with columns ['scoring','cv_cycle', ...methods...].
        :rtype: pd.DataFrame

        :raises Exception: Any unexpected exception is logged and re-raised.
        """
        try:
            if isinstance(scoring_list, str):
                scoring_list = [scoring_list]

            if isinstance(select_model, str):
                select_model = [select_model]

            X_data = data.drop([activity_col, id_col], axis=1)
            y_data = data[activity_col]

            task_type = _get_task_type(data, activity_col)
            model_map = _get_model_map(
                task_type, add_model, n_jobs, random_state=random_state
            )
            cv = _get_cv_strategy(
                task_type,
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
            )

            scoring_list = scoring_list or _get_cv_scoring(task_type)

            models_to_compare = {}

            if select_model is None:
                models_to_compare = model_map
            else:
                for name in select_model:
                    if name in model_map:
                        models_to_compare.update({name: model_map[name]})
                    else:
                        raise ValueError(f"Model '{name}' is not recognized.")

            result_df = ModelValidation._perform_cross_validation(
                models_to_compare,
                X_data,
                y_data,
                cv,
                scoring_list,
                include_stats,
                n_splits,
                n_repeats,
                n_jobs,
            )

            # Create a DataFrame in wide format
            result_df = result_df.pivot_table(
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
            # Optional saving of results to CSV
            if save_csv:
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                result_df.to_csv(f"{save_dir}/{csv_name}.csv", index=False)
                logging.info(
                    f"Cross validation report saved at: {save_dir}/{csv_name}.csv"
                )

            return result_df

        except Exception as e:
            logging.error(f"Error in cross-validation report generation {e}")
            raise

    @staticmethod
    def external_validation_report(
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        activity_col: str,
        id_col: str,
        add_model: Optional[dict] = None,
        select_model: Optional[List[str]] = None,
        scoring_list: Optional[Union[list, str]] = None,
        save_csv: bool = False,
        csv_name: str = "ev_report",
        save_dir: str = "Project/ModelDevelopment",
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        """
        Evaluate models on an external test set and return per-model evaluation metrics.

        :param data_train: Training dataset with features, activity_col and id_col.
        :type data_train: pd.DataFrame
        :param data_test: External test dataset with same feature columns.
        :type data_test: pd.DataFrame
        :param activity_col: Name of the target column.
        :type activity_col: str
        :param id_col: Identifier column name (dropped before training).
        :type id_col: str
        :param add_model: Extra models to include.
        :type add_model: Optional[dict]
        :param select_model: If provided, restrict evaluation to these models.
        :type select_model: Optional[List[str]]
        :param scoring_list: If provided, only include these metrics in the output.
        :type scoring_list: Optional[Union[list, str]]
        :param save_csv: If True, save results to CSV.
        :type save_csv: bool
        :param csv_name: Filename for saved CSV.
        :type csv_name: str
        :param save_dir: Directory for saving outputs.
        :type save_dir: str
        :param n_jobs: Number of threads for model libraries that accept it.
        :type n_jobs: int

        :return: DataFrame of evaluation metrics (rows = metric, columns = model).
        :rtype: pd.DataFrame

        :raises ValueError: If an unrecognized model or metric is requested.
        :raises Exception: Any unexpected exception is logged and re-raised.
        """
        try:
            if isinstance(scoring_list, str):
                scoring_list = [scoring_list]

            if isinstance(select_model, str):
                select_model = [select_model]

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
                y_test_proba = (
                    model.predict_proba(X_test)[:, 1] if task_type == "C" else None
                )

                scoring_dict = _get_ev_scoring(
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

            ev_df = ev_df.sort_index(axis=0).sort_index(axis=1)

            if save_csv:
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                ev_df.to_csv(f"{save_dir}/{csv_name}.csv")
                logging.info(
                    f"External validation report saved at: {save_dir}/{csv_name}.csv"
                )
            return ev_df

        except Exception as e:
            logging.error(f"Error during external validation report generation: {e}")
            raise

    @staticmethod
    def make_curve(
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        activity_col: str,
        id_col: str,
        curve_type: Union[str, List[str]] = [
            "roc",
            "pr",
        ],  # Can be a single string or a list of strings.
        select_model: Optional[Union[list, str]] = None,
        add_model: Optional[dict] = None,
        legend_loc: Optional[Union[str, Tuple[float, float]]] = "best",
        save_dir: Optional[str] = "Project/ModelDevelopment",
        fig_name: Optional[str] = None,
        n_jobs: int = 1,
    ) -> None:
        """
        Plot ROC and/or Precision-Recall curves for one or more models evaluated
        on an external test set (classification only).

        :param data_train: Training data to fit models.
        :type data_train: pd.DataFrame
        :param data_test: Test data used to compute curves.
        :type data_test: pd.DataFrame
        :param activity_col: Name of the target column.
        :type activity_col: str
        :param id_col: Identifier column name (dropped before training).
        :type id_col: str
        :param curve_type: Which curves to plot. Accepts 'roc' and/or 'pr'.
        :type curve_type: Union[str, List[str]]
        :param select_model: Subset of model names to evaluate (default: all).
        :type select_model: Optional[Union[list, str]]
        :param add_model: Extra models to include in the model map.
        :type add_model: Optional[dict]
        :param legend_loc: Legend location for matplotlib (default 'best').
        :type legend_loc: Optional[Union[str, Tuple[float, float]]]
        :param save_dir: Directory to save the resulting plot PDF. If None, no file is saved.
        :type save_dir: Optional[str]
        :param fig_name: Filename (without extension) for saved figure. If None, a default name is used.
        :type fig_name: Optional[str]
        :param n_jobs: Number of threads to pass to model libraries that accept it.
        :type n_jobs: int

        :raises ValueError: If non-classification task is passed or unsupported curve_type value.
        :raises Exception: Unexpected exceptions are logged and re-raised.

        :return: None
        :rtype: None
        """
        try:
            # Normalize curve_type to a list.
            if isinstance(curve_type, str):
                curve_types = [curve_type.lower()]
            elif isinstance(curve_type, list):
                curve_types = [ct.lower() for ct in curve_type]
            else:
                raise ValueError("curve_type must be a string or a list of strings.")

            # Prepare training and testing data.
            X_train = data_train.drop([activity_col, id_col], axis=1)
            y_train = data_train[activity_col]
            X_test = data_test.drop([activity_col, id_col], axis=1)
            y_test = data_test[activity_col]

            # Check if the task is classification.
            task_type = _get_task_type(data_train, activity_col)
            if task_type != "C":
                raise ValueError("This function only supports classification tasks.")

            # Get available models.
            model_map = _get_model_map(task_type, add_model, n_jobs=n_jobs)

            # Select models to compare.
            models_to_compare = {}
            if select_model is None:
                models_to_compare = model_map
            else:
                for name in select_model:
                    if name in model_map:
                        models_to_compare[name] = model_map[name]
                    else:
                        raise ValueError(f"Model '{name}' is not recognized.")

            # Precompute predictions for each model.
            model_predictions = {}
            for name, model in models_to_compare.items():
                model.fit(X=X_train, y=y_train)
                # Store probability predictions (assuming binary classification).
                model_predictions[name] = model.predict_proba(X_test)[:, 1]

            # Set up the plotting figure.
            n_plots = len(curve_types)
            if n_plots == 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                axes = [ax]
            else:
                fig, axes = plt.subplots(1, n_plots, figsize=(10 * n_plots, 8))

            # Loop through each curve type and plot.
            for idx, ct in enumerate(curve_types):
                ax = axes[idx]
                # Set defaults for legend location based on curve type if not provided.
                for name, y_test_proba in model_predictions.items():
                    if y_test_proba is not None:
                        if ct == "roc":
                            # Compute ROC curve and AUC.
                            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
                            auc = roc_auc_score(y_test, y_test_proba)
                            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
                        elif ct == "pr":
                            # Compute Precision-Recall curve and average precision score.
                            precision, recall, _ = precision_recall_curve(
                                y_test, y_test_proba
                            )
                            pr_auc = average_precision_score(y_test, y_test_proba)
                            ax.plot(
                                recall,
                                precision,
                                label=f"{name} (PR AUC = {pr_auc:.3f})",
                            )
                        else:
                            raise ValueError(
                                "curve_type values must be either 'roc' or 'pr'."
                            )

                # For ROC, add a diagonal line for a random classifier.
                if ct == "roc":
                    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.5)
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
                else:
                    ax.set_xlabel("Recall")
                    ax.set_ylabel("Precision")
                    ax.set_title("Precision-Recall Curve")

                ax.legend(loc=legend_loc)
                ax.grid(True, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

            plt.tight_layout()

            # Save the plot.
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                # If a single curve type is plotted, use the provided fig_name or default.
                if n_plots == 1:
                    final_name = fig_name or (f"{curve_types[0]}_curve_plot")
                else:
                    # For multiple curves, append the curve type to the base file name.
                    final_name = fig_name or "_".join(curve_types) + "_curve_plot"

                full_path = f"{save_dir}/{final_name}.pdf"
                plt.savefig(full_path, dpi=300, bbox_inches="tight")
                logging.info(f"Curve plot saved at: {full_path}")

            plt.show()

        except Exception as e:
            logging.error(f"Error during curve generation: {e}")
            raise

    @staticmethod
    def make_scatter_plot(
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        activity_col: str,
        id_col: str,
        select_model: Optional[Union[list, str]] = None,
        add_model: Optional[dict] = None,
        scoring_df: Optional[pd.DataFrame] = None,
        scoring_loc: Tuple[float, float] = (0.05, 0.95),
        save_dir: str = "Project/ModelDevelopment",
        fig_name: str = "scatter_plot",
        n_jobs: int = 1,
    ) -> None:
        """
        Produce scatter plots of predicted vs actual values for regression models.

        Each selected model is fitted on the training set and its predictions on
        the test set are plotted in a subplot. Optionally, per-model scores from
        ``scoring_df`` are displayed within each subplot.

        :param data_train: Training dataset with features and target.
        :type data_train: pd.DataFrame
        :param data_test: Test dataset with features and target.
        :type data_test: pd.DataFrame
        :param activity_col: Target column name.
        :type activity_col: str
        :param id_col: Identifier column name (dropped before training).
        :type id_col: str
        :param select_model: Names of models to compare (default: all available models).
        :type select_model: Optional[Union[list, str]]
        :param add_model: Additional models to include.
        :type add_model: Optional[dict]
        :param scoring_df: DataFrame containing per-model scoring information to display.
        :type scoring_df: Optional[pd.DataFrame]
        :param scoring_loc: Location (x,y in axes coords) where the scoring text is placed.
        :type scoring_loc: Tuple[float, float]
        :param save_dir: Directory to save the combined figure.
        :type save_dir: str
        :param fig_name: Filename (without extension) for saved plot.
        :type fig_name: str
        :param n_jobs: Number of threads to pass to model libraries that accept it.
        :type n_jobs: int

        :raises ValueError: If a non-regression task is provided or no valid models are selected.
        :raises Exception: Unexpected exceptions are logged and re-raised.

        :return: None
        :rtype: None
        """
        try:
            if isinstance(select_model, str):
                select_model = [select_model]

            # Prepare training and testing data.
            X_train = data_train.drop([activity_col, id_col], axis=1)
            y_train = data_train[activity_col]
            X_test = data_test.drop([activity_col, id_col], axis=1)
            y_test = data_test[activity_col]

            # Verify that the task is regression.
            task_type = _get_task_type(data_train, activity_col)
            if task_type != "R":
                raise ValueError("This function only supports regression tasks.")

            # Get available models.
            model_map = _get_model_map(task_type, add_model, n_jobs=n_jobs)

            # Select models to compare.
            models_to_compare = {}
            if select_model is None:
                models_to_compare = model_map
            else:
                for name in select_model:
                    if name in model_map:
                        models_to_compare[name] = model_map[name]
                    else:
                        raise ValueError(f"Model '{name}' is not recognized.")

            n_models = len(models_to_compare)
            if n_models == 0:
                raise ValueError("No valid models selected for plotting.")

            # Determine subplot grid dimensions.
            n_cols = int(np.ceil(np.sqrt(n_models)))
            n_rows = int(np.ceil(n_models / n_cols))

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
            # Flatten axes array for easier iteration if it's multi-dimensional.
            if n_models > 1:
                axes = np.array(axes).flatten()
            else:
                axes = [axes]

            # Loop through each model, fit, predict, and create its subplot.
            for idx, (name, model) in enumerate(models_to_compare.items()):
                ax = axes[idx]
                model.fit(X=X_train, y=y_train)
                y_pred = model.predict(X_test)

                # Scatter plot: actual vs. predicted values.
                ax.scatter(y_pred, y_test, alpha=0.4, color=plt.cm.tab10(idx % 10))

                # Determine min and max for the reference line.
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                ax.plot(
                    [min_val, max_val],
                    [min_val, max_val],
                    color="gray",
                    linestyle="--",
                    linewidth=1,
                )

                # Fetch and display all scores for this model
                if scoring_df is not None and name in scoring_df.columns:
                    scores = scoring_df[name]
                    score_text = "\n".join(
                        [
                            f"{metric}: {scores[metric]:.3f}"
                            for metric in scoring_df.index
                        ]
                    )

                    ax.text(
                        scoring_loc[0],
                        scoring_loc[1],
                        score_text,
                        transform=ax.transAxes,
                        fontsize=9,
                        verticalalignment="top",
                    )

                # Set labels and title.
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title(name)
                ax.grid(True, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

            # Hide any unused subplots.
            for j in range(idx + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()

            # Save the figure.
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                full_path = f"{save_dir}/{fig_name}.pdf"
                plt.savefig(full_path, dpi=300, bbox_inches="tight")
                logging.info(f"Scatter plot saved at: {full_path}")

            plt.show()

        except Exception as e:
            logging.error(f"Error during scatter plot generation: {e}")
            raise
