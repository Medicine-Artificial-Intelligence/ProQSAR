import os
import math
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Union, Tuple
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from ProQSAR.ModelDeveloper.model_developer_utils import (
    _get_task_type,
    _get_model_map,
    _get_cv_strategy,
    _get_cv_scoring,
    _get_ev_scoring,
)


class ModelValidation:
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
        models,
        X_data,
        y_data,
        cv,
        scoring_list,
        include_stats,
        n_splits,
        n_repeats,
        n_jobs,
    ):
        result = []

        for name, model in models.items():
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
    ):
        """
        Draws ROC and/or Precision-Recall curves for selected models and saves the plot.

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
        curve_type : Union[str, List[str]], default="roc"
            The type of curve(s) to generate. Can be a single value ("roc" or "pr")
            or a list of values (e.g., ["roc", "pr"]).
        select_model : Optional[List[str]]
            List of models to be selected for plotting. If None, all models are used.
        add_model : Optional[dict]
            Dictionary of additional models to include.
        legend_loc : Optional[Union[str, Tuple[float, float]]]
            The location for the legend in the plot. If not provided, defaults are set based on the curve type.
            For ROC: "lower right", for PR: "lower left".
        save_dir : str, default="Project/ModelDevelopment"
            Directory where the figure will be saved.
        fig_name : Optional[str]
            The base name of the file where the plot will be saved. If multiple curves are plotted,
            a suffix is added based on the curve type.
        n_jobs : int, default=1
            The number of jobs to run in parallel when fitting models.

        Returns:
        --------
        None
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
    ):
        """
        Generates separate scatter plots (using subplots) for regression tasks, each comparing
        actual vs. predicted values for a selected model.

        Parameters:
        -----------
        data_train : pd.DataFrame
            Training data used to fit the models.
        data_test : pd.DataFrame
            Test data used for evaluating the models.
        activity_col : str
            The target column (response variable) in the dataset.
        id_col : str
            The identifier column in the dataset.
        select_model : Optional[List[str]]
            List of model names to be selected for plotting. If None, all available models are used.
        add_model : Optional[dict]
            Dictionary of additional models to include.
        legend_loc : Union[str, Tuple[float, float]], default="best"
            The location for the legend in each subplot.
        save_dir : str, default="Project/ModelDevelopment"
            Directory where the figure will be saved.
        fig_name : str, default="scatter_plot"
            The base name of the file where the plot will be saved.
        n_jobs : int, default=1
            The number of jobs to run in parallel when fitting models.

        Returns:
        --------
        None
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
            # Here, we compute a grid with a reasonable layout.
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
