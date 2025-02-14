import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import math
import os
import warnings
from copy import deepcopy
from scipy import stats
from scipy.stats import levene
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.libqsturng import psturng, qsturng
from typing import Optional, Union
import scikit_posthocs as sp
import logging


class StatisticalAnalysis:

    @staticmethod
    def extract_scoring_dfs(
        report_df: pd.DataFrame,
        scoring_list: Optional[Union[list, str]] = None,
        method_list: Optional[Union[list, str]] = None,
        melt: bool = False,
    ) -> Union[pd.DataFrame, list, list]:
        """
        Extract scoring DataFrames from the report DataFrame based on the given scoring list and method list.

        Args:
            report_df (pd.DataFrame): The input report DataFrame.
            scoring_list (Optional[Union[list, str]]): The list of scoring metrics to extract. Default is None.
            method_list (Optional[Union[list, str]]): The list of methods to extract. Default is None.
            melt (bool): Whether to melt the DataFrame to long format. Default is False.

        Returns:
            Union[pd.DataFrame, list, list]: The extracted scoring DataFrames, scoring list, and method list.
        """
        try:
            if isinstance(scoring_list, str):
                scoring_list = [scoring_list]

            if scoring_list is None:
                scoring_list = report_df.index.get_level_values("scoring").unique()

            if isinstance(method_list, str):
                method_list = [method_list]

            if method_list is None:
                method_list = report_df.columns.tolist()

            scoring_list = [scoring.lower() for scoring in scoring_list]

            filtered_dfs = []

            for scoring in scoring_list:
                if scoring not in report_df.index.get_level_values("scoring").unique():
                    raise ValueError(f"Invalid scoring value: {scoring}.")
                score_df = deepcopy(
                    report_df[report_df.index.get_level_values("scoring") == scoring]
                )
                score_df = score_df[
                    method_list
                ]  # Select only the columns in method_list
                filtered_dfs.append(score_df)

            scoring_dfs = pd.concat(filtered_dfs)
            scoring_dfs = scoring_dfs[
                ~scoring_dfs.index.get_level_values("cv_cycle").isin(
                    ["mean", "median", "std"]
                )
            ]
            scoring_dfs = scoring_dfs.sort_index(axis=0).sort_index(axis=1)

            # Melt the dataframe to long format
            if melt:
                scoring_dfs.reset_index(inplace=True)
                scoring_dfs = scoring_dfs.melt(
                    id_vars=["scoring", "cv_cycle"],
                    var_name="method",
                    value_name="value",
                )
            logging.info("Successfully extracted scoring DataFrames.")
            return scoring_dfs, scoring_list, method_list

        except Exception as e:
            logging.error(f"Error in extracting scoring DataFrames: {e}")
            raise

    @staticmethod
    def check_variance_homogeneity(
        report_df: pd.DataFrame,
        scoring_list: Optional[Union[list, str]] = None,
        method_list: Optional[Union[list, str]] = None,
        levene_test: bool = True,
        save_csv: bool = True,
        save_dir: str = "Project/Analysis",
        csv_name: str = "check_variance_homogeneity",
    ) -> pd.DataFrame:
        """
        Check variance homogeneity across methods for the given scoring metrics.

        Args:
            report_df (pd.DataFrame): The input report DataFrame.
            scoring_list (Optional[Union[list, str]]): The list of scoring metrics to check. Default is None.
            method_list (Optional[Union[list, str]]): The list of methods to check. Default is None.
            levene_test (bool): Whether to perform Levene's test for variance homogeneity. Default is True.
            save_csv (bool): Whether to save the result as a CSV file. Default is True.
            save_dir (str): The directory to save the CSV file. Default is "Project/Analysis".
            csv_name (str): The name of the CSV file. Default is "check_variance_homogeneity".

        Returns:
            pd.DataFrame: The result DataFrame with variance fold difference and p-value (if applicable).
        """
        try:
            report_new, scoring_list, _ = StatisticalAnalysis.extract_scoring_dfs(
                report_df=report_df,
                scoring_list=scoring_list,
                method_list=method_list,
                melt=True,
            )

            result = []

            for scoring in scoring_list:
                # Filter data for the current metric
                scoring_data = report_new[report_new["scoring"] == scoring]

                # Calculate variance fold difference
                variances_by_method = scoring_data.groupby("method")["value"].var()
                max_fold_diff = variances_by_method.max() / variances_by_method.min()

                scoring_result = {
                    "scoring": scoring,
                    "variance_fold_difference": max_fold_diff,
                }
                # Perform Levene's test if specified
                p_value = None
                if levene_test:
                    groups = [
                        group["value"].values
                        for _, group in scoring_data.groupby("method")
                    ]
                    stat, p_value = levene(*groups)
                    scoring_result["p_value"] = p_value

                result.append(scoring_result)

            result_df = pd.DataFrame(result).set_index("scoring")

            if save_csv:
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                result_df.to_csv(f"{save_dir}/{csv_name}.csv")

            logging.info("Successfully checked variance homogeneity.")
            return result_df

        except Exception as e:
            logging.error(f"Error in checking variance homogeneity: {e}")
            raise

    @staticmethod
    def check_normality(
        report_df: pd.DataFrame,
        scoring_list: Optional[Union[list, str]] = None,
        method_list: Optional[Union[list, str]] = None,
        save_fig: bool = True,
        save_dir: str = "Project/Analysis",
        fig_name: str = "check_normality",
    ) -> None:
        """
        Check the normality of the data for the given scoring metrics and methods.

        Args:
            report_df (pd.DataFrame): The input report DataFrame.
            scoring_list (Optional[Union[list, str]]): The list of scoring metrics to check. Default is None.
            method_list (Optional[Union[list, str]]): The list of methods to check. Default is None.
            save_fig (bool): Whether to save the resulting figure as a file. Default is True.
            save_dir (str): The directory to save the figure. Default is "Project/Analysis".
            fig_name (str): The name of the figure file. Default is "check_normality".

        Returns:
            None
        """
        try:
            report_new, scoring_list, _ = StatisticalAnalysis.extract_scoring_dfs(
                report_df=report_df,
                scoring_list=scoring_list,
                method_list=method_list,
                melt=True,
            )

            # Normalize values by subtracting the mean for each method
            df_norm = deepcopy(report_new)
            df_norm["value"] = df_norm.groupby(["method", "scoring"])[
                "value"
            ].transform(lambda x: x - x.mean())

            sns.set_context("notebook")
            sns.set_style("whitegrid")

            fig, axes = plt.subplots(
                2, len(scoring_list), figsize=(5 * len(scoring_list), 10)
            )

            for i, scoring in enumerate(scoring_list):
                ax = axes[0, i]
                scoring_data = df_norm[df_norm["scoring"] == scoring]["value"]
                sns.histplot(scoring_data, kde=True, ax=ax)
                ax.set_title(f"{scoring.upper()}", fontsize=16)

                ax2 = axes[1, i]
                stats.probplot(scoring_data, dist="norm", plot=ax2)
                ax2.set_title("")

            plt.tight_layout()
            if save_fig:
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                plt.savefig(
                    f"{save_dir}/{fig_name}.pdf",
                    dpi=300,
                    bbox_inches="tight",
                )
            logging.info("Successfully checked normality.")
        except Exception as e:
            logging.error(f"Error in checking normality: {e}")
            raise

    @staticmethod
    def test(
        report_df: pd.DataFrame,
        scoring_list: Optional[Union[list, str]] = None,
        method_list: Optional[Union[list, str]] = None,
        select_test: str = "AnovaRM",
        showmeans: bool = True,
        save_fig: bool = True,
        save_dir: str = "Project/Analysis",
        fig_name: Optional[str] = None,
    ) -> None:
        """
        Perform a statistical test (AnovaRM or friedman) on the report dataframe and plot the results.

        Parameters
        ----------
        report_df : pd.DataFrame
            The input report dataframe.
        scoring_list : Optional[Union[list, str]], optional
            List or string of scoring metrics, by default None.
        method_list : Optional[Union[list, str]], optional
            List or string of methods, by default None.
        select_test : str, optional
            The statistical test to use ('AnovaRM' or 'friedman'), by default "AnovaRM".
        showmeans : bool, optional
            Whether to show means in the boxplot, by default True.
        save_fig : bool, optional
            Whether to save the figure, by default True.
        save_dir : str, optional
            Directory to save the figure, by default "Project/Analysis".
        fig_name : Optional[str], optional
            Name of the saved figure, by default None.

        Returns
        -------
        None
        """
        try:
            report_new, scoring_list, method_list = (
                StatisticalAnalysis.extract_scoring_dfs(
                    report_df=report_df,
                    scoring_list=scoring_list,
                    method_list=method_list,
                    melt=True,
                )
            )

            if select_test not in ["AnovaRM", "friedman"]:
                raise ValueError(
                    f"Unsupported test: {select_test}."
                    "Please choose 'AnovaRM' for a parametric test or 'friedman' for a non-parametric test."
                )

            sns.set_context("notebook")
            sns.set_style("whitegrid")

            nrow = math.ceil(len(scoring_list) / 2)
            nmethod = len(method_list)

            figure, axes = plt.subplots(
                nrow, 2, sharex=False, sharey=False, figsize=(3 * nmethod, 7 * nrow)
            )
            axes = axes.flatten()  # Turn 2D array to 1D array

            for i, scoring in enumerate(scoring_list):
                scoring_data = report_new[report_new["scoring"] == scoring]
                if select_test == "AnovaRM":
                    model = AnovaRM(
                        data=scoring_data,
                        depvar="value",
                        subject="cv_cycle",
                        within=["method"],
                    ).fit()
                    p_value = model.anova_table["Pr > F"].iloc[0]

                else:
                    p_value = pg.friedman(
                        data=scoring_data,
                        dv="value",
                        within="method",
                        subject="cv_cycle",
                    )["p-unc"].values[0]

                ax = sns.boxplot(
                    y="value",
                    x="method",
                    hue="method",
                    ax=axes[i],
                    showmeans=showmeans,
                    data=scoring_data,
                    palette="plasma",
                    legend=False,
                    width=0.5,
                )
                ax.set_title(f"p={p_value:.1e}")
                ax.set_xlabel("")
                ax.set_ylabel(scoring.upper())

                # Wrap labels
                labels = [item.get_text() for item in ax.get_xticklabels()]
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
                ax.set_xticks(list(range(0, len(labels))))
                ax.set_xticklabels(new_labels)
                ax.tick_params(axis="both", labelsize=12)

            # If there are less plots than cells in the grid, hide the remaining cells
            if (len(scoring_list) % 2) != 0:
                for i in range(len(scoring_list), nrow * 2):
                    axes[i].set_visible(False)

            plt.tight_layout()

            fig_name = fig_name or select_test
            if save_fig:
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                plt.savefig(
                    f"{save_dir}/{fig_name}.pdf",
                    dpi=300,
                    bbox_inches="tight",
                )
            logging.info(f"Figure saved at {save_dir}/{fig_name}.pdf")

        except Exception as e:
            logging.error(f"An error occurred during the test: {e}")
            raise

    @staticmethod
    def posthoc_conover_friedman(
        report_df: pd.DataFrame,
        scoring_list: Optional[Union[list, str]] = None,
        method_list: Optional[Union[list, str]] = None,
        plot: Optional[Union[list, str]] = None,
        save_fig: bool = True,
        save_result: bool = True,
        save_dir: str = "Project/Analysis",
    ) -> tuple:
        """
        Perform posthoc Conover-Friedman test on the report dataframe and generate plots.

        Parameters
        ----------
        report_df : pd.DataFrame
            The input report dataframe.
        scoring_list : Optional[Union[list, str]], optional
            List or string of scoring metrics, by default None.
        method_list : Optional[Union[list, str]], optional
            List or string of methods, by default None.
        plot : Optional[Union[list, str]], optional
            Type of plot to generate ('sign' or 'ccd'), by default None.
        save_fig : bool, optional
            Whether to save the figure, by default True.
        save_result : bool, optional
            Whether to save the results, by default True.
        save_dir : str, optional
            Directory to save the figure and results, by default "Project/Analysis".

        Returns
        -------
        tuple
            A tuple containing the posthoc Conover-Friedman results and the rank results.
        """
        try:
            report_new, scoring_list, method_list = (
                StatisticalAnalysis.extract_scoring_dfs(
                    report_df=report_df,
                    scoring_list=scoring_list,
                    method_list=method_list,
                    melt=False,
                )
            )

            # Precompute posthoc Conover-Friedman results for each metric
            pc_results = {}
            rank_results = {}

            for scoring in scoring_list:
                scoring_df_filtered = report_new[
                    report_new.index.get_level_values("scoring") == scoring
                ]
                scoring_df_filtered.reset_index(
                    level="scoring", drop=True, inplace=True
                )
                pc_results[scoring] = sp.posthoc_conover_friedman(
                    scoring_df_filtered, p_adjust="holm"
                )
                # Compute the mean rank for each method
                rank_results[scoring] = scoring_df_filtered.rank(
                    axis=1, method="average", pct=True
                ).mean(axis=0)

                if save_result:
                    if save_dir and not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    pc_results[scoring].to_csv(f"{save_dir}/cofried_pc_{scoring}.csv")
                    logging.info(
                        f"Posthoc Conover-Friedman results saved at {save_dir}/cofried_pc_{scoring}.csv"
                    )

            def make_sign_plots(scoring_list, pc_results, save_fig, save_dir):
                heatmap_args = {
                    "linewidths": 0.25,
                    "linecolor": "0.5",
                    "clip_on": True,
                    "square": True,
                }
                sns.set_context("notebook")
                sns.set_style("whitegrid")

                figure, axes = plt.subplots(
                    1,
                    len(scoring_list),
                    sharex=False,
                    sharey=True,
                    figsize=(6 * len(scoring_list), 10),
                )

                if not isinstance(axes, np.ndarray):
                    axes = np.array([axes])

                for i, scoring in enumerate(scoring_list):
                    pc = pc_results[scoring]
                    sub_ax, sub_c = sp.sign_plot(
                        pc, **heatmap_args, ax=axes[i], xticklabels=True
                    )
                    sub_ax.set_title(scoring.upper(), fontsize=16)

                if save_fig:
                    if save_dir and not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(
                        f"{save_dir}/cofried_sign_plot.pdf",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    logging.info(
                        f"Sign plots saved at {save_dir}/cofried_sign_plot.pdf"
                    )

            def make_critical_difference_diagrams(
                scoring_list, pc_results, rank_results, save_fig, save_dir
            ):
                sns.set_context("notebook")
                sns.set_style("whitegrid")
                figure, axes = plt.subplots(
                    len(scoring_list),
                    1,
                    sharex=True,
                    sharey=False,
                    figsize=(20, 3 * len(scoring_list)),
                )
                if not isinstance(axes, np.ndarray):
                    axes = np.array([axes])

                for i, scoring in enumerate(scoring_list):
                    pc = pc_results[scoring]
                    avg_rank = rank_results[scoring]
                    sp.critical_difference_diagram(avg_rank, pc, ax=axes[i])
                    axes[i].set_title(scoring.upper(), fontsize=16)

                plt.tight_layout()
                if save_fig:
                    if save_dir and not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(
                        f"{save_dir}/cofried_ccd.pdf",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    logging.info(
                        f"Critical difference diagrams saved at {save_dir}/cofried_ccd.pdf"
                    )

            if plot not in ["sign", "ccd", None]:
                raise ValueError(
                    f"Invalid plot type: {plot}. Please choose 'sign' or 'ccd'."
                )
            if plot is None or plot == "sign":
                make_sign_plots(scoring_list, pc_results, save_fig, save_dir)

            if plot is None or plot == "ccd":
                make_critical_difference_diagrams(
                    scoring_list, pc_results, rank_results, save_fig, save_dir
                )

            logging.info("Posthoc Conover-Friedman analysis completed successfully")
            return pc_results, rank_results

        except Exception as e:
            logging.error(
                f"An error occurred during the posthoc Conover-Friedman analysis: {e}"
            )

            return {}, {}

    @staticmethod
    def posthoc_tukeyhsd(
        report_df: pd.DataFrame,
        scoring_list: Optional[Union[list, str]] = None,
        method_list: Optional[Union[list, str]] = None,
        plot: Optional[Union[list, str]] = None,
        alpha: float = 0.05,
        direction_dict: dict = {},
        effect_dict: dict = {},
        title_size: int = 16,
        save_fig: bool = True,
        save_result: bool = True,
        save_dir: str = "Project/Analysis",
    ) -> dict:
        """
        Perform posthoc Tukey's HSD test on the report dataframe and generate plots.

        Parameters
        ----------
        report_df : pd.DataFrame
            The input report dataframe.
        scoring_list : Optional[Union[list, str]], optional
            List or string of scoring metrics, by default None.
        method_list : Optional[Union[list, str]], optional
            List or string of methods, by default None.
        plot : Optional[Union[list, str]], optional
            Type of plot to generate ('mcs_plot' or 'ci_plot'), by default None.
        alpha : float, optional
            Significance level for the test, by default 0.05.
        direction_dict : dict, optional
            Dictionary specifying the direction to maximize or minimize for each scoring metric, by default {}.
        effect_dict : dict, optional
            Dictionary specifying the effect size for each scoring metric, by default {}.
        save_fig : bool, optional
            Whether to save the figure, by default True.
        save_result : bool, optional
            Whether to save the results, by default True.
        save_dir : str, optional
            Directory to save the figure and results, by default "Project/Analysis".

        Returns
        -------
        dict
            A dictionary containing Tukey's HSD results.
        """
        try:
            report_new, scoring_list, method_list = (
                StatisticalAnalysis.extract_scoring_dfs(
                    report_df=report_df,
                    scoring_list=scoring_list,
                    method_list=method_list,
                    melt=True,
                )
            )

            # Set defaults
            for key in scoring_list:
                direction_dict.setdefault(
                    key, "maximize" if key != "max_error" else "minimize"
                )

            for key in scoring_list:
                effect_dict.setdefault(key, 0.1)

            direction_dict = {k.lower(): v for k, v in direction_dict.items()}
            effect_dict = {k.lower(): v for k, v in effect_dict.items()}

            tukey_results = {}

            for scoring in scoring_list:
                if direction_dict and scoring in direction_dict:
                    if direction_dict[scoring] == "maximize":
                        df_means = (
                            report_new.groupby("method")
                            .mean(numeric_only=True)
                            .sort_values("value", ascending=False)
                        )
                    elif direction_dict[scoring] == "minimize":
                        df_means = (
                            report_new.groupby("method")
                            .mean(numeric_only=True)
                            .sort_values("value", ascending=True)
                        )
                    else:
                        raise ValueError(
                            "Invalid direction. Expected 'maximize' or 'minimize'."
                        )
                else:
                    df_means = report_new.groupby("method").mean(numeric_only=True)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    aov = pg.rm_anova(
                        dv="value",
                        within="method",
                        subject="cv_cycle",
                        data=report_new[report_new["scoring"] == scoring],
                        detailed=True,
                    )

                mse = aov.loc[1, "MS"]
                df_resid = aov.loc[1, "DF"]

                methods = df_means.index
                n_groups = len(methods)
                n_per_group = report_new["method"].value_counts().mean()

                tukey_se = np.sqrt(2 * mse / n_per_group)
                q = qsturng(1 - alpha, n_groups, df_resid)

                num_comparisons = len(methods) * (len(methods) - 1) // 2
                result_tab = pd.DataFrame(
                    index=range(num_comparisons),
                    columns=["group1", "group2", "meandiff", "lower", "upper", "p-adj"],
                )

                df_means_diff = pd.DataFrame(index=methods, columns=methods, data=0.0)
                pc = pd.DataFrame(index=methods, columns=methods, data=1.0)

                row_idx = 0
                for i, method1 in enumerate(methods):
                    for j, method2 in enumerate(methods):
                        if i < j:
                            group1 = report_new[report_new["method"] == method1][
                                "value"
                            ]
                            group2 = report_new[report_new["method"] == method2][
                                "value"
                            ]
                            mean_diff = group1.mean() - group2.mean()
                            studentized_range = np.abs(mean_diff) / tukey_se
                            adjusted_p = psturng(
                                studentized_range * np.sqrt(2), n_groups, df_resid
                            )
                            if isinstance(adjusted_p, np.ndarray):
                                adjusted_p = adjusted_p[0]
                            lower = mean_diff - (q / np.sqrt(2) * tukey_se)
                            upper = mean_diff + (q / np.sqrt(2) * tukey_se)
                            result_tab.loc[row_idx] = [
                                method1,
                                method2,
                                mean_diff,
                                lower,
                                upper,
                                adjusted_p,
                            ]
                            pc.loc[method1, method2] = adjusted_p
                            pc.loc[method2, method1] = adjusted_p
                            df_means_diff.loc[method1, method2] = mean_diff
                            df_means_diff.loc[method2, method1] = -mean_diff
                            row_idx += 1

                df_means_diff = df_means_diff.astype(float)

                result_tab["group1_mean"] = result_tab["group1"].map(df_means["value"])
                result_tab["group2_mean"] = result_tab["group2"].map(df_means["value"])

                result_tab.index = result_tab["group1"] + " - " + result_tab["group2"]

                tukey_results[scoring] = {
                    "result_tab": result_tab,
                    "df_means": df_means,
                    "df_means_diff": df_means_diff,
                    "pc": pc,
                }
                if save_result:
                    if save_dir and not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    result_tab.to_csv(f"{save_dir}/tukey_result_tab_{scoring}.csv")
                    df_means.to_csv(f"{save_dir}/tukey_df_means_{scoring}.csv")
                    df_means_diff.to_csv(
                        f"{save_dir}/tukey_df_means_diff_{scoring}.csv"
                    )
                    pc.to_csv(f"{save_dir}/tukey_pc_{scoring}.csv")
                    logging.info(f"Tukey HSD results saved at {save_dir}")

            if plot is None or plot == "mcs_plot":
                StatisticalAnalysis._make_mcs_plot_grid(
                    tukey_results,
                    scoring_list,
                    method_list,
                    direction_dict,
                    effect_dict,
                    show_diff=True,
                    cell_text_size=16,
                    axis_text_size=12,
                    title_size=title_size,
                    save_fig=save_fig,
                    save_dir=save_dir,
                )

            if plot is None or plot == "ci_plot":
                StatisticalAnalysis._make_ci_plot_grid(
                    tukey_results,
                    scoring_list,
                    method_list,
                    title_size=title_size,
                    left_xlim=-0.5,
                    right_xlim=1,
                    save_fig=save_fig,
                    save_dir=save_dir,
                )

            logging.info("Posthoc Tukey HSD analysis completed successfully")
            return tukey_results

        except Exception as e:
            logging.error(
                f"An error occurred during the posthoc Tukey HSD analysis: {e}"
            )
            return {}

    @staticmethod
    def _mcs_plot(
        pc: pd.DataFrame,
        effect_size: pd.DataFrame,
        means: pd.DataFrame,
        labels: bool = True,
        cmap: Optional[str] = None,
        cbar_ax_bbox: Optional[tuple] = None,
        ax: Optional[plt.Axes] = None,
        show_diff: bool = True,
        cell_text_size: int = 16,
        axis_text_size: int = 12,
        show_cbar: bool = True,
        reverse_cmap: bool = False,
        vlim: Optional[float] = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Generate a heatmap plot of effect sizes with significance annotations.

        Parameters
        ----------
        pc : pd.DataFrame
            Pairwise comparison matrix containing p-values.
        effect_size : pd.DataFrame
            Matrix of effect sizes.
        means : pd.DataFrame
            Mean values for each method.
        labels : bool, optional
            Whether to show method labels, by default True.
        cmap : Optional[str], optional
            Colormap to use for the heatmap, by default None.
        cbar_ax_bbox : Optional[tuple], optional
            Bounding box for the colorbar, by default None.
        ax : Optional[plt.Axes], optional
            Axes object to draw the heatmap on, by default None.
        show_diff : bool, optional
            Whether to show effect size differences, by default True.
        cell_text_size : int, optional
            Text size for cell annotations, by default 16.
        axis_text_size : int, optional
            Text size for axis labels, by default 12.
        show_cbar : bool, optional
            Whether to show the colorbar, by default True.
        reverse_cmap : bool, optional
            Whether to reverse the colormap, by default False.
        vlim : Optional[float], optional
            Value limit for the heatmap, by default None.

        Returns
        -------
        plt.Axes
            Axes object with the heatmap.
        """
        try:
            for key in ["cbar", "vmin", "vmax", "center"]:
                if key in kwargs:
                    del kwargs[key]

            if not cmap:
                cmap = "coolwarm"
            if reverse_cmap:
                cmap = cmap + "_r"

            significance = pc.copy().astype(object)
            significance[(pc < 0.001) & (pc >= 0)] = "***"
            significance[(pc < 0.01) & (pc >= 0.001)] = "**"
            significance[(pc < 0.05) & (pc >= 0.01)] = "*"
            significance[(pc >= 0.05)] = ""

            np.fill_diagonal(significance.values, "")

            # Create a DataFrame for the annotations
            if show_diff:
                annotations = effect_size.round(3).astype(str) + significance
            else:
                annotations = significance

            hax = sns.heatmap(
                effect_size,
                cmap=cmap,
                annot=annotations,
                fmt="",
                cbar=show_cbar,
                ax=ax,
                annot_kws={"size": cell_text_size},
                vmin=-2 * vlim if vlim else None,
                vmax=2 * vlim if vlim else None,
                **kwargs,
            )

            if labels:
                label_list = list(means.index)
                x_label_list = [
                    x + f"\n{means.loc[x].values[0].round(2)}" for x in label_list
                ]
                y_label_list = [
                    x + f"\n{means.loc[x].values[0].round(2)}\n" for x in label_list
                ]
                hax.set_xticklabels(
                    x_label_list,
                    size=axis_text_size,
                    ha="center",
                    va="top",
                    rotation=0,
                    rotation_mode="anchor",
                )
                hax.set_yticklabels(
                    y_label_list,
                    size=axis_text_size,
                    ha="center",
                    va="center",
                    rotation=90,
                    rotation_mode="anchor",
                )

            hax.set_xlabel("")
            hax.set_ylabel("")

            return hax

        except Exception as e:
            logging.error(f"An error occurred in _mcs_plot: {e}")
            return ax

    @staticmethod
    def _make_mcs_plot_grid(
        tukey_results: dict,
        scoring_list: list,
        method_list: list,
        direction_dict: dict = {},
        effect_dict: dict = {},
        show_diff: bool = True,
        cell_text_size: int = 16,
        axis_text_size: int = 12,
        title_size: int = 16,
        save_fig: bool = True,
        save_dir: str = "Project/Analysis",
    ) -> None:
        """
        Generate a grid of MCS plots for each scoring metric.

        Parameters
        ----------
        tukey_results : dict
            Dictionary containing Tukey's HSD results.
        scoring_list : list
            List of scoring metrics.
        method_list : list
            List of methods.
        direction_dict : dict, optional
            Dictionary specifying the direction to maximize or minimize for each scoring metric, by default {}.
        effect_dict : dict, optional
            Dictionary specifying the effect size for each scoring metric, by default {}.
        show_diff : bool, optional
            Whether to show effect size differences, by default True.
        cell_text_size : int, optional
            Text size for cell annotations, by default 16.
        axis_text_size : int, optional
            Text size for axis labels, by default 12.
        title_size : int, optional
            Text size for titles, by default 16.
        save_fig : bool, optional
            Whether to save the figure, by default True.
        save_dir : str, optional
            Directory to save the figure, by default "Project/Analysis".

        Returns
        -------
        None
        """
        try:
            # Set defaults
            for key in scoring_list:
                direction_dict.setdefault(
                    key, "maximize" if key != "max_error" else "minimize"
                )

            for key in scoring_list:
                effect_dict.setdefault(key, 0.1)

            direction_dict = {k.lower(): v for k, v in direction_dict.items()}
            effect_dict = {k.lower(): v for k, v in effect_dict.items()}

            nrow = math.ceil(len(scoring_list) / 3)
            nmethod = len(method_list)
            fig, ax = plt.subplots(
                nrow, 3, figsize=(7.8 * nmethod, 2.3 * nmethod * nrow)
            )

            ax = ax.flatten()

            for i, scoring in enumerate(scoring_list):
                scoring = scoring.lower()

                if scoring not in direction_dict:
                    raise ValueError(
                        f"Stat '{scoring}' is missing in direction_dict. Please set its value."
                    )
                if scoring not in effect_dict:
                    raise ValueError(
                        f"Stat '{scoring}' is missing in effect_dict. Please set its value."
                    )

                reverse_cmap = False
                if direction_dict[scoring] == "minimize":
                    reverse_cmap = True

                df_means = tukey_results[scoring]["df_means"]
                df_means_diff = tukey_results[scoring]["df_means_diff"]
                pc = tukey_results[scoring]["pc"]

                hax = StatisticalAnalysis._mcs_plot(
                    pc,
                    effect_size=df_means_diff,
                    means=df_means,
                    show_diff=show_diff,
                    ax=ax[i],
                    cbar=True,
                    cell_text_size=cell_text_size,
                    axis_text_size=axis_text_size,
                    reverse_cmap=reverse_cmap,
                    vlim=effect_dict[scoring],
                )
                hax.set_title(scoring.upper(), fontsize=title_size)

            # If there are less plots than cells in the grid, hide the remaining cells
            if (len(scoring_list) % 3) != 0:
                for i in range(len(scoring_list), nrow * 3):
                    ax[i].set_visible(False)

            plt.tight_layout()

            if save_fig:
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                plt.savefig(
                    f"{save_dir}/tukey_mcs.pdf",
                    dpi=300,
                    bbox_inches="tight",
                )
                logging.info(f"MCS plot grid saved at {save_dir}/tukey_mcs.pdf")

        except Exception as e:
            logging.error(f"An error occurred in _make_mcs_plot_grid: {e}")

    @staticmethod
    def _make_ci_plot_grid(
        tukey_results: dict,
        scoring_list: list,
        method_list: list,
        title_size: int = 16,
        left_xlim: float = -0.5,
        right_xlim: float = 0.5,
        save_fig: bool = True,
        save_dir: str = "Project/Analysis",
    ) -> None:
        """
        Generate a grid of confidence interval plots for each scoring metric.

        Parameters
        ----------
        tukey_results : dict
            Dictionary containing Tukey's HSD results.
        scoring_list : list
            List of scoring metrics.
        method_list : list
            List of methods.
        title_size : int, optional
            Text size for titles, by default 16.
        left_xlim : float, optional
            Left limit for x-axis, by default -0.5.
        right_xlim : float, optional
            Right limit for x-axis, by default 0.5.
        save_fig : bool, optional
            Whether to save the figure, by default True.
        save_dir : str, optional
            Directory to save the figure, by default "Project/Analysis".

        Returns
        -------
        None
        """
        try:
            nmethod = len(method_list)
            ncouple = nmethod * (nmethod - 1) // 2
            figure, axes = plt.subplots(
                len(scoring_list),
                1,
                figsize=(12, 0.35 * ncouple * len(scoring_list)),
                sharex=False,
            )

            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])

            for i, scoring in enumerate(scoring_list):
                result_tab = tukey_results[scoring]["result_tab"]

                result_err = np.array(
                    [
                        result_tab["meandiff"] - result_tab["lower"],
                        result_tab["upper"] - result_tab["meandiff"],
                    ]
                )
                sns.set_context("notebook")
                sns.set_style("whitegrid")
                ax = sns.pointplot(
                    x=result_tab.meandiff,
                    y=result_tab.index,
                    marker="o",
                    linestyle="",
                    ax=axes[i],
                    color="red",
                    markersize=5,
                )
                ax.errorbar(
                    y=result_tab.index,
                    x=result_tab["meandiff"],
                    xerr=result_err,
                    fmt="ro",
                    capsize=4,
                    ecolor="red",
                    markerfacecolor="red",
                )
                ax.axvline(0, ls="--", lw=3, color="#ADD3ED")
                ax.set_xlabel("Mean Difference")
                ax.set_ylabel("")
                ax.set_title(scoring.upper(), fontsize=title_size)
                ax.set_xlim(left_xlim, right_xlim)
                ax.grid(True, axis="x")
            plt.tight_layout()

            if save_fig:
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                plt.savefig(
                    f"{save_dir}/tukey_ci.pdf",
                    dpi=300,
                    bbox_inches="tight",
                )
                logging.info(f"CI plot grid saved at {save_dir}/tukey_ci.pdf")

        except Exception as e:
            logging.error(f"An error occurred in _make_ci_plot_grid: {e}")
