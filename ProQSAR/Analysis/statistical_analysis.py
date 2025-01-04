import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import math
import warnings
from copy import deepcopy
from scipy import stats
from scipy.stats import levene
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.libqsturng import psturng, qsturng
from matplotlib import cm
from typing import Optional, Union
import scikit_posthocs as sp


class StatisticalAnalysis:

    @staticmethod
    def extract_scoring_dfs(
        report_df: pd.DataFrame,
        scoring_list: Optional[Union[list, str]] = None,
        method_list: Optional[Union[list, str]] = None,
        melt: bool = False,
    ):
        if isinstance(scoring_list, str):
            scoring_list = [scoring_list]

        if scoring_list is None:
            scoring_list = report_df.index.get_level_values("scoring").unique()

        if isinstance(method_list, str):
            method_list = [method_list]

        if method_list is None:
            method_list = report_df.columns.tolist()

        filtered_dfs = []

        for scoring in scoring_list:
            score_df = deepcopy(
                report_df[report_df.index.get_level_values("scoring") == scoring]
            )
            score_df = score_df[method_list]  # Select only the columns in method_list
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
                id_vars=["scoring", "cv_cycle"], var_name="method", value_name="value"
            )

        return scoring_dfs, scoring_list, method_list

    @staticmethod
    def check_variance_homogeneity(
        report_df: pd.DataFrame,
        scoring_list: Optional[Union[list, str]] = None,
        method_list: Optional[Union[list, str]] = None,
        levene_test: bool = True,
    ):

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
                    group["value"].values for _, group in scoring_data.groupby("method")
                ]
                stat, p_value = levene(*groups)
                scoring_result["p_value"] = p_value

            result.append(scoring_result)

        result_df = pd.DataFrame(result).set_index("scoring")

        return result_df

    @staticmethod
    def check_normality(
        report_df: pd.DataFrame,
        scoring_list: Optional[Union[list, str]] = None,
        method_list: Optional[Union[list, str]] = None,
    ):
        """
        Check the normality of the scoring metrics in the report DataFrame.

        This method normalizes values by subtracting the mean for each method,
        and then visualizes the distribution of the scoring metrics through histograms
        and Q-Q plots to assess normality.

        Parameters:
            report_df (pd.DataFrame): DataFrame containing scoring metrics for different methods.
            scoring (list | str): Scoring metric(s) to check normality (e.g., 'precision', 'f1', 'roc_auc').

        Returns:
            None: Displays histograms and Q-Q plots for each scoring metric.
        """

        report_new, scoring_list, _ = StatisticalAnalysis.extract_scoring_dfs(
            report_df=report_df,
            scoring_list=scoring_list,
            method_list=method_list,
            melt=True,
        )

        # Normalize values by subtracting the mean for each method
        df_norm = deepcopy(report_new)
        df_norm["value"] = df_norm.groupby(["method", "scoring"])["value"].transform(
            lambda x: x - x.mean()
        )

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

    @staticmethod
    def test(
        report_df: pd.DataFrame,
        scoring_list: Optional[Union[list, str]] = None,
        method_list: Optional[Union[list, str]] = None,
        select_test: str = "AnovaRM",
        showmeans: bool = True,
    ):
        report_new, scoring_list, method_list = StatisticalAnalysis.extract_scoring_dfs(
            report_df=report_df,
            scoring_list=scoring_list,
            method_list=method_list,
            melt=True,
        )

        if select_test not in ["AnovaRM", "friedman"]:
            raise ValueError(
                f"Unsupported test: {select_test}. Please choose 'AnovaRM' for a parametric test or 'friedman' for a non-parametric test."
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
            if select_test == "AnovaRM":
                scoring_data = report_new[report_new["scoring"] == scoring]
                model = AnovaRM(
                    data=scoring_data,
                    depvar="value",
                    subject="cv_cycle",
                    within=["method"],
                ).fit()
                p_value = model.anova_table["Pr > F"].iloc[0]

            else:
                p_value = pg.friedman(
                    data=scoring_data, dv="value", within="method", subject="cv_cycle"
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

    @staticmethod
    def posthoc_conover_friedman(
        report_df: pd.DataFrame,
        scoring_list: Optional[Union[list, str]] = None,
        method_list: Optional[Union[list, str]] = None,
        plot: Optional[Union[list, str]] = None,
    ):

        report_new, scoring_list, method_list = StatisticalAnalysis.extract_scoring_dfs(
            report_df=report_df,
            scoring_list=scoring_list,
            method_list=method_list,
            melt=False,
        )

        # Precompute posthoc Conover-Friedman results for each metric
        pc_results = {}
        rank_results = {}

        for scoring in scoring_list:
            scoring_df_filtered = report_new[
                report_new.index.get_level_values("scoring") == scoring
            ]
            scoring_df_filtered.reset_index(level="scoring", drop=True, inplace=True)
            pc_results[scoring] = sp.posthoc_conover_friedman(
                scoring_df_filtered, p_adjust="holm"
            )
            # Compute the mean rank for each method
            rank_results[scoring] = scoring_df_filtered.rank(
                axis=1, method="average", pct=True
            ).mean(axis=0)

        def make_sign_plots(scoring_list, pc_results):
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

            axes = [axes] if len(scoring_list) == 1 else axes

            for i, scoring in enumerate(scoring_list):
                pc = pc_results[scoring]
                sub_ax, sub_c = sp.sign_plot(
                    pc, **heatmap_args, ax=axes[i], xticklabels=True
                )
                sub_ax.set_title(scoring.upper(), fontsize=16)

        def make_critical_difference_diagrams(scoring_list, pc_results, rank_results):
            sns.set_context("notebook")
            sns.set_style("whitegrid")
            figure, axes = plt.subplots(
                len(scoring_list),
                1,
                sharex=True,
                sharey=False,
                figsize=(20, 3 * len(scoring_list)),
            )
            axes = [axes] if len(scoring_list) == 1 else axes
            for i, scoring in enumerate(scoring_list):
                pc = pc_results[scoring]
                avg_rank = rank_results[scoring]
                sp.critical_difference_diagram(avg_rank, pc, ax=axes[i])
                axes[i].set_title(scoring.upper(), fontsize=16)

            plt.tight_layout()

        if plot is None or plot == "sign":
            make_sign_plots(scoring_list, pc_results)

        if plot is None or plot == "ccd":
            make_critical_difference_diagrams(scoring_list, pc_results, rank_results)

        return pc_results, rank_results

    def posthoc_conover_friedman2(
        report_df: pd.DataFrame,
        scoring_list: Optional[Union[list, str]] = None,
        method_list: Optional[Union[list, str]] = None,
        plot: Optional[Union[list, str]] = None,
    ):

        report_new, scoring_list, method_list = StatisticalAnalysis.extract_scoring_dfs(
            report_df=report_df,
            scoring_list=scoring_list,
            method_list=method_list,
            melt=False,
        )

        # Precompute posthoc Conover-Friedman results for each metric
        pc_results = {}
        for scoring in scoring_list:
            scoring_df_filtered = report_new[
                report_new.index.get_level_values("scoring") == scoring
            ]
            scoring_df_filtered.reset_index(level="scoring", drop=True, inplace=True)
            pc_results[scoring] = sp.posthoc_conover_friedman(
                scoring_df_filtered, p_adjust="holm"
            )

        return pc_results

    ########## refine code ###########
    @staticmethod
    def posthoc_tukeyhsd(
        report_df: pd.DataFrame,
        scoring_list: Optional[Union[list, str]] = None,
        method_list: Optional[Union[list, str]] = None,
        plot: Optional[Union[list, str]] = None,
        alpha: float = 0.05,
        direction_dict: Optional[dict] = None,
    ):
        """
        Perform Tukey HSD test and generate corresponding plots.

        Parameters:
        report_df (pd.DataFrame): Dataframe containing the report data.
        scoring_list (Optional[Union[list, str]]): List or string of metrics to be analyzed. If None, all metrics are analyzed.
        plot (Optional[Union[list, str]]): Type of plot to generate. Options are 'mcs_plot', 'ci_plot'. If None, both plots are generated.
        alpha (float): Significance level for the test. Default is 0.05.
        direction_dict (Optional[dict]): Dictionary indicating whether to minimize or maximize each metric.
        """

        report_new, scoring_list, method_list = StatisticalAnalysis.extract_scoring_dfs(
            report_df=report_df,
            scoring_list=scoring_list,
            method_list=method_list,
            melt=True,
        )

        tukey_results = {}

        for scoring in scoring_list:
            if direction_dict and scoring in direction_dict:
                sort_order = direction_dict[scoring]
                df_means = (
                    report_new.groupby("method")
                    .mean(numeric_only=True)
                    .sort_values(scoring, ascending=(sort_order == "minimize")) # The ascending parameter is set to True if sort_order is "minimize"
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
                        group1 = report_new[report_new["method"] == method1][scoring]
                        group2 = report_new[report_new["method"] == method2][scoring]
                        mean_diff = group1.mean() - group2.mean()
                        studentized_range = np.abs(mean_diff) / tukey_se
                        adjusted_p = qsturng(
                            studentized_range * np.sqrt(2), n_groups, df_resid
                        )
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

            tukey_results[scoring] = {
                "result_tab": result_tab,
                "df_means": df_means,
                "df_means_diff": df_means_diff,
                "pc": pc,
            }

        if plot is None or plot == "mcs_plot":
            make_mcs_plot(scoring_list, tukey_results)

        if plot is None or plot == "ci_plot":
            make_ci_plot(scoring_list, tukey_results)

        return tukey_results

    def mcs_plot(pc, effect_size, means, labels=True, cmap=None, cbar_ax_bbox=None,
                ax=None, show_diff=True, cell_text_size=16, axis_text_size=12,
                show_cbar=True, reverse_cmap=False, vlim=None, **kwargs):


        for key in ['cbar', 'vmin', 'vmax', 'center']:
            if key in kwargs:
                del kwargs[key]

        if not cmap:
            cmap = "coolwarm"
        if reverse_cmap:
            cmap = cmap + "_r"

        significance = pc.copy().astype(object)
        significance[(pc < 0.001) & (pc >= 0)] = '***'
        significance[(pc < 0.01) & (pc >= 0.001)] = '**'
        significance[(pc < 0.05) & (pc >= 0.01)] = '*'
        significance[(pc >= 0.05)] = ''

        np.fill_diagonal(significance.values, '')

        # Create a DataFrame for the annotations
        if show_diff:
            annotations = effect_size.round(3).astype(str) + significance
        else:
            annotations = significance

        hax = sns.heatmap(effect_size, cmap=cmap, annot=annotations, fmt='', cbar=show_cbar, ax=ax,
                        annot_kws={"size": cell_text_size},
                        vmin=-2*vlim if vlim else None, vmax=2*vlim if vlim else None, **kwargs)

        if labels:
            label_list = list(means.index)
            x_label_list = [x + f'\n{means.loc[x].round(2)}' for x in label_list]
            y_label_list = [x + f'\n{means.loc[x].round(2)}\n' for x in label_list]
            hax.set_xticklabels(x_label_list, size=axis_text_size, ha='center', va='top', rotation=0,
                                rotation_mode='anchor')
            hax.set_yticklabels(y_label_list, size=axis_text_size, ha='center', va='center', rotation=90,
                                rotation_mode='anchor')

        hax.set_xlabel('')
        hax.set_ylabel('')

        return hax


    def make_mcs_plot_grid(df, stats, group_col, alpha=.05,
                        figsize=(20, 10), direction_dict={}, effect_dict={}, show_diff=True,
                        cell_text_size=16, axis_text_size=12, title_text_size=16, sort_axes=False):

        nrow = math.ceil(len(stats) / 3)
        fig, ax = plt.subplots(nrow, 3, figsize=figsize)

        # Set defaults
        for key in ['r2', 'rho', 'prec', 'recall', 'mae', 'mse']:
            direction_dict.setdefault(key, 'maximize' if key in ['r2', 'rho', 'prec', 'recall'] else 'minimize')

        for key in ['r2', 'rho', 'prec', 'recall']:
            effect_dict.setdefault(key, 0.1)

        direction_dict = {k.lower(): v for k, v in direction_dict.items()}
        effect_dict = {k.lower(): v for k, v in effect_dict.items()}

        for i, stat in enumerate(stats):
            stat = stat.lower()

            row = i // 3
            col = i % 3

            if stat not in direction_dict:
                raise ValueError(f"Stat '{stat}' is missing in direction_dict. Please set its value.")
            if stat not in effect_dict:
                raise ValueError(f"Stat '{stat}' is missing in effect_dict. Please set its value.")

            reverse_cmap = False
            if direction_dict[stat] == 'minimize':
                reverse_cmap = True

            _, df_means, df_means_diff, pc = StatisticalAnalysis.posthoc_tukeyhsd(df, stat, group_col, alpha,
                                                        sort_axes, direction_dict)

            hax = mcs_plot(pc, effect_size=df_means_diff, means=df_means[stat],
                        show_diff=show_diff, ax=ax[row, col], cbar=True,
                        cell_text_size=cell_text_size, axis_text_size=axis_text_size,
                        reverse_cmap=reverse_cmap, vlim=effect_dict[stat])
            hax.set_title(stat.upper(), fontsize=title_text_size)

        # If there are less plots than cells in the grid, hide the remaining cells
        if (len(stats) % 3) != 0:
            for i in range(len(stats), nrow * 3):
                row = i // 3
                col = i % 3
                ax[row, col].set_visible(False)

        plt.tight_layout()

    def ci_plot(result_tab, ax_in, name):
        """
        Create a confidence interval plot for the given result table.

        Parameters:
        result_tab (pd.DataFrame): DataFrame containing the results with columns 'meandiff', 'lower', and 'upper'.
        ax_in (matplotlib.axes.Axes): The axes on which to plot the confidence intervals.
        name (str): The title of the plot.

        Returns:
        None
        """
        result_err = np.array([result_tab['meandiff'] - result_tab['lower'],
                            result_tab['upper'] - result_tab['meandiff']])
        sns.set(rc={'figure.figsize': (6, 2)})
        sns.set_context('notebook')
        sns.set_style('whitegrid')
        ax = sns.pointplot(x=result_tab.meandiff, y=result_tab.index, marker='o', linestyle='', ax=ax_in)
        ax.errorbar(y=result_tab.index, x=result_tab['meandiff'], xerr=result_err, fmt='o', capsize=5)
        ax.axvline(0, ls="--", lw=3)
        ax.set_xlabel("Mean Difference")
        ax.set_ylabel("")
        ax.set_title(name)
        ax.set_xlim(-0.2, 0.2) 


    def make_ci_plot_grid(df_in, metric_list, group_col="method"):
        """
        Create a grid of confidence interval plots for multiple metrics using Tukey HSD test results.

        Parameters:
        df_in (pd.DataFrame): Input dataframe containing the data.
        metric_list (list of str): List of metric column names to create confidence interval plots for.
        group_col (str): The column name indicating the groups. Default is "method".

        Returns:
        None
        """
        figure, axes = plt.subplots(len(metric_list), 1, figsize=(8, 2 * len(metric_list)), sharex=False)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        for i, metric in enumerate(metric_list):
            df_tukey, _, _, _ = rm_tukey_hsd(df_in, metric, group_col=group_col)
            ci_plot(df_tukey, ax_in=axes[i], name=metric)
        figure.suptitle("Multiple Comparison of Means\nTukey HSD, FWER=0.05")
        plt.tight_layout()
