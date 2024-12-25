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
    def extract_scoring_dfs(report_df: pd.DataFrame, scoring_list: list | str):

        if isinstance(scoring_list, str):
            scoring_list = [scoring_list]

        filtered_dfs = []

        for scoring in scoring_list:
            score_df = deepcopy(report_df[report_df.index.str.startswith(f"{scoring}_fold")])
            score_df["scoring"] = scoring
            filtered_dfs.append(score_df)

        scoring_dfs = pd.concat(filtered_dfs)

        # Melt the dataframe to long format
        df_long = scoring_dfs.reset_index().melt(
            id_vars=["scoring", "index"], var_name="method", value_name="value"
        )
        df_long.rename(columns={"index": "cv_cycle"}, inplace=True)

        return df_long

    @staticmethod
    def check_variance_homogeneity(
        report_df: pd.DataFrame,
        scoring_list: Optional[Union[list, str]] = None,
        levene_test: bool = True,
    ):
        if scoring_list is None:
            if "scoring" in report_df.columns:
                scoring_list = report_df["scoring"].unique().tolist()  # Convert to list
            else:
                raise ValueError(
                    "The 'scoring' column is not present in the DataFrame."
                )
        elif isinstance(scoring_list, str):
            scoring_list = [scoring_list]

        result = []

        for scoring in scoring_list:
            # Filter data for the current metric
            scoring_data = report_df[report_df["scoring"] == scoring]

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
        report_df: pd.DataFrame, scoring_list: Optional[Union[list, str]] = None
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

        if scoring_list is None:
            if "scoring" in report_df.columns:
                scoring_list = report_df["scoring"].unique().tolist()  # Convert to list
            else:
                raise ValueError(
                    "The 'scoring' column is not present in the DataFrame."
                )
        elif isinstance(scoring_list, str):
            scoring_list = [scoring_list]

        # Normalize values by subtracting the mean for each method
        df_norm = deepcopy(report_df)
        df_norm["value"] = df_norm.groupby(["method", "scoring"])["value"].transform(
            lambda x: x - x.mean()
        )

        sns.set_context("notebook", font_scale=1.5)
        sns.set_style("whitegrid")

        fig, axes = plt.subplots(2, len(scoring_list), figsize=(40, 10))

        for i, scoring in enumerate(scoring_list):
            ax = axes[0, i]
            sns.histplot(
                df_norm[df_norm["scoring"] == scoring]["value"], kde=True, ax=ax
            )
            ax.set_title(f"{scoring.upper()}", fontsize=16)

        for i, scoring in enumerate(scoring_list):
            ax = axes[1, i]
            scoring_data = df_norm[df_norm["scoring"] == scoring]["value"]
            stats.probplot(scoring_data, dist="norm", plot=ax)
            ax.set_title("")

        plt.tight_layout()

    @staticmethod
    def test(
        report_df: pd.DataFrame,
        scoring_list: Optional[Union[list, str]] = None,
        select_test: str = "AnovaRM",
    ):

        if scoring_list is None:
            if "scoring" in report_df.columns:
                scoring_list = report_df["scoring"].unique().tolist()  # Convert to list
            else:
                raise ValueError(
                    "The 'scoring' column is not present in the DataFrame."
                )
        elif isinstance(scoring_list, str):
            scoring_list = [scoring_list]

        if select_test not in ["AnovaRM", "friedman"]:
            raise ValueError(
                f"Unsupported test: {select_test}. Please choose 'AnovaRM' for a parametric test or 'friedman' for a non-parametric test."
            )

        sns.set_context("notebook")
        sns.set_theme(rc={"figure.figsize": (4, 3)}, font_scale=1.5)
        sns.set_style("whitegrid")
        
        num_rows = round(len(scoring_list)/2)
        figure, axes = plt.subplots(
            num_rows, 2, sharex=False, sharey=False, figsize=(3*len(report_df["method"].unique()), 7*num_rows)
        )
        axes = axes.flatten() #Turn 2D array to 1D array

        for i, scoring in enumerate(scoring_list):
            if select_test == "AnovaRM":
                scoring_data = report_df[report_df["scoring"] == scoring]
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
                    subject="cv_cycle"
                )["p-unc"].values[0]

            ax = sns.boxplot(
                y="value",
                x="method",
                hue="method",
                ax=axes[i],
                data=scoring_data,
                palette="plasma",
                legend=False,
                width=.5
            )
            ax.set_title(f"p={p_value:.1e}")
            ax.set_xlabel("")
            ax.set_ylabel(scoring.upper())
            
            
            #Wrap labels
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
                
        plt.tight_layout()

    @staticmethod
    def posthoc_conover_friedman(
        report_df: pd.DataFrame,
        scoring_list: Optional[Union[list, str]] = None,
        plot: Optional[Union[list, str]] = None,
    ):
    
        if scoring_list is None:
            if "scoring" in report_df.columns:
                scoring_list = report_df["scoring"].unique().tolist()
            else:
                raise ValueError(
                    "The 'scoring' column is not present in the DataFrame."
                )
        elif isinstance(scoring_list, str):
            scoring_list = [scoring_list]
        
        # Precompute posthoc Conover-Friedman results for each metric
        pc_results = {}
        for scoring in scoring_list:
            scoring_df_filtered = report_df[report_df["scoring"] == scoring]
            scoring_df_wide = scoring_df_filtered.pivot(index="cv_cycle", columns="method", values="value")
            pc_results[scoring] = sp.posthoc_conover_friedman(scoring_df_wide, p_adjust="holm")
        
        def make_sign_plots(scoring_list, pc_results):
            heatmap_args = {
                "linewidths": 0.25,
                "linecolor": "0.5",
                "clip_on": True,
                "square": True,
            }
            sns.set_theme(rc={"figure.figsize": (4, 3)}, font_scale=1.5)
            
            num_rows = round(len(scoring_list)/2)
            figure, axes = plt.subplots(
                num_rows, 2, sharex=False, sharey=False, figsize=(20, 8*num_rows)
            )
        
            axes = axes.flatten() #Turn 2D array to 1D array
                   
            for i, scoring in enumerate(scoring_list):
                pc = pc_results[scoring]
                sub_ax, sub_c = sp.sign_plot(
                    pc, **heatmap_args, ax=axes[i], xticklabels=True
                )
                sub_ax.set_title(scoring.upper())
                sub_ax.tick_params(axis="both", labelsize=12)
            plt.tight_layout()

        def make_critical_difference_diagrams(scoring_list, pc_results):
            figure, axes = plt.subplots(
                len(scoring_list), 1, sharex=True, sharey=False, figsize=(20, 4*len(scoring_list))
            )
            for i, scoring in enumerate(scoring_list):
                avg_rank = (
                    report_df[report_df["scoring"] == scoring].groupby("cv_cycle")["value"]
                    .rank(pct=True)
                    .groupby(report_df.method)
                    .mean()
                )
                pc = pc_results[scoring]
                sp.critical_difference_diagram(avg_rank, pc, ax=axes[i])
                axes[i].set_title(scoring.upper())
                axes[i].tick_params(axis="y", labelsize=8)
                
            plt.tight_layout()

        if plot is None or plot == "sign_plot":
            make_sign_plots(scoring_list, pc_results)

        if plot is None or plot == "critical_difference_diagrams":
            make_critical_difference_diagrams(scoring_list, pc_results)
        
        return pc_results


########## refine code ###########
    @staticmethod
    def posthoc_tukeyhsd(report_df: pd.DataFrame, scoring_list: Optional[Union[list, str]] = None, plot: Optional[Union[list, str]] = None, alpha: float = 0.05, direction_dict: Optional[dict] = None):
        """
        Perform Tukey HSD test and generate corresponding plots.

        Parameters:
        report_df (pd.DataFrame): Dataframe containing the report data.
        scoring_list (Optional[Union[list, str]]): List or string of metrics to be analyzed. If None, all metrics are analyzed.
        plot (Optional[Union[list, str]]): Type of plot to generate. Options are 'mcs_plot', 'ci_plot'. If None, both plots are generated.
        alpha (float): Significance level for the test. Default is 0.05.
        direction_dict (Optional[dict]): Dictionary indicating whether to minimize or maximize each metric.
        """
        if scoring_list is None:
            if "scoring" in report_df.columns:
                scoring_list = report_df["scoring"].unique().tolist()
            else:
                raise ValueError("The 'scoring' column is not present in the DataFrame.")
        elif isinstance(scoring_list, str):
            scoring_list = [scoring_list]

        tukey_results = {}

        for scoring in scoring_list:
            if direction_dict and scoring in direction_dict:
                sort_order = direction_dict[scoring]
                df_means = report_df.groupby("method").mean(numeric_only=True).sort_values(scoring, ascending=(sort_order == 'minimize'))
            else:
                df_means = report_df.groupby("method").mean(numeric_only=True)

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                aov = pg.rm_anova(dv='value', within='method', subject='cv_cycle', data=report_df[report_df['scoring'] == scoring], detailed=True)

            mse = aov.loc[1, 'MS']
            df_resid = aov.loc[1, 'DF']

            methods = df_means.index
            n_groups = len(methods)
            n_per_group = report_df["method"].value_counts().mean()

            tukey_se = np.sqrt(2 * mse / n_per_group)
            q = qsturng(1 - alpha, n_groups, df_resid)

            num_comparisons = len(methods) * (len(methods) - 1) // 2
            result_tab = pd.DataFrame(index=range(num_comparisons),
                                    columns=["group1", "group2", "meandiff", "lower", "upper", "p-adj"])

            df_means_diff = pd.DataFrame(index=methods, columns=methods, data=0.0)
            pc = pd.DataFrame(index=methods, columns=methods, data=1.0)

            row_idx = 0
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods):
                    if i < j:
                        group1 = report_df[report_df['method'] == method1][scoring]
                        group2 = report_df[report_df['method'] == method2][scoring]
                        mean_diff = group1.mean() - group2.mean()
                        studentized_range = np.abs(mean_diff) / tukey_se
                        adjusted_p = qsturng(studentized_range * np.sqrt(2), n_groups, df_resid)
                        lower = mean_diff - (q / np.sqrt(2) * tukey_se)
                        upper = mean_diff + (q / np.sqrt(2) * tukey_se)
                        result_tab.loc[row_idx] = [method1, method2, mean_diff, lower, upper, adjusted_p]
                        pc.loc[method1, method2] = adjusted_p
                        pc.loc[method2, method1] = adjusted_p
                        df_means_diff.loc[method1, method2] = mean_diff
                        df_means_diff.loc[method2, method1] = -mean_diff
                        row_idx += 1

            tukey_results[scoring] = {
                "result_tab": result_tab,
                "df_means": df_means,
                "df_means_diff": df_means_diff,
                "pc": pc
            }

        def make_mcs_plot(scoring_list, tukey_results):
            fig, axes = plt.subplots(1, len(scoring_list), figsize=(20, 10))
            for i, scoring in enumerate(scoring_list):
                result = tukey_results[scoring]
                pc = result["pc"]
                effect_size = result["df_means_diff"]
                means = result["df_means"][scoring]
                sns.heatmap(effect_size, annot=True, fmt='.2f', ax=axes[i], cmap="coolwarm")
                axes[i].set_title(scoring.upper())
            plt.tight_layout()

        def make_ci_plot(scoring_list, tukey_results):
            fig, axes = plt.subplots(len(scoring_list), 1, figsize=(8, 2 * len(scoring_list)))
            for i, scoring in enumerate(scoring_list):
                result = tukey_results[scoring]
                result_tab = result["result_tab"]
                result_err = np.array([result_tab['meandiff'] - result_tab['lower'],
                                    result_tab['upper'] - result_tab['meandiff']])
                sns.pointplot(x=result_tab.meandiff, y=result_tab.index, ax=axes[i])
                axes[i].errorbar(y=result_tab.index, x=result_tab['meandiff'], xerr=result_err, fmt='o', capsize=5)
                axes[i].axvline(0, ls="--", lw=3)
                axes[i].set_title(scoring.upper())
            plt.tight_layout()

        if plot is None or plot == 'mcs_plot':
            make_mcs_plot(scoring_list, tukey_results)

        if plot is None or plot == 'ci_plot':
            make_ci_plot(scoring_list, tukey_results)

        return tukey_results