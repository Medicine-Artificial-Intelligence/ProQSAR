import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from copy import deepcopy
from scipy import stats
from scipy.stats import spearmanr, levene
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.libqsturng import psturng, qsturng
from matplotlib import cm
from sklearn.preprocessing import LabelEncoder
import scikit_posthocs as sp


class StatisticalAnalysis:
    @staticmethod
    def extract_scoring_dfs_2(report_df: pd.DataFrame, scoring: list | str):
        """
        Extract separate DataFrames for each scoring metric.

        Parameters:
            report_df (pd.DataFrame): Input report DataFrame.
            scoring (list | str): Scoring metric(s) to extract (e.g., 'precision', 'f1', 'roc_auc').

        Returns:
            dict: A dictionary containing DataFrames keyed by scoring names.
        """
        if isinstance(scoring, str):
            scoring = [scoring]
    
        scoring_dfs = {}

        for score in scoring:
            score_df = report_df[report_df.index.str.startswith(f"{score}_fold")]
            scoring_dfs[f"{score}"] = score_df
    
        return scoring_dfs
    
    @staticmethod
    def extract_scoring_dfs(report_df: pd.DataFrame, scoring: list | str):

        if isinstance(scoring, str):
            scoring = [scoring]
    
        filtered_dfs = []

        for score in scoring:
            score_df = report_df[report_df.index.str.startswith(f"{score}_fold")]
            score_df["Scoring"] = score
            filtered_dfs.append(score_df)

        scoring_dfs = pd.concat(filtered_dfs)

        # Melt the dataframe to long format
        df_long = scoring_dfs.melt(
            id_vars=["Scoring", "index"],
            var_name="Method",
            value_name="Value"
        )

        return df_long
    
    @staticmethod
    def check_variance_homogeneity(report_df: pd.DataFrame, scoring: list | str, levene_test: bool = True):

        if isinstance(scoring, str):
            scoring = [scoring]

        result = []

        for metric in scoring:
            # Filter data for the current metric
            metric_data = report_df[report_df["Scoring"] == metric]

            # Calculate variance fold difference
            variances_by_method = metric_data.groupby("Method")["Value"].var()
            max_fold_diff = variances_by_method.max() / variances_by_method.min()

            result.append({
                'metric': metric,
                'variance_fold_difference': max_fold_diff,
            })
            # Perform Levene's test if specified
            p_value = None
            if levene_test:
                groups = [group["Value"].values for _, group in metric_data.groupby('method')]
                stat, p_value = levene(*groups)
                result.append({'p_value': p_value})
                
        result_df = pd.DataFrame(result).set_index('metric')

        return result_df

    @staticmethod
    def check_normality(report_df: pd.DataFrame, scoring: list | str):
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

        if isinstance(scoring, str):
            scoring = [scoring]

        # Normalize values by subtracting the mean for each method
        df_norm = deepcopy(report_df)
        df_norm["Value"] = df_norm.groupby(["Method", "Scoring"])["Value"].transform(lambda x: x - x.mean())

        sns.set_context('notebook', font_scale=1.5)
        sns.set_style('whitegrid')
    
        scorings = df_norm['Scoring'].unique()
        n_scorings = len(scorings)
    
        fig, axes = plt.subplots(2, n_scorings, figsize=(20, 10))
    
        for i, metric in enumerate(scorings):
            ax = axes[0, i]
            sns.histplot(df_norm[df_norm["Scoring"] == metric]['Value'], kde=True, ax=ax)
            ax.set_title(f'{metric}', fontsize=16)
    
        for i, metric in enumerate(scorings):
            ax = axes[1, i]
            metric_data = df_norm[df_norm["Scoring"] == metric]['Value']
            stats.probplot(metric_data, dist="norm", plot=ax)
            ax.set_title("")
    
        plt.tight_layout()

