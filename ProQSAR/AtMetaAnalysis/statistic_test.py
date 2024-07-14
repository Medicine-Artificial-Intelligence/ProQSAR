import os
import numpy as np
import pandas as pd
import seaborn as sns
import scikit_posthocs as sp
from scipy.stats import t
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, NoNorm
from typing import Union
from matplotlib.pyplot import Axes

sns.set("notebook")


class StatisticTest:
    """
    Class for statistical comparison and posthoc analysis.

    Attributes:
    ----------
    meta_data : DataFrame
        The DataFrame containing the data on which statistical tests are to be performed.
    save_data : bool
        If True, the resulting p-values of the posthoc tests are saved as a CSV file.
    save_fig : bool
        If True, the generated heatmap is saved as an image file.
    scoring : str
        The metric used for scoring the statistical test results.
    posthoc_method : str
        The method of posthoc analysis to be used. Options are 'Wilcoxon' or 'Mannwhitney'.
    kind_analysis : str
        The kind of analysis to be performed, typically 'Meta'.
    save_dir : str
        The directory where the generated files will be saved.

    Methods:
    -------
    posthoc()
        Performs the specified posthoc test on the data and generates a heatmap of the results.
    _plot_heatmap()
        Generates a heatmap from the posthoc test results.
    """

    def __init__(
        self,
        meta_data: pd.DataFrame,
        save_data: bool = False,
        save_fig: bool = False,
        scoring: str = "f1",
        posthoc_method: str = "Wilcoxon",
        kind_analysis: str = "Meta",
        save_dir: str = None,
    ) -> None:
        self.meta_data = meta_data
        self.results = meta_data.values.T
        self.names = meta_data.columns
        self.save_data = save_data
        self.save_fig = save_fig
        self.scoring = scoring
        self.posthoc_method = posthoc_method
        self.kind_analysis = kind_analysis
        self.save_dir = save_dir or "Meta_folder"
        os.makedirs(self.save_dir, exist_ok=True)

    def posthoc(self) -> None:
        """Perform posthoc statistical analysis and generate a heatmap for visualization."""
        df_metrics: pd.DataFrame = pd.DataFrame(self.results.T, columns=self.names)
        df_melt: pd.DataFrame = pd.melt(
            df_metrics.reset_index(), id_vars=["index"], value_vars=df_metrics.columns
        )
        df_melt.columns = ["index", "Method", "Scores"]

        if self.posthoc_method == "Wilcoxon":
            self.pc: pd.DataFrame = sp.posthoc_wilcoxon(
                df_melt, val_col="Scores", group_col="Method", p_adjust="holm"
            )
        elif self.posthoc_method == "Mannwhitney":
            self.pc: pd.DataFrame = sp.posthoc_mannwhitney(
                df_melt, val_col="Scores", group_col="Method", p_adjust="holm"
            )

        self._plot_heatmap()

    def _plot_heatmap(self) -> None:
        """Internal method to plot a heatmap based on the posthoc test results."""
        fig, ax = plt.subplots(figsize=(15, 10))  # Use subplots for better control
        heatmap_ax: Axes = sns.heatmap(
            self.pc, annot=True, fmt=".2f", linewidths=0.5, cmap="coolwarm", ax=ax
        )

        # Improve the colorbar
        cbar: Colorbar = heatmap_ax.collections[0].colorbar
        cbar.set_ticks([0.001, 0.01, 0.05, 1])
        cbar.set_ticklabels(["***", "**", "*", "ns"])

        # Aesthetics
        plt.title(
            f"Posthoc {self.posthoc_method} Analysis", fontsize=24, weight="semibold"
        )
        plt.xticks(
            rotation=45, ha="right", fontsize=10
        )  # Rotate x labels for better fit
        plt.yticks(fontsize=10)
        sns.despine()  # Remove the top and right spines

        # Save figure if needed
        if self.save_fig:
            plt.savefig(
                os.path.join(
                    self.save_dir,
                    f"{self.posthoc_method}_{self.kind_analysis}_heatmap.png",
                ),
                dpi=300,
                bbox_inches="tight",
            )
        if self.save_data:
            self.pc.to_csv(
                os.path.join(
                    self.save_dir, f"{self.posthoc_method}_{self.kind_analysis}.csv"
                )
            )
        plt.show()
