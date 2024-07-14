import os
import glob
import pandas as pd
from typing import Optional
from pandas import DataFrame


class MetaDataGenerator:
    """
    Class for generating meta data and running statistical tests.

    Attributes:
    -----------
    data_dir : str
        Directory containing data to run posthoc analysis.
    save_data : bool
        Flag to save the posthoc analysis data. Default is False.
    scoring : str
        Scoring metric used in analysis ('f1', 'average_precision', 'recall', etc.).
    posthoc_method : str
        Method for posthoc analysis ('Wilcoxon', 'Mannwhitney').
    kind_analysis : str
        Type of analysis ('Meta', 'Subgroup').
    meta_tab : pandas.DataFrame
        DataFrame to store the meta data.
    """

    def __init__(
        self,
        data_dir: str,
        save_data: bool = False,
        scoring: str = "f1",
        posthoc_method: str = "Wilcoxon",
        kind_analysis: str = "Meta",
    ) -> None:
        self.data_dir: str = data_dir
        self.save_data: bool = save_data
        self.scoring: str = scoring
        self.posthoc_method: str = posthoc_method
        self.kind_analysis: str = kind_analysis
        self.meta_tab: DataFrame = pd.DataFrame()

    def _load_data(self, filepath: str) -> DataFrame:
        data: DataFrame = pd.read_csv(filepath)
        if "Unnamed: 0" in data.columns:
            data = data.drop(["Unnamed: 0"], axis=1)
        return data

    def _process_data(self, data: DataFrame, data_name: str) -> DataFrame:
        if self.kind_analysis == "Meta":
            flat_data: DataFrame = pd.DataFrame(
                data.values.reshape(-1), columns=[data_name]
            )
        else:
            col: int = (
                data.mean().argmax()
                if self.kind_analysis == "best"
                else data.mean().argmin()
            )
            flat_data: DataFrame = pd.DataFrame(
                data.iloc[:, col].values, columns=[data_name]
            )
        return flat_data

    def fit(self) -> DataFrame:
        """
        Main method to process data and perform posthoc analysis.
        """
        for file in glob.glob(os.path.join(self.data_dir, "*.csv")):
            base_name: str = os.path.basename(file)
            data_name: str = base_name.split("_")[0]
            data: DataFrame = self._load_data(file)
            processed_data: DataFrame = self._process_data(data, data_name)
            self.meta_tab = pd.concat([self.meta_tab, processed_data], axis=1)

        return self.meta_tab
