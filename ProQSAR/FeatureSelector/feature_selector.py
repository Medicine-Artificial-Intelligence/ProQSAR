import os
import pickle
import pandas as pd
from sklearn.exceptions import NotFittedError
from typing import Optional

from ProQSAR.FeatureSelector.feature_selector_utils import (
    _get_method_map,
    evaluate_feature_selectors,
)
from ProQSAR.ModelDeveloper.model_developer_utils import (
    _get_task_type,
    _get_cv_strategy,
)


class FeatureSelector:
    """
    A class for selecting features from a dataset based on specified criteria.

    Attributes:
        activity_col (str): Column name for the target variable.
        id_col (str): Column name for the unique identifier.
        select_method (str): Feature selection method. Defaults to "best".
        add_method (Optional[dict]): Additional feature selection methods.
        scoring (Optional[str]): Scoring metric for model evaluation.
        n_splits (int): Number of splits for cross-validation.
        n_repeats (int): Number of repeats for cross-validation.
        save_method (bool): Whether to save the fitted feature selector.
        save_trans_data (bool): Whether to save the transformed data.
        trans_data_name (str): File name for saved transformed data.
        save_dir (Optional[str]): Directory to save outputs.
        save_cv_report (bool): Whether to save a CV report.
        cv_report_name (str): Name for the CV report file.
        visualize (Optional[str]): Visualization options.
        save_fig (bool): Whether to save figures.
        fig_prefix (str): Prefix for saved figure files.
        n_jobs (int): Number of jobs to run in parallel.

    Methods:
        fit(data: pd.DataFrame) -> object:
            Fits the feature selector to the provided data.
        transform(data: pd.DataFrame) -> pd.DataFrame:
            Transforms the data based on the fitted selector.
        fit_transform(data: pd.DataFrame) -> pd.DataFrame:
            Fits and transforms the data in one step.
        static_transform(data: pd.DataFrame, save_dir: str, save_trans_data: bool = False,
            trans_data_name: str = "fs_trans_data") -> pd.DataFrame:
            Transforms data using a previously fitted selector loaded from disk.
    """

    def __init__(
        self,
        activity_col: str,
        id_col: str,
        select_method: str = "best",
        add_method: Optional[dict] = None,
        scoring: Optional[str] = None,
        n_splits: int = 10,
        n_repeats: int = 3,
        save_method: bool = False,
        save_trans_data: bool = False,
        trans_data_name: str = "fs_trans_data",
        save_dir: Optional[str] = "Project/FeatureSelector",
        save_cv_report: bool = False,
        cv_report_name: str = "fs_cv_report",
        visualize: Optional[str] = None,
        save_fig: bool = False,
        fig_prefix: str = "fs_cv_graph",
        n_jobs: int = -1,
    ):
        """
        Initializes the FeatureSelector with the specified parameters.

        Parameters are set based on selection and cross-validation criteria.
        """

        self.activity_col = activity_col
        self.id_col = id_col
        self.select_method = select_method
        self.add_method = add_method
        self.scoring = scoring
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.save_method = save_method
        self.save_trans_data = save_trans_data
        self.trans_data_name = trans_data_name
        self.save_dir = save_dir
        self.save_cv_report = save_cv_report
        self.cv_report_name = cv_report_name
        self.visualize = visualize
        self.save_fig = save_fig
        self.fig_prefix = fig_prefix
        self.n_jobs = n_jobs
        self.feature_selector = None
        self.task_type = None
        self.cv = None

    def fit(self, data: pd.DataFrame) -> object:
        """
        Fits the feature selector to the data.

        Determines the best feature selection method based on cross-validation or
        uses a specified method, then fits the data.

        Parameters:
            data (pd.DataFrame): Input data containing features and target column.

        Returns:
            object: Fitted feature selector object.

        Raises:
            ValueError: If an unrecognized selection method is specified.
        """

        X_data = data.drop([self.activity_col, self.id_col], axis=1)
        y_data = data[self.activity_col]

        self.task_type = _get_task_type(data, self.activity_col)
        self.method_map = _get_method_map(self.task_type, self.add_method, self.n_jobs)
        self.cv = _get_cv_strategy(
            self.task_type, n_splits=self.n_splits, n_repeats=self.n_repeats
        )

        if self.select_method == "best":
            self.scoring = self.scoring or "f1" if self.task_type == "C" else "r2"
            comparison_df = evaluate_feature_selectors(
                data=data,
                activity_col=self.activity_col,
                id_col=self.id_col,
                add_method=self.add_method,
                scoring_list=[self.scoring],
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
                visualize=self.visualize,
                save_fig=self.save_fig,
                fig_prefix=self.fig_prefix,
                save_csv=self.save_cv_report,
                csv_name=self.cv_report_name,
                save_dir=self.save_dir,
                n_jobs=self.n_jobs,
            )

            self.select_method = comparison_df.loc[
                comparison_df[f"{self.scoring}_mean"].idxmax(), "FeatureSelector"
            ]
            self.feature_selector = self.method_map[self.select_method].fit(
                X=X_data, y=y_data
            )

        elif self.select_method in self.method_map:
            self.feature_selector = self.method_map[self.select_method].fit(
                X=X_data, y=y_data
            )
        else:
            raise ValueError(f"Method '{self.select_method}' not recognized.")

        if self.save_method:
            if self.save_dir and not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir, exist_ok=True)
            with open(f"{self.save_dir}/activity_col.pkl", "wb") as file:
                pickle.dump(self.activity_col, file)
            with open(f"{self.save_dir}/id_col.pkl", "wb") as file:
                pickle.dump(self.id_col, file)
            with open(f"{self.save_dir}/feature_selector.pkl", "wb") as file:
                pickle.dump(self.feature_selector, file)

        return self.feature_selector

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms data using the fitted feature selector.

        Parameters:
            data (pd.DataFrame): Input data containing features and target column.

        Returns:
            pd.DataFrame: Transformed data with selected features.

        Raises:
            NotFittedError: If the feature selector is not fitted yet.
        """
        if self.feature_selector is None:
            raise NotFittedError(
                "FeatureSelector is not fitted yet. Call 'fit' before using this method."
            )

        X_data = data.drop([self.activity_col, self.id_col], axis=1)
        data_selected = pd.DataFrame(self.feature_selector.transform(X_data))
        transformed_data = pd.concat(
            [data_selected, data[[self.id_col, self.activity_col]]], axis=1
        )
        if self.save_trans_data:
            if self.save_dir and not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir, exist_ok=True)
            transformed_data.to_csv(f"{self.save_dir}/{self.trans_data_name}.csv")

        return transformed_data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fits and transforms the data in one step.

        Parameters:
            data (pd.DataFrame): Input data containing features and target column.

        Returns:
            pd.DataFrame: Transformed data with selected features.
        """
        self.fit(data)
        return self.transform(data)

    @staticmethod
    def static_transform(
        data: pd.DataFrame,
        save_dir: str,
        save_trans_data: bool = False,
        trans_data_name: str = "fs_trans_data",
    ) -> pd.DataFrame:
        """
        Transforms data using a previously fitted selector loaded from disk.

        Parameters:
            data (pd.DataFrame): Input data containing features and target column.
            save_dir (str): Directory to load the fitted selector and related data.
            save_trans_data (bool): Whether to save the transformed data to a file.
            trans_data_name (str): File name for saved transformed data.

        Returns:
            pd.DataFrame: Transformed data with selected features.

        Raises:
            NotFittedError: If the feature selector has not been previously fitted and saved.
        """
        if not os.path.exists(f"{save_dir}/feature_selector.pkl"):
            raise NotFittedError(
                "FeatureSelector is not fitted yet. Call 'fit' before using this method."
            )

        with open(f"{save_dir}/activity_col.pkl", "rb") as file:
            activity_col = pickle.load(file)
        with open(f"{save_dir}/id_col.pkl", "rb") as file:
            id_col = pickle.load(file)
        with open(f"{save_dir}/feature_selector.pkl", "rb") as file:
            feature_selector = pickle.load(file)

        X_data = data.drop(
            [activity_col, id_col],
            axis=1,
            errors="ignore",
        )
        data_selected = pd.DataFrame(feature_selector.transform(X_data))
        transformed_data = pd.concat(
            [data_selected, data[[id_col, activity_col]]], axis=1
        )
        if save_trans_data:
            transformed_data.to_csv(f"{save_dir}/{trans_data_name}.csv")
        return transformed_data
