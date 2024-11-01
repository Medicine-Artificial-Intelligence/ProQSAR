import os
import pickle
import pandas as pd
from sklearn.exceptions import NotFittedError
from typing import Optional

from ProQSAR.FeatureSelector.feature_selector_utils import (_get_method_map, evaluate_feature_selectors)
from ProQSAR.ModelDeveloper.model_developer_utils import (
    _get_task_type,
    _get_cv_strategy,
)


class FeatureSelector:
    def __init__(
        self,
        activity_col: str,
        id_col: str,
        select_method: str = "best",
        add_method: Optional[dict] = None,
        scoring: Optional[str] = None,
        n_splits: int = 10,
        n_repeats: int = 3,
        save_method: bool = True,
        save_trans_data: bool = True,
        trans_data_name: str = "fs_trans_data",
        save_dir: Optional[str] = "Project/FeatureSelector",
        save_cv_report: bool = False,
        cv_report_name: str = "fs_cv_report",
        visualize: Optional[str] = "box",
        save_fig: bool = False,
        fig_prefix: str = "fs_cv_graph",
        n_jobs: int = -1,
    ):

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

        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    def fit(self, data: pd.DataFrame):

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
            self.feature_selector = self.method_map[self.select_method].fit(X=X_data, y=y_data)
            
        elif self.select_method in self.method_map:
            self.feature_selector = self.method_map[self.select_method].fit(X=X_data, y=y_data)
        else:
            raise ValueError(f"Method '{self.select_method}' not recognized.")

        if self.save_method:
            with open(f"{self.save_dir}/activity_col.pkl", "wb") as file:
                pickle.dump(self.activity_col, file)
            with open(f"{self.save_dir}/id_col.pkl", "wb") as file:
                pickle.dump(self.id_col, file)
            with open(f"{self.save_dir}/feature_selector.pkl", "wb") as file:
                pickle.dump(self.feature_selector, file)
        
        return self.feature_selector

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        if self.feature_selector is None:
            raise NotFittedError("FeatureSelector is not fitted yet. Call 'fit' before using this method.")

        X_data = data.drop([self.activity_col, self.id_col], axis=1)
        data_selected = pd.DataFrame(self.feature_selector.transform(X_data))
        transformed_data = pd.concat(
            [data_selected, data[[self.id_col, self.activity_col]]], axis=1
        )
        if self.save_trans_data:
            transformed_data.to_csv(f"{self.save_dir}/{self.trans_data_name}.csv")
            
        return transformed_data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:

        self.fit(data)
        return self.transform(data)

    @staticmethod
    def static_transform(
        data: pd.DataFrame, 
        save_dir: str,
        save_trans_data,
        trans_data_name) -> pd.DataFrame:

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
