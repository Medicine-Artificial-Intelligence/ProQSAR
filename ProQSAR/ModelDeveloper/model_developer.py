import os
import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from typing import Optional

from ProQSAR.ModelDeveloper.model_developer_utils import (
    _get_task_type,
    _get_method_map,
    _get_cv_strategy,
)
from ProQSAR.ModelDeveloper.model_validation import iv_report


class ModelDeveloper:
    """
    Class to handle model development for machine learning tasks.

    Attributes:
    -----------
    activity_col : str
        Name of the activity column in the dataset.
    id_col : str
        Name of the ID column in the dataset.
    method : str, optional
        The method to use for model development, 'best' or a specific model (default: 'best').
    add_method : Optional[dict], optional
        Additional methods to be added (default: None).
    scoring_target : Optional[str], optional
        Scoring metric to target for comparison (default: None).
    n_splits : int, optional
        Number of cross-validation splits (default: 10).
    n_repeats : int, optional
        Number of cross-validation repeats (default: 3).
    save_model : bool, optional
        Whether to save the model (default: True).
    save_pred_result : bool, optional
        Whether to save prediction results (default: True).
    pred_result_name : str, optional
        Name of the prediction result file (default: 'pred_result').
    save_dir : str, optional
        Directory to save model and results (default: 'Project/Model_Development').
    save_iv_report : bool, optional
        Whether to save the IV report (default: False).
    iv_report_name : str, optional
        Name of the IV report file (default: 'comparison_iv_report').
    visualize : Optional[str], optional
        Visualization option for report (default: None).
    save_fig : bool, optional
        Whether to save figures (default: False).
    fig_name : str, optional
        Name of the figure file (default: 'comparison_iv_graph').
    n_jobs : int, optional
        Number of jobs to run in parallel (default: -1).

    Methods:
    --------
    fit(data: pd.DataFrame) -> BaseEstimator
        Fits the model to the provided data.

    predict(data: pd.DataFrame) -> pd.DataFrame
        Predicts outcomes using the fitted model.

    static_predict(data: pd.DataFrame, save_dir: str,
        save_pred_result: bool = True, pred_result_name: str = "pred_result") -> pd.DataFrame
        Loads a pre-trained model from disk and predicts outcomes.
    """

    def __init__(
        self,
        activity_col: str,
        id_col: str,
        method: str = "best",
        add_method: Optional[dict] = None,
        scoring_target: Optional[str] = None,
        n_splits: int = 10,
        n_repeats: int = 3,
        save_model: bool = True,
        save_pred_result: bool = True,
        pred_result_name: str = "pred_result",
        save_dir: str = "Project/Model_Development",
        save_iv_report: bool = False,
        iv_report_name: str = "comparison_iv_report",
        visualize: Optional[str] = None,
        save_fig: bool = False,
        fig_name: str = "comparison_iv_graph",
        n_jobs: int = -1,
    ):
        """Initializes the ModelDeveloper with necessary attributes."""
        self.activity_col = activity_col
        self.id_col = id_col
        self.method = method
        self.add_method = add_method
        self.scoring_target = scoring_target
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.save_model = save_model
        self.save_pred_result = save_pred_result
        self.pred_result_name = pred_result_name
        self.save_dir = save_dir
        self.save_iv_report = save_iv_report
        self.iv_report_name = iv_report_name
        self.visualize = visualize
        self.save_fig = save_fig
        self.fig_name = fig_name
        self.n_jobs = n_jobs
        self.model = None
        self.task_type = None
        self.cv = None
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    def fit(self, data: pd.DataFrame) -> BaseEstimator:
        """
        Fits a machine learning model based on the specified method.

        Parameters:
        -----------
        data : pd.DataFrame
            The training dataset including features, activity, and ID columns.

        Returns:
        --------
        BaseEstimator
            The fitted machine learning model.
        """
        X_data = data.drop([self.activity_col, self.id_col], axis=1)
        y_data = data[self.activity_col]

        self.task_type = _get_task_type(data, self.activity_col)
        self.method_map = _get_method_map(self.task_type, self.add_method, self.n_jobs)
        self.cv = _get_cv_strategy(
            self.task_type, n_splits=self.n_splits, n_repeats=self.n_repeats
        )

        if self.method == "best":
            if self.scoring_target is None:
                self.scoring_target = "f1" if self.task_type == "C" else "r2"
            comparison_df = iv_report(
                data=data,
                activity_col=self.activity_col,
                id_col=self.id_col,
                add_method=self.add_method,
                scoring_list=[self.scoring_target],
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
                visualize=self.visualize,
                save_fig=self.save_fig,
                fig_name=self.fig_name,
                save_csv=self.iv_report_name,
                csv_name=self.iv_report_name,
                save_dir=self.save_dir,
                n_jobs=self.n_jobs,
            )

            self.method = comparison_df.loc[
                comparison_df[f"{self.scoring_target}_mean"].idxmax(), "Method"
            ]
            self.model = self.method_map[self.method].fit(X=X_data, y=y_data)
        elif self.method in self.method_map:
            self.model = self.method_map[self.method].fit(X=X_data, y=y_data)
        else:
            raise ValueError(f"Method '{self.method}' is not recognized.")

        if self.save_model:
            with open(f"{self.save_dir}/activity_col.pkl", "wb") as file:
                pickle.dump(self.activity_col, file)
            with open(f"{self.save_dir}/id_col.pkl", "wb") as file:
                pickle.dump(self.id_col, file)
            with open(f"{self.save_dir}/model.pkl", "wb") as file:
                pickle.dump(self.model, file)
            with open(f"{self.save_dir}/task_type.pkl", "wb") as file:
                pickle.dump(self.task_type, file)

        return self.model

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts outcomes using the fitted model.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset to predict on, including features, activity, and ID columns.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the predicted outcomes and optionally the probabilities.
        """
        if self.model is None:
            raise NotFittedError(
                "ModelDeveloper is not fitted yet. Call 'fit' before using this method."
            )

        X_data = data.drop([self.activity_col, self.id_col], axis=1)
        y_pred = self.model.predict(X_data)
        result = {
            "ID": data[self.id_col].values,
            "Predicted values": y_pred,
        }

        if self.task_type == "C":
            y_proba = self.model.predict_proba(X_data)[:, 1] * 100
            result["Probability"] = np.round(y_proba, 2)

        self.pred_result = pd.DataFrame(result)

        if self.save_pred_result:
            self.pred_result.to_csv(f"{self.save_dir}/{self.pred_result_name}.csv")

        return self.pred_result

    @staticmethod
    def static_predict(
        data: pd.DataFrame,
        save_dir: str,
        save_pred_result: bool = True,
        pred_result_name: str = "pred_result",
    ) -> pd.DataFrame:
        """
        Loads a pre-trained model from disk and predicts outcomes.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset to predict on, including features, activity, and ID columns.
        save_dir : str
            The directory where the model is saved.
        save_pred_result : bool, optional
            Whether to save the prediction result (default: True).
        pred_result_name : str, optional
            Name of the prediction result file (default: 'pred_result').

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the predicted outcomes and optionally the probabilities.
        """

        if not os.path.exists(f"{save_dir}/model.pkl"):
            raise NotFittedError(
                "ModelDeveloper is not fitted yet. Call 'fit' before using this method."
            )

        with open(f"{save_dir}/activity_col.pkl", "rb") as file:
            activity_col = pickle.load(file)
        with open(f"{save_dir}/id_col.pkl", "rb") as file:
            id_col = pickle.load(file)
        with open(f"{save_dir}/model.pkl", "rb") as file:
            model = pickle.load(file)
        with open(f"{save_dir}/task_type.pkl", "rb") as file:
            task_type = pickle.load(file)

        X_data = data.drop(
            [activity_col, id_col],
            axis=1,
            errors="ignore",
        )
        y_pred = model.predict(X_data)
        result = {
            "ID": data[id_col].values,
            "Predicted values": y_pred,
        }
        if task_type == "C":
            y_proba = model.predict_proba(X_data)[:, 1] * 100
            result["Probability"] = np.round(y_proba, 2)

        pred_result = pd.DataFrame(result)

        if save_pred_result:
            pred_result.to_csv(f"{save_dir}/{pred_result_name}.csv")

        return pred_result
