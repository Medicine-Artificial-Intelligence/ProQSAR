import os
import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from typing import Optional

from ProQSAR.ModelDeveloper.model_developer_utils import (
    _get_task_type,
    _get_model_map,
    _get_cv_strategy,
)
from ProQSAR.ModelDeveloper.model_validation import cross_validation_report


class ModelDeveloper:
    """
    Class to handle model development for machine learning tasks.

    Attributes:
    -----------
    activity_col : str
        Name of the activity column in the dataset.
    id_col : str
        Name of the ID column in the dataset.
    select_model : str, optional
        The model to use for model development, 'best' or a specific model (default: 'best').
    add_model : Optional[dict], optional
        Additional models to be added (default: None).
    scoring : Optional[str], optional
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
    save_cv_report : bool, optional
        Whether to save the IV report (default: False).
    cv_report_name : str, optional
        Name of the IV report file (default: 'comparison_cv_report').
    visualize : Optional[str], optional
        Visualization option for report (default: None).
    save_fig : bool, optional
        Whether to save figures (default: False).
    fig_prefix : str, optional
        Name of the figure file (default: 'comparison_cv_graph').
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
        select_model: str = "best",
        add_model: Optional[dict] = None,
        scoring: Optional[str] = None,
        n_splits: int = 10,
        n_repeats: int = 3,
        save_model: bool = True,
        save_pred_result: bool = True,
        pred_result_name: str = "pred_result",
        save_dir: Optional[str] = "Project/ModelDeveloper",
        save_cv_report: bool = False,
        cv_report_name: str = "md_cv_report",
        visualize: Optional[str] = None,
        save_fig: bool = False,
        fig_prefix: str = "md_cv_graph",
        n_jobs: int = -1,
    ):
        """Initializes the ModelDeveloper with necessary attributes."""
        self.activity_col = activity_col
        self.id_col = id_col
        self.select_model = select_model
        self.add_model = add_model
        self.scoring = scoring
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.save_model = save_model
        self.save_pred_result = save_pred_result
        self.pred_result_name = pred_result_name
        self.save_dir = save_dir
        self.save_cv_report = save_cv_report
        self.cv_report_name = cv_report_name
        self.visualize = visualize
        self.save_fig = save_fig
        self.fig_prefix = fig_prefix
        self.n_jobs = n_jobs
        self.model = None
        self.task_type = None
        self.cv = None

    def fit(self, data: pd.DataFrame) -> BaseEstimator:
        """
        Fits a machine learning model based on the specified model.

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
        self.model_map = _get_model_map(self.task_type, self.add_model, self.n_jobs)
        self.cv = _get_cv_strategy(
            self.task_type, n_splits=self.n_splits, n_repeats=self.n_repeats
        )

        if self.select_model == "best":
            self.scoring = self.scoring or "f1" if self.task_type == "C" else "r2"
            comparison_df = cross_validation_report(
                data=data,
                activity_col=self.activity_col,
                id_col=self.id_col,
                add_model=self.add_model,
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

            self.select_model = comparison_df.loc[
                comparison_df[f"{self.scoring}_mean"].idxmax(), "Model"
            ]
            self.model = self.model_map[self.select_model].fit(X=X_data, y=y_data)
        elif self.select_model in self.model_map:
            self.model = self.model_map[self.select_model].fit(X=X_data, y=y_data)
        else:
            raise ValueError(f"Model '{self.select_model}' is not recognized.")

        if self.save_model:
            if self.save_dir and not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir, exist_ok=True)
            with open(f"{self.save_dir}/model.pkl", "wb") as file:
                pickle.dump(self, file)

        return self

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
                "ModelDeveloper is not fitted yet. Call 'fit' before using this model."
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
            if self.save_dir and not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir, exist_ok=True)
            self.pred_result.to_csv(f"{self.save_dir}/{self.pred_result_name}.csv")

        return self.pred_result
