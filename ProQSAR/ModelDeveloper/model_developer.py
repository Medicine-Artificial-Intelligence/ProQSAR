import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from typing import Union, Optional, List
from sklearn.model_selection import (
    cross_val_score,
    cross_validate,
    RepeatedStratifiedKFold,
    RepeatedKFold,
)
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    ElasticNetCV,
    Ridge,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
)
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    log_loss,
    brier_score_loss,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    mean_absolute_percentage_error,
    max_error,
)
from ProQSAR.ModelDeveloper.model_developer_utils import (
    _get_task_type,
    _get_method_map,
    _get_cv_strategy,
    _get_scoring_target,
    _get_iv_scoring_list,
    _get_ev_scoring_dict,
)
from ProQSAR. ModelDeveloper.model_validation import _get_best_method

class ModelDeveloper:

    def __init__(
        self,
        activity_col: str,
        id_col: str,
        method: str = "best",
        add_method: Optional[dict] = None,
        scoring_target: Optional[str] = None,
        n_jobs: int = -1,
        save_dir: Optional[str] = None,
        comparison_report: bool = False,
        comparison_visual: Optional[str] = None,
        save_fig: bool = False,
    ):
        """
        Initializes the ModelDeveloper class with the given parameters.

        Parameters
        ----------
        activity_col : str
            The name of the target variable column in the dataset.
        id_col : str
            The name of the ID column in the dataset.
        method : str, optional
            The method to use for model selection ("best" or specific model name).
        add_method : dict, optional
            Additional models to add to the method map.
        scoring_target : str, optional
            The target scoring metric.
        n_jobs : int, optional
            Number of parallel jobs for cross-validation.
        save_dir : str, optional
            Directory to save models and results.
        comparison_report : bool, optional
            Whether to display a comparison report.
        comparison_visual : str, optional
            Type of visualization for model comparison.
        save_fig : bool, optional
            Whether to save the comparison plot.
        """
        self.activity_col = activity_col
        self.id_col = id_col
        self.method = method
        self.add_method = add_method
        self.scoring_target = scoring_target
        self.n_jobs = n_jobs
        self.save_dir = save_dir
        self.comparison_report = comparison_report
        self.comparison_visual = comparison_visual
        self.save_fig = save_fig
        self.model = None
        self.task_type = None
        self.cv = None

    def fit(self, data: pd.DataFrame) -> BaseEstimator:
        """
        Fits the model using the provided dataset.

        Parameters:
            data (pd.DataFrame): The dataset including features and target column.

        Returns:
            BaseEstimator: The trained model.

        Raises:
            ValueError: If the specified method is not recognized.
        """
        X_data = data.drop([self.activity_col, self.id_col], axis=1)
        y_data = data[self.activity_col]

        self.task_type = _get_task_type(data, self.activity_col)
        self.method_map = _get_method_map(self.task_type, self.add_method)
        self.cv = _get_cv_strategy(self.task_type)

        if self.method == "best":
            if self.scoring_target is None:
                self.scoring_target = "f1" if self.task_type == "C" else "r2"                
            self.method = self._get_best_method(data, self.scoring_target)
            self.model = self.method_map[self.method].fit(X=X_data, y=y_data)
        elif self.method in self.method_map:
            self.model = self.method_map[self.method].fit(X=X_data, y=y_data)
        else:
            raise ValueError(f"Method '{self.method}' is not recognized.")

        if self.save_dir:
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
        Makes predictions using the trained model.

        Parameters:
            data (pd.DataFrame): The dataset including features and ID column.

        Returns:
            pd.DataFrame: A DataFrame with IDs and predicted values. Includes probabilities for classification tasks.

        Raises:
            NotFittedError: If the model has not been trained yet.
        """
        if self.model is None:
            raise NotFittedError("ModelDeveloper is not fitted yet.")

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
        if self.save_dir:
            self.pred_result.to_csv(f"{self.save_dir}/pred_result.csv")

        return self.pred_result

    @staticmethod
    def static_predict(data: pd.DataFrame, save_dir: str) -> pd.DataFrame:
        """
        Makes predictions using a previously saved model.

        Parameters:
            data (pd.DataFrame): The dataset including features and ID column.
            save_dir (str): Directory where the model and other necessary files are saved.

        Returns:
            pd.DataFrame: A DataFrame with IDs and predicted values. Includes probabilities for classification tasks.

        Raises:
            NotFittedError: If the necessary files are not found in the specified directory.
        """
        if not os.path.exists(f"{save_dir}/feature_selector.pkl"):
            raise NotFittedError(
                "The FeatureSelector instance is not fitted yet. Call 'fit' before using this method."
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
        if save_dir:
            pred_result.to_csv(f"{save_dir}/pred_result.csv")

        return pred_result

