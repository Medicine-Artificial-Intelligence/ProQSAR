import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from typing import Optional, List
from ProQSAR.ModelDeveloper.model_developer_utils import _get_task_type
from ProQSAR.Optimizer.optimizer_utils import _get_model_list, _get_model_and_params


class Optimizer:
    def __init__(
        self,
        activity_col: str,
        id_col: str,
        select_model: Optional[List[str]] = None,
        param_ranges: Optional[dict] = None,
        add_model: Optional[dict] = None,
        scoring: Optional[str] = None,
        n_trials: int = 10,
        cv: int = 5,
        n_jobs: int = -1,
    ):

        self.activity_col = activity_col
        self.id_col = id_col
        self.select_model = select_model
        self.param_ranges = param_ranges if param_ranges else {}
        self.add_model = add_model if add_model else {}
        self.scoring = scoring
        self.n_trials = n_trials
        self.cv = cv
        self.n_jobs = n_jobs
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.task_type = None
        self.param_ranges.update(
            {name: params for name, (model, params) in self.add_model.items()}
        )

    def optimize(self, data: pd.DataFrame):

        X_data = data.drop([self.activity_col, self.id_col], axis=1)
        y_data = data[self.activity_col]

        self.task_type = _get_task_type(data, self.activity_col)
        if self.select_model is None:
            self.select_model = _get_model_list(self.task_type, self.add_model)

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self.objective(trial, X_data, y_data),
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
        )

        self.best_model = study.best_trial.user_attrs["best_model"]
        self.best_params = study.best_trial.params
        self.best_score = study.best_value

        print(f"Best model: {self.best_model}")
        print(f"Best parameters: {self.best_params}")
        print(f"Best {self.scoring} score: {self.best_score}")

    def objective(self, trial, X, y):
        """
        Objective function for the optimization.

        Parameters:
            trial (optuna.trial.Trial): An Optuna trial object.
            X (pd.DataFrame or np.ndarray): Feature data.
            y (pd.Series or np.ndarray): Target data.

        Returns:
            float: Cross-validated score of the model.
        """
        model_name = trial.suggest_categorical("model", self.select_model)
        model, params = _get_model_and_params(
            trial, trial, model_name, self.param_ranges, self.add_model
        )

        model.set_params(**params)
        score = cross_val_score(
            model, X, y, scoring=self.scoring, cv=self.cv, n_jobs=self.n_jobs
        ).mean()

        trial.set_user_attr("best_model", model_name)
        return score

    def get_best_model(self):
        """
        Get the best model selected by Optuna.

        Returns:
            str: Name of the best model.
        """
        return self.best_model

    def get_best_params(self):
        """
        Get the best parameters found for the selected model.

        Returns:
            dict: Best parameter dictionary.
        """
        return self.best_params

    def get_best_score(self):
        """
        Get the best cross-validated score found for the selected model.

        Returns:
            float: Best score.
        """
        return self.best_score
