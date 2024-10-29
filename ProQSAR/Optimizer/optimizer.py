import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score
from typing import Optional, List
from ProQSAR.ModelDeveloper.model_developer_utils import (
    _get_task_type,
    _get_cv_strategy,
)
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
        n_splits: int = 5,
        n_repeats: int = 2,
        n_jobs: int = -1,
    ):

        self.activity_col = activity_col
        self.id_col = id_col
        self.select_model = select_model
        self.param_ranges = param_ranges if param_ranges else {}
        self.add_model = add_model if add_model else {}
        self.scoring = scoring
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.n_jobs = n_jobs
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.task_type = None
        self.cv = None
        self.param_ranges.update(
            {name: params for name, (model, params) in self.add_model.items()}
        )

    def optimize(self, data: pd.DataFrame):

        X = data.drop([self.activity_col, self.id_col], axis=1)
        y = data[self.activity_col]

        self.task_type = _get_task_type(data, self.activity_col)
        self.cv = _get_cv_strategy(
            self.task_type, n_splits=self.n_splits, n_repeats=self.n_repeats
        )
        if self.scoring is None:
            self.scoring = "f1" if self.task_type == "C" else "r2"

        model_list = (
            self.select_model
            if self.select_model
            else _get_model_list(self.task_type, self.add_model)
        )

        def objective(trial):
            model_name = trial.suggest_categorical("model", model_list)
            model, params = _get_model_and_params(
                trial, model_name, self.param_ranges, self.add_model
            )

            model.set_params(**params)
            score = cross_val_score(
                model, X, y, scoring=self.scoring, cv=self.cv, n_jobs=self.n_jobs
            ).mean()
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs)

        self.best_params = study.best_trial.params
        self.best_score = study.best_value

        return self.best_params, self.best_score

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
