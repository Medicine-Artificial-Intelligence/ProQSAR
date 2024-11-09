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
    """
    A class to optimize machine learning models using Optuna for hyperparameter tuning.

    Parameters:
    ----------
    activity_col : str
        The name of the target column in the dataset.
    id_col : str
        The name of the identifier column in the dataset (not used in modeling).
    select_model : Optional[List[str]], default=None
        A list of model names to optimize. If None, all compatible models will be considered.
    scoring : Optional[str], default=None
        The scoring metric to use for model evaluation. If None, defaults to "f1" for classification
        tasks and "r2" for regression tasks.
    param_ranges : dict, default={}
        A dictionary specifying the ranges of hyperparameters for each model.
    add_model : dict, default={}
        A dictionary of additional models and their parameter ranges to consider during optimization.
    n_trials : int, default=100
        The number of trials to run for hyperparameter optimization.
    n_splits : int, default=5
        The number of splits for cross-validation.
    n_repeats : int, default=2
        The number of times to repeat cross-validation.
    n_jobs : int, default=-1
        The number of parallel jobs to run for cross-validation. -1 means using all processors.

    Attributes:
    ----------
    best_model : Optional
        The best model found during optimization.
    best_params : Optional[dict]
        The best hyperparameters found during optimization.
    best_score : Optional[float]
        The best score achieved with the best parameters.
    task_type : Optional[str]
        The type of task (classification or regression).
    cv : Optional
        The cross-validation strategy used.

    Methods:
    -------
    optimize(data: pd.DataFrame) -> tuple:
        Optimizes the model based on the provided dataset.
    get_best_params() -> dict:
        Retrieves the best hyperparameters found during optimization.
    get_best_score() -> float:
        Retrieves the best score achieved during optimization.
    """

    def __init__(
        self,
        activity_col: str,
        id_col: str,
        select_model: Optional[List[str]] = None,
        scoring: Optional[str] = None,
        param_ranges: dict = {},
        add_model: dict = {},
        n_trials: int = 100,
        n_splits: int = 5,
        n_repeats: int = 2,
        n_jobs: int = -1,
    ):

        self.activity_col = activity_col
        self.id_col = id_col
        self.select_model = select_model
        self.param_ranges = param_ranges
        self.add_model = add_model
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

    def optimize(self, data: pd.DataFrame) -> tuple:
        """
        Optimizes the model based on the provided dataset.

        Parameters:
        ----------
        data : pd.DataFrame
            The input data containing features and target variable.

        Returns:
        -------
        tuple
            A tuple containing the best parameters and the best score achieved.
        """

        X = data.drop([self.activity_col, self.id_col], axis=1)
        y = data[self.activity_col]

        self.task_type = _get_task_type(data, self.activity_col)
        self.cv = _get_cv_strategy(
            self.task_type, n_splits=self.n_splits, n_repeats=self.n_repeats
        )
        self.scoring = self.scoring or "f1" if self.task_type == "C" else "r2"

        model_list = self.select_model or _get_model_list(
            self.task_type, self.add_model
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

    def get_best_params(self) -> dict:
        """
        Retrieves the best model and hyperparameters found during optimization.

        Returns:
        -------
        dict
            The best model and hyperparameters.

        Raises:
        ------
        AttributeError
            If called before running 'optimize'.
        """
        if self.best_params:
            return self.best_params
        else:
            raise AttributeError(
                "Attempted to access 'best_params' before running 'optimize'. "
                "Run 'optimize' to obtain the best parameters."
            )

    def get_best_score(self) -> float:
        """
        Retrieves the best score achieved during optimization.

        Returns:
        -------
        float
            The best score.

        Raises:
        ------
        AttributeError
            If called before running 'optimize'.
        """
        if self.best_params:
            return self.best_score
        else:
            raise AttributeError(
                "Attempted to access 'best_score' before running 'optimize'. "
                "Run 'optimize' to obtain the best score."
            )
