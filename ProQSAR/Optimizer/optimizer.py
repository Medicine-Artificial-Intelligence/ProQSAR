import optuna
import pandas as pd
import logging
from sklearn.model_selection import cross_val_score
from typing import Optional, List, Tuple, Dict, Any, Union
from ProQSAR.ModelDeveloper.model_developer_utils import (
    _get_task_type,
    _get_cv_strategy,
)
from ProQSAR.Optimizer.optimizer_utils import _get_model_list, _get_model_and_params
from sklearn.base import BaseEstimator


class Optimizer(BaseEstimator):
    """
    Optimize hyperparameters for one or more candidate models using Optuna.

    The Optimizer supports:
      - specifying which models to search over (select_model),
      - custom parameter ranges for each model (param_ranges),
      - adding custom models (add_model: mapping model_name -> (estimator, param_ranges)),
      - repeated cross-validation for robust scoring,
      - retrieving best parameters and score after optimization.

    Parameters
    ----------
    activity_col : str
        Column name for the target variable (default "activity").
    id_col : str
        Column name for the identifier column (default "id").
    select_model : list[str] | None
        Optional list of model names to evaluate. If None, the default model
        list for the detected task will be used.
    scoring : str | None
        Scoring metric name used by sklearn (e.g., 'f1', 'r2'). If None, defaults
        to 'f1' for classification and 'r2' for regression.
    param_ranges : dict
        Mapping model_name -> parameter ranges used by the trial sampler.
        Example: {"RandomForestClassifier": {"n_estimators": (50,200)}}.
    add_model : dict
        Mapping of custom models to add. Expected format:
        {name: (estimator_instance, param_range_dict)}.
    n_trials : int
        Number of Optuna trials to run (default 50).
    n_splits : int
        Number of CV folds (default 5).
    n_repeats : int
        Number of CV repeats (default 2).
    n_jobs : int
        Number of parallel jobs passed to cross_val_score and some estimators.
    random_state : int
        Random seed used for reproducibility (default 42).
    study_name : str
        Optuna study name / storage key base (default 'my_study').
    deactivate : bool
        If True, optimization is skipped and the instance is returned as-is.

    Attributes
    ----------
    best_model : Any
        Best model discovered (not always set; kept for compatibility).
    best_params : dict | None
        Best parameters found by the study after optimization.
    best_score : float | None
        Best cross-validated score achieved.
    task_type : str | None
        Inferred task type: 'C' or 'R'.
    cv : object | None
        Cross-validation splitter used during optimization.
    """

    def __init__(
        self,
        activity_col: str = "activity",
        id_col: str = "id",
        select_model: Optional[List[str]] = None,
        scoring: Optional[str] = None,
        param_ranges: Dict[str, Dict[str, Any]] = {},
        add_model: Dict[str, Tuple[Any, Dict[str, Any]]] = {},
        n_trials: int = 50,
        n_splits: int = 5,
        n_repeats: int = 2,
        n_jobs: int = 1,
        random_state: int = 42,
        study_name: str = "my_study",
        deactivate: bool = False,
    ) -> None:
        """
        Initializes the Optimizer class with user-defined parameters.
        """
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
        self.random_state = random_state
        self.study_name = study_name
        self.deactivate = deactivate
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.task_type = None
        self.cv = None

        # Merge additional model parameters into param_ranges
        self.param_ranges.update(
            {name: params for name, (model, params) in self.add_model.items()}
        )

    def optimize(
        self, data: pd.DataFrame
    ) -> Union[Tuple[Dict[str, Any], float], "Optimizer"]:
        """
        Run the Optuna optimization process to find the best hyperparameters.

        Steps:
          - Infer task type and CV splitting strategy.
          - Build the list of candidate models (either user-provided or the
            default from _get_model_list).
          - Define an Optuna objective that samples model name (if multiple)
            and hyperparameters, sets them on the model, and evaluates via
            cross_val_score using the configured CV splitter.
          - Create or load an Optuna study (SQLite storage 'example.db') and
            run the specified number of trials.
          - Store `best_params` and `best_score` on the instance and return them.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing feature columns and the activity/id columns.

        Returns
        -------
        tuple(dict, float) | Optimizer
            Returns (best_params, best_score) upon successful optimization.
            If `deactivate` is True, returns `self` without performing optimization.

        Raises
        ------
        Exception
            Any unexpected exceptions are logged and re-raised.
        """
        if self.deactivate:
            logging.info("Optimizer is deactivated. Skipping optimize.")
            return self

        try:
            X = data.drop([self.activity_col, self.id_col], axis=1)
            y = data[self.activity_col]

            self.task_type = _get_task_type(data, self.activity_col)
            self.cv = _get_cv_strategy(
                self.task_type,
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
                random_state=self.random_state,
            )
            if self.scoring is None:
                self.scoring = "f1" if self.task_type == "C" else "r2"

            model_list = self.select_model or _get_model_list(
                self.task_type, self.add_model
            )
            if isinstance(model_list, str):
                model_list = [model_list]

            def objective(trial):
                try:
                    # If only one model is provided, use it directly.
                    if len(model_list) == 1:
                        model_name = model_list[0]
                    else:
                        model_name = trial.suggest_categorical("model", model_list)

                    model, params = _get_model_and_params(
                        trial, model_name, self.param_ranges, self.add_model
                    )
                    model.set_params(**params)

                    # Ensure the model is set with a random state if applicable
                    if "random_state" in model.get_params():
                        model.set_params(random_state=self.random_state)

                    # Set thread count or n_jobs if applicable
                    if "thread_count" in model.get_params():
                        model.set_params(thread_count=self.n_jobs)
                    if "n_jobs" in model.get_params():
                        model.set_params(n_jobs=self.n_jobs)

                    logging.info(f"Starting trial with parameters: {trial.params}")

                    score = cross_val_score(
                        model,
                        X,
                        y,
                        scoring=self.scoring,
                        cv=self.cv,
                        n_jobs=self.n_jobs,
                    ).mean()

                    return score

                except Exception as e:
                    logging.error(f"Error in objective function: {e}")
                    raise

            # Setting the logging level WARNING, the INFO logs are suppressed.
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            # Simple SQLite storage to persist studies; load_if_exists=True resumes studies.
            storage = "sqlite:///example.db"
            study = optuna.create_study(
                study_name=self.study_name,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
                storage=storage,
                load_if_exists=True,
            )
            study.optimize(objective, n_trials=self.n_trials, n_jobs=2)
            self.best_params = study.best_trial.params
            self.best_score = study.best_value

            logging.info(
                f"Optimizer: best_params are {self.best_params}, best_score is {self.best_score}."
            )

            return self.best_params, self.best_score

        except Exception as e:
            logging.error(f"Optimization failed: {e}")
            raise

    def get_best_params(self) -> Dict[str, Any]:
        """
        Return the best hyperparameter dictionary found by the last optimize() call.

        Raises
        ------
        AttributeError
            If optimize() has not been run and best_params is not set.
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
        Return the best cross-validated score found by the last optimize() call.

        Raises
        ------
        AttributeError
            If optimize() has not been run and best_score is not set.
        """
        if self.best_params:
            return self.best_score
        else:
            raise AttributeError(
                "Attempted to access 'best_score' before running 'optimize'. "
                "Run 'optimize' to obtain the best score."
            )
