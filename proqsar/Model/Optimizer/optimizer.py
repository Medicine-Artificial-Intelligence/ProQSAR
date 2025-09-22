# proqsar/Model/Optimizer/optimizer.py
import optuna
import pandas as pd
import logging
from sklearn.model_selection import cross_val_score
from typing import Optional, List, Tuple, Dict, Any, Union
from ..ModelDeveloper.model_developer_utils import (
    _get_task_type,
    _get_cv_strategy,
)
from .optimizer_utils import _get_model_list, _get_model_and_params
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

    :param activity_col: Column name for the target variable (default: "activity").
    :type activity_col: str
    :param id_col: Column name for the identifier column (default: "id").
    :type id_col: str
    :param select_model: Optional list of model names to evaluate. If None, the
        default model list for the detected task will be used.
    :type select_model: list[str] | None
    :param scoring: Scoring metric name used by sklearn (e.g., 'f1', 'r2').
        If None, defaults to 'f1' for classification and 'r2' for regression.
    :type scoring: str | None
    :param param_ranges: Mapping model_name -> parameter ranges used by the trial sampler.
        Example: {"RandomForestClassifier": {"n_estimators": (50,200)}}.
    :type param_ranges: dict
    :param add_model: Mapping of custom models to add. Expected format:
        {name: (estimator_instance, param_range_dict)}.
    :type add_model: dict
    :param n_trials: Number of Optuna trials to run (default: 50).
    :type n_trials: int
    :param n_splits: Number of CV folds (default: 5).
    :type n_splits: int
    :param n_repeats: Number of CV repeats (default: 2).
    :type n_repeats: int
    :param n_jobs: Number of parallel jobs passed to cross_val_score and some estimators.
    :type n_jobs: int
    :param random_state: Random seed used for reproducibility (default: 42).
    :type random_state: int
    :param study_name: Optuna study name / storage key base (default: 'my_study').
    :type study_name: str
    :param deactivate: If True, optimization is skipped and the instance is returned as-is.
    :type deactivate: bool
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
        Initialize the Optimizer instance.

        :param activity_col: Column name for the target variable.
        :param id_col: Column name for the identifier column.
        :param select_model: List of model names to evaluate or None to use defaults.
        :param scoring: Scoring metric for cross-validation.
        :param param_ranges: Dictionary of parameter ranges for models.
        :param add_model: Additional custom models mapping.
        :param n_trials: Number of optimization trials.
        :param n_splits: CV fold count.
        :param n_repeats: CV repeat count.
        :param n_jobs: Parallel job count.
        :param random_state: Random seed.
        :param study_name: Optuna study name.
        :param deactivate: Skip optimization if True.
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

        # Merge additional model parameters into param_ranges so they are available in search.
        self.param_ranges.update(
            {name: params for name, (model, params) in self.add_model.items()}
        )

    # --- Private helpers -------------------------------------------------
    def _prepare_xy(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract feature matrix X and target y from a DataFrame.
        """
        X = data.drop([self.activity_col, self.id_col], axis=1)
        y = data[self.activity_col]
        return X, y

    def _infer_task_and_cv(self, data: pd.DataFrame) -> None:
        """
        Infer task type (C/R) and prepare CV splitter. Mutates self.task_type and self.cv.
        """
        self.task_type = _get_task_type(data, self.activity_col)
        self.cv = _get_cv_strategy(
            self.task_type,
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )
        if self.scoring is None:
            self.scoring = "f1" if self.task_type == "C" else "r2"

    def _build_model_list(self) -> List[str]:
        """
        Build and normalize the candidate model list.
        """
        model_list = self.select_model or _get_model_list(self.task_type, self.add_model)
        if isinstance(model_list, str):
            model_list = [model_list]
        return model_list

    def _set_runtime_params(self, model: Any) -> None:
        """
        Set runtime parameters on the model if parameter names exist.
        Attempts to set random_state, thread_count, n_jobs when present.
        """
        try:
            params = model.get_params()
        except Exception:
            return

        if "random_state" in params:
            try:
                model.set_params(random_state=self.random_state)
            except Exception:
                pass

        if "thread_count" in params:
            try:
                model.set_params(thread_count=self.n_jobs)
            except Exception:
                pass

        if "n_jobs" in params:
            try:
                model.set_params(n_jobs=self.n_jobs)
            except Exception:
                pass

    def _make_objective(self, X: pd.DataFrame, y: pd.Series, model_list: List[str]):
        """
        Return an Optuna objective function bound to X, y and model_list.
        """

        def objective(trial: optuna.Trial) -> float:
            # choose model (or single-model mode)
            if len(model_list) == 1:
                model_name = model_list[0]
            else:
                model_name = trial.suggest_categorical("model", model_list)

            # obtain model instance and sampled params
            model, params = _get_model_and_params(
                trial, model_name, self.param_ranges, self.add_model
            )
            model.set_params(**params)

            # set runtime params (n_jobs/thread_count/random_state) when available
            self._set_runtime_params(model)

            logging.info(f"Starting trial with model={model_name} params={trial.params}")

            # evaluate with cross_val_score and return mean
            score = cross_val_score(
                model,
                X,
                y,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=self.n_jobs,
            ).mean()
            return score

        return objective

    def _create_or_load_study(self) -> optuna.Study:
        """
        Create or load an Optuna study using a small SQLite file for persistence.
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        storage = "sqlite:///example.db"
        study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            storage=storage,
            load_if_exists=True,
        )
        return study

    # --- Public API -----------------------------------------------------
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

        :param data: DataFrame containing feature columns and the activity/id columns.
        :type data: pd.DataFrame
        :returns: (best_params, best_score) tuple on success or self if deactivated.
        :rtype: Tuple[Dict[str, Any], float] | Optimizer
        :raises Exception: Any unexpected exceptions are logged and re-raised.
        """
        if self.deactivate:
            logging.info("Optimizer is deactivated. Skipping optimize.")
            return self

        try:
            # prepare XY and CV
            X, y = self._prepare_xy(data)
            self._infer_task_and_cv(data)
            model_list = self._build_model_list()

            # build objective and run study
            objective = self._make_objective(X, y, model_list)
            study = self._create_or_load_study()
            study.optimize(objective, n_trials=self.n_trials, n_jobs=2)

            # store best results
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

        :returns: Best parameters dictionary.
        :rtype: Dict[str, Any]
        :raises AttributeError: If optimize() has not been run and best_params is not set.
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

        :returns: Best cross-validated score.
        :rtype: float
        :raises AttributeError: If optimize() has not been run and best_score is not set.
        """
        if self.best_score is not None:
            return self.best_score
        else:
            raise AttributeError(
                "Attempted to access 'best_score' before running 'optimize'. "
                "Run 'optimize' to obtain the best score."
            )
