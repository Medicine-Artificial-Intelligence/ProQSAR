import os
import pickle
import logging
import pandas as pd
from typing import Optional, Union, List
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator
from ProQSAR.ModelDeveloper.model_developer_utils import (
    _get_task_type,
    _get_model_map,
    _get_cv_strategy,
)
from ProQSAR.ModelDeveloper.model_validation import ModelValidation
from ProQSAR.Config.validation_config import CrossValidationConfig


class ModelDeveloper(CrossValidationConfig, BaseEstimator):
    """
    Wrapper for model selection, cross-validated evaluation, model fitting and prediction.

    This class:
      - infers the task type (classification/regression) from the data,
      - constructs a default model map (mergeable with `add_model`),
      - optionally cross-validates candidate models and selects the best one,
      - fits the selected model on the full provided dataset,
      - exposes `predict` to create a predictions DataFrame,
      - optionally saves the fitted ModelDeveloper instance or prediction results.

    Parameters
    ----------
    activity_col : str
        Column name for the target variable (default "activity").
    id_col : str
        Column name for the identifier column (default "id").
    select_model : str | list[str] | None
        Name of the model to use or a list of candidate names to evaluate (default None).
    add_model : dict
        Additional models to include in the model map (name -> estimator or (estimator, ...)).
    cross_validate : bool
        Whether to run cross-validation to select among candidate models (default True).
    save_model : bool
        If True, save the fitted ModelDeveloper object (pickle) to `save_dir`.
    save_pred_result : bool
        If True, save prediction results to CSV when `predict` is called.
    pred_result_name : str
        Filename (without directory) for saved prediction results (default "pred_result").
    save_dir : str | None
        Directory for saving model/prediction files (default "Project/ModelDeveloper").
    n_jobs : int
        Number of parallel jobs passed to underlying estimators (default 1).
    random_state : Optional[int]
        Random seed for reproducible estimators (default 42).
    **kwargs
        Forwarded to CrossValidationConfig for CV-related parameters (e.g., n_splits, scoring_target, scoring_list).
    """

    def __init__(
        self,
        activity_col: str = "activity",
        id_col: str = "id",
        select_model: Optional[Union[str, List[str]]] = None,
        add_model: dict = {},
        cross_validate: bool = True,
        save_model: bool = False,
        save_pred_result: bool = False,
        pred_result_name: str = "pred_result",
        save_dir: Optional[str] = "Project/ModelDeveloper",
        n_jobs: int = 1,
        random_state: Optional[int] = 42,
        **kwargs,
    ):
        """Initializes the ModelDeveloper with necessary attributes."""
        CrossValidationConfig.__init__(self, **kwargs)
        self.activity_col = activity_col
        self.id_col = id_col
        self.select_model = select_model
        self.add_model = add_model
        self.cross_validate = cross_validate
        self.save_model = save_model
        self.save_pred_result = save_pred_result
        self.pred_result_name = pred_result_name
        self.save_dir = save_dir
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.model = None
        self.task_type = None
        self.cv = None
        self.report = None
        self.classes_ = None

    def fit(self, data: pd.DataFrame) -> "ModelDeveloper":
        """
        Fit (or select and fit) the model on the provided dataset.

        Behavior:
          - Infers task type and CV strategy,
          - Builds the model map merged with `add_model`,
          - If `select_model` is None or a list and `cross_validate` is True,
            runs cross-validation to select the best model and fits it on full data.
          - If `select_model` is a string, fits that model directly and optionally runs CV.
          - Saves the fitted ModelDeveloper instance if `save_model` is True.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing features and the activity/id columns.

        Returns
        -------
        ModelDeveloper
            The fitted ModelDeveloper instance.

        Raises
        ------
        Exception
            Any unexpected exception is logged and re-raised.
        """
        try:
            X_data = data.drop([self.activity_col, self.id_col], axis=1)
            y_data = data[self.activity_col]

            self.task_type = _get_task_type(data, self.activity_col)
            self.cv = _get_cv_strategy(
                self.task_type,
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
                random_state=self.random_state,
            )
            model_map = _get_model_map(
                self.task_type,
                self.add_model,
                self.n_jobs,
                random_state=self.random_state,
            )
            # Set scorings
            if self.scoring_target is None:
                self.scoring_target = "f1" if self.task_type == "C" else "r2"

            if self.scoring_list:
                if isinstance(self.scoring_list, str):
                    self.scoring_list = [self.scoring_list]

                if self.scoring_target not in self.scoring_list:
                    self.scoring_list.append(self.scoring_target)

            if isinstance(self.select_model, list) or not self.select_model:
                if self.cross_validate:
                    logging.info(
                        "ModelDeveloper: Selecting the optimal model "
                        f"among {self.select_model or list(model_map.keys())}, "
                        f"scoring target: '{self.scoring_target}'."
                    )
                    self.report = ModelValidation.cross_validation_report(
                        data=data,
                        activity_col=self.activity_col,
                        id_col=self.id_col,
                        add_model=self.add_model,
                        select_model=self.select_model,
                        scoring_list=self.scoring_list,
                        n_splits=self.n_splits,
                        n_repeats=self.n_repeats,
                        visualize=self.visualize,
                        save_fig=self.save_fig,
                        fig_prefix=self.fig_prefix,
                        save_csv=self.save_cv_report,
                        csv_name=self.cv_report_name,
                        save_dir=self.save_dir,
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                    )

                    self.select_model = (
                        self.report.set_index(["scoring", "cv_cycle"])
                        .loc[(f"{self.scoring_target}", "mean")]
                        .idxmax()
                    )
                    self.model = model_map[self.select_model].fit(X=X_data, y=y_data)
                    logging.info(f"ModelDeveloper: Selected model: {self.select_model}")

                else:
                    raise AttributeError(
                        "'select_model' is entered as a list."
                        "To evaluate and use the best method among the entered methods, turn 'cross_validate = True'."
                        "Otherwise, select_model must be a string as the name of the method."
                    )

            elif isinstance(self.select_model, str):
                if self.select_model not in model_map:
                    raise ValueError(
                        f"ModelDeveloper: Model '{self.select_model}' is not recognized."
                    )
                else:
                    logging.info(f"ModelDeveloper: Using model: {self.select_model}")
                    self.model = model_map[self.select_model].fit(X=X_data, y=y_data)

                    if self.cross_validate:
                        self.report = ModelValidation.cross_validation_report(
                            data=data,
                            activity_col=self.activity_col,
                            id_col=self.id_col,
                            add_model=self.add_model,
                            select_model=self.select_model,
                            scoring_list=self.scoring_list,
                            n_splits=self.n_splits,
                            n_repeats=self.n_repeats,
                            visualize=self.visualize,
                            save_fig=self.save_fig,
                            fig_prefix=self.fig_prefix,
                            save_csv=self.save_cv_report,
                            csv_name=self.cv_report_name,
                            save_dir=self.save_dir,
                            n_jobs=self.n_jobs,
                            random_state=self.random_state,
                        )
            else:
                raise AttributeError(
                    f"'select_model' is entered as a {type(self.select_model)}"
                    "Please input a string or a list or None."
                )

            self.classes_ = self.model.classes_ if self.task_type == "C" else None

            if self.save_model:
                if self.save_dir and not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir, exist_ok=True)
                with open(f"{self.save_dir}/model.pkl", "wb") as file:
                    pickle.dump(self, file)
                logging.info(f"Model saved at: {self.save_dir}/model.pkl")

            return self

        except Exception as e:
            logging.error(f"Error in model fitting: {e}")
            raise

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for the provided data using the fitted model.

        The method returns a DataFrame that always contains the id column and a
        'Predicted value' column, and includes the true activity values if available.
        For classification tasks, probability columns for each class are also included.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing features and id/activity columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with prediction results and optionally saved to CSV if
            `save_pred_result` is True.

        Raises
        ------
        NotFittedError
            If `fit` has not been called and the internal model is not present.
        Exception
            Any unexpected exception is logged and re-raised.
        """
        try:
            if self.model is None:
                raise NotFittedError(
                    "ModelDeveloper is not fitted yet. Call 'fit' before using this model."
                )

            X_data = data.drop(
                [self.activity_col, self.id_col], axis=1, errors="ignore"
            )
            y_pred = self.model.predict(X_data)
            result = {
                f"{self.id_col}": data[self.id_col].values,
                "Predicted value": y_pred,
            }
            # Get actual value if available
            if self.activity_col in data.columns:
                result[self.activity_col] = data[self.activity_col].values

            if self.task_type == "C":
                y_proba = self.model.predict_proba(X_data)
                result[f"Probability for class {self.classes_[0]}"] = y_proba[:, 0]
                result[f"Probability for class {self.classes_[1]}"] = y_proba[:, 1]

            pred_result = pd.DataFrame(result)

            if self.save_pred_result:
                if self.save_dir and not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir, exist_ok=True)
                pred_result.to_csv(
                    f"{self.save_dir}/{self.pred_result_name}.csv", index=False
                )
                logging.info(
                    f"Prediction results saved to {self.save_dir}/{self.pred_result_name}.csv"
                )

            return pred_result

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise

    def set_params(self, **kwargs):
        """
        Update attributes of the ModelDeveloper instance.

        Only existing attributes may be updated; unknown keys raise KeyError.
        Returns self to allow fluent chaining.

        Parameters
        ----------
        **kwargs : dict
            Attribute names and their new values.

        Returns
        -------
        ModelDeveloper
            The same instance with updated attributes.

        Raises
        ------
        KeyError
            If a provided key does not correspond to an existing attribute.
        """
        valid_keys = self.__dict__.keys()
        for key in kwargs:
            if key not in valid_keys:
                raise KeyError(f"'{key}' is not a valid attribute.")
        self.__dict__.update(**kwargs)

        return self
