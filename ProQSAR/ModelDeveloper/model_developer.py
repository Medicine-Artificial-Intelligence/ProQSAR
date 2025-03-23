import os
import pickle
import logging
import pandas as pd
from sklearn.exceptions import NotFittedError
from typing import Optional, Union, List

from ProQSAR.ModelDeveloper.model_developer_utils import (
    _get_task_type,
    _get_model_map,
    _get_cv_strategy,
)
from ProQSAR.ModelDeveloper.model_validation import ModelValidation
from ProQSAR.validation_config import CrossValidationConfig


class ModelDeveloper(CrossValidationConfig):
    """
    Class to handle model development for machine learning tasks.

    Attributes:
    -----------
    activity_col : str
        Name of the activity column in the dataset.
    id_col : str
        Name of the ID column in the dataset.
    select_model : str, optional
        The model to use for model development.
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
        Name of the IV report file (default: 'cv_report').
    visualize : Optional[str], optional
        Visualization option for report (default: None).
    save_fig : bool, optional
        Whether to save figures (default: False).
    fig_prefix : str, optional
        Name of the figure file (default: 'cv_graph').
    n_jobs : int, optional
        Number of jobs to run in parallel (default: -1).

    Methods:
    --------
    fit(data: pd.DataFrame) -> BaseEstimator
        Fits the model to the provided data.

    predict(data: pd.DataFrame) -> pd.DataFrame
        Predicts outcomes using the fitted model.

    """

    def __init__(
        self,
        activity_col: str = "activity",
        id_col: str = "id",
        select_model: Optional[Union[str, List[str]]] = None,
        add_model: dict = {},
        compare: bool = True,
        save_model: bool = False,
        save_pred_result: bool = False,
        pred_result_name: str = "pred_result",
        save_dir: Optional[str] = "Project/ModelDeveloper",
        n_jobs: int = -1,
        **kwargs,
    ):
        """Initializes the ModelDeveloper with necessary attributes."""
        super().__init__(**kwargs)
        self.activity_col = activity_col
        self.id_col = id_col
        self.select_model = select_model
        self.add_model = add_model
        self.compare = compare
        self.save_model = save_model
        self.save_pred_result = save_pred_result
        self.pred_result_name = pred_result_name
        self.save_dir = save_dir
        self.n_jobs = n_jobs
        self.model = None
        self.task_type = None
        self.cv = None
        self.report = None
        self.classes_ = None

    def fit(self, data: pd.DataFrame) -> "ModelDeveloper":
        """
        Fits a machine learning model based on the specified model.

        Parameters:
        -----------
        data : pd.DataFrame
            The training dataset including features, activity, and ID columns.
        """
        try:
            logging.info("Starting model fitting.")
            X_data = data.drop([self.activity_col, self.id_col], axis=1)
            y_data = data[self.activity_col]

            self.task_type = _get_task_type(data, self.activity_col)
            self.cv = _get_cv_strategy(
                self.task_type, n_splits=self.n_splits, n_repeats=self.n_repeats
            )
            model_map = _get_model_map(self.task_type, self.add_model, self.n_jobs)
            # Set scorings
            self.scoring_target = (
                self.scoring_target or "f1" if self.task_type == "C" else "r2"
            )
            if self.scoring_list:
                if isinstance(self.scoring_list, str):
                    self.scoring_list = [self.scoring_list]

                if self.scoring_target not in self.scoring_list:
                    self.scoring_list.append(self.scoring_target)

            if isinstance(self.select_model, list) or not self.select_model:
                if self.compare:
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
                    )

                    self.select_model = (
                        self.report.set_index(["scoring", "cv_cycle"])
                        .loc[(f"{self.scoring_target}", "mean")]
                        .idxmax()
                    )
                    self.model = model_map[self.select_model].fit(X=X_data, y=y_data)
                else:
                    raise AttributeError(
                        "'select_model' is entered as a list."
                        "To evaluate and use the best method among the entered methods, turn 'compare = True'."
                        "Otherwise, select_model must be a string as the name of the method."
                    )

            elif isinstance(self.select_model, str):
                if self.select_model not in model_map:
                    raise ValueError(f"Model '{self.select_model}' is not recognized.")
                else:
                    self.model = model_map[self.select_model].fit(X=X_data, y=y_data)
                    self.report = ModelValidation.cross_validation_report(
                        data=data,
                        activity_col=self.activity_col,
                        id_col=self.id_col,
                        add_model=self.add_model,
                        select_model=None if self.compare else self.select_model,
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

            logging.info("Model fitting completed successfully.")

            return self

        except Exception as e:
            logging.error(f"Error in model fitting: {e}")
            raise

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
        try:
            if self.model is None:
                raise NotFittedError(
                    "ModelDeveloper is not fitted yet. Call 'fit' before using this model."
                )

            logging.info("Starting predictions.")
            X_data = data.drop(
                [self.activity_col, self.id_col], axis=1, errors="ignore"
            )
            y_pred = self.model.predict(X_data)
            result = {
                "ID": data[self.id_col].values,
                "Predicted values": y_pred,
            }

            if self.task_type == "C":
                y_proba = self.model.predict_proba(X_data)
                result[f"Probability for class {self.classes_[0]}"] = y_proba[:, 0]
                result[f"Probability for class {self.classes_[1]}"] = y_proba[:, 1]

            self.pred_result = pd.DataFrame(result)

            if self.save_pred_result:
                if self.save_dir and not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir, exist_ok=True)
                self.pred_result.to_csv(
                    f"{self.save_dir}/{self.pred_result_name}.csv", index=False
                )
                logging.info(
                    f"Prediction results saved to {self.save_dir}/{self.pred_result_name}.csv"
                )

            logging.info("Predictions completed successfully.")
            return self.pred_result

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise

    def setting(self, **kwargs):
        valid_keys = self.__dict__.keys()
        for key in kwargs:
            if key not in valid_keys:
                raise KeyError(f"'{key}' is not a valid attribute of ModelDeveloper.")
        self.__dict__.update(**kwargs)

        return self

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, return the parameters of sub-estimators as well.

        Returns
        -------
        params : dict
            Dictionary of parameter names mapped to their values.
        """
        out = {}
        for key in self.__dict__:
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                for sub_key, sub_value in deep_items:
                    out[f"{key}__{sub_key}"] = sub_value
            out[key] = value

        return out

    def __repr__(self):
        """Return a string representation of the estimator."""
        class_name = self.__class__.__name__
        params = self.get_params(deep=False)
        param_str = ", ".join(f"{key}={repr(value)}" for key, value in params.items())
        return f"{class_name}({param_str})"
