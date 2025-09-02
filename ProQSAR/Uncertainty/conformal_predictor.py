import os
import pickle
import logging
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from typing import Optional, Union, Iterable
from ProQSAR.ModelDeveloper.model_developer_utils import _get_task_type
from ProQSAR.ModelDeveloper.model_developer import ModelDeveloper
from mapie.regression import MapieRegressor
from mapie.classification import MapieClassifier


class ConformalPredictor(BaseEstimator):
    """
    Calibrate and query conformal predictors using MAPIE.

    Parameters
    ----------
    model : Optional[Union[ModelDeveloper, BaseEstimator]]
        The base estimator to wrap. May be an sklearn-compatible estimator or
        a ProQSAR ModelDeveloper instance (in which case the underlying
        estimator is extracted).
    activity_col : str, optional
        Column name of the target/label in input DataFrames (default "activity").
    id_col : str, optional
        Column name of the identifier in input DataFrames (default "id").
    n_jobs : int, optional
        Number of parallel jobs for MAPIE when supported (default 1).
    random_state : Optional[int], optional
        Random seed passed to MAPIE and possibly the estimator (default 42).
    save_dir : Optional[str], optional
        Directory to save fitted ConformalPredictor and results (default None).
    deactivate : bool, optional
        If True, ConformalPredictor becomes a no-op (fit/predict return early).
    **kwargs : dict
        Additional keyword arguments passed to the MAPIE estimator (e.g., `method`, `cv`).

    Attributes
    ----------
    model : BaseEstimator | None
        The underlying estimator used by MAPIE (extracted from ModelDeveloper if provided).
    cp : MapieClassifier | MapieRegressor | None
        The fitted MAPIE wrapper after calling `fit`.
    task_type : Optional[str]
        'C' for classification or 'R' for regression (inferred during fit).
    cp_kwargs : dict
        Additional kwargs passed to the MAPIE wrapper at `set_params`.
    """

    def __init__(
        self,
        model: Optional[Union[ModelDeveloper, BaseEstimator]] = None,
        activity_col: str = "activity",
        id_col: str = "id",
        n_jobs: int = 1,
        random_state: Optional[int] = 42,
        save_dir: Optional[str] = None,
        deactivate: bool = False,
        **kwargs,
    ) -> None:

        self.model = model
        self.activity_col = activity_col
        self.id_col = id_col
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.save_dir = save_dir
        self.deactivate = deactivate
        self.cp_kwargs = kwargs
        self.task_type = None
        self.cp_kwargs = kwargs
        self.cp = None

    def fit(self, data: pd.DataFrame) -> "ConformalPredictor":
        """
        Fit (calibrate) the MAPIE conformal predictor on the provided dataset.

        If `model` is an instance of ProQSAR's ModelDeveloper, its fitted
        estimator (`model.model`) is used. The method determines whether the
        task is classification or regression and uses MapieClassifier or
        MapieRegressor accordingly.

        The fitted ConformalPredictor stores MAPIE's `conformity_scores_`
        as float64 for consistency and optionally saves the fitted object
        to disk.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing features and the activity/id columns named
            according to `activity_col` and `id_col`.

        Returns
        -------
        ConformalPredictor
            The fitted ConformalPredictor instance.

        Raises
        ------
        Exception
            Any unexpected errors are logged and re-raised.
        """

        if self.deactivate:
            logging.info("ConformalPredictor is deactivated. Skipping calibrate.")
            return self

        if isinstance(self.model, ModelDeveloper):
            self.model = self.model.model

        try:
            # get X & y
            X_data = data.drop(
                [self.activity_col, self.id_col], axis=1, errors="ignore"
            )
            y_data = data[self.activity_col]

            # get task_type
            self.task_type = _get_task_type(data, self.activity_col)

            if self.task_type == "C":
                self.cp = MapieClassifier()
            elif self.task_type == "R":
                self.cp = MapieRegressor()
            else:
                raise ValueError("Unsupported task type detected.")

            self.cp.set_params(
                estimator=self.model,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                **self.cp_kwargs,
            )
            self.cp.fit(X=X_data, y=y_data)
            self.cp.conformity_scores_ = self.cp.conformity_scores_.astype(np.float64)

            logging.info(
                f"ConformalPredictor: Fitted a MAPIE {'Classifier' if self.task_type == 'C' else 'Regressor'}."
            )

            if self.save_dir:
                os.makedirs(self.save_dir, exist_ok=True)
                with open(f"{self.save_dir}/conformal_predictor.pkl", "wb") as file:
                    pickle.dump(self, file)

            return self

        except Exception as e:
            logging.error(f"Error during calibration: {e}")
            raise

    def predict(
        self,
        data: pd.DataFrame,
        alpha: Optional[Union[float, Iterable[float]]] = None,
    ) -> pd.DataFrame:
        """
        Produce conformal predictions for the input DataFrame.

        For classification, MAPIE returns prediction sets (arrays indicating
        membership of classes for each alpha). For regression, MAPIE returns
        prediction intervals. This method organizes MAPIE outputs into a
        readable pandas DataFrame where each row corresponds to an input sample,
        with columns for ID, actual value (if available), point prediction and
        the prediction set / interval for each alpha supplied.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing features and optional id/activity columns.
        alpha : float or iterable of float, optional
            Significance level(s) for conformal prediction (e.g., 0.1 for 90%
            prediction sets/intervals). If a single float is provided it is
            converted to a list internally. If None, only point predictions
            are returned (behavior depends on MAPIE's predict signature).

        Returns
        -------
        pd.DataFrame
            DataFrame with prediction results. Columns include:
              - id_col (if present)
              - activity_col (if present)
              - 'Predicted value'
              - 'Prediction Set (alpha=...)' or 'Prediction Interval (alpha=...)'

        Raises
        ------
        NotFittedError
            If called before `fit` / calibration.
        Exception
            Any unexpected errors are logged and re-raised.
        """
        # check_is_fitted(self.predictor)
        if self.cp is None:
            raise NotFittedError(
                "ConformalPredictor is not calibrated yet. Call 'calibrate' before using this function."
            )
        try:
            X_data = data.drop(
                [self.activity_col, self.id_col], axis=1, errors="ignore"
            )
            y_pred = self.cp.predict(X=X_data, alpha=alpha)

            results_list = []
            if isinstance(alpha, float):
                alpha = [alpha]

            for i in range(len(X_data)):
                # Get ID
                sample_data = {self.id_col: data[self.id_col].iloc[i]}

                # Get actual value if available
                if self.activity_col in data.columns:
                    sample_data[self.activity_col] = data[self.activity_col].iloc[i]

                if alpha:
                    sample_data["Predicted value"] = y_pred[0][i]
                    sample_cal = y_pred[1][i, :, :]
                    for k, a in enumerate(alpha):
                        if self.task_type == "C":
                            class_labels = self.cp.classes_
                            set_indices = np.where(sample_cal[:, k])[0]
                            result = class_labels[set_indices]

                        elif self.task_type == "R":
                            result = np.round(sample_cal[:, k], decimals=3)

                        sample_data[
                            (
                                f"Prediction Set (alpha={a})"
                                if self.task_type == "C"
                                else f"Prediction Interval (alpha={a})"
                            )
                        ] = result
                else:
                    sample_data["Predicted value"] = y_pred[i]

                results_list.append(sample_data)

            pred_result = pd.DataFrame(results_list)

            if self.save_dir:
                os.makedirs(self.save_dir, exist_ok=True)
                pred_result.to_csv(
                    f"{self.save_dir}/conformal_pred_result.csv", index=False
                )

            return pred_result

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise

    def set_params(self, **kwargs):
        """
        Set attributes on the ConformalPredictor. Only keys already present in
        the instance's __dict__ are accepted; attempting to set unknown keys
        raises a KeyError.

        Returns
        -------
        ConformalPredictor
            The updated instance (self).

        Raises
        ------
        KeyError
            If an unknown attribute name is passed in kwargs.
        """
        valid_keys = self.__dict__.keys()
        for key in kwargs:
            if key not in valid_keys:
                raise KeyError(f"'{key}' is not a valid attribute.")
        self.__dict__.update(**kwargs)

        return self
