import os
import pickle
import logging
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from typing import Optional, Union, Iterable
from proqsar.Model.ModelDeveloper.model_developer_utils import _get_task_type
from proqsar.Model.ModelDeveloper.model_developer import ModelDeveloper
from mapie.regression import MapieRegressor
from mapie.classification import MapieClassifier


class ConformalPredictor(BaseEstimator):
    """
    Calibrate and query conformal predictors using MAPIE.

    Wraps MAPIE's `MapieClassifier` or `MapieRegressor` depending on whether the
    task is classification or regression. Supports integration with ProQSAR's
    `ModelDeveloper` or any scikit-learn compatible estimator.

    :param model: Base estimator or a ProQSAR `ModelDeveloper`. If `ModelDeveloper`
                  is provided, its internal estimator is used.
    :type model: Optional[Union[ModelDeveloper, BaseEstimator]]
    :param activity_col: Column name for the target variable.
    :type activity_col: str, optional
    :param id_col: Column name for unique identifiers.
    :type id_col: str, optional
    :param n_jobs: Number of parallel jobs for MAPIE (where supported).
    :type n_jobs: int, optional
    :param random_state: Random seed for reproducibility.
    :type random_state: Optional[int], optional
    :param save_dir: Directory to save fitted predictor and results.
    :type save_dir: Optional[str], optional
    :param deactivate: If True, disables calibration and prediction (no-op).
    :type deactivate: bool, optional
    :param kwargs: Extra keyword arguments forwarded to MAPIE (e.g., `method`, `cv`).
    :type kwargs: dict

    :ivar model: Underlying estimator after initialization.
    :vartype model: BaseEstimator or None
    :ivar cp: Fitted MAPIE wrapper (classifier or regressor).
    :vartype cp: MapieClassifier or MapieRegressor or None
    :ivar task_type: 'C' for classification, 'R' for regression.
    :vartype task_type: Optional[str]
    :ivar cp_kwargs: Additional keyword arguments passed to MAPIE.
    :vartype cp_kwargs: dict
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
        self.cp = None

    def fit(self, data: pd.DataFrame) -> "ConformalPredictor":
        """
        Fit (calibrate) the MAPIE conformal predictor.

        Determines task type automatically and selects either
        `MapieClassifier` (classification) or `MapieRegressor` (regression).
        If a ProQSAR `ModelDeveloper` is provided as the base model, its
        underlying estimator is extracted.

        :param data: Training dataset containing features and target column.
        :type data: pandas.DataFrame
        :return: Self with fitted MAPIE conformal predictor.
        :rtype: ConformalPredictor
        :raises Exception: Logs and re-raises unexpected errors.

        .. note::
           Fitted predictor stores conformity scores as `float64` for consistency.
        """
        if self.deactivate:
            logging.info("ConformalPredictor is deactivated. Skipping calibrate.")
            return self

        if isinstance(self.model, ModelDeveloper):
            self.model = self.model.model

        try:
            X_data = data.drop(
                [self.activity_col, self.id_col], axis=1, errors="ignore"
            )
            y_data = data[self.activity_col]

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
        Generate conformal predictions for new samples.

        For regression, returns prediction intervals. For classification,
        returns prediction sets. If ``alpha`` is None, only point predictions
        are returned.

        :param data: Dataset with features and optional ID/target columns.
        :type data: pandas.DataFrame
        :param alpha: Significance level(s) for conformal prediction. Single
                      float or iterable of floats (e.g., 0.1 for 90% intervals).
        :type alpha: Optional[Union[float, Iterable[float]]]
        :return: DataFrame containing predictions, IDs, true values (if present),
                 and conformal sets/intervals for each ``alpha``.
        :rtype: pandas.DataFrame
        :raises NotFittedError: If called before calibration with `fit`.
        :raises Exception: Logs and re-raises unexpected errors.

        .. code-block:: python

            cp = ConformalPredictor(model=some_model)
            cp.fit(train_df)
            preds = cp.predict(test_df, alpha=0.1)
            print(preds.head())
        """
        if self.cp is None:
            raise NotFittedError(
                "ConformalPredictor is not calibrated yet. Call 'fit' before prediction."
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
                sample_data = {self.id_col: data[self.id_col].iloc[i]}
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
                        else:  # regression
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
        Update ConformalPredictor attributes with keyword arguments.

        Only keys already present in the instance are accepted.

        :param kwargs: Attributes to update.
        :type kwargs: dict
        :return: Updated instance.
        :rtype: ConformalPredictor
        :raises KeyError: If an unknown attribute is provided.
        """
        valid_keys = self.__dict__.keys()
        for key in kwargs:
            if key not in valid_keys:
                raise KeyError(f"'{key}' is not a valid attribute.")
        self.__dict__.update(**kwargs)
        return self
