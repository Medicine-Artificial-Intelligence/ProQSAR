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
    A class to perform conformal prediction for classification and regression models.
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
        """
        Initialize the ConformalPredictor class.

        Args:
            model (Union[ModelDeveloper, object]): The model to be used for conformal prediction.
            activity_col (str): The target variable column name.
            id_col (str): The identifier column name.
            save_dir (Optional[str]): Directory to save calibration and prediction results.
        """
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

    def fit(self, data: pd.DataFrame):

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
        Generate conformal prediction intervals or sets.

        Args:
            data (pd.DataFrame): Dataset for making predictions.
            confidence (float): Confidence level for prediction intervals.
            **kwargs: Additional parameters for prediction.

        Returns:
            pd.DataFrame: DataFrame containing prediction results.
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
        valid_keys = self.__dict__.keys()
        for key in kwargs:
            if key not in valid_keys:
                raise KeyError(f"'{key}' is not a valid attribute.")
        self.__dict__.update(**kwargs)

        return self
