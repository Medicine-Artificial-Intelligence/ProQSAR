import os
import pickle
import logging
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from typing import Optional, Union, List
from crepes import WrapClassifier, WrapRegressor
from ProQSAR.ModelDeveloper.model_developer import ModelDeveloper
from ProQSAR.ModelDeveloper.model_developer_utils import _get_task_type


class ConformalPredictor:
    """
    A class to perform conformal prediction for classification and regression models.
    """

    def __init__(
        self,
        model: Union[ModelDeveloper, BaseEstimator],
        activity_col: str,
        id_col: str,
        save_dir: Optional[str] = "Project/ConformalPredictor",
    ) -> None:
        """
        Initialize the ConformalPredictor class.

        Args:
            model (Union[ModelDeveloper, object]): The model to be used for conformal prediction.
            activity_col (str): The target variable column name.
            id_col (str): The identifier column name.
            save_dir (Optional[str]): Directory to save calibration and prediction results.
        """
        self.model = model.model if isinstance(model, ModelDeveloper) else model
        self.activity_col = activity_col
        self.id_col = id_col
        self.save_dir = save_dir
        self.task_type = None
        self.cp = None
        self.pred_result = None
        self.evaluate_result = None

    def calibrate(self, data, **kwargs) -> "ConformalPredictor":
        """
        Calibrate the conformal predictor using the provided dataset.

        Args:
            data (pd.DataFrame): Training dataset for calibration.
            **kwargs: Additional parameters for calibration.

        Returns:
            ConformalPredictor: The instance itself after calibration.
        """
        try:
            # check if model is fitted or not
            check_is_fitted(self.model)

            # get X & y
            X_data = data.drop(
                [self.activity_col, self.id_col], axis=1, errors="ignore"
            )
            y_data = data[self.activity_col]

            # get task_type
            self.task_type = _get_task_type(data, self.activity_col)

            if self.task_type == "C":
                self.cp = WrapClassifier(self.model)
            elif self.task_type == "R":
                self.cp = WrapRegressor(self.model)
            else:
                raise ValueError("Unsupported task type detected.")

            self.cp.calibrate(X=X_data, y=y_data, **kwargs)

            if self.save_dir:
                os.makedirs(self.save_dir, exist_ok=True)
                with open(f"{self.save_dir}/conformal_predictor.pkl", "wb") as file:
                    pickle.dump(self, file)

            logging.info("Conformal predictor calibrated successfully.")
            return self

        except Exception as e:
            logging.error(f"Error during calibration: {e}")
            raise

    def predict(
        self, data: pd.DataFrame, confidence: float = 0.95, **kwargs
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

            if self.task_type == "C":
                classes = self.model.classes_

                # predict p
                self.p_values = self.cp.predict_p(X_data, **kwargs)

                # predict set
                self.sets = self.cp.predict_set(
                    X=X_data, confidence=confidence, **kwargs
                )

                predicted_labels = []
                for i in range(len(self.sets)):
                    present_labels = [
                        str(classes[j])
                        for j in range(len(classes))
                        if self.sets[i][j] == 1
                    ]
                    predicted_labels.append(", ".join(present_labels))

                # return result in dataframe format
                self.pred_result = pd.DataFrame(
                    {
                        "ID": data[self.id_col].values,
                        "Predicted set": predicted_labels,
                        f"P-value for class {classes[0]}": self.p_values[:, 0],
                        f"P-value for class {classes[1]}": self.p_values[:, 1],
                    }
                )

            elif self.task_type == "R":
                y_pred = self.model.predict(X_data)

                self.interval = self.cp.predict_int(X=X_data, **kwargs)

                self.pred_result = pd.DataFrame(
                    {
                        "ID": data[self.id_col].values,
                        "Predicted values": y_pred,
                        "Lower Bound": self.interval[:, 0],
                        "Upper Bound": self.interval[:, 1],
                    }
                )

            if self.activity_col in data.columns:
                self.pred_result["Actual values"] = data[self.activity_col]

            if self.save_dir:
                os.makedirs(self.save_dir, exist_ok=True)
                self.pred_result.to_csv(f"{self.save_dir}/conformal_pred_result.csv")

            logging.info("Prediction completed successfully.")
            return self.pred_result

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise

    def evaluate(
        self, data: pd.DataFrame, confidence: Union[float, List[float]] = 0.95, **kwargs
    ) -> pd.DataFrame:
        """
        Evaluate the conformal predictor performance on a dataset.

        Args:
            data (pd.DataFrame): Dataset for evaluation.
            confidence (Union[float, List[float]]): Confidence levels to evaluate.
            **kwargs: Additional parameters for evaluation.

        Returns:
            pd.DataFrame: DataFrame containing evaluation results.
        """
        if self.activity_col not in data.columns:
            raise KeyError(
                f"'{self.activity_col}' column is not found in the provided data. "
                "Please ensure that the data contains this column in order to use this function."
            )

        if isinstance(confidence, float):
            confidence = [confidence]

        try:
            X_data = data.drop(
                [self.activity_col, self.id_col], axis=1, errors="ignore"
            )
            y_data = data[self.activity_col]

            result = {
                conf: self.cp.evaluate(X=X_data, y=y_data, confidence=conf, **kwargs)
                for conf in confidence
            }

            self.evaluate_result = pd.DataFrame(result)

            if self.save_dir:
                os.makedirs(self.save_dir, exist_ok=True)
                self.evaluate_result.to_csv(
                    f"{self.save_dir}/conformal_evaluate_result.csv"
                )

            logging.info("Evaluation completed successfully.")
            return self.evaluate_result

        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            raise
