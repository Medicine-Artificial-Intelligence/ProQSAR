import os
import pickle
import pandas as pd
from copy import deepcopy
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from typing import Optional
from crepes import WrapClassifier, WrapRegressor
from ProQSAR.ModelDeveloper.model_developer import ModelDeveloper
from ProQSAR.ModelDeveloper.model_developer_utils import _get_task_type


class ConformalPredictor:
    def __init__(
            self, 
            model, 
            activity_col, 
            id_col,
            save_dir: Optional[str] = "Project/ConformalPredictor"
            ):
        self.model = model.model if isinstance(model, ModelDeveloper) else model
        self.activity_col = activity_col
        self.id_col = id_col
        self.save_dir = save_dir
        self.task_type = None
        self.cp = None
        self.pred_result = None
        self.evaluate_result = None

    def calibrate(self, data, **kwargs):

        # check if model is fitted or not
        check_is_fitted(self.model)

        # get X & y
        X_data = data.drop([self.activity_col, self.id_col], axis=1, errors='ignore')
        y_data = data[self.activity_col]

        # get task_type
        self.task_type = _get_task_type(data, self.activity_col)

        if self.task_type == 'C':
            self.cp = WrapClassifier(self.model)
        elif self.task_type == 'R':
            self.cp = WrapRegressor(self.model)

        self.cp.calibrate(X=X_data, y=y_data, **kwargs)

        if self.save_dir:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir, exist_ok=True)
            with open(f"{self.save_dir}/conformal_predictor.pkl", "wb") as file:
                pickle.dump(self, file)

        return self

    def predict(self, data, confidence=0.95, **kwargs):
        #check_is_fitted(self.predictor)
        if self.cp is None:
            raise NotFittedError(
                "ConformalPredictor is not calibrated yet. Call 'calibrate' before using this function."
            )

        X_data = data.drop([self.activity_col, self.id_col], axis=1, errors='ignore')

        if self.task_type == 'C':
            classes = self.model.classes_

            # predict p
            self.p_values = self.cp.predict_p(X_data, **kwargs)

            # predict set
            self.sets = self.cp.predict_set(X=X_data, confidence=confidence, **kwargs)

            predicted_labels = []
            for i in range(len(self.sets)):
                present_labels = [str(classes[j]) for j in range(len(classes)) if self.sets[i][j] == 1]
                predicted_labels.append(", ".join(present_labels))

            # return result in dataframe format
            self.pred_result = pd.DataFrame({
                'ID': data[self.id_col].values,
                'Predicted set': predicted_labels,
                f'P-value for class {classes[0]}': self.p_values[:, 0],
                f'P-value for class {classes[1]}': self.p_values[:, 1]
            })


        elif self.task_type == 'R':
            y_pred = self.model.predict(X_data)

            interval = self.cp.predict_int(X=X_data, **kwargs)

            self.pred_result = pd.DataFrame({
                'ID': data[self.id_col].values,
                'Predicted values': y_pred,
                'Lower Bound': interval[:, 0],
                'Upper Bound': interval[:, 1]
            })
        
        if self.activity_col in data.columns:
            self.pred_result['Actual values'] = data[self.activity_col]

        if self.save_dir:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir, exist_ok=True)
            self.pred_result.to_csv(f"{self.save_dir}/conformal_pred_result.csv")

        return self.pred_result
    
    def evaluate(self, data, confidence=0.95, **kwargs):
        if not self.activity_col in data.columns:
            raise KeyError(f"'{self.activity_col}' column is not found in the provided data. "
                            "Please ensure that the data contains this column in order to use this function.")

        if isinstance(confidence, float):
            confidence = [confidence]

        X_data = data.drop([self.activity_col, self.id_col], axis=1, errors='ignore')
        y_data = data[self.activity_col]

        result = {}
        for i in confidence:
            result.update({i: self.cp.evaluate(X=X_data, y=y_data, confidence=i, **kwargs)})

        self.evaluate_result = pd.DataFrame(result)

        if self.save_dir:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir, exist_ok=True)
            self.evaluate_result.to_csv(f"{self.save_dir}/conformal_evaluate_result.csv")

        return self.evaluate_result


