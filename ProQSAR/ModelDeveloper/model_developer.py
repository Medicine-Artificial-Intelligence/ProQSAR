import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from typing import Union, Optional, List
from sklearn.model_selection import (
    cross_val_score,
    cross_validate,
    RepeatedStratifiedKFold,
    RepeatedKFold,
)
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    ElasticNetCV,
    Ridge,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
)
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    log_loss,
    brier_score_loss,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    mean_absolute_percentage_error,
    max_error,
)
import warnings

warnings.filterwarnings("ignore")


class ModelDeveloper:
    """
    A class for developing and evaluating machine learning models.

    Attributes
    ----------
    activity_col : str
        The name of the column in the dataset that represents the target variable.
    id_col : str
        The name of the column in the dataset that represents the ID or unique identifier.
    method : str, optional
        The method used to select the model. If "best", it will select the best model based on cross-validation.
    add_method : dict, optional
        A dictionary of additional models to add to the method map.
    scoring_target : str, optional
        The scoring metric used to evaluate model performance.
    n_jobs : int, optional
        The number of jobs to run in parallel for cross-validation. Default is -1 (use all processors).
    save_dir : str, optional
        Directory path to save model and results.
    comparison_report : bool, optional
        Whether to display a comparison report of models.
    comparison_visual : str, optional
        Type of visualization for model comparison ("box", "bar", or "violin").
    save_fig : bool, optional
        Whether to save the comparison plot as a file.

    Methods
    -------
    fit(data: pd.DataFrame) -> BaseEstimator
        Fits the model to the given data.
    predict(data: pd.DataFrame) -> pd.DataFrame
        Predicts values using the fitted model.
    static_predict(data: pd.DataFrame, save_dir: str) -> pd.DataFrame
        Static method to predict values using a saved model.
    internal_validation_report(
        data: pd.DataFrame,
        model: Optional[str] = None,
        scoring_list: Optional[List[str]] = None,
        save_name: Optional[str] = None
    ) -> pd.DataFrame
        Generates an internal validation report for the model.
    iv_model_comparison_report(
        data: pd.DataFrame,
        select_model: Optional[List[str]] = None,
        save_name: Optional[str] = None
    ) -> pd.DataFrame
        Generates a report comparing multiple models based on internal validation.
    external_validation_report(
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        select_model: Optional[List[str]] = None,
        scoring_list: Optional[List[str]] = None,
        save_name: Optional[str] = None
    ) -> pd.DataFrame
        Generates a report comparing multiple models based on external validation.
    """

    def __init__(
        self,
        activity_col: str,
        id_col: str,
        method: str = "best",
        add_method: Optional[dict] = None,
        scoring_target: Optional[str] = None,
        n_jobs: int = -1,
        save_dir: Optional[str] = None,
        comparison_report: bool = False,
        comparison_visual: Optional[str] = None,
        save_fig: bool = False,
    ):
        """
        Initializes the ModelDeveloper class with the given parameters.

        Parameters
        ----------
        activity_col : str
            The name of the target variable column in the dataset.
        id_col : str
            The name of the ID column in the dataset.
        method : str, optional
            The method to use for model selection ("best" or specific model name).
        add_method : dict, optional
            Additional models to add to the method map.
        scoring_target : str, optional
            The target scoring metric.
        n_jobs : int, optional
            Number of parallel jobs for cross-validation.
        save_dir : str, optional
            Directory to save models and results.
        comparison_report : bool, optional
            Whether to display a comparison report.
        comparison_visual : str, optional
            Type of visualization for model comparison.
        save_fig : bool, optional
            Whether to save the comparison plot.
        """
        self.activity_col = activity_col
        self.id_col = id_col
        self.method = method
        self.add_method = add_method
        self.scoring_target = scoring_target
        self.n_jobs = n_jobs
        self.save_dir = save_dir
        self.comparison_report = comparison_report
        self.comparison_visual = comparison_visual
        self.save_fig = save_fig
        self.model = None
        self.task_type = None
        self.cv = None
        self.method_map = {}
        self.class_method_map = {
            "Logistic": LogisticRegression(
                max_iter=10000, solver="liblinear", random_state=42
            ),
            "KNN": KNeighborsClassifier(n_neighbors=20),
            "SVM": SVC(probability=True, max_iter=10000),
            "RF": RandomForestClassifier(random_state=42),
            "ExT": ExtraTreesClassifier(random_state=42),
            "Ada": AdaBoostClassifier(n_estimators=100, random_state=42),
            "Grad": GradientBoostingClassifier(random_state=42),
            "XGB": XGBClassifier(random_state=42, verbosity=0, eval_metric="logloss"),
            "CatB": CatBoostClassifier(random_state=42, verbose=0),
            "MLP": MLPClassifier(
                alpha=0.01, max_iter=10000, hidden_layer_sizes=(150,), random_state=42
            ),
        }
        self.reg_method_map = {
            "Linear": LinearRegression(),
            "KNN": KNeighborsRegressor(),
            "SVM": SVR(),
            "RF": RandomForestRegressor(random_state=42),
            "ExT": ExtraTreesRegressor(random_state=42),
            "Ada": AdaBoostRegressor(random_state=42),
            "Grad": GradientBoostingRegressor(random_state=42),
            "XGB": XGBRegressor(
                random_state=42, verbosity=0, objective="reg:squarederror"
            ),
            "CatB": CatBoostRegressor(random_state=42, verbose=0),
            "MLP": MLPRegressor(
                alpha=0.01, max_iter=10000, hidden_layer_sizes=(150,), random_state=42
            ),
            "Ridge": Ridge(),
            "ElasticNet": ElasticNetCV(cv=5),
        }

        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    def _determine_task_type(self, data: pd.DataFrame) -> str:
        """
        Determines the task type (classification or regression) based on the target variable.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset containing the target variable.

        Returns
        -------
        str
            The type of task: "C" for classification or "R" for regression.

        Raises
        ------
        ValueError
            If the number of unique categories in the target variable is insufficient.
        """
        y_data = data[self.activity_col]
        unique_targets = len(np.unique(y_data))
        if unique_targets == 2:
            return "C"
        elif unique_targets > 2:
            return "R"
        else:
            raise ValueError(
                "Insufficient number of categories to determine model type."
            )

    def _get_method_map(self, task_type: str) -> dict[str, BaseEstimator]:
        """
        Gets the method map based on the task type.

        Parameters
        ----------
        task_type : str
            The type of task ("C" or "R").

        Returns
        -------
        dict[str, BaseEstimator]
            The method map for the task type.
        """
        if task_type == "C":
            self.method_map = self.class_method_map
        else:
            self.method_map = self.reg_method_map

        if self.add_method:
            self.method_map.update(self.add_method)

        return self.method_map

    def _determine_cv_strategy(
        self, task_type: str
    ) -> Union[RepeatedStratifiedKFold, RepeatedKFold]:
        """
        Determines the cross-validation strategy based on the task type.

        Parameters
        ----------
        task_type : str
            The type of task ("C" or "R").

        Returns
        -------
        Union[RepeatedStratifiedKFold, RepeatedKFold]
            The cross-validation strategy.
        """
        if task_type == "C":
            return RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
        else:
            return RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)

    def fit(self, data: pd.DataFrame) -> BaseEstimator:
        """
        Fits the model using the provided dataset.

        Parameters:
            data (pd.DataFrame): The dataset including features and target column.

        Returns:
            BaseEstimator: The trained model.

        Raises:
            ValueError: If the specified method is not recognized.
        """
        X_data = data.drop([self.activity_col, self.id_col], axis=1)
        y_data = data[self.activity_col]

        self.task_type = self._determine_task_type(data)
        self.method_map = self._get_method_map(self.task_type)
        self.cv = self._determine_cv_strategy(self.task_type)
        if self.add_method is not None:
            self.method_map.update(self.add_method)

        if self.method == "best":
            self.method = self._best_method(data)
            self.model = self.method_map[self.method].fit(X=X_data, y=y_data)
        elif self.method in self.method_map:
            self.model = self.method_map[self.method].fit(X=X_data, y=y_data)
        else:
            raise ValueError(f"Method '{self.method}' is not recognized.")

        if self.save_dir:
            with open(f"{self.save_dir}/activity_col.pkl", "wb") as file:
                pickle.dump(self.activity_col, file)
            with open(f"{self.save_dir}/id_col.pkl", "wb") as file:
                pickle.dump(self.id_col, file)
            with open(f"{self.save_dir}/model.pkl", "wb") as file:
                pickle.dump(self.model, file)
            with open(f"{self.save_dir}/task_type.pkl", "wb") as file:
                pickle.dump(self.task_type, file)

        return self.model

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Makes predictions using the trained model.

        Parameters:
            data (pd.DataFrame): The dataset including features and ID column.

        Returns:
            pd.DataFrame: A DataFrame with IDs and predicted values. Includes probabilities for classification tasks.

        Raises:
            NotFittedError: If the model has not been trained yet.
        """
        if self.model is None:
            raise NotFittedError("This ModelDeveloper instance is not fitted yet.")

        X_data = data.drop([self.activity_col, self.id_col], axis=1)
        y_pred = self.model.predict(X_data)
        result = {
            "ID": data[self.id_col].values,
            "Predicted values": y_pred,
        }

        if self.task_type == "C":
            y_proba = self.model.predict_proba(X_data)[:, 1] * 100
            result["Probability"] = np.round(y_proba, 2)

        self.pred_result = pd.DataFrame(result)
        if self.save_dir:
            self.pred_result.to_csv(f"{self.save_dir}/pred_result.csv")

        return self.pred_result

    @staticmethod
    def static_predict(data: pd.DataFrame, save_dir: str) -> pd.DataFrame:
        """
        Makes predictions using a previously saved model.

        Parameters:
            data (pd.DataFrame): The dataset including features and ID column.
            save_dir (str): Directory where the model and other necessary files are saved.

        Returns:
            pd.DataFrame: A DataFrame with IDs and predicted values. Includes probabilities for classification tasks.

        Raises:
            NotFittedError: If the necessary files are not found in the specified directory.
        """
        if not os.path.exists(f"{save_dir}/feature_selector.pkl"):
            raise NotFittedError(
                "The FeatureSelector instance is not fitted yet. Call 'fit' before using this method."
            )

        with open(f"{save_dir}/activity_col.pkl", "rb") as file:
            activity_col = pickle.load(file)
        with open(f"{save_dir}/id_col.pkl", "rb") as file:
            id_col = pickle.load(file)
        with open(f"{save_dir}/model.pkl", "rb") as file:
            model = pickle.load(file)
        with open(f"{save_dir}/task_type.pkl", "rb") as file:
            task_type = pickle.load(file)

        X_data = data.drop(
            [activity_col, id_col],
            axis=1,
            errors="ignore",
        )
        y_pred = model.predict(X_data)
        result = {
            "ID": data[id_col].values,
            "Predicted values": y_pred,
        }
        if task_type == "C":
            y_proba = model.predict_proba(X_data)[:, 1] * 100
            result["Probability"] = np.round(y_proba, 2)

        pred_result = pd.DataFrame(result)
        if save_dir:
            pred_result.to_csv(f"{save_dir}/pred_result.csv")

        return pred_result

    def internal_validation_report(
        self,
        data: pd.DataFrame,
        model: Optional[str] = None,
        scoring_list: Optional[List[str]] = None,
        save_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generates a report of internal validation scores for the model.

        Parameters:
            data (pd.DataFrame): The dataset including features and target column.
            model (Optional[str], optional): The model to evaluate. If None, the currently fitted model is used.
            Defaults to None.
            scoring_list (Optional[List[str]], optional): List of scoring metrics to include in the report.
            Defaults to None.
            save_name (Optional[str], optional): The name for the saved report file. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing internal validation scores.

        """
        X_data = data.drop([self.activity_col, self.id_col], axis=1)
        y_data = data[self.activity_col]

        self.task_type = self._determine_task_type(data)
        self.method_map = self._get_method_map(self.task_type)
        self.cv = self._determine_cv_strategy(self.task_type)

        if scoring_list is None:
            scoring_list = self._get_iv_scoring_list(self.task_type)
        else:
            scoring_list = scoring_list

        if model is None:
            model = self.model
        else:
            model = self.method_map[model]

        scores = cross_validate(
            model, X_data, y_data, scoring=scoring_list, cv=self.cv, n_jobs=self.n_jobs
        )

        iv_score = {}
        for metric in scoring_list:
            metric_scores = scores[f"test_{metric}"]
            iv_score[metric] = {
                "Mean": round(np.mean(metric_scores), 3),
                "Std": round(np.std(metric_scores), 3),
                "Median": round(np.median(metric_scores), 3),
            }

            for i, score in enumerate(metric_scores, 1):
                iv_score[metric][f"Score_{i}"] = score

        self.internal_validation_report = pd.DataFrame(iv_score)

        if self.save_dir:
            save_name = "internal_validation_report" if save_name is None else save_name
            self.internal_validation_report.to_csv(f"{self.save_dir}/{save_name}.csv")

        return self.internal_validation_report

    def _get_iv_scoring_list(self, task_type: str) -> str:
        """
        Determines the scoring target for model selection based on the task type.

        Parameters:
            task_type (str): The type of task ('C' for classification or 'R' for regression).

        Returns:
            str: The scoring target metric.
        """
        if task_type == "C":
            return [
                "roc_auc",
                "average_precision",
                "accuracy",
                "recall",
                "precision",
                "f1",
                "neg_log_loss",
                "neg_brier_score",
            ]
        else:
            return [
                "r2",
                "neg_mean_squared_error",
                "neg_root_mean_squared_error",
                "neg_mean_absolute_error",
                "neg_median_absolute_error",
                "neg_mean_absolute_percentage_error",
                "max_error",
            ]

    def _get_scoring_target(self, task_type: str) -> str:
        """
        Determines the scoring target for model selection based on the task type.

        Parameters:
            task_type (str): The type of task ('C' for classification or 'R' for regression).

        Returns:
            str: The scoring target metric.
        """
        if task_type == "C":
            return "f1" if self.scoring_target is None else self.scoring_target
        else:
            return "r2" if self.scoring_target is None else self.scoring_target

    def iv_model_comparison_report(
        self,
        data: pd.DataFrame,
        select_model: Optional[List] = None,
        save_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generates a report comparing the performance of different models using internal validation.

        Parameters:
            data (pd.DataFrame): The dataset including features and target column.
            select_model (Optional[List[str]], optional): List of models to compare. Defaults to None.
            save_name (Optional[str], optional): The name for the saved report file. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing model comparison results.

        """
        X_data = data.drop([self.activity_col, self.id_col], axis=1)
        y_data = data[self.activity_col]

        self.task_type = self._determine_task_type(data)
        self.method_map = self._get_method_map(self.task_type)
        self.cv = self._determine_cv_strategy(self.task_type)
        self.scoring_target = self._get_scoring_target(self.task_type)

        comparison_result = []
        models_to_compare = {}

        if select_model is None:
            models_to_compare = self.method_map
        else:
            for name in select_model:
                if name in self.method_map:
                    models_to_compare.update({name: self.method_map[name]})
                else:
                    raise ValueError(f"Method '{name}' is not recognized.")

        for name, model in models_to_compare.items():
            scores = cross_val_score(
                model,
                X_data,
                y_data,
                cv=self.cv,
                scoring=self.scoring_target,
                n_jobs=self.n_jobs,
            )
            method_result = {
                "Method": name,
                "Mean": round(np.mean(scores), 3),
                "Std": round(np.std(scores), 3),
                "Median": round(np.median(scores), 3),
            }
            for i, score in enumerate(scores):
                method_result[f"Score_{i+1}"] = score
            comparison_result.append(method_result)

        self.iv_comparison_report = pd.DataFrame(comparison_result)

        if self.comparison_report:
            display(self.iv_comparison_report)

        if self.comparison_visual:
            self._plot_compare_models()

        if self.save_dir:
            save_name = (
                f"iv_model_comparison_{self.scoring_target}"
                if save_name is None
                else save_name
            )
            self.iv_comparison_report.to_csv(f"{self.save_dir}/{save_name}.csv")

        return self.iv_comparison_report

    def _plot_compare_models(self) -> None:
        """
        Plots the comparison of model performance based on internal validation scores.

        Raises:
            ValueError: If an invalid comparison visualization type is specified.
        """
        sns.set_style("whitegrid")
        plt.figure(figsize=(20, 10))

        score_columns = [
            col for col in self.iv_comparison_report.columns if col.startswith("Score_")
        ]
        melted_result = self.iv_comparison_report.melt(
            id_vars=["Method"],
            value_vars=score_columns,
            var_name="Score",
            value_name="Value",
        )

        if self.comparison_visual == "box":
            plot = sns.boxplot(
                x="Method",
                y="Value",
                data=melted_result,
                showmeans=True,
                width=0.5,
                palette="plasma",
                meanprops={
                    "marker": "o",
                    "markerfacecolor": "red",
                    "markeredgecolor": "black",
                },
                medianprops={"color": "red", "linewidth": 2},
                boxprops={"edgecolor": "w"},
            )
        elif self.comparison_visual == "bar":
            plot = sns.barplot(
                x="Method",
                y="Value",
                data=melted_result,
                errorbar="sd",
                palette="plasma",
                width=0.5,
                color="black",
            )
        elif self.comparison_visual == "violin":
            plot = sns.violinplot(
                x="Method", y="Value", data=melted_result, inner=None, palette="plasma"
            )
            sns.stripplot(
                x="Method",
                y="Value",
                data=melted_result,
                color="white",
                size=5,
                jitter=True,
            )
        else:
            raise ValueError(
                f"Invalid comparison_visual '{self.comparison_visual}'. Choose 'box', 'bar' or 'violin'."
            )

        plot.set_title("Compare performance of different models", fontsize=16)
        plot.set_xlabel("Model", fontsize=14)
        plot.set_ylabel(f"{self.scoring_target.capitalize()} Score", fontsize=14)

        # Adding the mean values to the plot
        for i, row in self.iv_comparison_report.iterrows():
            position = (
                0.05 if self.comparison_visual == "bar" else (row["Mean"] + 0.015)
            )
            plot.text(
                i,
                position,
                str(row["Mean"]),
                horizontalalignment="center",
                size="x-large",
                color="w",
                weight="semibold",
            )

        if self.save_fig and self.save_dir:
            plt.savefig(
                f"{self.save_dir}/iv_model_comparison_{self.scoring_target}_{self.comparison_visual}.png",
                dpi=300,
            )

        plt.show()

    def _best_method(self, data: pd.DataFrame) -> str:
        """
        Determine the best method based on internal validation report.

        Args:
            data (pd.DataFrame): The dataset for evaluation.

        Returns:
            str: The name of the best method.
        """
        result_df = self.iv_model_comparison_report(data)
        best_method = result_df.loc[result_df["Mean"].idxmax(), "Method"]
        return best_method

    def _get_ev_scoring_dict(
        self, task_type: str, y_test, y_test_pred, y_test_proba
    ) -> str:
        """
        Compute external validation scoring metrics.

        Args:
            task_type (str): The type of task ('C' for classification, 'R' for regression).
            y_test (np.ndarray): True labels.
            y_test_pred (np.ndarray): Predicted labels.
            y_test_proba (Optional[np.ndarray]): Predicted probabilities (for classification).

        Returns:
            dict: A dictionary with scoring metrics.
        """
        if task_type == "C":
            scoring_dict = {
                "roc_auc": roc_auc_score(y_test, y_test_proba),
                "average_precision": average_precision_score(y_test, y_test_proba),
                "accuracy": accuracy_score(y_test, y_test_pred),
                "recall": recall_score(y_test, y_test_pred),
                "precision": precision_score(y_test, y_test_pred),
                "f1": f1_score(y_test, y_test_pred, average="binary"),
                "log_loss": log_loss(y_test, y_test_proba),
                "brier_score": brier_score_loss(y_test, y_test_proba),
            }

        else:
            scoring_dict = {
                "r2": r2_score(y_test, y_test_pred),
                "mean_squared_error": mean_squared_error(y_test, y_test_pred),
                "root_mean_squared_error": mean_squared_error(
                    y_test, y_test_pred, squared=False
                ),
                "mean_absolute_error": mean_absolute_error(y_test, y_test_pred),
                "median_absolute_error": median_absolute_error(y_test, y_test_pred),
                "mean_absolute_percentage_error": mean_absolute_percentage_error(
                    y_test, y_test_pred
                ),
                "max_error": max_error(y_test, y_test_pred),
            }

        return scoring_dict

    def external_validation_report(
        self,
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        select_model: Optional[List[str]] = None,
        scoring_list: Optional[List[str]] = None,
        save_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate an external validation report for selected models.

        Args:
            data_train (pd.DataFrame): Training dataset.
            data_test (pd.DataFrame): Testing dataset.
            select_model (Optional[List[str]]): List of model names to evaluate. If None, evaluate all models.
            scoring_list (Optional[List[str]]): List of metrics to include in the report.
            save_name (Optional[str]): The name for saving the report.

        Returns:
            pd.DataFrame: DataFrame containing the external validation report.
        """
        X_train = data_train.drop([self.activity_col, self.id_col], axis=1)
        y_train = data_train[self.activity_col]
        X_test = data_test.drop([self.activity_col, self.id_col], axis=1)
        y_test = data_test[self.activity_col]

        self.task_type = self._determine_task_type(data_train)
        self.method_map = self._get_method_map(self.task_type)

        models_to_compare = {}
        if select_model is None:
            models_to_compare = self.method_map
        else:
            for name in select_model:
                if name in self.method_map:
                    models_to_compare.update({name: self.method_map[name]})
                else:
                    raise ValueError(f"Method '{name}' is not recognized.")

        ev_score = {}
        for name, model in models_to_compare.items():
            model.fit(X=X_train, y=y_train)
            y_test_pred = model.predict(X_test)
            y_test_proba = (
                model.predict_proba(X_test)[:, 1] if self.task_type == "C" else None
            )

            scoring_dict = self._get_ev_scoring_dict(
                self.task_type, y_test, y_test_pred, y_test_proba
            )

            if scoring_list is None:
                ev_score[name] = scoring_dict
            else:
                ev_score[name] = {}
                for metric in scoring_list:
                    if metric in scoring_dict:
                        ev_score[name].update({metric: scoring_dict[metric]})
                    else:
                        raise ValueError(f"'{metric}' is not recognized.")

        self.external_validation_report = pd.DataFrame(ev_score).T

        if self.save_dir:
            save_name = "internal_validation_report" if save_name is None else save_name
            self.external_validation_report.to_csv(f"{self.save_dir}/{save_name}.csv")

        return self.external_validation_report
