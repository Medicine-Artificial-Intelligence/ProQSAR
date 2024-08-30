import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from typing import Union, Any, Optional
from sklearn.model_selection import (
    cross_val_score,
    RepeatedStratifiedKFold,
    RepeatedKFold,
)
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    mutual_info_classif,
    f_regression,
    mutual_info_regression,
    f_classif,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LassoCV
from xgboost import XGBClassifier, XGBRegressor
import warnings

warnings.filterwarnings("ignore")


class FeatureSelector:
    def __init__(
        self,
        activity_col: str,
        id_col: str,
        method: str = "best",
        scoring: Optional[str] = None,
        n_jobs: int = -1,
        save_dir: Optional[str] = None,
        compare_table: bool = False,
        compare_visual: Optional[str] = "box",
        save_fig: bool = False,
    ):
        """
        Initialize the FeatureSelector object.

        Parameters:
        ----------
        activity_col : str
            Name of the column containing the target variable.
        id_col : str
            Name of the column containing unique identifiers.
        method : str, optional
            The feature selection method to use, by default "best".
        scoring : Optional[str], optional
            Scoring metric for model evaluation, by default None.
        n_jobs : int, optional
            Number of jobs to run in parallel, by default -1.
        save_dir : Optional[str], optional
            Directory to save the selected features and models, by default None.
        compare_table : bool, optional
            If True, display the comparison of feature selection methods in a table, by default False.
        compare_visual : Optional[str], optional
            Visualization type for method comparison ("box" or "violin"), by default "box".
        save_fig : bool, optional
            If True, save the visualization of the comparison, by default False.
        """
        self.activity_col = activity_col
        self.id_col = id_col
        self.method = method
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.save_dir = save_dir
        self.compare_table = compare_table
        self.compare_visual = compare_visual
        self.save_fig = save_fig
        self.feature_selector: Optional[BaseEstimator] = None
        self.task_type: Optional[str] = None
        self.cv: Optional[Union[RepeatedStratifiedKFold, RepeatedKFold]] = None
        self.method_map = {
            "Anova": self._apply_anova,
            "Mutual information": self._apply_mutual_inf,
            "RF": self._apply_random_forest,
            "ExT": self._apply_extra_tree,
            "Ada": self._apply_ada_boost,
            "Grad": self._apply_gradient_boosting,
            "XGB": self._apply_xgboost,
            "LogisticRegression": self._apply_linear_method,
            "Lasso": self._apply_linear_method,
        }

        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    def _determine_task_type(self, data: pd.DataFrame) -> str:
        """
        Determine the type of task based on the target variable.

        Parameters:
        ----------
        data : pd.DataFrame
            The dataset containing the target column.

        Returns:
        -------
        str
            Task type: 'C' for Classification or 'R' for Regression.
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

    def _determine_cv_strategy(
        self, task_type: str
    ) -> Union[RepeatedStratifiedKFold, RepeatedKFold]:
        """
        Determine the cross-validation strategy based on the task type.

        Parameters:
        ----------
        task_type : str
            Task type: 'C' for Classification or 'R' for Regression.

        Returns:
        -------
        Union[RepeatedStratifiedKFold, RepeatedKFold]
            The cross-validation strategy.
        """
        if task_type == "C":
            return RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
        else:
            return RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)

    @staticmethod
    def _apply_anova(X: pd.DataFrame, y: pd.Series, task_type: str) -> BaseEstimator:
        """
        Apply ANOVA feature selection method.

        Parameters:
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
        task_type : str
            Task type: 'C' for Classification or 'R' for Regression.

        Returns:
        -------
        BaseEstimator
            The fitted feature selector.
        """
        if task_type == "C":
            return SelectKBest(score_func=f_classif, k=20).fit(X, y)
        else:
            return SelectKBest(score_func=f_regression, k=20).fit(X, y)

    @staticmethod
    def _apply_mutual_inf(
        X: pd.DataFrame, y: pd.Series, task_type: str
    ) -> BaseEstimator:
        """
        Apply Mutual Information feature selection method.

        Parameters:
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
        task_type : str
            Task type: 'C' for Classification or 'R' for Regression.

        Returns:
        -------
        BaseEstimator
            The fitted feature selector.
        """
        if task_type == "C":
            return SelectKBest(score_func=mutual_info_classif, k=20).fit(X, y)
        else:
            return SelectKBest(score_func=mutual_info_regression, k=20).fit(X, y)

    @staticmethod
    def _apply_random_forest(
        X: pd.DataFrame, y: pd.Series, task_type: str
    ) -> BaseEstimator:
        """
        Apply RandomForest feature selection method.

        Parameters:
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
        task_type : str
            Task type: 'C' for Classification or 'R' for Regression.

        Returns:
        -------
        BaseEstimator
            The fitted feature selector.
        """
        if task_type == "C":
            return SelectFromModel(RandomForestClassifier(random_state=42)).fit(X, y)
        else:
            return SelectFromModel(RandomForestRegressor(random_state=42)).fit(X, y)

    @staticmethod
    def _apply_extra_tree(
        X: pd.DataFrame, y: pd.Series, task_type: str
    ) -> BaseEstimator:
        """
        Apply ExtraTrees feature selection method.

        Parameters:
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
        task_type : str
            Task type: 'C' for Classification or 'R' for Regression.

        Returns:
        -------
        BaseEstimator
            The fitted feature selector.
        """
        if task_type == "C":
            return SelectFromModel(ExtraTreesClassifier(random_state=42)).fit(X, y)
        else:
            return SelectFromModel(ExtraTreesRegressor(random_state=42)).fit(X, y)

    @staticmethod
    def _apply_ada_boost(
        X: pd.DataFrame, y: pd.Series, task_type: str
    ) -> BaseEstimator:
        """
        Apply AdaBoost feature selection method.

        Parameters:
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
        task_type : str
            Task type: 'C' for Classification or 'R' for Regression.

        Returns:
        -------
        BaseEstimator
            The fitted feature selector.
        """
        if task_type == "C":
            return SelectFromModel(AdaBoostClassifier(random_state=42)).fit(X, y)
        else:
            return SelectFromModel(AdaBoostRegressor(random_state=42)).fit(X, y)

    @staticmethod
    def _apply_gradient_boosting(
        X: pd.DataFrame, y: pd.Series, task_type: str
    ) -> BaseEstimator:
        """
        Apply Gradient Boosting feature selection method.

        Parameters:
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
        task_type : str
            Task type: 'C' for Classification or 'R' for Regression.

        Returns:
        -------
        BaseEstimator
            The fitted feature selector.
        """
        if task_type == "C":
            return SelectFromModel(GradientBoostingClassifier(random_state=42)).fit(
                X, y
            )
        else:
            return SelectFromModel(GradientBoostingRegressor(random_state=42)).fit(X, y)

    @staticmethod
    def _apply_xgboost(X: pd.DataFrame, y: pd.Series, task_type: str) -> BaseEstimator:
        """
        Apply XGBoost feature selection method.

        Parameters:
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
        task_type : str
            Task type: 'C' for Classification or 'R' for Regression.

        Returns:
        -------
        BaseEstimator
            The fitted feature selector.
        """
        if task_type == "C":
            return SelectFromModel(
                XGBClassifier(random_state=42, verbosity=0, eval_metric="logloss")
            ).fit(X, y)
        else:
            return SelectFromModel(
                XGBRegressor(random_state=42, verbosity=0, eval_metric="rmse")
            ).fit(X, y)

    @staticmethod
    def _apply_linear_method(
        X: pd.DataFrame, y: pd.Series, task_type: str
    ) -> BaseEstimator:
        """
        Apply linear models like Logistic Regression or Lasso.

        Parameters:
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
        task_type : str
            Task type: 'C' for Classification or 'R' for Regression.

        Returns:
        -------
        BaseEstimator
            The fitted feature selector.
        """
        if task_type == "C":
            return SelectFromModel(
                LogisticRegression(
                    random_state=42,
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=0.5,
                    max_iter=1000,
                )
            ).fit(X, y)
        else:
            return SelectFromModel(LassoCV(random_state=42)).fit(X, y)

    def fit(self, data: pd.DataFrame) -> "FeatureSelector":
        """
        Fit the feature selection model.

        Parameters:
        ----------
        data : pd.DataFrame
            The dataset containing features and the target column.

        Returns:
        -------
        FeatureSelector
            The fitted FeatureSelector instance.
        """
        X_data = data.drop([self.activity_col, self.id_col], axis=1)
        y_data = data[self.activity_col]

        self.task_type = self._determine_task_type(data)
        self.cv = self._determine_cv_strategy(self.task_type)

        if self.method == "best":
            self.method = self._select_best_method
            self.feature_selector = self.method_map[self.method](
                X=X_data, y=y_data, task_type=self.task_type
            )
        elif self.method in self.method_map:
            self.feature_selector = self.method_map[self.method](
                X=X_data, y=y_data, task_type=self.task_type
            )
        else:
            raise ValueError(f"Method '{self.method}' not recognized.")

        if self.save_dir:
            with open(f"{self.save_dir}/activity_col.pkl", "wb") as file:
                pickle.dump(self.activity_col, file)
            with open(f"{self.save_dir}/id_col.pkl", "wb") as file:
                pickle.dump(self.id_col, file)
            with open(f"{self.save_dir}/feature_selector.pkl", "wb") as file:
                pickle.dump(self.feature_selector, file)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataset using the fitted feature selection model.

        Parameters:
        ----------
        data : pd.DataFrame
            The dataset containing features and the target column.

        Returns:
        -------
        pd.DataFrame
            The dataset with selected features.
        """
        if self.feature_selector is None:
            raise NotFittedError("This FeatureSelector instance is not fitted yet.")

        X_data = data.drop([self.activity_col, self.id_col], axis=1)
        data_selected = pd.DataFrame(self.feature_selector.transform(X_data))
        transformed_data = pd.concat(
            [data_selected, data[[self.id_col, self.activity_col]]], axis=1
        )

        return transformed_data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the model and transform the dataset.

        Parameters:
        ----------
        data : pd.DataFrame
            The dataset containing features and the target column.

        Returns:
        -------
        pd.DataFrame
            The dataset with selected features.
        """
        self.fit(data)
        return self.transform(data)

    @staticmethod
    def static_transform(data: pd.DataFrame, save_dir: str) -> pd.DataFrame:
        """
        Static method to transform the dataset using the selected feature selection method without fitting.

        Parameters:
        ----------
        data : pd.DataFrame
            The dataset to transform.

        Returns:
        --------
        pd.DataFrame:
            The transformed dataset.
        """
        if not os.path.exists(f"{save_dir}/feature_selector.pkl"):
            raise NotFittedError(
                "The FeatureSelector instance is not fitted yet. Call 'fit' before using this method."
            )

        with open(f"{save_dir}/activity_col.pkl", "rb") as file:
            activity_col = pickle.load(file)
        with open(f"{save_dir}/id_col.pkl", "rb") as file:
            id_col = pickle.load(file)
        with open(f"{save_dir}/feature_selector.pkl", "rb") as file:
            feature_selector = pickle.load(file)

        X_data = data.drop(
            [activity_col, id_col],
            axis=1,
            errors="ignore",
        )
        data_selected = pd.DataFrame(feature_selector.transform(X_data))
        transformed_data = pd.concat(
            [data_selected, data[[id_col, activity_col]]], axis=1
        )

        return transformed_data

    @staticmethod
    def evaluate_model(
        model: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        scoring: str,
        cv: Any,
        n_jobs: int = -1,
    ) -> np.ndarray:
        """
        Evaluate the performance of a given model using cross-validation.

        Parameters:
        ----------
        model : BaseEstimator
            The model to evaluate.
        X_train : pd.DataFrame
            The training data features.
        y_train : pd.DataFrame
            The training data target.
        scoring : str
            The scoring method for evaluation.
        cv : Any
            Cross-validation strategy.
        n_jobs : int, optional
            Number of jobs to run in parallel, by default -1.

        Returns:
        -------
        np.ndarray
            The cross-validation scores.
        """
        return cross_val_score(
            model, X_train, y_train, scoring=scoring, cv=cv, n_jobs=n_jobs
        )

    def _train_model(self, task_type: str) -> BaseEstimator:
        """
        Train a model based on the task type.

        Parameters:
        ----------
        task_type : str
            Task type: 'C' for Classification or 'R' for Regression.

        Returns:
        -------
        BaseEstimator
            The trained model.
        """
        if task_type == "C":
            return RandomForestClassifier(random_state=42)
        else:
            return RandomForestRegressor(random_state=42)

    def _get_scoring_method(self, task_type: str) -> str:
        """
        Get the scoring method based on the task type.

        Parameters:
        ----------
        task_type : str
            Task type: 'C' for Classification or 'R' for Regression.

        Returns:
        -------
        str
            The scoring method.
        """
        if task_type == "C":
            return "f1" if self.scoring is None else self.scoring
        else:
            return "r2" if self.scoring is None else self.scoring

    def compare_feature_selectors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compare different feature selection methods and evaluate their performance.

        Parameters:
        ----------
        data : pd.DataFrame
            The dataset containing features and the target variable.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the comparison results for each method.
        """

        X_data = data.drop([self.activity_col, self.id_col], axis=1)
        y_data = data[self.activity_col]

        task_type = self._determine_task_type(data)
        cv = self._determine_cv_strategy(task_type)
        scoring = self._get_scoring_method(task_type)

        result = []

        for method, func in self.method_map.items():
            selector = func(X_data, y_data, task_type)
            selected_X = selector.transform(X_data)
            model = self._train_model(task_type)
            scores = cross_val_score(
                model, selected_X, y_data, cv=cv, scoring=scoring, n_jobs=self.n_jobs
            )
            method_result = {
                "Method": method,
                "Mean": round(np.mean(scores), 3),
                "Std": round(np.std(scores), 3),
                "Median": round(np.median(scores), 3),
            }
            for i, score in enumerate(scores):
                method_result[f"Score_{i+1}"] = score
            result.append(method_result)

        result_df = pd.DataFrame(result)

        if self.compare_table:
            print(result_df)

        if self.compare_visual:
            self._plot_compare_method(result_df)

        return result_df

    def _plot_compare_method(self, result: pd.DataFrame) -> None:
        """
        Plot a comparison of feature selection methods.

        Parameters:
        ----------
        result : pd.DataFrame
            A DataFrame containing the comparison results for each method.

        Raises:
        ------
        ValueError
            If `compare_visual` is not 'box' or 'violin'.
        """
        sns.set_style("whitegrid")
        plt.figure(figsize=(20, 10))

        score_columns = [col for col in result.columns if col.startswith("Score_")]
        melted_result = result.melt(
            id_vars=["Method"],
            value_vars=score_columns,
            var_name="Score",
            value_name="Value",
        )

        if self.compare_visual == "box":
            plot = sns.boxplot(
                x="Method", y="Value", data=melted_result, showmeans=True, width=0.5
            )
        elif self.compare_visual == "violin":
            plot = sns.violinplot(x="Method", y="Value", data=melted_result, inner=None)
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
                f"Invalid compare_visual '{self.compare_visual}'. Choose 'box' or 'violin'."
            )

        plot.set_title("Comparison of Feature Selection Methods", fontsize=16)
        plot.set_xlabel("Method", fontsize=14)
        plot.set_ylabel("Cross-Validation Score", fontsize=14)

        # Adding the mean values to the plot
        vertical_offset = melted_result["Value"].max() * 0.01
        for i, row in result.iterrows():
            plot.text(
                i,
                row["Mean"] + vertical_offset,
                str(row["Mean"]),
                horizontalalignment="center",
                size="x-large",
                color="w",
                weight="semibold",
            )

        if self.save_fig:
            plt.savefig("Feature Selection Methods", dpi=300)

        plt.show()

    def _select_best_method(self, data: pd.DataFrame) -> str:
        """
        Select the best feature selection method based on performance.

        Parameters:
        ----------
        data : pd.DataFrame
            The dataset containing features and the target variable.

        Returns:
        -------
        str
            The name of the best feature selection method.
        """
        result_df = self.compare_feature_selectors(data)
        best_method = result_df.loc[result_df["Mean"].idxmax(), "Method"]
        return best_method
