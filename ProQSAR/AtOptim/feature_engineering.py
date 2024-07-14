import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
from typing import Tuple, List, Union, Any, Optional
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    cross_val_score,
    RepeatedStratifiedKFold,
    RepeatedKFold,
)
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    mutual_info_classif,
    chi2,
    f_regression,
    mutual_info_regression,
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
from sklearn.svm import LinearSVC, LinearSVR


class FeatureEngineering:
    """
    A class for performing feature engineering using various feature selection methods.

    Parameters:
    ----------
    data_train : pd.DataFrame
        The training dataset.
    data_test : pd.DataFrame
        The testing dataset.
    activity_col : str
        The name of the target column.
    n_jobs : int, optional
        Number of jobs to run in parallel (default is -1, which means using all processors).

    Attributes:
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    X_test : pd.DataFrame
        Testing features.
    y_test : pd.Series
        Testing target.
    results : List[np.ndarray]
        List to store evaluation results.
    names : List[str]
        Names of feature selection methods.
    models : List[BaseEstimator]
        List of models for feature selection.
    scoring : str
        Scoring metric for model evaluation.
    cv : Union[RepeatedStratifiedKFold, RepeatedKFold]
        Cross-validation strategy.
    """

    def __init__(
        self,
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        activity_col: str,
        n_jobs: int = -1,
        plot_type="box",
        save_fig_path=None,
    ):
        self.X_train = data_train.drop([activity_col], axis=1)
        self.y_train = data_train[activity_col]
        self.X_test = data_test.drop([activity_col], axis=1)
        self.y_test = data_test[activity_col]
        self.n_jobs = n_jobs
        self.plot_type = plot_types
        self.save_fig_path = save_fig_path
        self.results = []
        self.names = []
        self.models = []
        self.scoring = ""
        self.cv = None
        self._initialize_model_case()

    def _initialize_model_case(self) -> None:
        """
        Determines whether the task is classification or regression, and initializes the respective models.
        """
        unique_targets = len(np.unique(self.y_train))
        if unique_targets == 2:
            self.models, self.names = self._init_classification_models()
            self.cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

        elif unique_targets > 2:
            self.models, self.names = self._init_regression_models()
            self.cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)

        else:
            raise ValueError(
                "Insufficient number of categories to determine model type."
            )

    @staticmethod
    def _init_regression_models() -> Tuple[List[Pipeline], List[str]]:
        """
        Initializes regression models with different feature selection methods.

        Returns:
        --------
        Tuple[List[Pipeline], List[str]]:
            A tuple of the list of regression pipelines and their names.
        """
        models, names = [], []

        # 1. Anova test select k best
        select = SelectKBest(score_func=f_regression, k=40)
        model = RandomForestRegressor(random_state=42)
        models.append(Pipeline([("s", select), ("m", model)]))
        names.append("Anova")

        # 2. Mutual_info select k best
        select = SelectKBest(score_func=mutual_info_regression, k=40)
        models.append(Pipeline([("s", select), ("m", model)]))
        names.append("Mutual_info")

        # 3. Random Forest
        rf = RandomForestRegressor(random_state=42)
        models.append(Pipeline([("s", SelectFromModel(rf)), ("m", model)]))
        names.append("RF")

        # 4. ExtraTree
        ext = ExtraTreesRegressor(random_state=42)
        models.append(Pipeline([("s", SelectFromModel(ext)), ("m", model)]))
        names.append("ExT")

        # 5. AdaBoost
        ada = AdaBoostRegressor(random_state=42)
        models.append(Pipeline([("s", SelectFromModel(ada)), ("m", model)]))
        names.append("Ada")

        # 6. Gradient Boosting
        grad = GradientBoostingRegressor(random_state=42)
        models.append(Pipeline([("s", SelectFromModel(grad)), ("m", model)]))
        names.append("Grad")

        # 7. XGB
        xgb = XGBRegressor(random_state=42, verbosity=0)
        models.append(Pipeline([("s", SelectFromModel(xgb)), ("m", model)]))
        names.append("XGB")

        # 8. Lasso
        lasso = LassoCV(random_state=42)
        models.append(Pipeline([("s", SelectFromModel(lasso)), ("m", model)]))
        names.append("Lasso")

        return models, names

    @staticmethod
    def _init_classification_models() -> Tuple[List[Pipeline], List[str]]:
        """
        Initializes classification models with different feature selection methods.

        Returns:
        --------
        Tuple[List[Pipeline], List[str]]:
            A tuple of the list of classification pipelines and their names.
        """
        models, names = [], []

        # 1. Anova test select k best
        select = SelectKBest(score_func=chi2, k=40)
        model = RandomForestClassifier(random_state=42)
        models.append(Pipeline([("s", select), ("m", model)]))
        names.append("Chi2")

        # 2. Mutual_info select k best
        select = SelectKBest(score_func=mutual_info_classif, k=40)
        models.append(Pipeline([("s", select), ("m", model)]))
        names.append("Mutual_info")

        # 3. Random Forest
        rf = RandomForestClassifier(random_state=42)
        models.append(Pipeline([("s", SelectFromModel(rf)), ("m", model)]))
        names.append("RF")

        # 4. ExtraTree
        ext = ExtraTreesClassifier(random_state=42)
        models.append(Pipeline([("s", SelectFromModel(ext)), ("m", model)]))
        names.append("ExT")

        # 5. AdaBoost
        ada = AdaBoostClassifier(random_state=42)
        models.append(Pipeline([("s", SelectFromModel(ada)), ("m", model)]))
        names.append("AdaBoost")

        # 6. Gradient Boosting
        grad = GradientBoostingClassifier(random_state=42)
        models.append(Pipeline([("s", SelectFromModel(grad)), ("m", model)]))
        names.append("GradBoost")

        # 7. XGBoost
        xgb = XGBClassifier(random_state=42, verbosity=0)
        models.append(Pipeline([("s", SelectFromModel(xgb)), ("m", model)]))
        names.append("XGB")

        # 8. Logistic Regression
        log_reg = LogisticRegression(random_state=42, max_iter=10000)
        models.append(Pipeline([("s", SelectFromModel(log_reg)), ("m", model)]))
        names.append("LogisticRegression")

        return models, names

    @staticmethod
    def evaluate_model(
        model: BaseEstimator,
        X_train: Any,
        y_train: Any,
        scoring: str,
        cv: Any,
        n_jobs: int = -1,
    ) -> np.ndarray:
        """
        Evaluates a given model using cross-validation.

        Parameters:
        ----------
        model : BaseEstimator
            The model to be evaluated.
        X_train : Any
            Training feature set.
        y_train : Any
            Training target set.
        scoring : str
            Scoring metric for model evaluation.
        cv : Any
            Cross-validation splitting strategy.
        n_jobs : int, default -1
            Number of jobs to run in parallel.

        Returns:
        --------
        np.ndarray:
            The array of cross-validation scores.
        """
        return cross_val_score(
            model, X_train, y_train, scoring=scoring, cv=cv, n_jobs=n_jobs
        )

    def compare_feature_selection_methods(self) -> None:
        """
        Compares different feature selection methods and prints out their performance.
        """
        for i, model in enumerate(self.models):
            scores = self.evaluate_model(model)
            self.results.append(scores)
            print(
                f">{self.names[i]}: Mean={np.mean(scores):.3f}, Std={np.std(scores):.3f}, Median={np.median(scores):.3f}"
            )
        self.visualize_comparison(
            self.results, self.names, self.scoring, self.plot_type, self.save_fig_path
        )

    @staticmethod
    def visualize_comparison(
        results: List[np.ndarray],
        names: List[str],
        scoring: str,
        plot_type: str = "box",
        save_fig_path: Optional[str] = None,
    ) -> None:
        """
        Visualizes the comparison of feature selection methods.

        Parameters:
        ----------
        results : List[np.ndarray]
            The cross-validation results for each feature selection method.
        names : List[str]
            Names of the feature selection methods.
        scoring : str
            The scoring metric used in model evaluation.
        plot_type : str, default 'box'
            Type of plot to use for visualization ('box' or 'violin').
        save_fig_path : Optional[str], default None
            The path to save the figure. If None, the figure is not saved.
        """
        mean_scores = [np.mean(scores).round(3) for scores in results]

        sns.set_style("whitegrid")
        plt.figure(figsize=(20, 10))

        if plot_type == "box":
            plot = sns.boxplot(
                data=results,
                showmeans=True,
                meanprops={
                    "marker": "d",
                    "markerfacecolor": "white",
                    "markeredgecolor": "black",
                    "markersize": "10",
                },
            )
        elif plot_type == "violin":
            plot = sns.violinplot(data=results, inner=None)
            sns.stripplot(data=results, color="white", size=5, jitter=True)

        plot.axes.set_title("Comparison of Feature Selection Methods", fontsize=16)
        plot.set_xlabel("Method", fontsize=14)
        plot.set_ylabel(f"{scoring.upper()} Score", fontsize=14)

        vertical_offset = max(mean_scores) * 0.01
        for i, mean_score in enumerate(mean_scores):
            plot.text(
                i,
                mean_score + vertical_offset,
                str(mean_score),
                horizontalalignment="center",
                size="x-large",
                color="w",
                weight="semibold",
            )

        plot.set_xticklabels(names, rotation="horizontal")

        if save_fig_path:
            plt.savefig(save_fig_path, dpi=300)

        plt.show()
