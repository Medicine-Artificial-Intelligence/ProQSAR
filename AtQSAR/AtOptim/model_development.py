import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Union, Dict, Any
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    cross_val_score,
    RepeatedStratifiedKFold,
    RepeatedKFold,
)
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    ElasticNetCV,
    Ridge,
    LassoCV,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
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
    classification_report,
    log_loss,
    brier_score_loss,
    hamming_loss,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    max_error,
    mean_squared_log_error,
)


class ModelDevelopment:
    """
    A class to perform model selection and evaluation for classification or regression tasks.

    Attributes:
    ----------
    X_train : np.ndarray
        The training feature set.
    y_train : np.ndarray
        The training target set.
    save_prefix : str
        Directory path to save the results.
    data_name : str
        Name of the dataset.
    task_type : str, default 'C'
        Type of task: 'C' for classification, 'R' for regression.
    scoring : str, default 'f1'
        Scoring metric for model evaluation.
    n_jobs : int, default 4
        Number of jobs to run in parallel.
    plot_type : str, default 'box'
        Type of plot for visualizations.
    random_state : int, default 42
        Random state for reproducibility.
    cv_splits : int, default 10
        Number of splits for cross-validation.
    cv_repeats : int, default 3
        Number of repeats for cross-validation.
    models : List[BaseEstimator]
        List of machine learning models for evaluation.
    names : List[str]
        Names of the machine learning models.

    Methods:
    -------
    evaluate_model(model: BaseEstimator) -> np.ndarray:
        Evaluates a given model using cross-validation.
    compare_models() -> Dict[str, Any]:
        Compares all models and logs their performance metrics.
    _generate_visualizations(results: List[np.ndarray]) -> None:
        Generates visualizations for model comparison.
    """

    def __init__(
        self,
        X_train: np.ndarray,  # Argument: Training data features (numpy array)
        y_train: np.ndarray,  # Argument: Training data labels (numpy array)
        save_prefix: str,  # Argument: Prefix for saving files (string)
        data_name: str,  # Argument: Name of the data (string)
        task_type: str = "C",  # Argument: Type of task (classification by default) (string)
        scoring: str = "f1",  # Argument: Scoring metric (string)
        n_jobs: int = 4,  # Argument: Number of parallel jobs (integer)
        plot_type: str = "box",  # Argument: Type of plot (box plot by default) (string)
        random_state: int = 42,  # Argument: Random state (integer)
        cv_splits: int = 10,  # Argument: Number of cross-validation splits (integer)
        cv_repeats: int = 3,  # Argument: Number of times cross-validation is repeated (integer)
    ) -> None:  # Return type: None
        """
        Initializes the class with the given parameters.

        Args:
            X_train (np.ndarray): Training data features.
            y_train (np.ndarray): Training data labels.
            save_prefix (str): Prefix for saving files.
            data_name (str): Name of the data.
            task_type (str, optional): Type of task (classification by default).
            scoring (str, optional): Scoring metric.
            n_jobs (int, optional): Number of parallel jobs.
            plot_type (str, optional): Type of plot (box plot by default).
            random_state (int, optional): Random state.
            cv_splits (int, optional): Number of cross-validation splits.
            cv_repeats (int, optional): Number of times cross-validation is repeated.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.save_prefix = save_prefix
        self.data_name = data_name
        self.task_type = task_type
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.plot_type = plot_type
        self.random_state = random_state
        self.cv_splits = cv_splits
        self.cv_repeats = cv_repeats
        self.models, self.names = self._select_models(task_type)

    def _select_models(self, task_type: str) -> List[str]:
        """
        Selects the appropriate models based on the given task_type.

        Args:
            task_type (str): The type of task ('C' for classification, 'R' for regression).

        Returns:
            List[str]: The list of selected models.
        """
        if task_type == "C":
            return self._classification_models()
        else:
            return self._regression_models()

    def _classification_models(self) -> Tuple[List[BaseEstimator], List[str]]:
        """
        Returns a list of classification models and their corresponding names.

        Returns:
            models (List[BaseEstimator]): A list of classification models.
            names (List[str]): A list of names corresponding to the models.
        """
        models: List[BaseEstimator] = []
        names: List[str] = []

        models.append(
            LogisticRegression(
                max_iter=100000,
                penalty="l2",
                solver="liblinear",
                random_state=self.random_state,
            )
        )
        names.append("Logistic")

        models.append(KNeighborsClassifier(n_neighbors=20))
        names.append("KNN")

        models.append(
            SVC(
                probability=True,
                kernel="rbf",
                gamma="scale",
                max_iter=10000,
                random_state=self.random_state,
            )
        )
        names.append("SVM")

        models.append(
            RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        )
        names.append("RF")

        models.append(
            ExtraTreesClassifier(n_estimators=100, random_state=self.random_state)
        )
        names.append("ExT")

        models.append(
            AdaBoostClassifier(n_estimators=100, random_state=self.random_state)
        )
        names.append("Ada")

        models.append(
            GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)
        )
        names.append("Grad")

        models.append(
            XGBClassifier(
                random_state=self.random_state, verbosity=0, eval_metric="logloss"
            )
        )
        names.append("XGB")

        # models.append(CatBoostClassifier(
        #     verbose=0,
        #     random_state=self.random_state
        # ))
        # names.append('CatB')

        # models.append(MLPClassifier(
        #     alpha=0.01,
        #     max_iter=10000,
        #     random_state=self.random_state,
        #     hidden_layer_sizes=(150,)
        # ))
        # names.append('MLP')

        return models, names

    def _regression_models(self) -> Tuple[List[BaseEstimator], List[str]]:
        """
        Returns a tuple containing a list of regression models and their corresponding names.

        Returns:
            Tuple[List[BaseEstimator], List[str]]: A tuple containing the list of regression models and the list of their names.
        """
        models: List[BaseEstimator] = []
        names: List[str] = []

        models.append(LinearRegression())
        names.append("LR")

        models.append(Ridge(alpha=1))
        names.append("Ridge")

        models.append(ElasticNetCV(cv=5))
        names.append("ElasticNet")

        models.append(KNeighborsRegressor(n_neighbors=5))
        names.append("KNN")

        models.append(SVR(kernel="rbf", gamma="scale", C=1.0, epsilon=0.1))
        names.append("SVR")

        models.append(
            RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        )
        names.append("RF")

        models.append(
            GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, random_state=self.random_state
            )
        )
        names.append("Grad")

        models.append(
            XGBRegressor(
                random_state=self.random_state,
                verbosity=0,
                objective="reg:squarederror",
            )
        )
        names.append("XGB")

        models.append(CatBoostRegressor(verbose=0, random_state=self.random_state))
        names.append("CatB")

        models.append(
            MLPRegressor(
                alpha=0.01,
                max_iter=10000,
                random_state=self.random_state,
                hidden_layer_sizes=(150,),
            )
        )
        names.append("MLP")

        return models, names

    def evaluate_model(self, model: BaseEstimator) -> float:
        """Evaluate the performance of a machine learning model using cross-validation.

        Args:
            model (BaseEstimator): The machine learning model to evaluate.

        Returns:
            float: The average score of the model across all cross-validation folds.
        """
        scores = cross_val_score(
            model,
            self.X_train,
            self.y_train,
            scoring=self.scoring,
            cv=self._get_cv_strategy(),
            n_jobs=self.n_jobs,
        )
        return scores.mean()

    def _get_cv_strategy(self) -> Union[RepeatedKFold, RepeatedStratifiedKFold]:
        """
        Returns the cross-validation strategy based on the task type.

        Args:
            self (object): The instance of the class.

        Returns:
            Union[RepeatedKFold, RepeatedStratifiedKFold]: The cross-validation strategy object.
        """
        if self.task_type == "C":
            return RepeatedStratifiedKFold(
                n_splits=self.cv_splits,
                n_repeats=self.cv_repeats,
                random_state=self.random_state,
            )
        else:
            return RepeatedKFold(
                n_splits=self.cv_splits,
                n_repeats=self.cv_repeats,
                random_state=self.random_state,
            )

    def compare_models(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compare the performance of multiple models on the given data and generate visualizations.

        Returns:
            model_performance_log (Dict): A dictionary containing the performance metrics for each model.
                The structure of the dictionary is as follows:
                {
                    data_name: {
                        model_name: {
                            'mean': float,
                            'std': float,
                            'median': float
                        }
                    }
                }
        """
        results = []
        model_performance_log = {self.data_name: {}}

        for i, model in enumerate(self.models):
            model_name = self.names[i]
            scores = self.evaluate_model(model)
            results.append(scores)
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            median_score = np.median(scores)

            model_performance_log[self.data_name][model_name] = {
                "mean": mean_score,
                "std": std_score,
                "median": median_score,
            }

            print(
                f"{model_name}: Mean={mean_score:.3f}, Std={std_score:.3f}, Median={median_score:.3f}"
            )

        self._generate_visualizations(results)

        if self.save_prefix:
            performance_df = pd.DataFrame(results).T
            performance_df.columns = self.names
            performance_df.to_csv(
                f"{self.save_prefix}/{self.data_name}_model_performance.csv"
            )

        return model_performance_log

    def _generate_visualizations(self, results: List[List[float]]) -> None:
        """
        Generate visualizations based on the results.

        Args:
            results (List[List[float]]): A list of lists containing the performance results of different models.

        Returns:
            None
        """

        plt.figure(figsize=(15, 8))
        num_models = len(results)
        palette = sns.color_palette("coolwarm", num_models)

        if self.plot_type == "box":
            sns.boxplot(
                data=results,
                palette=palette,
                showmeans=True,
                meanprops={
                    "marker": "o",
                    "markerfacecolor": "red",
                    "markeredgecolor": "black",
                },
            )
        elif self.plot_type == "bar":
            sns.barplot(data=np.array(results).T, palette=palette)
        elif self.plot_type == "violin":
            sns.violinplot(data=results, palette=palette, inner="point", scale="width")
            means = [np.mean(result) for result in results]
            for i, mean in enumerate(means):
                plt.text(
                    i,
                    mean,
                    f"{mean:.2f}",
                    horizontalalignment="center",
                    color="black",
                    fontsize=10,
                    weight="semibold",
                )

        plt.title(f"Comparison of Model Performances", fontsize=16, weight="semibold")
        plt.xlabel("Model", fontsize=14)
        plt.ylabel(f"{self.scoring} Score", fontsize=14, weight="semibold")
        plt.xticks(
            ticks=np.arange(len(self.names)),
            labels=self.names,
            rotation=45,
            ha="right",
            weight="semibold",
        )
        plt.tight_layout()

        try:
            if self.save_prefix:
                plt.savefig(
                    f"{self.save_prefix}/{self.data_name}_model_comparison_{self.plot_type}.png",
                    dpi=300,
                )
        except Exception as e:
            print(f"Error saving the plot: {e}")

        plt.show()
