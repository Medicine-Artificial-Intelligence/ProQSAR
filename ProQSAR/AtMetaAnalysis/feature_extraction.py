from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    mutual_info_classif,
    f_classif,
    mutual_info_regression,
    f_regression,
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
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
import warnings

warnings.filterwarnings("ignore")


class FeatureExtraction:
    """
    A pipeline for feature selection in machine learning models.

    Parameters:
    ----------
    data_train: pandas.DataFrame
        The training dataset.
    data_test: pandas.DataFrame
        The test dataset.
    activity_col: str
        The name of the target variable column.
    id_col: str
        The name of the identifier column.
    task_type: str
        Task type: 'C' for Classification or 'R' for Regression.
    method: str
        Feature selection method to use.
    """

    def __init__(
        self, data_train, data_test, activity_col, id_col, task_type="C", method="RF"
    ):
        self.activity_col = activity_col
        self.task_type = task_type
        self.method = method

        # Store IDs separately and remove them from the feature set
        self.id_train = data_train[id_col]
        self.id_test = data_test[id_col]

        # Prepare the feature and target sets
        self.X_train = data_train.drop([activity_col, id_col], axis=1)
        self.y_train = data_train[activity_col]
        self.X_test = data_test.drop([activity_col, id_col], axis=1)
        self.y_test = data_test[activity_col]

        # Set up cross-validation based on task type
        self.cv = self._determine_cv_strategy(task_type)

    def _determine_cv_strategy(self, task_type):
        """Determine the cross-validation strategy based on the task type."""
        if task_type == "C":
            return RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
        else:  # task_type == 'R'
            return RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)

    @staticmethod
    def apply_statistic_method(X, y, method="Anova", task_type="C"):
        if task_type == "C":
            if method == "Anova":
                selector = SelectKBest(score_func=f_classif, k=20)
            elif method == "Mutual information":
                selector = SelectKBest(score_func=mutual_info_classif, k=20)
        else:  # For regression tasks
            if method == "Anova":
                selector = SelectKBest(score_func=f_regression, k=20)
            elif method == "Mutual information":
                selector = SelectKBest(score_func=mutual_info_regression, k=20)

        selector.fit(X, y)
        return selector

    @staticmethod
    def apply_random_forest(X, y, task_type):
        model = (
            RandomForestClassifier(random_state=42)
            if task_type == "C"
            else RandomForestRegressor(random_state=42)
        )
        model.fit(X, y)
        return SelectFromModel(model, prefit=True)

    @staticmethod
    def apply_extra_tree(X, y, task_type):
        model = (
            ExtraTreesClassifier(random_state=42)
            if task_type == "C"
            else ExtraTreesRegressor(random_state=42)
        )
        model.fit(X, y)
        return SelectFromModel(model, prefit=True)

    @staticmethod
    def apply_ada_boost(X, y, task_type):
        model = (
            AdaBoostClassifier(random_state=42)
            if task_type == "C"
            else AdaBoostRegressor(random_state=42)
        )
        model.fit(X, y)
        return SelectFromModel(model, prefit=True)

    @staticmethod
    def apply_gradient_boosting(X, y, task_type):
        model = (
            GradientBoostingClassifier(random_state=42)
            if task_type == "C"
            else GradientBoostingRegressor(random_state=42)
        )
        model.fit(X, y)
        return SelectFromModel(model, prefit=True)

    @staticmethod
    def apply_xgboost(X, y, task_type):
        model = (
            XGBClassifier(random_state=42, verbosity=0, eval_metric="logloss")
            if task_type == "C"
            else XGBRegressor(random_state=42, verbosity=0, eval_metric="logloss")
        )
        model.fit(X, y)
        return SelectFromModel(model, prefit=True)

    @staticmethod
    def apply_linear_method(X, y, task_type):
        model = (
            LogisticRegression(
                random_state=42,
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.5,
                max_iter=10000,
            )
            if task_type == "C"
            else LassoCV(random_state=42)
        )
        model.fit(X, y)
        return SelectFromModel(model, prefit=True)

    def fit(self):
        method_mapping = {
            "Anova": self.apply_statistic_method,
            "Mutual information": self.apply_statistic_method,
            "RF": self.apply_random_forest,
            "ExT": self.apply_extra_tree,
            "Ada": self.apply_ada_boost,
            "Grad": self.apply_gradient_boosting,
            "XGB": self.apply_xgboost,
            "Linear": self.apply_linear_method,
        }

        if self.method in method_mapping:
            self.select = method_mapping[self.method](
                self.X_train, self.y_train, self.task_type
            )
            self.X_train_new = self.select.transform(self.X_train)
            self.X_test_new = self.select.transform(self.X_test)
        return self.X_train_new, self.X_test_new, self.y_train, self.y_test
