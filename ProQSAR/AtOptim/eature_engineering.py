import warnings

warnings.filterwarnings("ignore")
# Classification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    mutual_info_classif,
    chi2,
    RFE,
    RFECV,
    f_classif,
    mutual_info_regression,
    f_regression,
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    RepeatedStratifiedKFold,
)
from sklearn.svm import LinearSVC, SVC

# Regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, cross_val_score, RepeatedKFold
from sklearn.svm import LinearSVR, SVR

import warnings

warnings.filterwarnings("ignore")


class feature_engineering:
    """
    - Remove unnecessary features
    - Show a chart that compares the effectiveness of each method.
    - Based on the chart, choose the best method.

    Input:
    ------
    Rescaled Data_train and test

    Returns:
    --------
    Completed Data_train and Data_test

    """

    def __init__(self, data_train, data_test, activity_col):
        self.Y_name = activity_col
        self.X_train = data_train.drop([self.Y_name], axis=1)
        self.y_train = data_train[self.Y_name]
        self.X_test = data_test.drop([self.Y_name], axis=1)
        self.y_test = data_test[self.Y_name]
        self.results = list()
        self.names = list()

    # 1. Choose Regression or Classification
    # case = 2: Classification
    # case > 2: Regression
    def case_model(self):
        self.case = len(np.unique(self.y_train))
        if self.case == 2:
            self.models, self.names = self.C_feature_model()
            # define evaluation procedure
            self.cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
            while True:
                try:
                    input_scores = int(
                        input(
                            "Please select metric you want to compare:\n\t1:Average Precision\n\t2:F1_score\n\t3:Recall\n"
                        )
                    )
                    break
                except:
                    print("Error value")
            if input_scores == 1:
                self.scoring = "average_precision"
            elif input_scores == 2:
                self.scoring = "f1"
            else:
                self.scoring = "recall"

        elif self.case > 2:
            self.models, self.names = self.R_feature_model()
            # define evaluation procedure
            self.cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
            while True:
                try:
                    input_scores = int(
                        input(
                            "Please select metric you want to compare:\n\t1:R2\n\t2:RMSE\n\t3:MAE\n"
                        )
                    )
                    break
                except:
                    print("Error value")
            if input_scores == 1:
                self.scoring = "r2"
            elif input_scores == 2:
                self.scoring = "neg_mean_squared_error"
            else:
                self.scoring = "neg_mean_absolute_error"
            self.scoring = "r2"
        else:
            raise "too few case"

    # 2. Regression Feature selection
    def R_feature_model(self):
        # baseline model: random forest
        models, names = list(), list()

        # 1. Anova test select k best
        select = SelectKBest(score_func=f_regression, k=40)
        model = RandomForestRegressor(random_state=42)
        steps = [("s", select), ("m", model)]
        models.append(Pipeline(steps=steps))
        names.append("Anova")

        # 2. Mutual_info select k best
        select = SelectKBest(score_func=mutual_info_regression, k=40)
        model = RandomForestRegressor(random_state=42)
        steps = [("s", select), ("m", model)]
        models.append(Pipeline(steps=steps))
        names.append("Mutual_info")

        # 3. Random Forest
        rf = RandomForestRegressor(random_state=42)
        select = SelectFromModel(rf)
        model = RandomForestRegressor(random_state=42)
        steps = [("s", select), ("m", model)]
        models.append(Pipeline(steps=steps))
        names.append("Random Forest")

        # 4. ExtraTree
        ext = ExtraTreesRegressor(random_state=42)
        select = SelectFromModel(ext)
        model = RandomForestRegressor(random_state=42)
        steps = [("s", select), ("m", model)]
        models.append(Pipeline(steps=steps))
        names.append("Extra Tree")

        # 5. AdaBoost
        ada = AdaBoostRegressor(random_state=42)
        select = SelectFromModel(ada)
        model = RandomForestRegressor(random_state=42)
        steps = [("s", select), ("m", model)]
        models.append(Pipeline(steps=steps))
        names.append("AdaBoost")

        # 6. GradBoost
        grad = GradientBoostingRegressor(random_state=42)
        select = SelectFromModel(grad)
        model = RandomForestRegressor(random_state=42)
        steps = [("s", select), ("m", model)]
        models.append(Pipeline(steps=steps))
        names.append("GradBoost")

        # 7. XGB
        xgb = XGBRegressor(
            random_state=42,
            verbosity=0,
            # eval_metrics ='logloss'
        )
        select = SelectFromModel(xgb)
        model = RandomForestRegressor(random_state=42)
        steps = [("s", select), ("m", model)]
        models.append(Pipeline(steps=steps))
        names.append("XGBoost")

        # 8. Lasso
        lasso = LassoCV(random_state=42)
        select = SelectFromModel(lasso)
        model = RandomForestRegressor(random_state=42)
        steps = [("s", select), ("m", model)]
        models.append(Pipeline(steps=steps))
        names.append("Lasso")

        return models, names

    # 2. Classification Feature selection
    def C_feature_model(self):
        # baseline model: random forest
        models, names = list(), list()

        model = RandomForestClassifier(random_state=42)
        steps = [("m", model)]
        models.append(Pipeline(steps=steps))
        names.append("Baseline")

        # 1. Chi2 select k best
        select = SelectKBest(score_func=chi2, k=50)
        model = RandomForestClassifier(random_state=42)
        steps = [("s", select), ("m", model)]
        models.append(Pipeline(steps=steps))
        names.append("Chi2")

        # 2. Mutual_info select k best
        select = SelectKBest(score_func=mutual_info_classif, k=50)
        model = RandomForestClassifier(random_state=42)
        steps = [("s", select), ("m", model)]
        models.append(Pipeline(steps=steps))
        names.append("Mutual_info")

        # 3. Random Forest
        rf = RandomForestClassifier(random_state=42)
        select = SelectFromModel(rf)
        model = RandomForestClassifier(random_state=42)
        steps = [("s", select), ("m", model)]
        models.append(Pipeline(steps=steps))
        names.append("RF")

        # 4. ExtraTree
        ext = ExtraTreesClassifier(random_state=42)
        select = SelectFromModel(ext)
        model = RandomForestClassifier(random_state=42)
        steps = [("s", select), ("m", model)]
        models.append(Pipeline(steps=steps))
        names.append("ExT")

        # 5. AdaBoost
        ada = AdaBoostClassifier(random_state=42)
        select = SelectFromModel(ada)
        model = RandomForestClassifier(random_state=42)
        steps = [("s", select), ("m", model)]
        models.append(Pipeline(steps=steps))
        names.append("Ada")

        # 6. GradBoost
        grad = GradientBoostingClassifier(random_state=42)
        select = SelectFromModel(grad)
        model = RandomForestClassifier(random_state=42)
        steps = [("s", select), ("m", model)]
        models.append(Pipeline(steps=steps))
        names.append("Grad")

        # 7. XGB
        xgb = XGBClassifier(random_state=42, verbosity=0, eval_metrics="logloss")
        select = SelectFromModel(xgb)
        model = RandomForestClassifier(random_state=42)
        steps = [("s", select), ("m", model)]
        models.append(Pipeline(steps=steps))
        names.append("XGB")

        # 8. Logistics
        lgr = LogisticRegression(
            random_state=42,
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            max_iter=10000,
        )
        select = SelectFromModel(lgr)
        model = RandomForestClassifier(random_state=42)
        steps = [("s", select), ("m", model)]
        models.append(Pipeline(steps=steps))
        names.append("Logistics")

        return models, names

    def evaluate_model(self, X, y, model):
        # evaluate model: đánh giá hiệu quả của phương pháp
        scores = cross_val_score(
            model, X, y, scoring=self.scoring, cv=self.cv, n_jobs=-1
        )
        return scores

    def compare(self):
        # Thêm điểm của từng phương pháp vào list_results.
        for i in range(len(self.models)):
            scores = self.evaluate_model(self.X_train, self.y_train, self.models[i])
            self.results.append(scores)
            print(
                ">%s %.3f ± %.3f (%.3f)"
                % (self.names[i], np.mean(scores), np.std(scores), np.median(scores))
            )
        # self.features_compare = statical_test(self.results, self.names,X_train = self.X_train, y_train = self.y_train)
        # self.features_compare.visualize()
        self.visualize()

    def visualize(self):
        mean = list()
        for i in range(len(self.results)):
            x = self.results[i].mean().round(3)
            mean.append(x)
        data = np.array(mean)
        ser = pd.Series(data, index=self.names)

        dict_columns = {
            "Mean": mean,
            "Method": self.names,
        }
        df = pd.DataFrame(dict_columns)

        sns.set_style("whitegrid")
        plt.figure(figsize=(20, 10))
        box_plot = sns.boxplot(
            data=self.results,
            showmeans=True,
            meanprops={
                "marker": "d",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": "10",
            },
        )
        box_plot.axes.set_title("Compare feature selection method", fontsize=16)
        box_plot.set_xlabel("Method", fontsize=14)
        box_plot.set_ylabel(f"{self.scoring}", fontsize=14)
        vertical_offset = df["Mean"].median() * 0.01

        for xtick in box_plot.get_xticks():
            box_plot.text(
                xtick,
                ser[xtick] + vertical_offset,
                ser[xtick],
                horizontalalignment="center",
                size="x-large",
                color="w",
                weight="semibold",
            )

        # box_plot.get_xticks(range(len(self.results)))
        box_plot.set_xticklabels(self.names, rotation="horizontal")
