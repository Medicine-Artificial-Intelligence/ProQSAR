from sklearn.metrics import log_loss, brier_score_loss, hamming_loss

# Classification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    FunctionTransformer,
)

# Library in Model Selection class
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
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    RepeatedStratifiedKFold,
)
from sklearn.model_selection import KFold, cross_val_score, RepeatedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNetCV, Ridge
from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import (
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import (
    mean_absolute_percentage_error,
    max_error,
    mean_squared_log_error,
)
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import (
    SGDRegressor,
    HuberRegressor,
    TheilSenRegressor,
    RANSACRegressor,
)
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from imblearn.pipeline import Pipeline
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor


class Classification_report:
    def __init__(self, X_train, X_test, y_train, y_test, metric=None, create_df=True):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.metric = metric
        self.df_compare_train = pd.DataFrame(
            columns=[
                "roc_auc_score",
                "average_precision_score",
                "accuracy_score",
                "recall_score",
                "precision_score",
                "f1_score",
                "log_loss",
                "brier_score_loss",
                "hamming_loss",
                "No. of obs.",
            ]
        )
        self.df_compare_test = self.df_compare_train.copy()
        self.create_df = create_df
        self.logit = LogisticRegression(penalty="l2", max_iter=10000)
        self.kNN = KNeighborsClassifier()
        self.SVC = SVC(probability=True, max_iter=10000)
        # self.NuSVC  = NuSVC(probability = True)
        self.DT = DecisionTreeClassifier(ccp_alpha=0.02)
        self.RF = RandomForestClassifier(random_state=42)
        self.ada = AdaBoostClassifier(random_state=42)
        self.grad = GradientBoostingClassifier(random_state=42)
        self.ext = ExtraTreesClassifier(random_state=42)
        self.xgb = XGBClassifier(
            random_state=42,
            verbosity=0,
            use_label_encoder=False,
            eval_metrics="logloss",
        )
        self.cbs = CatBoostClassifier(verbose=0, random_state=42)
        self.mlp = MLPClassifier(
            alpha=0.01,
            max_iter=10000,
            validation_fraction=0.1,
            random_state=42,
            hidden_layer_sizes=150,
        )
        self.gNB = GaussianNB()
        self.bNB = BernoulliNB()

    def model(self):
        classifiers = [
            ("Logistic Reg.", self.logit),
            ("Knn", self.kNN),
            ("SVM", self.SVC),  # ('nuSVM'            , self.NuSVC),
            ("Decision Tree", self.DT),
            ("Random Forest", self.RF),
            ("Extra Tree", self.ext),
            ("Ada Boost", self.ada),
            ("Gradient Boost", self.grad),
            ("XGBoost", self.xgb),
            ("CAT Boost", self.cbs),
            ("MLP", self.mlp),
            ("Gaussian", self.gNB),
            ("Bernoulli", self.bNB),
        ]

        for self.name, self.estimator in classifiers:
            self.estimator.fit(self.X_train, self.y_train)
            self.Report_metrics()

    def Report_metrics(self):
        background_color = "#F0F6FC"
        hue_color = ["#FF5003", "#428bca"]
        color = "#000000"

        self.P_train = self.estimator.predict(self.X_train)
        self.P_test = self.estimator.predict(self.X_test)
        self.PB_train = self.estimator.predict_proba(self.X_train)[:, 1]
        self.PB_test = self.estimator.predict_proba(self.X_test)[:, 1]

        if self.metric == "raw_metric":
            f = plt.figure(figsize=(8, 2.5), dpi=200)
            f.patch.set_facecolor(background_color)
            a = f.add_axes([0.0, 0.0, 1.0, 0.5])

            a.axis("off")
            for i in np.arange(0, 1, 0.08):
                a.text(0.50, i, "|", verticalalignment="bottom")
                a.text(i, 0.75, "_________", verticalalignment="bottom")

            a.text(
                0.13,
                0.75,
                " Training Data                                 Test Data",
                family="monospace",
                verticalalignment="bottom",
            )

            dic = {
                f1_score: ["f1_score   = ", (P_train, P_test)],
                recall_score: ["Recall     = ", (P_train, P_test)],
                precision_score: ["Precision  = ", (P_train, P_test)],
                accuracy_score: ["Accuracy   = ", (P_train, P_test)],
                roc_auc_score: ["AUC        = ", (PB_train, PB_test)],
            }

            g = 0
            for k, v in dic.items():
                a.text(
                    0.15,
                    0.00 + g,
                    v[0] + str(round(k(y_train, v[1][0]), 3)),
                    horizontalalignment="left",
                    family="monospace",
                    fontsize=8,
                )
                a.text(
                    0.70,
                    0.00 + g,
                    v[0] + str(round(k(y_test, v[1][1]), 3)),
                    horizontalalignment="left",
                    family="monospace",
                    fontsize=8,
                )
                g = g + 0.15

        elif self.metric == "classification_report":
            f = plt.figure(figsize=(12, 5), dpi=200)
            f.patch.set_facecolor(background_color)
            a = f.add_axes([0.0, 0.0, 1.0, 0.5])

            a.axis("off")
            for i in np.arange(0, 1, 0.05):
                a.text(0.50, i, "|", verticalalignment="bottom")
                a.text(i, 0.75, "_________", verticalalignment="bottom")
                a.text(i, 0.64, "_________", verticalalignment="bottom")

            a.text(
                0.2,
                0.8,
                "Training Data                         Test Data",
                family="monospace",
                verticalalignment="bottom",
                fontsize=20,
            )
            a.text(
                1.00,
                0.05,
                classification_report(self.y_test, self.P_test),
                horizontalalignment="right",
                family="monospace",
            )
            a.text(
                0.45,
                0.05,
                classification_report(self.y_train, self.P_train),
                horizontalalignment="right",
                family="monospace",
            )

        elif self.metric == "confusion_matrix":
            f, a = plt.subplots(1, 2, figsize=(12, 4), dpi=200)
            f.patch.set_facecolor(background_color)

            ls = [
                ["Training Data", self.X_train, self.y_train],
                ["Test Data", self.X_test, self.y_test],
            ]

            for i in range(len(ls)):
                a[i].grid(False)
                a[i].set_title(ls[i][0], fontsize=20)
                plot_confusion_matrix(
                    self.estimator, ls[i][1], ls[i][2], ax=a[i], cmap=plt.cm.Blues
                )

        elif self.metric == "Roc_Precision Recall":
            f = plt.figure()
            f, a = plt.subplots(1, 2, figsize=(14, 4), dpi=200)
            f.patch.set_facecolor(background_color)

            ls = [
                ["Train", self.X_train, self.y_train],
                ["Test.", self.X_test, self.y_test],
            ]
            ti = ["Roc Curve", "Precision Recall Curve"]

            a[0].plot([0, 1], [0, 1], "r--")
            for i in range(len(ls)):
                a[i].set_title(ti[i], fontsize=20)
                a[i].set_frame_on(False)
                a[i].grid(color=background_color)
                a[i].grid(color=color, linestyle=":", axis="y", zorder=0, dashes=(1, 5))

                plot_roc_curve(
                    self.estimator,
                    ls[i][1],
                    ls[i][2],
                    ax=a[0],
                    name=ls[i][0],
                    color=hue_color[i],
                )
                plot_precision_recall_curve(
                    self.estimator,
                    ls[i][1],
                    ls[i][2],
                    ax=a[1],
                    name=ls[i][0],
                    color=hue_color[i],
                )
                f.savefig(f"{self.name}-Roc.png", dpi=600, transparent=True)

        if self.create_df == True:
            ind = [
                "roc_auc_score",
                "average_precision_score",
                "accuracy_score",
                "recall_score",
                "precision_score",
                "f1_score",
                "log_loss",
                "brier_score_loss",
                "hamming_loss",
                "No. of obs.",
            ]
            self.metrics_df = pd.DataFrame(
                index=ind,
                data=[
                    [
                        roc_auc_score(self.y_train, self.PB_train),
                        roc_auc_score(self.y_test, self.PB_test),
                    ],
                    [
                        average_precision_score(self.y_train, self.PB_train),
                        average_precision_score(self.y_test, self.PB_test),
                    ],
                    [
                        accuracy_score(self.y_train, self.P_train),
                        accuracy_score(self.y_test, self.P_test),
                    ],
                    [
                        recall_score(self.y_train, self.P_train),
                        recall_score(self.y_test, self.P_test),
                    ],
                    [
                        precision_score(self.y_train, self.P_train),
                        precision_score(self.y_test, self.P_test),
                    ],
                    [
                        f1_score(self.y_train, self.P_train, average="binary"),
                        f1_score(self.y_test, self.P_test, average="binary"),
                    ],
                    [
                        log_loss(self.y_train, self.PB_train),
                        log_loss(self.y_test, self.PB_test),
                    ],
                    [
                        brier_score_loss(self.y_train, self.PB_train),
                        brier_score_loss(self.y_test, self.PB_test),
                    ],
                    [
                        hamming_loss(self.y_train, self.P_train),
                        hamming_loss(self.y_test, self.P_test),
                    ],
                    [len(self.y_train), len(self.y_test)],
                ],
                columns=["Train", "Test"],
            ).round(3)

            def names_of_object(arg):
                results = [
                    n
                    for n, v in globals().items()
                    if v is arg and not n.startswith("_")
                ]
                return results[0] if len(results) == 1 else results if results else None

            # train
            self.metrics_df["Estimator Name"] = self.name
            df_compared_train = self.metrics_df.drop(
                ["Test", "Estimator Name"], axis=1, inplace=False
            )
            df_compared_train = df_compared_train.rename(
                columns={"Train": self.metrics_df["Estimator Name"][0]}
            ).T
            self.df_compare_train = self.df_compare_train.append(df_compared_train)

            # test
            self.metrics_df["Estimator Name"] = self.name
            df_compared_test = self.metrics_df.drop(
                ["Train", "Estimator Name"], axis=1, inplace=False
            )
            df_compared_test = df_compared_test.rename(
                columns={"Test": self.metrics_df["Estimator Name"][0]}
            ).T
            self.df_compare_test = self.df_compare_test.append(df_compared_test)

        else:
            self.metrics_df = "File not created"
