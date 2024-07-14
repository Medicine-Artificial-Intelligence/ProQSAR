import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt
import seaborn as sns


class MultivariateOutliers:
    """
    Class for removing multivariate outliers using various methods.

    Parameters:
    ----------
    data_train: pandas.DataFrame
        Training data after handling univariate outliers.
    data_test: pandas.DataFrame
        Test data after handling univariate outliers.
    method: str
        Method for outlier detection. Options include 'LocalOutlierFactor',
        'IsolationForest', 'OneClassSVM', 'RobustCovariance', 'EmpiricalCovariance', 'Compare'.
    n_jobs: int
        Number of CPUs to use for outlier detection (-1 uses all available CPUs).
    id_col: str
        Column name of the identifier in the dataset.
    activity_col: str
        Name of the activity column in the dataset.
    save_path: str or None
        Path to save the comparison chart (if None, chart is not saved).

    Attributes:
    ----------
    data_train: pandas.DataFrame
        Training data after removing multivariate outliers.
    data_test: pandas.DataFrame
        Test data after removing multivariate outliers.
    """

    def __init__(
        self,
        data_train,
        data_test,
        method="LocalOutlierFactor",
        n_jobs=4,
        id_col=None,
        activity_col=None,
        save_path=None,
    ):
        self.data_train = data_train
        self.data_test = data_test
        self.method = method
        self.n_jobs = n_jobs
        self.id_col = id_col
        self.activity_col = activity_col
        self.save_path = save_path

    def fit(self):
        # Exclude id_col and activity_col during outlier detection
        features_to_use = self.data_train.drop(
            [self.id_col, self.activity_col], axis=1
        ).columns

        # Apply the chosen method
        if self.method == "LocalOutlierFactor":
            self.data_train, self.data_test = self._apply_method(
                LocalOutlierFactor(n_neighbors=20, n_jobs=self.n_jobs), features_to_use
            )
        elif self.method == "IsolationForest":
            self.data_train, self.data_test = self._apply_method(
                IsolationForest(
                    n_estimators=100,
                    contamination="auto",
                    random_state=42,
                    n_jobs=self.n_jobs,
                ),
                features_to_use,
            )
        elif self.method == "OneClassSVM":
            self.data_train, self.data_test = self._apply_method(
                OneClassSVM(), features_to_use
            )
        elif self.method == "RobustCovariance":
            self.data_train, self.data_test = self._apply_method(
                EllipticEnvelope(contamination=0.1, random_state=42), features_to_use
            )
        elif self.method == "EmpiricalCovariance":
            self.data_train, self.data_test = self._apply_method(
                EllipticEnvelope(
                    contamination=0.1, support_fraction=0.1, random_state=42
                ),
                features_to_use,
            )
        else:
            self.compare_multivariate_method()
        return self.data_train, self.data_test

    def _apply_method(self, model, features):
        """
        Applies the given outlier detection model to the specified features.
        """
        # Special handling for Local Outlier Factor
        if isinstance(model, LocalOutlierFactor):
            # Fit the model to the training data
            train_data = self.data_train[features].to_numpy()
            test_data = self.data_test[features].to_numpy()
            # model.fit(train_data)

            # Apply to training data using fit_predict
            train_outliers = model.fit_predict(train_data) == -1

            filtered_train_data = self.data_train[~train_outliers]

            # For LOF, the model needs to be refitted for novelty detection on test data
            lof_novelty = LocalOutlierFactor(
                n_neighbors=20, novelty=True, n_jobs=self.n_jobs
            )
            lof_novelty.fit(train_data)  # Fit on training data
            test_outliers = lof_novelty.predict(test_data) == -1
            filtered_test_data = self.data_test[~test_outliers]

        else:
            # Fit the model to the training data
            model.fit(self.data_train[features])

            # Apply to training data
            train_outliers = model.predict(self.data_train[features]) == -1
            filtered_train_data = self.data_train[~train_outliers]

            # Apply to test data
            test_outliers = model.predict(self.data_test[features]) == -1
            filtered_test_data = self.data_test[~test_outliers]

        return filtered_train_data, filtered_test_data

    def compare_multivariate_method(self):
        """
        Compares different multivariate outlier detection methods visually.
        """
        features_to_use = self.data_train.columns.drop([self.id_col, self.activity_col])

        # Initialize models with appropriate settings
        methods = {
            "LocalOutlierFactor": LocalOutlierFactor(
                n_neighbors=20, n_jobs=self.n_jobs
            ),
            "IsolationForest": IsolationForest(
                n_estimators=100,
                contamination="auto",
                random_state=42,
                n_jobs=self.n_jobs,
            ),
            "OneClassSVM": OneClassSVM(),
            "RobustCovariance": EllipticEnvelope(contamination=0.1, random_state=42),
            "EmpiricalCovariance": EllipticEnvelope(
                contamination=0.1, support_fraction=0.1, random_state=42
            ),
        }

        outliers_count = {}
        for method_name, model in methods.items():
            if method_name == "LocalOutlierFactor":
                # Use fit_predict for LocalOutlierFactor
                outliers = self.data_train[features_to_use][
                    model.fit_predict(self.data_train[features_to_use]) == -1
                ]
            else:
                model.fit(self.data_train[features_to_use])
                outliers = self.data_train[features_to_use][
                    model.predict(self.data_train[features_to_use]) == -1
                ]
            outliers_count[method_name] = len(outliers)

        # Visualization
        plt.figure(figsize=(12, 6))
        sns.barplot(
            x=list(outliers_count.keys()),
            y=list(outliers_count.values()),
            palette="viridis",
        )
        plt.title(
            "Comparison of Multivariate Outlier Detection Methods",
            fontsize=18,
            fontweight="bold",
        )
        plt.xlabel("Methods", fontsize=14)
        plt.ylabel("Number of Outliers Detected", fontsize=14)
        plt.xticks(rotation=45)
        sns.despine()

        # Optionally, save the figure
        if self.save_path is not None:
            plt.savefig(
                f"{self.save_path}/multivariate_outlier_comparison.png",
                dpi=300,
                bbox_inches="tight",
            )

        plt.show()
