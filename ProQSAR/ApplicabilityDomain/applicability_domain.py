import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.svm import OneClassSVM
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator


class ApplicabilityDomain(BaseEstimator):
    """
    A class to determine the applicability domain of a model using different methods.

    Methods:
        - ocsvm: One-Class Support Vector Machine
        - knn: k-Nearest Neighbors
        - lof: Local Outlier Factor
    """

    def __init__(
        self,
        activity_col: Optional[str] = None,
        id_col: Optional[str] = None,
        method: str = "lof",
        rate_of_outliers: float = 0.01,
        gamma="auto",
        nu=0.5,
        n_neighbors=10,
        metric="minkowski",
        p=2,
        save_dir: Optional[str] = "Project/ApplicabilityDomain",
        deactivate: bool = False,
    ):
        """
        Initialize the applicability domain with the specified method and parameters.
        ----------
        Parameters
        ----------
        method_name: str, default 'ocsvm'
            The name of method to set AD. 'knn', 'lof', or 'ocsvm'
        rate_of_outliers: float, default 0.01
            Rate of outlier samples. This is used to set threshold.
        gamma : (only for 'ocsvm') float, default ’auto’
            Kernel coefficient for ‘rbf’.
            Current default is ‘auto’ which optimize gamma to maximize variance in Gram matrix
        nu : (only for 'ocsvm') float, default 0.5
            An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
            Should be in the interval (0, 1]. By default 0.5 will be taken.
            https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM
        n_neighbors: (only for 'knn' and 'lof') int, default 10
            Number of neighbors to use for each query
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
        metric : string or callable, default ‘minkowski’
            Metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
        p : integer, default 2
            Parameter for the Minkowski metric from sklearn.metrics.pairwise.pairwise_distances.
            When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2.
            For arbitrary p, minkowski_distance (l_p) is used.
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
        save_dir: (Optional[str]):
            Directory to save fitted ApplicabilityDomain and prediction results.
        """
        if method not in ["knn", "lof", "ocsvm"]:
            logging.error(
                f"Invalid method: {method}. Choose from 'knn', 'lof', or 'ocsvm'."
            )
            raise ValueError(f"Invalid method: {method}.")

        self.method = method
        self.activity_col = activity_col
        self.id_col = id_col
        self.rate_of_outliers = rate_of_outliers
        self.gamma = gamma
        self.nu = nu
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.save_dir = save_dir
        self.deactivate = deactivate
        self.ad = None
        self.offset = None

    def fit(self, data: pd.DataFrame):
        """
        Fit the applicability domain model based on the selected method.

        Args:
            data (pd.DataFrame): Training dataset for fitting the model.
        """
        if self.deactivate:
            logging.info("ApplicabilityDomain is deactivated. Skipping fit.")
            return None
        try:
            X_data = data.drop(
                [self.activity_col, self.id_col], axis=1, errors="ignore"
            )
            x = np.array(X_data)

            if self.method == "ocsvm":
                if self.gamma == "auto":
                    ocsvm_gammas = 2 ** np.arange(-20, 11, dtype=float)
                    variance_of_gram_matrix = []
                    for index, ocsvm_gamma in enumerate(ocsvm_gammas):
                        gram_matrix = np.exp(
                            -ocsvm_gamma * cdist(x, x, metric="seuclidean")
                        )
                        variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
                    self.optimal_gamma = ocsvm_gammas[
                        variance_of_gram_matrix.index(max(variance_of_gram_matrix))
                    ]
                else:
                    self.optimal_gamma = self.gamma
                self.ad = OneClassSVM(
                    kernel="rbf", gamma=self.optimal_gamma, nu=self.nu
                )
                self.ad.fit(x)
                ad_values = np.ndarray.flatten(self.ad.decision_function(x))

            elif self.method == "knn":
                self.ad = NearestNeighbors(n_neighbors=self.n_neighbors)
                self.ad.fit(x)
                knn_dist_all, _ = self.ad.kneighbors()
                ad_values = 1 / (knn_dist_all.mean(axis=1) + 1)
            elif self.method == "lof":
                self.ad = LocalOutlierFactor(
                    novelty=True, contamination=self.rate_of_outliers
                )
                self.ad.fit(x)
                ad_values = self.ad.negative_outlier_factor_ - self.ad.offset_

            self.offset = np.percentile(ad_values, 100 * self.rate_of_outliers)

            if self.save_dir:
                os.makedirs(self.save_dir, exist_ok=True)
                with open(f"{self.save_dir}/applicability_domain.pkl", "wb") as file:
                    pickle.dump(self, file)

            logging.info("ApplicabilityDomain model fitted successfully.")
            return self

        except Exception as e:
            logging.error(f"Error fitting the model: {e}")
            raise

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the applicability domain for new data.

        Args:
            data (pd.DataFrame): Dataset for making predictions.

        Returns:
            pd.DataFrame: DataFrame containing prediction results.
        """
        if self.ad is None:
            raise NotFittedError("Model is not fitted. Call 'fit' before predicting.")

        try:
            X_data = data.drop(
                [self.activity_col, self.id_col], axis=1, errors="ignore"
            )
            x = np.array(X_data)

            if self.method == "ocsvm":
                ad_values = np.ndarray.flatten(self.ad.decision_function(x))

            elif self.method == "knn":
                knn_dist_all, _ = self.ad.kneighbors(x)
                ad_values = 1 / (knn_dist_all.mean(axis=1) + 1)
            elif self.method == "lof":
                ad_values = np.ndarray.flatten(self.ad.decision_function(x))

            result = [
                "in" if (value - self.offset) > 0 else "out" for value in ad_values
            ]
            result_df = pd.DataFrame({"Applicability domain": result})

            if self.id_col in data.columns:
                result_df[self.id_col] = data[self.id_col].values

            if self.save_dir:
                os.makedirs(self.save_dir, exist_ok=True)
                result_df.to_csv(f"{self.save_dir}/ad_pred_result.csv", index=False)

            logging.info("Prediction completed successfully.")
            return result_df

        except Exception as e:
            logging.error(f"Error predicting applicability domain: {e}")
            raise
