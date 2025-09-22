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
    Determine the applicability domain (AD) of a dataset using one of several methods.

    Supported methods:
      - ``ocsvm``: One-Class Support Vector Machine
      - ``knn``: k-Nearest Neighbors
      - ``lof``: Local Outlier Factor

    :param activity_col: Column name for activity or target variable.
    :type activity_col: Optional[str], optional
    :param id_col: Column name for unique identifier.
    :type id_col: Optional[str], optional
    :param method: Method for AD estimation. One of {"ocsvm", "knn", "lof"}.
                   Default is "lof".
    :type method: str, optional
    :param rate_of_outliers: Proportion of samples considered outliers. Used to set threshold.
    :type rate_of_outliers: float, optional
    :param gamma: Kernel coefficient for RBF kernel in ``ocsvm``.
                  If "auto", optimized to maximize variance in the Gram matrix.
    :type gamma: str or float, optional
    :param nu: ``ocsvm`` parameter controlling training error fraction and support vector fraction.
               Must be in (0, 1].
    :type nu: float, optional
    :param n_neighbors: Number of neighbors for ``knn`` and ``lof`` methods.
    :type n_neighbors: int, optional
    :param metric: Distance metric for ``knn`` (see scikit-learn NearestNeighbors docs).
    :type metric: str or callable, optional
    :param p: Parameter for Minkowski metric (1 = Manhattan, 2 = Euclidean).
    :type p: int, optional
    :param save_dir: Directory to save fitted AD object and prediction results.
    :type save_dir: Optional[str], optional
    :param deactivate: If True, AD estimation is skipped.
    :type deactivate: bool, optional

    :ivar ad: The fitted model object (e.g., OneClassSVM, NearestNeighbors, LocalOutlierFactor).
    :vartype ad: object or None
    :ivar offset: Decision threshold based on `rate_of_outliers`.
    :vartype offset: float or None
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
        save_dir: Optional[str] = None,
        deactivate: bool = False,
    ):
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
        Fit the applicability domain model using the chosen method.

        :param data: Training dataset for fitting the model.
        :type data: pandas.DataFrame
        :return: Self with fitted AD model and computed threshold.
        :rtype: ApplicabilityDomain
        :raises ValueError: If method is invalid.
        :raises Exception: If fitting fails for other reasons.

        .. note::
           - For ``ocsvm``, ``gamma="auto"`` selects the gamma value maximizing Gram matrix variance.
           - For ``knn``, inverse mean neighbor distance is used.
           - For ``lof``, scikit-learn’s LocalOutlierFactor in novelty mode is used.
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
                    for ocsvm_gamma in ocsvm_gammas:
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

            logging.info(f"ApplicabilityDomain: Using '{self.method}' method.")

            if self.save_dir:
                os.makedirs(self.save_dir, exist_ok=True)
                with open(f"{self.save_dir}/applicability_domain.pkl", "wb") as file:
                    pickle.dump(self, file)

            return self

        except Exception as e:
            logging.error(f"Error fitting the model: {e}")
            raise

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict whether new samples fall inside or outside the applicability domain.

        :param data: Dataset for making predictions.
        :type data: pandas.DataFrame
        :return: DataFrame with prediction results. Includes a column
        ``"Applicability domain"`` with values {"in", "out"}.
        :rtype: pandas.DataFrame
        :raises NotFittedError: If ``fit`` has not been called before.
        :raises Exception: If prediction fails.

        .. code-block:: python

            ad = ApplicabilityDomain(method="ocsvm")
            ad.fit(train_df)
            results = ad.predict(test_df)
            print(results.head())
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

            return result_df

        except Exception as e:
            logging.error(f"Error predicting applicability domain: {e}")
            raise
