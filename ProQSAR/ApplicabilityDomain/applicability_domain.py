import sys
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from scipy.spatial.distance import cdist


class ApplicabilityDomain():
    def __init__(
        self, 
        method: str = 'lof',
        activity_col: Optional[str] = None,
        id_col: Optional[str] = None,
        rate_of_outliers: float = 0.01, 
        gamma='auto', 
        nu=0.5, 
        n_neighbors=10,
        metric='minkowski', 
        p=2):

        """
        Applicability Domain (AD)
        
        Parameters
        ----------
        method_name: str, default 'ocsvm'
            The name of method to set AD. 'knn', 'lof', or 'ocsvm'
        rate_of_outliers: float, default 0.01
            Rate of outlier samples. This is used to set threshold
        gamma : (only for 'ocsvm') float, default ’auto’
            Kernel coefficient for ‘rbf’. Current default is ‘auto’ which optimize gamma to maximize variance in Gram matrix
        nu : (only for 'ocsvm') float, default 0.5
            An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. Should be in the interval (0, 1]. By default 0.5 will be taken.
            https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM
        n_neighbors: (only for 'knn' and 'lof') int, default 10
            Number of neighbors to use for each query
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
        metric : string or callable, default ‘minkowski’
            Metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
        p : integer, default 2
            Parameter for the Minkowski metric from sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
        """
        
        if method != 'knn' and method != 'lof' and method != 'ocsvm':
            sys.exit('There is no ad method named \'{0}\'. Please check the variable of method_name.'.format(method))
            
        self.method = method
        self.activity_col = activity_col
        self.id_col = id_col
        self.rate_of_outliers = rate_of_outliers
        self.gamma = gamma
        self.nu = nu
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p

    def fit(self, data):

        X_data = data.drop([self.activity_col, self.id_col], axis=1, errors='ignore')
        x = np.array(X_data)
        
        if self.method == 'ocsvm':
            if self.gamma == 'auto':
                ocsvm_gammas = 2 ** np.arange(-20, 11, dtype=float)
                variance_of_gram_matrix = []
                for index, ocsvm_gamma in enumerate(ocsvm_gammas):
                    gram_matrix = np.exp(-ocsvm_gamma * cdist(x, x, metric='seuclidean'))
                    variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
                self.optimal_gamma = ocsvm_gammas[variance_of_gram_matrix.index(max(variance_of_gram_matrix))]
            else:
                self.optimal_gamma = self.gamma
            self.ad = OneClassSVM(kernel='rbf', gamma=self.optimal_gamma, nu=self.nu)
            self.ad.fit(x)
            ad_values = np.ndarray.flatten(self.ad.decision_function(x))
            
        elif self.method == 'knn':
            self.ad = NearestNeighbors(n_neighbors=self.n_neighbors)
            self.ad.fit(x)
            knn_dist_all, _ = self.ad.kneighbors()
            ad_values = 1 / (knn_dist_all.mean(axis=1) + 1)
        elif self.method == 'lof':
            self.ad = LocalOutlierFactor(novelty=True, contamination=self.rate_of_outliers)
            self.ad.fit(x)
            ad_values = self.ad.negative_outlier_factor_ - self.ad.offset_
            
        self.offset = np.percentile(ad_values, 100 * self.rate_of_outliers)

        return self


    def predict(self, data):

        X_data = data.drop([self.activity_col, self.id_col], axis=1, errors='ignore')
        x = np.array(X_data)
        
        if self.method == 'ocsvm':
            ad_values = np.ndarray.flatten(self.ad.decision_function(x))

        elif self.method == 'knn':
            knn_dist_all, _ = self.ad.kneighbors(x)
            ad_values = 1 / (knn_dist_all.mean(axis=1) + 1)
        elif self.method == 'lof':
            ad_values = np.ndarray.flatten(self.ad.decision_function(x))

        result = ['in' if (value - self.offset) > 0 else 'out' for value in ad_values]
        result_df = pd.DataFrame({'Applicability domain': result})

        if self.id_col in data.columns:
            result_df[self.id_col] = data[self.id_col].values
        
        return result_df
