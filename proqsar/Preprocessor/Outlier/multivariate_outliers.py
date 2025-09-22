import os
import pickle
import logging
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, List


class MultivariateOutliersHandler(BaseEstimator, TransformerMixin):
    """
    Detect and remove multivariate outliers from tabular datasets.

    The handler supports several algorithms for multivariate outlier detection:

      - ``"LocalOutlierFactor"``
      - ``"IsolationForest"``
      - ``"OneClassSVM"``
      - ``"RobustCovariance"`` (EllipticEnvelope with contamination=0.1)
      - ``"EmpiricalCovariance"`` (EllipticEnvelope with support_fraction=1)

    Outliers are identified during :meth:`fit` and removed during :meth:`transform`.

    :param activity_col: Name of the activity/target column to ignore when fitting.
    :type activity_col: Optional[str]
    :param id_col: Name of the identifier column to ignore when fitting.
    :type id_col: Optional[str]
    :param select_method: Algorithm name to use for detection.
                          One of {"LocalOutlierFactor","IsolationForest","OneClassSVM",
                          "RobustCovariance","EmpiricalCovariance"}.
    :type select_method: str
    :param n_jobs: Number of parallel jobs (where supported). Default is 1.
    :type n_jobs: int
    :param random_state: Random seed for reproducibility where applicable.
                         Default is 42.
    :type random_state: Optional[int]
    :param save_method: If True, save the fitted handler as a pickle.
    :type save_method: bool
    :param save_dir: Directory to store pickled handler / transformed data.
                     Default is "Project/MultivOutlierHandler".
    :type save_dir: Optional[str]
    :param save_trans_data: If True, save transformed DataFrame to CSV.
    :type save_trans_data: bool
    :param trans_data_name: Base filename for saving transformed CSV.
    :type trans_data_name: str
    :param deactivate: If True, disables the handler. Methods become no-ops.
    :type deactivate: bool

    :ivar multi_outlier_handler: The fitted estimator instance, or None.
    :vartype multi_outlier_handler: object | None
    :ivar features: List of feature column names used in fitting.
    :vartype features: pandas.Index | None
    :ivar data_fit: The feature matrix used at fit time.
    :vartype data_fit: pandas.DataFrame
    :ivar transformed_data: Last transformed DataFrame.
    :vartype transformed_data: pandas.DataFrame | None
    """

    def __init__(
        self,
        activity_col: Optional[str] = None,
        id_col: Optional[str] = None,
        select_method: str = "LocalOutlierFactor",
        n_jobs: int = 1,
        random_state: Optional[int] = 42,
        save_method: bool = False,
        save_dir: Optional[str] = "Project/MultivOutlierHandler",
        save_trans_data: bool = False,
        trans_data_name: str = "trans_data",
        deactivate: bool = False,
    ) -> None:
        self.activity_col = activity_col
        self.id_col = id_col
        self.select_method = select_method
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.save_method = save_method
        self.save_dir = save_dir
        self.save_trans_data = save_trans_data
        self.trans_data_name = trans_data_name
        self.deactivate = deactivate
        self.multi_outlier_handler = None
        self.features = None

    def fit(self, data: pd.DataFrame, y=None) -> "MultivariateOutliersHandler":
        """
        Fit the selected outlier detector on the given dataset.

        :param data: Input DataFrame containing features and optional id/activity columns.
        :type data: pandas.DataFrame
        :param y: Ignored (sklearn API compatibility).
        :type y: Any
        :return: Fitted handler (self).
        :rtype: MultivariateOutliersHandler
        :raises ValueError: If `select_method` is not supported.
        :raises Exception: If fitting fails unexpectedly.
        """
        if self.deactivate:
            logging.info("MultivariateOutliersHandler is deactivated. Skipping fit.")
            return self

        try:
            self.features = data.drop(
                columns=[self.id_col, self.activity_col], errors="ignore"
            ).columns
            self.data_fit = data[self.features]

            method_map = {
                "LocalOutlierFactor": LocalOutlierFactor(
                    n_neighbors=20, n_jobs=self.n_jobs
                ),
                "IsolationForest": IsolationForest(
                    n_estimators=100,
                    contamination="auto",
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                ),
                "OneClassSVM": OneClassSVM(),
                "RobustCovariance": EllipticEnvelope(
                    contamination=0.1, random_state=self.random_state
                ),
                "EmpiricalCovariance": EllipticEnvelope(
                    contamination=0.1,
                    support_fraction=1,
                    random_state=self.random_state,
                ),
            }

            if self.select_method not in method_map:
                raise ValueError(
                    f"Unsupported method: {self.select_method}. "
                    f"Choose from {list(method_map.keys())}."
                )

            self.multi_outlier_handler = method_map[self.select_method].fit(
                self.data_fit.values
            )

            if self.save_method:
                os.makedirs(self.save_dir, exist_ok=True)
                with open(f"{self.save_dir}/multi_outlier_handler.pkl", "wb") as file:
                    pickle.dump(self, file)
                logging.info(
                    f"Handler saved at {self.save_dir}/multi_outlier_handler.pkl"
                )

        except Exception as e:
            logging.error(f"Error fitting MultivariateOutliersHandler: {e}")
            raise

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows flagged as outliers.

        - For ``LocalOutlierFactor``, supports both in-sample and novelty detection.
        - For other estimators, uses :meth:`predict` with outliers = -1.

        :param data: DataFrame with the same feature columns as used in :meth:`fit`.
        :type data: pandas.DataFrame
        :return: DataFrame with outlier rows removed.
        :rtype: pandas.DataFrame
        :raises NotFittedError: If called before :meth:`fit`.
        :raises Exception: If transformation fails unexpectedly.
        """
        if self.deactivate:
            self.transformed_data = data
            logging.info("MultivariateOutliersHandler is deactivated. Returning input.")
            return data

        try:
            if self.multi_outlier_handler is None:
                raise NotFittedError("Handler not fitted. Call 'fit' first.")

            if self.select_method == "LocalOutlierFactor":
                novelty = not data[self.features].equals(self.data_fit)
                self.multi_outlier_handler.set_params(novelty=novelty)

                if novelty:
                    self.multi_outlier_handler.fit(self.data_fit.values)
                    outliers = (
                        self.multi_outlier_handler.predict(data[self.features].values)
                        == -1
                    )
                else:
                    outliers = (
                        self.multi_outlier_handler.fit_predict(
                            data[self.features].values
                        )
                        == -1
                    )
            else:
                outliers = (
                    self.multi_outlier_handler.predict(data[self.features].values) == -1
                )

            transformed_data = data[~outliers]

            if self.save_trans_data:
                os.makedirs(self.save_dir, exist_ok=True)
                csv_name = self.trans_data_name
                if os.path.exists(f"{self.save_dir}/{csv_name}.csv"):
                    base, ext = os.path.splitext(self.trans_data_name)
                    counter = 1
                    while os.path.exists(
                        f"{self.save_dir}/{base} ({counter}){ext}.csv"
                    ):
                        counter += 1
                    csv_name = f"{base} ({counter}){ext}"
                transformed_data.to_csv(f"{self.save_dir}/{csv_name}.csv", index=False)
                logging.info(
                    f"Transformed data saved at {self.save_dir}/{csv_name}.csv"
                )

            self.transformed_data = transformed_data
            return transformed_data

        except Exception as e:
            logging.error(
                f"Error transforming data in MultivariateOutliersHandler: {e}"
            )
            raise

    def fit_transform(self, data: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit the outlier detector and immediately transform the data.

        :param data: Input dataset to fit and filter.
        :type data: pandas.DataFrame
        :param y: Ignored (sklearn API compatibility).
        :type y: Any
        :return: Transformed DataFrame with outliers removed.
        :rtype: pandas.DataFrame
        """
        if self.deactivate:
            logging.info("MultivariateOutliersHandler is deactivated. Returning input.")
            return data

        self.fit(data)
        return self.transform(data)

    @staticmethod
    def compare_multivariate_methods(
        data1: pd.DataFrame,
        data2: Optional[pd.DataFrame] = None,
        data1_name: str = "data1",
        data2_name: str = "data2",
        activity_col: Optional[str] = None,
        id_col: Optional[str] = None,
        methods_to_compare: Optional[List[str]] = None,
        save_dir: Optional[str] = "Project/OutlierHandler",
    ) -> pd.DataFrame:
        """
        Compare multiple outlier detection methods across datasets.

        :param data1: Primary dataset.
        :type data1: pandas.DataFrame
        :param data2: Optional second dataset for evaluation.
        :type data2: Optional[pandas.DataFrame]
        :param data1_name: Label for dataset1 in results.
        :type data1_name: str
        :param data2_name: Label for dataset2 in results.
        :type data2_name: str
        :param activity_col: Activity/target column name to exclude.
        :type activity_col: Optional[str]
        :param id_col: Identifier column name to exclude.
        :type id_col: Optional[str]
        :param methods_to_compare: List of algorithms to compare. If None, defaults to all.
        :type methods_to_compare: Optional[List[str]]
        :param save_dir: If set, saves comparison results CSV to this directory.
        :type save_dir: Optional[str]
        :return: Summary table with rows removed for each method and dataset.
        :rtype: pandas.DataFrame
        :raises Exception: If comparison fails unexpectedly.
        """
        try:
            comparison_data = []
            default_methods = [
                "LocalOutlierFactor",
                "IsolationForest",
                "OneClassSVM",
                "RobustCovariance",
                "EmpiricalCovariance",
            ]
            methods_to_compare = methods_to_compare or default_methods

            for method in methods_to_compare:
                handler = MultivariateOutliersHandler(
                    id_col=id_col, activity_col=activity_col, select_method=method
                )
                handler.fit(data1)

                transformed_data1 = handler.transform(data1)
                comparison_data.append(
                    {
                        "Method": method,
                        "Dataset": data1_name,
                        "Original Rows": data1.shape[0],
                        "After Handling Rows": transformed_data1.shape[0],
                        "Removed Rows": data1.shape[0] - transformed_data1.shape[0],
                    }
                )

                if data2 is not None:
                    transformed_data2 = handler.transform(data2)
                    comparison_data.append(
                        {
                            "Method": method,
                            "Dataset": data2_name,
                            "Original Rows": data2.shape[0],
                            "After Handling Rows": transformed_data2.shape[0],
                            "Removed Rows": data2.shape[0] - transformed_data2.shape[0],
                        }
                    )

            comparison_table = pd.DataFrame(comparison_data)

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                comparison_table.to_csv(
                    f"{save_dir}/compare_multivariate_methods.csv", index=False
                )
                logging.info(
                    f"Comparison table saved at {save_dir}/compare_multivariate_methods.csv"
                )

            return comparison_table

        except Exception as e:
            logging.error(f"Error comparing multivariate methods: {e}")
            raise
