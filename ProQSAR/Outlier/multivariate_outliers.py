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

    The handler supports fitting one of several multivariate outlier detectors
    and then removing rows flagged as outliers in `transform`.

    Supported methods (select_method):
      - "LocalOutlierFactor"
      - "IsolationForest"
      - "OneClassSVM"
      - "RobustCovariance" (EllipticEnvelope with contamination=0.1)
      - "EmpiricalCovariance" (EllipticEnvelope with support_fraction=1)

    Parameters
    ----------
    activity_col : Optional[str]
        Name of the activity/target column to ignore when fitting (default None).
    id_col : Optional[str]
        Name of the id column to ignore when fitting (default None).
    select_method : str
        The chosen multivariate outlier detection method (default "LocalOutlierFactor").
    n_jobs : int
        Number of parallel jobs to use where supported (default 1).
    random_state : Optional[int]
        Random seed for algorithms that accept it (default 42).
    save_method : bool
        If True, save the fitted handler object as a pickle to `save_dir`.
    save_dir : Optional[str]
        Directory where pickles / transformed data will be saved (default "Project/MultivOutlierHandler").
    save_trans_data : bool
        If True, save transformed data to CSV after `transform`.
    trans_data_name : str
        Base filename to use when saving transformed data (default "trans_data").
    deactivate : bool
        If True, the handler is deactivated and `fit`/`transform` become no-ops.

    Attributes
    ----------
    multi_outlier_handler : object | None
        The fitted outlier detection estimator instance.
    features : Index | None
        List of feature column names used during fit (excludes id/activity).
    data_fit : pd.DataFrame
        The feature matrix used for fitting (kept so some detectors can operate in novelty mode).
    transformed_data : pd.DataFrame | None
        Stores the last transformed DataFrame after `transform()` is called.
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
        Fit the selected multivariate outlier detection method on the provided data.

        The handler will ignore `id_col` and `activity_col` (if present) when fitting.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing feature columns (plus optional id/activity columns).
        y : Optional[pd.Series]
            Ignored — present for sklearn compatibility.

        Returns
        -------
        MultivariateOutliersHandler
            The fitted handler (self).

        Raises
        ------
        Exception
            Unexpected exceptions are logged and re-raised.
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
                    n_neighbors=20,
                    n_jobs=self.n_jobs,
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
                    f"MultivariateOutliersHandler: Unsupported method: {self.select_method}"
                )

            self.multi_outlier_handler = method_map[self.select_method].fit(
                self.data_fit.values
            )

            logging.info(
                f"MultivariateOutliersHandler: Using '{self.select_method}' method."
            )

            if self.save_method:
                if self.save_dir and not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir, exist_ok=True)
                with open(f"{self.save_dir}/multi_outlier_handler.pkl", "wb") as file:
                    pickle.dump(self, file)
                logging.info(
                    f"MultivariateOutliersHandler saved at: {self.save_dir}/multi_outlier_handler.pkl"
                )

        except Exception as e:
            logging.error(f"Error in fitting: {e}")
            raise

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows detected as multivariate outliers from `data`.

        Behavior varies slightly depending on the chosen method:
          - For LocalOutlierFactor, the handler attempts to detect whether the
            incoming `data` is novel (different from the data used to fit). If so,
            it sets novelty mode and re-fits if necessary.
          - For other estimators, it uses `predict` to mark outliers (-1).

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame to be filtered of outliers. Must contain the same feature
            columns that were present at `fit` time.

        Returns
        -------
        pd.DataFrame
            The input DataFrame with outlier rows removed.

        Raises
        ------
        NotFittedError
            If called before the handler was fitted.
        Exception
            Unexpected exceptions are logged and re-raised.
        """
        if self.deactivate:
            self.transformed_data = data
            logging.info(
                "MultivariateOutlierHandler is deactivated. Returning unmodified data."
            )
            return data

        try:
            if self.multi_outlier_handler is None:
                raise NotFittedError(
                    "MultivariateOutlierHandler is not fitted yet. Call 'fit' before using this method."
                )

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
                if self.save_dir and not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir, exist_ok=True)
                if os.path.exists(f"{self.save_dir}/{self.trans_data_name}.csv"):
                    base, ext = os.path.splitext(self.trans_data_name)
                    counter = 1
                    new_filename = f"{base} ({counter}){ext}"

                    while os.path.exists(f"{self.save_dir}/{new_filename}.csv"):
                        counter += 1
                        new_filename = f"{base} ({counter}){ext}"

                    csv_name = new_filename

                else:
                    csv_name = self.trans_data_name

                transformed_data.to_csv(f"{self.save_dir}/{csv_name}.csv")
                logging.info(
                    f"MultivariateOutliersHandler: Transformed data saved at: {self.save_dir}/{csv_name}.csv"
                )

            self.transformed_data = transformed_data

            return transformed_data
        except Exception as e:
            logging.error(f"Error in transforming the data: {e}")
            raise

    def fit_transform(self, data: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Convenience method that fits the outlier detector on `data` and then
        transforms `data` to remove detected outliers.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame to fit and transform.
        y : Optional[pd.Series]
            Ignored — present for sklearn compatibility.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame with outliers removed.
        """
        if self.deactivate:
            logging.info(
                "MultivariateOutlierHandler is deactivated. Returning unmodified data."
            )
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
        Compare several multivariate outlier detection methods by applying each
        to `data1` (and optionally `data2`) and returning a summary table with
        the number of rows removed by each method.

        Parameters
        ----------
        data1 : pd.DataFrame
            Primary dataset to evaluate outlier removal on.
        data2 : Optional[pd.DataFrame]
            Optional second dataset to evaluate using the same fitted handlers.
        data1_name : str
            Name label for data1 in the output table.
        data2_name : str
            Name label for data2 in the output table (if provided).
        activity_col : Optional[str]
            Activity/target column name (optional).
        id_col : Optional[str]
            Identifier column name (optional).
        methods_to_compare : Optional[List[str]]
            List of method names to compare. If None, a default set is used.
        save_dir : Optional[str]
            If provided, saves the comparison table CSV into this directory.

        Returns
        -------
        pd.DataFrame
            A DataFrame summarizing, for each method and dataset, the original
            row count, the row count after handling, and the number of removed rows.
        """
        try:
            comparison_data = []
            methods = [
                "LocalOutlierFactor",
                "IsolationForest",
                "OneClassSVM",
                "RobustCovariance",
                "EmpiricalCovariance",
            ]
            methods_to_compare = methods_to_compare or methods

            for method in methods_to_compare:
                multi_outlier_handler = MultivariateOutliersHandler(
                    id_col=id_col,
                    activity_col=activity_col,
                    select_method=method,
                )
                multi_outlier_handler.fit(data1)

                transformed_data1 = multi_outlier_handler.transform(data1)
                comparison_data.append(
                    {
                        "Method": method,
                        "Dataset": data1_name,
                        "Original Rows": data1.shape[0],
                        "After Handling Rows": transformed_data1.shape[0],
                        "Removed Rows": data1.shape[0] - transformed_data1.shape[0],
                    }
                )

                comparison_table = pd.DataFrame(comparison_data)

                if data2 is not None:
                    transformed_data2 = multi_outlier_handler.transform(data2)
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
                        (f"{save_dir}/compare_multivariate_methods.csv")
                    )
                    logging.info(
                        f"Transformed data saved at: {save_dir}/compare_multivariate_methods.csv"
                    )

            logging.info("Comparison of multivariate methods completed.")
            return comparison_table

        except Exception as e:
            logging.error(f"Error in comparing methods: {e}")
            raise
