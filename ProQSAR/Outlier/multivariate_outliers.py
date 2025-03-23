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
    A class to handle multivariate outlier detection using various algorithms
    including Local Outlier Factor, Isolation Forest, One-Class SVM, and
    Elliptic Envelope methods.

    Attributes:
        id_col (Optional[str]): Column name for the unique identifier.
        activity_col (Optional[str]): Column name for activity or target variable.
        select_method (str): Method to use for outlier detection.
        novelty (bool): If True, enables novelty detection for certain methods.
        n_jobs (int): Number of parallel jobs to run (-1 uses all processors).
        save_method (bool): If True, saves the fitted model to disk.
        save_dir (Optional[str]): Directory path to save model and transformed data.
        save_trans_data (bool): If True, saves the transformed data to disk.
        trans_data_name (str): Name for the transformed data file.
        multi_outlier_handler (Optional[object]): Initialized outlier detection model.
        features (Optional[pd.Index]): Feature columns used in outlier detection.
    """

    def __init__(
        self,
        activity_col: Optional[str] = None,
        id_col: Optional[str] = None,
        select_method: str = "LocalOutlierFactor",
        novelty: bool = False,
        n_jobs: int = -1,
        save_method: bool = False,
        save_dir: Optional[str] = "Project/MultivOutlierHandler",
        save_trans_data: bool = False,
        trans_data_name: str = "trans_data",
        deactivate: bool = False,
    ) -> None:
        """
        Initializes MultivariateOutliersHandler with given parameters.

        Args:
            activity_col (Optional[str]): Column name for activity or target variable.
            id_col (Optional[str]): Column name for the unique identifier.
            select_method (str): Method to use for outlier detection.
            novelty (bool): If True, enables novelty detection for certain methods.
            n_jobs (int): Number of parallel jobs to run (-1 uses all processors).
            save_method (bool): If True, saves the fitted model to disk.
            save_dir (Optional[str]): Directory path to save model and transformed data.
            save_trans_data (bool): If True, saves the transformed data to disk.
            trans_data_name (str): Name for the transformed data file.
            deactivate (bool): Flag to deactivate the process.
        """
        self.activity_col = activity_col
        self.id_col = id_col
        self.select_method = select_method
        self.novelty = novelty
        self.n_jobs = n_jobs
        self.save_method = save_method
        self.save_dir = save_dir
        self.save_trans_data = save_trans_data
        self.trans_data_name = trans_data_name
        self.deactivate = deactivate
        self.multi_outlier_handler = None
        self.features = None

    def fit(self, data: pd.DataFrame, y=None) -> None:
        """
        Fits the selected outlier detection model to the provided data.

        Args:
            data (pd.DataFrame): The input dataset.

        Raises:
            ValueError: If an unsupported outlier detection method is provided.
        """
        if self.deactivate:
            logging.info("MultivariateOutliersHandler is deactivated. Skipping fit.")
            return self

        try:
            self.features = data.drop(
                columns=[self.id_col, self.activity_col], errors="ignore"
            ).columns

            method_map = {
                "LocalOutlierFactor": LocalOutlierFactor(
                    n_neighbors=20, n_jobs=self.n_jobs, novelty=self.novelty
                ),
                "IsolationForest": IsolationForest(
                    n_estimators=100,
                    contamination="auto",
                    random_state=42,
                    n_jobs=self.n_jobs,
                ),
                "OneClassSVM": OneClassSVM(),
                "RobustCovariance": EllipticEnvelope(
                    contamination=0.1, random_state=42
                ),
                "EmpiricalCovariance": EllipticEnvelope(
                    contamination=0.1, support_fraction=1, random_state=42
                ),
            }
            if self.select_method not in method_map:
                raise ValueError(f"Unsupported method: {self.select_method}")

            self.multi_outlier_handler = method_map[self.select_method]
            if self.select_method == "LocalOutlierFactor" and self.novelty:
                self.multi_outlier_handler.fit(data[self.features])
            else:
                self.multi_outlier_handler.fit(data[self.features])

            if self.save_method:
                if self.save_dir and not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir, exist_ok=True)
                with open(f"{self.save_dir}/multi_outlier_handler.pkl", "wb") as file:
                    pickle.dump(self, file)
                logging.info("Multivariate outlier handler saved successfully.")

            logging.info("Model fitting complete.")

        except Exception as e:
            logging.error(f"Error in fitting: {e}")
            raise

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data by removing detected outliers based on the fitted model.

        Args:
            data (pd.DataFrame): The input dataset to be transformed.

        Returns:
            pd.DataFrame: The transformed dataset with outliers removed.

        Raises:
            NotFittedError: If the model has not been fitted yet.
        """
        if self.deactivate:
            logging.info(
                "MultivariateOutlierHandler is deactivated. Returning unmodified data."
            )
            return data

        try:
            if self.multi_outlier_handler is None:
                raise NotFittedError(
                    "MultivariateOutlierHandler is not fitted yet. Call 'fit' before using this method."
                )

            if self.select_method == "LocalOutlierFactor" and not self.novelty:
                outliers = (
                    self.multi_outlier_handler.fit_predict(data[self.features]) == -1
                )
            else:
                outliers = self.multi_outlier_handler.predict(data[self.features]) == -1

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
                    f"Transformed data saved at: {self.save_dir}/{csv_name}.csv"
                )

            return transformed_data
        except Exception as e:
            logging.error(f"Error in transforming the data: {e}")
            raise

    def fit_transform(self, data: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fits the model and transforms the data in a single step.

        Args:
            data (pd.DataFrame): The input dataset to fit and transform.

        Returns:
            pd.DataFrame: The transformed dataset with outliers removed.
        """
        if self.deactivate:
            logging.info(
                "MultivariateOutlierHandler is deactivated. Returning unmodified data."
            )
            return data

        self.fit(data)
        return self.transform(data)

    def setting(self, **kwargs):
        valid_keys = self.__dict__.keys()
        for key in kwargs:
            if key not in valid_keys:
                raise KeyError(
                    f"'{key}' is not a valid attribute of MultivariateOutlierHandler."
                )
        self.__dict__.update(**kwargs)

        return self

    @staticmethod
    def compare_multivariate_methods(
        data1: pd.DataFrame,
        data2: Optional[pd.DataFrame] = None,
        data1_name: str = "data1",
        data2_name: str = "data2",
        activity_col: Optional[str] = None,
        id_col: Optional[str] = None,
        novelty: bool = False,
        methods_to_compare: List[str] = None,
        save_dir: Optional[str] = "Project/OutlierHandler",
    ) -> pd.DataFrame:
        """
        Compares the effect of different outlier detection methods on one or two datasets.

        Args:
            data1 (pd.DataFrame): The primary dataset for comparison.
            data2 (Optional[pd.DataFrame]): The secondary dataset for comparison, if any.
            data1_name (str): Name for the primary dataset in output.
            data2_name (str): Name for the secondary dataset in output.
            activity_col (Optional[str]): Column name for activity or target variable.
            id_col (Optional[str]): Column name for unique identifier.
            novelty (bool): If True, enables novelty detection for methods that support it.
            methods_to_compare (List[str]): List of methods to include in comparison.

        Returns:
            pd.DataFrame: A DataFrame summarizing the outlier removal effects of each method.
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
                    novelty=novelty,
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
