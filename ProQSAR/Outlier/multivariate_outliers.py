import os
import pickle
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.exceptions import NotFittedError
from typing import Optional


class MultivariateOutliersHandler:
    """
    A class for handling multivariate outliers in a dataset using various methods.

    Attributes:
    -----------
    id_col : Optional[str]
        The column name of the ID feature.
    activity_col : Optional[str]
        The column name of the activity feature.
    method : str
        The method used for outlier detection. Options are "LocalOutlierFactor", "IsolationForest",
        "OneClassSVM", "RobustCovariance", "EmpiricalCovariance".
    novelty : bool
        Whether the model is used for novelty detection (only for LocalOutlierFactor).
    n_jobs : int
        The number of jobs to run in parallel.
    save_dir : Optional[str]
        Directory where fitted models will be saved.
    model : Optional[object]
        The fitted outlier detection model.
    features : Optional[pd.Index]
        The feature columns used for fitting the model.
    """

    def __init__(
        self,
        id_col: Optional[str] = None,
        activity_col: Optional[str] = None,
        method: str = "LocalOutlierFactor",
        novelty: bool = False,
        n_jobs: int = 4,
        save_dir: Optional[str] = None,
    ) -> None:
        self.id_col = id_col
        self.activity_col = activity_col
        self.method = method
        self.novelty = novelty
        self.n_jobs = n_jobs
        self.save_dir = save_dir
        self.model = None

        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the outlier detection model to the data.

        Parameters:
        -----------
        data : pd.DataFrame
            The input dataframe containing the data to fit the model.
        """
        self.features = data.drop(
            columns=[self.id_col, self.activity_col], errors="ignore"
        ).columns

        if self.method == "LocalOutlierFactor":
            self.model = LocalOutlierFactor(
                n_neighbors=20, n_jobs=self.n_jobs, novelty=self.novelty
            )
            if self.novelty:
                self.model.fit(data[self.features])
            else:
                self.model.fit_predict(data[self.features])
        elif self.method == "IsolationForest":
            self.model = IsolationForest(
                n_estimators=100,
                contamination="auto",
                random_state=42,
                n_jobs=self.n_jobs,
            )
            self.model.fit(data[self.features])
        elif self.method == "OneClassSVM":
            self.model = OneClassSVM()
            self.model.fit(data[self.features])
        elif self.method == "RobustCovariance":
            self.model = EllipticEnvelope(contamination=0.1, random_state=42)
            self.model.fit(data[self.features])
        elif self.method == "EmpiricalCovariance":
            self.model = EllipticEnvelope(
                contamination=0.1, support_fraction=1, random_state=42
            )
            self.model.fit(data[self.features])
        else:
            raise ValueError(f"Unsupported method: {self.method}")

        # Save the model if a save directory is specified
        if self.save_dir:
            with open(f"{self.save_dir}/model.pkl", "wb") as file:
                pickle.dump(self.model, file)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted outlier detection model.

        Parameters:
        -----------
        data : pd.DataFrame
            The input dataframe to be transformed.

        Returns:
        --------
        transformed_data : pd.DataFrame
            The dataframe with outliers removed.

        Raises:
        -------
        NotFittedError
            If the model has not been fitted before calling this method.
        """
        if not self.model:
            raise NotFittedError(
                "Model is not fitted. Call 'fit' before using 'transform'."
            )

        if self.method == "LocalOutlierFactor" and not self.novelty:
            outliers = self.model.fit_predict(data[self.features]) == -1
        else:
            outliers = self.model.predict(data[self.features]) == -1

        transformed_data = data[~outliers]

        return transformed_data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the model and transform the data in one step.

        Parameters:
        -----------
        data : pd.DataFrame
            The input dataframe to be fitted and transformed.

        Returns:
        --------
        transformed_data : pd.DataFrame
            The dataframe with outliers removed.
        """
        self.fit(data)
        return self.transform(data)

    @staticmethod
    def static_transform(
        data: pd.DataFrame,
        save_dir: str,
        id_col: Optional[str] = None,
        activity_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Transform the data using a previously saved model.

        Parameters:
        -----------
        data : pd.DataFrame
            The input dataframe to be transformed.
        save_dir : str
            The directory where the fitted model is saved.

        Returns:
        --------
        transformed_data : pd.DataFrame
            The dataframe with outliers removed.

        Raises:
        -------
        NotFittedError
            If no saved model is found in the specified directory.
        """
        if not os.path.exists(f"{save_dir}/model.pkl"):
            raise NotFittedError(
                "No saved model found. Fit the model first or check save_dir."
            )

        with open(f"{save_dir}/model.pkl", "rb") as file:
            model = pickle.load(file)

        features = data.drop(columns=[id_col, activity_col], errors="ignore").columns
        if isinstance(model, LocalOutlierFactor) and not model.novelty:
            outliers = model.fit_predict(data[features]) == -1
        else:
            outliers = model.predict(data[features]) == -1
        return data[~outliers]

    @staticmethod
    def compare_multivariate_methods(
        data1: pd.DataFrame,
        data2: Optional[pd.DataFrame] = None,
        data1_name: str = "data1",
        data2_name: str = "data2",
        activity_col: Optional[str] = None,
        id_col: Optional[str] = None,
        novelty: bool = False,
    ) -> pd.DataFrame:
        """
        Compare different multivariate outlier handling methods.

        Parameters:
        -----------
        data1 : pd.DataFrame
            The primary dataframe for fitting the models.
        data2 : Optional[pd.DataFrame], optional
            The secondary dataframe for transformation. Defaults to None.
        data1_name : str, optional
            The name of the first dataset (used in the comparison table). Defaults to "data1".
        data2_name : str, optional
            The name of the second dataset (used in the comparison table). Defaults to "data2".
        activity_col : Optional[str], optional
            The name of the activity column, if present. Defaults to None.
        id_col : Optional[str], optional
            The name of the ID column, if present. Defaults to None.
        novelty : bool, optional
            Whether to use the models for novelty detection. Defaults to False.

        Returns:
        --------
        comparison_table : pd.DataFrame
            A dataframe summarizing the results of different outlier handling methods.
        """

        comparison_data = []
        methods = [
            "LocalOutlierFactor",
            "IsolationForest",
            "OneClassSVM",
            "RobustCovariance",
            "EmpiricalCovariance",
        ]

        for method in methods:
            handler = MultivariateOutliersHandler(
                id_col=id_col, activity_col=activity_col, method=method, novelty=novelty
            )
            handler.fit(data1)

            if data2 is None:
                transformed_data1 = handler.transform(data1)
                comparison_data.append(
                    {
                        "Method": method,
                        "Original Rows": data1.shape[0],
                        "After Handling Rows": transformed_data1.shape[0],
                        "Removed Rows": data1.shape[0] - transformed_data1.shape[0],
                    }
                )

                comparison_table = pd.DataFrame(comparison_data)
                comparison_table.name = (
                    f"Comparison of different outlier handling methods on {data1_name}"
                )

            else:
                transformed_data2 = handler.transform(data2)
                comparison_data.append(
                    {
                        "Method": method,
                        "Original Rows": data2.shape[0],
                        "After Handling Rows": transformed_data2.shape[0],
                        "Removed Rows": data2.shape[0] - transformed_data2.shape[0],
                    }
                )
                comparison_table = pd.DataFrame(comparison_data)
                comparison_table.name = (
                    f"Methods fitted on {data1_name} & transformed on {data2_name}"
                )
        return comparison_table
