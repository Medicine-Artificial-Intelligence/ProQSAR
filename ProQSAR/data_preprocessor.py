import os
from typing import Optional, Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from ProQSAR.Config.config import Config


class DataPreprocessor(BaseEstimator):
    """
    High-level data preprocessor that builds a sklearn Pipeline from components
    provided by a ProQSAR `Config` object.

    The pipeline order is:
      1. duplicate (duplicate removal)
      2. missing (missing value handling)
      3. lowvar (low-variance feature removal)
      4. univ_outlier (univariate outlier handling)
      5. kbin (KBins discretization for flagged features)
      6. multiv_outlier (multivariate outlier detection/removal)
      7. rescaler (feature rescaling / normalization)

    Parameters
    ----------
    activity_col : str
        Column name for the activity/target column.
    id_col : str
        Column name for the identifier column.
    save_dir : Optional[str]
        Directory where an optionally saved preprocessed CSV will be written.
        Default: "Project/DataGenerator".
    data_name : Optional[str]
        Optional base name used when saving the preprocessed CSV.
    config : Optional[Config]
        A ProQSAR Config instance describing which components to use. If None,
        a default Config() is created.

    Attributes
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The composed sklearn Pipeline of preprocessing transformers.
    duplicate, missing, lowvar, univ_outlier, kbin, multiv_outlier, rescaler :
        Instances of corresponding handlers from Config, each configured with
        `activity_col` and `id_col`.
    """

    def __init__(
        self,
        activity_col: str,
        id_col: str,
        save_dir: Optional[str] = "Project/DataGenerator",
        data_name: Optional[str] = None,
        config=None,
    ):
        """
        Initialize the DataPreprocessor by instantiating the configured
        preprocessing components and composing them into a Pipeline.
        """
        self.activity_col = activity_col
        self.id_col = id_col
        self.save_dir = save_dir
        self.data_name = data_name
        self.config = config or Config()

        # instantiate and configure each step from the Config
        for attr in [
            "duplicate",
            "missing",
            "lowvar",
            "univ_outlier",
            "kbin",
            "multiv_outlier",
            "rescaler",
        ]:
            setattr(
                self,
                attr,
                getattr(self.config, attr).set_params(
                    activity_col=self.activity_col, id_col=self.id_col
                ),
            )

        self.pipeline = Pipeline(
            [
                ("duplicate", self.duplicate),
                ("missing", self.missing),
                ("lowvar", self.lowvar),
                ("univ_outlier", self.univ_outlier),
                ("kbin", self.kbin),
                ("multiv_outlier", self.multiv_outlier),
                ("rescaler", self.rescaler),
            ]
        )

    def fit(self, data):
        """
        Fit all preprocessing steps on the provided training data.

        Parameters
        ----------
        data : pd.DataFrame or compatible
            The training dataset to fit the pipeline components on.

        Returns
        -------
        DataPreprocessor
            The fitted DataPreprocessor (self).
        """

        self.pipeline.fit(data)
        return self

    def transform(self, data):
        """
        Apply the composed preprocessing Pipeline to `data`.

        After transformation, if `save_dir` is set the resulting DataFrame is
        saved as a CSV file named '{data_name}_preprocessed.csv' (or
        'preprocessed.csv' when data_name is None).

        Parameters
        ----------
        data : pd.DataFrame or compatible
            Dataset to transform using the fitted pipeline.

        Returns
        -------
        pd.DataFrame
            The transformed dataset produced by the pipeline.
        """

        transformed_data = self.pipeline.transform(data)

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            name_prefix = f"{self.data_name}_" if self.data_name else ""
            transformed_data.to_csv(
                f"{self.save_dir}/{name_prefix}preprocessed.csv", index=False
            )

        return transformed_data

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Return all hyperparameters as a dictionary, similar to sklearn's API.

        When `deep=True`, nested estimators' parameters are expanded using the
        '<component>__<param>' naming convention.

        Parameters
        ----------
        deep : bool, optional
            If True, include parameters of nested objects (default True).

        Returns
        -------
        dict
            Mapping of parameter names to values.
        """
        out: Dict[str, Any] = {}
        for key in self.__dict__:
            if key == "pipeline":
                continue
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                for sub_key, sub_value in deep_items:
                    out[f"{key}__{sub_key}"] = sub_value
            out[key] = value

        return out
