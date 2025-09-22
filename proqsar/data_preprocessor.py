import os
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from proqsar.Config.config import Config


class DataPreprocessor(BaseEstimator):
    """
    High-level data preprocessor that builds a sklearn `Pipeline` from components
    provided by a ProQSAR :class:`proqsar.Config.config.Config` object.

    The pipeline order is:

    1. ``duplicate`` (duplicate removal)
    2. ``missing`` (missing value handling)
    3. ``lowvar`` (low-variance feature removal)
    4. ``univ_outlier`` (univariate outlier handling)
    5. ``kbin`` (KBins discretization for flagged features)
    6. ``multiv_outlier`` (multivariate outlier detection/removal)
    7. ``rescaler`` (feature rescaling / normalization)

    Each step is obtained from the provided :class:`Config` instance and is
    configured with the provided ``activity_col`` and ``id_col``.

    :param activity_col: Column name for the activity/target column.
    :type activity_col: str
    :param id_col: Column name for the identifier column.
    :type id_col: str
    :param save_dir: Directory where an optionally saved preprocessed CSV will be written.
                     Default: ``"Project/DataGenerator"``.
    :type save_dir: Optional[str]
    :param data_name: Optional base name used when saving the preprocessed CSV.
    :type data_name: Optional[str]
    :param config: A ProQSAR :class:`Config` instance describing which components to use.
                   If ``None``, a default :class:`Config()` is created.
    :type config: Optional[Config]

    Attributes
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The composed sklearn Pipeline of preprocessing transformers.
    duplicate, missing, lowvar, univ_outlier, kbin, multiv_outlier, rescaler :
        Instances of corresponding handlers from ``config``, each configured with
        ``activity_col`` and ``id_col``.
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
        preprocessing components and composing them into a :class:`Pipeline`.

        :param activity_col: Column name for the activity/target column.
        :type activity_col: str
        :param id_col: Column name for the identifier column.
        :type id_col: str
        :param save_dir: Directory where an optionally saved preprocessed CSV will be written.
                         Default: ``"Project/DataGenerator"``.
        :type save_dir: Optional[str]
        :param data_name: Optional base name used when saving the preprocessed CSV.
        :type data_name: Optional[str]
        :param config: A ProQSAR :class:`Config` instance describing which components to use.
                       If ``None``, a default :class:`Config()` is created.
        :type config: Optional[Config]

        :returns: None
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

        The method delegates to the composed ``Pipeline``'s ``fit`` method which
        calls ``fit`` for each transformer in the sequence.

        :param data: The training dataset to fit the pipeline components on.
                     Typically a pandas DataFrame or an object supported by the
                     individual transformers.
        :type data: Any
        :returns: The fitted DataPreprocessor instance (``self``).
        :rtype: DataPreprocessor
        """
        self.pipeline.fit(data)
        return self

    def transform(self, data):
        """
        Apply the composed preprocessing ``Pipeline`` to ``data``.

        After transformation, if ``save_dir`` is set the resulting DataFrame
        (or array-like object returned by the pipeline) is saved as a CSV file
        named ``'{data_name}_preprocessed.csv'`` or ``'preprocessed.csv'`` when
        ``data_name`` is ``None``.

        NOTE: The pipeline may return either a pandas DataFrame or a NumPy
        array depending on configured transformers. If a NumPy array is
        returned it is saved directly using pandas.DataFrame(...) wrapper for
        CSV export.

        :param data: Dataset to transform using the fitted pipeline.
        :type data: Any
        :returns: The transformed dataset produced by the pipeline (DataFrame or array-like).
        :rtype: Any
        :raises ValueError: If the pipeline returns an unsupported type for saving.
        """
        transformed_data = self.pipeline.transform(data)

        # Attempt to persist results if requested
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            name_prefix = f"{self.data_name}_" if self.data_name else ""
            save_path = f"{self.save_dir}/{name_prefix}preprocessed.csv"

            # If pipeline returned a DataFrame-like object, save directly.
            if hasattr(transformed_data, "to_csv"):
                transformed_data.to_csv(save_path, index=False)
            else:
                # Try to wrap array-like into DataFrame for saving.
                try:
                    pd.DataFrame(transformed_data).to_csv(save_path, index=False)
                except Exception as e:
                    raise ValueError(
                        f"DataPreprocessor: Unable to save transformed data to CSV: {e}"
                    )

        return transformed_data

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Return all hyperparameters as a dictionary, similar to sklearn's API.

        When ``deep=True``, nested estimators' parameters are expanded using the
        ``'<component>__<param>'`` naming convention.

        :param deep: If True, include parameters of nested objects (default True).
        :type deep: bool
        :returns: Mapping of parameter names to values.
        :rtype: dict
        """
        out: Dict[str, Any] = {}
        for key in self.__dict__:
            if key == "pipeline":
                # pipeline is derived from components; skip direct expansion here
                continue
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                for sub_key, sub_value in deep_items:
                    out[f"{key}__{sub_key}"] = sub_value
            out[key] = value

        return out
