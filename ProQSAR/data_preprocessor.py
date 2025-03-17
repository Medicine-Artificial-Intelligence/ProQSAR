import logging
from sklearn.pipeline import Pipeline
from ProQSAR.Preprocessor.duplicate_handler import DuplicateHandler
from ProQSAR.Preprocessor.missing_handler import MissingHandler
from ProQSAR.Preprocessor.low_variance_handler import LowVarianceHandler
from ProQSAR.Outlier.univariate_outliers import UnivariateOutliersHandler
from ProQSAR.Outlier.kbin_handler import KBinHandler
from ProQSAR.Outlier.multivariate_outliers import MultivariateOutliersHandler
from ProQSAR.Rescaler.rescaler import Rescaler


class DataPreprocessor:
    def __init__(self, activity_col, id_col, deactivate: bool = False):
        self.activity_col = activity_col
        self.id_col = id_col
        self.deactivate = deactivate

        self.duplicatehandler = DuplicateHandler(activity_col, id_col)
        self.missinghandler = MissingHandler(activity_col, id_col)
        self.lowvariancehandler = LowVarianceHandler(activity_col, id_col)
        self.univariateoutliershandler = UnivariateOutliersHandler(activity_col, id_col)
        self.kbinhandler = KBinHandler(activity_col, id_col)
        self.multivariateoutliershandler = MultivariateOutliersHandler(activity_col, id_col)
        self.rescaler = Rescaler(activity_col, id_col)

        self.stages = [
            ("duplicatehandler", self.duplicatehandler),
            ("missinghandler", self.missinghandler),
            ("lowvariancehandler", self.lowvariancehandler),
            ("univariateoutliershandler", self.univariateoutliershandler),
            ("kbinhandler", self.kbinhandler),
            ("multivariateoutliershandler", self.multivariateoutliershandler),
            ("rescaler", self.rescaler),
        ]

        self.datapreprocessor = Pipeline(self.stages)

    def fit(self, data):
        """Fit all preprocessing steps on the training data."""
        if self.deactivate:
            logging.info("DataPreprocessor is deactivated. Skipping fit.")
            return self

        logging.info("Fitting data preprocessing pipeline.")
        self.datapreprocessor.fit(data)
        return self

    def transform(self, data):
        """Apply transformations to the dataset."""
        if self.deactivate:
            logging.info("DataPreprocessor is deactivated. Returning unmodified data.")
            return data

        logging.info("Transforming data using the preprocessing pipeline.")
        transformed_data = self.datapreprocessor.transform(data)

        # Identify columns and IDs that were deleted during preprocessing
        deleted_col = [
            col for col in data.columns if col not in transformed_data.columns
        ]
        deleted_id = [
            id
            for id in data[self.id_col].values
            if id not in transformed_data[self.id_col].values
        ]

        return transformed_data, deleted_col, deleted_id

    def fit_transform(self, data):
        """Fit and transform in one step."""
        if self.deactivate:
            logging.info("DataPreprocessor is deactivated. Returning unmodified data.")
            return data

        self.fit(data)
        return self.transform(data)

    def set_params(self, stage, **kwargs):
        """Set parameters for a specific preprocessing stage."""
        component = getattr(self, stage.lower(), None)
        if component and isinstance(
            component,
            (
                DuplicateHandler,
                MissingHandler,
                LowVarianceHandler,
                UnivariateOutliersHandler,
                KBinHandler,
                MultivariateOutliersHandler,
                Rescaler,
            ),
        ):
            for key, value in kwargs.items():
                if hasattr(component, key):
                    setattr(component, key, value)
                else:
                    raise AttributeError(
                        f"{stage} does not have a parameter named '{key}'"
                    )
        else:
            raise ValueError(f"Invalid stage name: {stage}")

    def get_params(self, deep=False):
        """
        Get parameters of the DataGenerator class as a dictionary.

        Parameters
        ----------
        deep : bool, optional, default=True
            If True, will include parameters of the underlying components.

        Returns
        -------
        params : dict
            Dictionary containing parameters and their values.
        """
        # Parameters from the DataGenerator itself
        params = {
            "datapreprocessor__activity_col": self.activity_col,
            "datapreprocessor__id_col": self.id_col,
            "datapreprocessor__deactivate": self.deactivate,
        }

        pipeline_params = self.datapreprocessor.get_params()
        stage_params = {
            k: v
            for k, v in pipeline_params.items()
            if k not in ["memory", "steps", "verbose"]
        }

        params.update(stage_params)

        return params
