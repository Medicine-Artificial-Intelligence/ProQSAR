import logging
from sklearn.pipeline import Pipeline
from ProQSAR.config import Config


class DataPreprocessor:
    def __init__(
            self, 
            activity_col: str, 
            id_col: str, 
            deactivate: bool = False, 
            config=None
        ):
        
        self.activity_col = activity_col
        self.id_col = id_col
        self.deactivate = deactivate
        self.config = config or Config()

        for attr in ["duplicate", "missing", "lowvar", "univ_outlier", "kbin", "multiv_outlier", "rescaler"]:
            setattr(self, attr, getattr(self.config, attr).setting(activity_col=self.activity_col, id_col=self.id_col))

        self.datapreprocessor = Pipeline(
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


    def get_params(self, deep=True) -> dict:
        """Return all hyperparameters as a dictionary."""
        out = {}
        for key in self.__dict__:
            if key == "datapreprocessor":
                continue
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                for sub_key, sub_value in deep_items:
                    out[f"{key}__{sub_key}"] = sub_value
            out[key] = value

        return out

