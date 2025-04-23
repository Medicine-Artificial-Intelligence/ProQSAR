import os
import logging
from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from ProQSAR.Config.config import Config


class DataPreprocessor(BaseEstimator):
    def __init__(
        self,
        activity_col: str,
        id_col: str,
        save_dir: Optional[str] = "Project/DataGenerator",
        data_name: Optional[str] = None,
        config=None,
    ):

        self.activity_col = activity_col
        self.id_col = id_col
        self.save_dir = save_dir
        self.data_name = data_name
        self.config = config or Config()

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
        """Fit all preprocessing steps on the training data."""

        self.pipeline.fit(data)
        return self

    def transform(self, data):
        """Apply transformations to the dataset."""

        transformed_data = self.pipeline.transform(data)

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            name_prefix = f"{self.data_name}_" if self.data_name else ""
            transformed_data.to_csv(
                f"{self.save_dir}/{name_prefix}preprocessed.csv", index=False
            )

        return transformed_data

    def get_params(self, deep=True) -> dict:
        """Return all hyperparameters as a dictionary."""
        out = {}
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
