import os
import gc
import logging
import pandas as pd
from typing import Optional, Union
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from ProQSAR.data_generator import DataGenerator
from ProQSAR.data_preprocessor import DataPreprocessor
from ProQSAR.ModelDeveloper.model_validation import ModelValidation
from ProQSAR.ModelDeveloper.model_developer_utils import (
    _get_task_type,
    _get_cv_strategy,
    _get_cv_scoring,
)
from ProQSAR.Config.validation_config import CrossValidationConfig
from ProQSAR.Config.config import Config
from copy import deepcopy


class OptimalDataset(CrossValidationConfig):
    def __init__(
        self,
        activity_col: str,
        id_col: str,
        smiles_col: str,
        mol_col: str = "mol",
        keep_all_train: bool = False,
        save_dir: Optional[str] = "Project/OptimalDataset",
        n_jobs: int = 1,
        random_state: int = 42,
        config=None,
        **kwargs,
    ):

        CrossValidationConfig.__init__(self, **kwargs)
        self.activity_col = activity_col
        self.id_col = id_col
        self.save_dir = save_dir
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.config = config or Config()
        self.smiles_col = (
            smiles_col
            if self.config.standardizer.deactivate
            else f"standardized_{smiles_col}"
        )
        self.mol_col = (
            mol_col if self.config.standardizer.deactivate else f"standardized_mol"
        )
        self.data_features = None
        self.train, self.test = {}, {}
        self.report = None
        self.optimal_set = None
        self.shape_summary = {}

        self.datagenerator = DataGenerator(
            activity_col,
            id_col,
            smiles_col,
            mol_col,
            n_jobs=self.n_jobs,
            save_dir=save_dir,
            config=config,
        )
        self.splitter = self.config.splitter.set_params(
            activity_col=activity_col,
            smiles_col=self.smiles_col,
            mol_col=self.mol_col,
            save_dir=save_dir,
            random_state=self.random_state,
        )
        self.datapreprocessor = DataPreprocessor(
            activity_col, id_col, save_dir=save_dir, config=config
        )
        if keep_all_train:
            self.datapreprocessor.duplicate.set_params(rows=False)
            self.datapreprocessor.multiv_outlier.set_params(deactivate=True)

    def run(self, data):
        # Generate all features
        self.data_features = self.datagenerator.generate(data)

        # Set metrics to perform cross validation
        self.task_type = _get_task_type(data, self.activity_col)
        self.cv = _get_cv_strategy(
            self.task_type,
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )
        self.scoring_list = self.scoring_list or _get_cv_scoring(self.task_type)

        # Set scorings
        if self.scoring_target is None:
            self.scoring_target = "f1" if self.task_type == "C" else "r2"

        if self.scoring_list:
            if isinstance(self.scoring_list, str):
                self.scoring_list = [self.scoring_list]

            if self.scoring_target not in self.scoring_list:
                self.scoring_list.append(self.scoring_target)

        # Set methods
        if self.task_type == "C":
            method = RandomForestClassifier(
                random_state=self.random_state, n_jobs=self.n_jobs
            )
        else:
            method = RandomForestRegressor(
                random_state=self.random_state, n_jobs=self.n_jobs
            )

        fs = SelectFromModel(method)

        result = []
        self.dataprep_fitted = {}

        for i in self.data_features.keys():
            logging.info(f"----------{i}----------")
            # Split each feature set into train set & test set
            self.splitter.set_params(data_name=i)
            self.train[i], self.test[i] = self.splitter.fit(self.data_features[i])
            self._record_shape("original", i, "train", self.train[i])
            # self._record_shape("original", i, "test", self.test[i])

            # Apply DataPreprocessor pipeline to train set
            self.datapreprocessor.fit(self.train[i])
            self.dataprep_fitted[i] = deepcopy(self.datapreprocessor)

            self.datapreprocessor.set_params(data_name=f"train_{i}")
            self.train[i + "_preprocessed"] = self.datapreprocessor.transform(
                self.train[i]
            )
            for step, transformer in self.datapreprocessor.pipeline.steps:
                self._record_shape(step, i, "train", transformer.transformed_data)

            # Apply DataPreprocessor pipeline to test set
            # self.datapreprocessor.set_params(data_name=f"test_{i}")
            # self.test[i + "_preprocessed"] = self.datapreprocessor.transform(
            #    self.test[i]
            # )
            # for step, transformer in self.datapreprocessor.pipeline.steps:
            #    self._record_shape(step, i, "test", transformer.transformed_data)

            # Apply feature selection & perform cross validation, both using RF algorithm
            X_data = self.train[i + "_preprocessed"].drop(
                [self.activity_col, self.id_col], axis=1
            )
            y_data = self.train[i + "_preprocessed"][self.activity_col]

            if not self.config.feature_selector.deactivate:
                X_data = fs.fit_transform(X_data, y_data)

            # Record data shape transformation
            self._record_shape(
                "feature_selector (rf)",
                i,
                "train",
                (X_data.shape[0], X_data.shape[1] + 2),
            )
            # self._record_shape("feature_selector (rf)", i, "test")

            result.append(
                ModelValidation._perform_cross_validation(
                    models={i: method},
                    X_data=X_data,
                    y_data=y_data,
                    cv=self.cv,
                    scoring_list=self.scoring_list,
                    include_stats=True,
                    n_splits=self.n_splits,
                    n_repeats=self.n_repeats,
                    n_jobs=self.n_jobs,
                )
            )
            del X_data, y_data
            del self.train[i]
            #del self.data_features[i]
            gc.collect()



        # Pivot the DataFrame so that each model becomes a separate column
        self.report = pd.concat(result).pivot_table(
            index=["scoring", "cv_cycle"],
            columns="method",
            values="value",
            aggfunc="first",
        )
        # Sort index and columns to maintain a consistent order
        self.report = self.report.sort_index(axis=0).sort_index(axis=1)

        # Reset index
        self.report = self.report.reset_index().rename_axis(None, axis="columns")

        # Identify optimal feature set
        self.optimal_set = (
            self.report.set_index(["scoring", "cv_cycle"])
            .loc[(f"{self.scoring_target}", "mean")]
            .idxmax()
        )
        # Visualization if requested
        if self.visualize is not None:
            if isinstance(self.visualize, str):
                self.visualize = [self.visualize]

            for graph_type in self.visualize:
                ModelValidation._plot_cv_report(
                    report_df=self.report,
                    scoring_list=self.scoring_list,
                    graph_type=graph_type,
                    save_fig=self.save_fig,
                    fig_prefix=self.save_fig,
                    save_dir=self.save_dir,
                )
        if self.save_cv_report:
            if self.save_dir and not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir, exist_ok=True)
            self.report.to_csv(
                f"{self.save_dir}/{self.cv_report_name}.csv", index=False
            )

        return self.optimal_set

    def _record_shape(
        self,
        stage_name: str,
        feature_set_name: str,
        data_name: str,
        data: Optional[Union[pd.DataFrame, tuple]] = None,
    ):
        """Helper method to record shapes at different pipeline stages in a dictionary format."""

        if isinstance(data, tuple):
            data_shape = data
        elif isinstance(data, pd.DataFrame):
            data_shape = data.shape
        else:
            data_shape = "N/A"

        if feature_set_name not in self.shape_summary:
            self.shape_summary[feature_set_name] = {"Data": {}}

        if data_name not in self.shape_summary[feature_set_name]["Data"]:
            self.shape_summary[feature_set_name]["Data"][data_name] = {}

        self.shape_summary[feature_set_name]["Data"][data_name][stage_name] = data_shape

    def get_shape_summary_df(self) -> pd.DataFrame:
        """Converts the shape_summary dictionary into a structured pandas DataFrame."""
        records = []
        for feature_set, data_entries in self.shape_summary.items():
            for data_name, stages in data_entries["Data"].items():
                record = {"Feature Set": feature_set, "Data": data_name, **stages}
                records.append(record)

        return pd.DataFrame(records)

    def get_params(self, deep=True) -> dict:
        """Return all hyperparameters as a dictionary, filtering nested parameters for specific attributes."""
        excluded_keys = {"data_features", "train", "test", "report", "optimal_set"}
        out = {}

        for key, value in self.__dict__.items():
            if key in excluded_keys:
                continue

            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()

                if key in {"datagenerator", "datapreprocessor"}:
                    out.update(
                        {sub_key: sub_value for sub_key, sub_value in deep_items}
                    )
                else:
                    out.update(
                        {
                            f"{key}_{sub_key}": sub_value
                            for sub_key, sub_value in deep_items
                        }
                    )

            out[key] = value

        return out

    def __repr__(self):
        """Return a string representation of the estimator."""
        class_name = self.__class__.__name__
        params = self.get_params(deep=False)
        param_str = ", ".join(f"{key}={repr(value)}" for key, value in params.items())
        return f"{class_name}({param_str})"
