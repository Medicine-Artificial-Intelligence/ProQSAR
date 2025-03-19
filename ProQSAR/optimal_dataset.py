import os
import logging
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from ProQSAR.data_generator import DataGenerator
from ProQSAR.data_preprocessor import DataPreprocessor
from ProQSAR.Splitter.data_splitter import Splitter
from ProQSAR.ModelDeveloper.model_validation import ModelValidation
from ProQSAR.ModelDeveloper.model_developer_utils import (
    _get_task_type,
    _get_cv_strategy,
    _get_cv_scoring,
)
from ProQSAR.validation_config import CrossValidationConfig
from ProQSAR.config import Config


class OptimalDataset(CrossValidationConfig):
    def __init__(
            self, 
            activity_col: str, 
            id_col: str, 
            smiles_col: str, 
            save_features: bool = False,
            save_dir: str = "Project/OptimalDataset",
            n_jobs: int = -1,
            config=None,
            **kwargs
            ):
        
        super().__init__(**kwargs)
        self.activity_col = activity_col
        self.id_col = id_col
        self.smiles_col = smiles_col
        self.save_features = save_features
        self.save_dir = save_dir
        self.n_jobs = n_jobs
        self.config = config or Config()

        self.features = None
        self.train, self.test = {}, {}
        self.report = None
        self.optimal_set = None
        self.standardized_smiles_col = f"standardized_{self.smiles_col}"

        self.datagenerator = DataGenerator(activity_col, id_col, smiles_col, config=self.config)
        self.splitter = self.config.splitter.setting(activity_col=self.activity_col, smiles_col=self.standardized_smiles_col)
        self.datapreprocessor = DataPreprocessor(activity_col, id_col, config=self.config)
    

    def run(self, data):
        
        # Generate all features
        self.features = self.datagenerator.generate(data)

        # Set metrics to perform cross validation
        self.task_type = _get_task_type(data, self.activity_col)
        self.cv = _get_cv_strategy(self.task_type, n_splits=self.n_splits, n_repeats=self.n_repeats)
        self.scoring_list = self.scoring_list or _get_cv_scoring(self.task_type)

        # Set scorings
        self.scoring_target = (
            self.scoring_target or "f1" if self.task_type == "C" else "r2"
        )
        if self.scoring_list:
            if isinstance(self.scoring_list, str):
                self.scoring_list = [self.scoring_list]

            if self.scoring_target not in self.scoring_list:
                self.scoring_list.append(self.scoring_target)


        # Set methods
        if self.task_type == "C":
            method = RandomForestClassifier(random_state=42, n_jobs=self.n_jobs)
        else:
            method = RandomForestRegressor(random_state=42, n_jobs=self.n_jobs)

        fs = SelectFromModel(method)

        result = []

        for i in self.features.keys():
            # Split each feature set into train set & test set
            self.train[i], self.test[i] = self.splitter.fit(self.features[i])

            # Apply DataPreprocessor pipeline to train set
            self.train[i+'_clean'], _, _ = self.datapreprocessor.fit_transform(self.train[i])

            # Apply feature selection & perform cross validation, both using RF algorithm
            X_data = self.train[i+'_clean'].drop([self.activity_col, self.id_col], axis=1)
            y_data = self.train[i+'_clean'][self.activity_col]

            selected_X = fs.fit_transform(X_data, y_data)

            result.append(
                ModelValidation._perform_cross_validation(
                    models={i: method},
                    X_data=selected_X,
                    y_data=y_data,
                    cv=self.cv,
                    scoring_list=self.scoring_list,
                    include_stats=True,
                    n_splits=self.n_splits,
                    n_repeats=self.n_repeats,
                    n_jobs=self.n_jobs,
                )
            )
            
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
            self.report.to_csv(f"{self.save_dir}/{self.cv_report_name}.csv", index=False)

        if self.save_features:
            if self.save_dir and not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir, exist_ok=True)
            for key, value in self.features.items():
                value.to_csv(f"{self.save_dir}/{key}.csv", index=False)
            for key, value in self.train.items():
                value.to_csv(f"{self.save_dir}/train_{key}.csv", index=False)
            for key, value in self.test.items():
                value.to_csv(f"{self.save_dir}/test_{key}.csv", index=False)

        return self.optimal_set
    

###### LAM TOI DAY NHA - QUA MET ROI #############
    def get_params(self, deep=True) -> dict:
        """Return all hyperparameters as a dictionary, filtering nested parameters for specific attributes."""
        excluded_keys = {"features", "train", "test", "report"}
        out = {}

        for key, value in self.__dict__.items():
            if key in excluded_keys:
                continue
            
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_params = value.get_params().items()

                if key in {"datagenerator", "datapreprocessor"}:
                    out.update({k: v for k, v in deep_params if "__" in k})
                else:
                    out[key] = value

        return out

