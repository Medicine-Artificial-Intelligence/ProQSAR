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
)
from typing import Union, Optional


class OptimalDataset:
    def __init__(
            self, 
            activity_col: str, 
            id_col: str, 
            smiles_col: str, 
            feature_types: list = ["ECFP4", "RDK5", "FCFP4"],
            scoring_target: str = None, 
            scoring_list: Optional[Union[list, str]] = None,
            n_splits: int = 5,
            n_repeats: int = 5,
            save_dir: str = "Project/OptimalDataset",
            n_jobs: int = -1,
            ):
        
        self.activity_col = activity_col
        self.id_col = id_col
        self.smiles_col = smiles_col
        self.feature_types = feature_types
        self.scoring_target = scoring_target
        self.scoring_list = scoring_list
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.save_dir = save_dir
        self.n_jobs = n_jobs

        self.comparison = None
        self.optimal_set = None
        self.standardized_smiles_col = f"standardized_{self.smiles_col}"

        self.datagenerator = DataGenerator(activity_col, id_col, smiles_col, feature_types=self.feature_types)
        self.splitter = Splitter(activity_col, smiles_col=self.standardized_smiles_col)
        self.datapreprocessor = DataPreprocessor(activity_col, id_col)
    

    def run(self, data):
        
        # Generate all features
        features = self.datagenerator.generate(data)

        # Set metrics to perform cross validation
        self.task_type = _get_task_type(data, self.activity_col)
        self.cv = _get_cv_strategy(self.task_type, n_splits=self.n_splits, n_repeats=self.n_repeats)

        # Set scorings
        self.scoring_target = (
            self.scoring_target or "f1" if self.task_type == "C" else "r2"
        )
        if self.scoring_list:
            if isinstance(self.scoring_list, str):
                self.scoring_list = [self.scoring_list]

            if self.scoring_target not in self.scoring_list:
                self.scoring_list.append(self.scoring_target)

        else:
            self.scoring_list = [self.scoring_target]


        # Set methods
        if self.task_type == "C":
            method = RandomForestClassifier(random_state=42, n_jobs=self.n_jobs)
        else:
            method = RandomForestRegressor(random_state=42, n_jobs=self.n_jobs)

        fs = SelectFromModel(method)

        train, test = {}, {}
        result = []

        for i in self.feature_types:
            # Split each feature set into train set & test set
            train[i], test[i] = self.splitter.fit(features[i])

            # Apply DataPreprocessor pipeline to train set
            train[i+'_clean'], _, _ = self.datapreprocessor.fit_transform(train[i])

            # Apply feature selection & perform cross validation, both using RF algorithm
            X_data = train[i+'_clean'].drop([self.activity_col, self.id_col], axis=1)
            y_data = train[i+'_clean'][self.activity_col]

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
        self.comparison = pd.concat(result).pivot_table(
            index=["scoring", "cv_cycle"],
            columns="method",
            values="value",
            aggfunc="first",
        )
        # Sort index and columns to maintain a consistent order
        self.comparison = self.comparison.sort_index(axis=0).sort_index(axis=1)

        # Reset index
        self.comparison = self.comparison.reset_index().rename_axis(None, axis="columns")

        # Identify optimal feature set

        self.optimal_set = (
            self.comparison.set_index(["scoring", "cv_cycle"])
            .loc[(f"{self.scoring_target}", "mean")]
            .idxmax()
        )

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            for key, value in features.items():
                value.to_csv(f"{self.save_dir}/{key}.csv", index=False)
            for key, value in train.items():
                value.to_csv(f"{self.save_dir}/train_{key}.csv", index=False)
            for key, value in test.items():
                value.to_csv(f"{self.save_dir}/test_{key}.csv", index=False)

            self.comparison.to_csv(f"{self.save_dir}/optimal_data_ report.csv", index=False)

        return self.optimal_set
