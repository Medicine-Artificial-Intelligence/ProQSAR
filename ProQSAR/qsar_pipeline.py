import os
import logging
import pandas as pd
from ProQSAR.FeatureSelector.feature_selector import FeatureSelector
from ProQSAR.ModelDeveloper.model_validation import ModelValidation
from ProQSAR.ModelDeveloper.model_developer import ModelDeveloper
from ProQSAR.ModelDeveloper.model_developer_utils import _get_model_map
from ProQSAR.validation_config import CrossValidationConfig
from ProQSAR.data_generator import DataGenerator
from ProQSAR.data_preprocessor import DataPreprocessor
from ProQSAR.optimal_dataset import OptimalDataset
from ProQSAR.config import Config
from typing import Optional, Union
from copy import deepcopy


class QSARPipeline(CrossValidationConfig):
    def __init__(
        self,
        activity_col: str,
        id_col: str,
        smiles_col: str,
        mol_col: str = "mol",
        save_data: bool = False,
        save_dir: str = "Project/QSARPipeline",
        n_jobs: int = -1,
        config=None,
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.activity_col = activity_col
        self.id_col = id_col
        self.smiles_col = smiles_col
        self.save_data = save_data
        self.save_dir = save_dir
        self.n_jobs = n_jobs
        self.config = config or Config()

        self.optimaldata = OptimalDataset(
            activity_col,
            id_col,
            smiles_col,
            mol_col,
            save_data,
            save_dir,
            n_jobs,
            config,
            **kwargs,
        )

        self.datagenerator = DataGenerator(
            activity_col, id_col, smiles_col, mol_col, n_jobs=self.n_jobs, config=self.config
        )
        self.datapreprocessor = DataPreprocessor(activity_col, id_col, config=self.config)

        self.splitter = self.config.splitter
        
        self.feature_selector = self.config.feature_selector.setting(
            activity_col=activity_col, id_col=id_col, n_jobs=self.n_jobs, **kwargs
        )

        self.model_dev = self.config.model_dev.setting(
            activity_col=activity_col, id_col=id_col, n_jobs=self.n_jobs, **kwargs
        )

        self.optimizer = self.config.optimizer.setting(
            activity_col=activity_col,
            id_col=id_col,
            n_jobs=self.n_jobs,
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            scoring=self.scoring_target,
        ) if not self.config.optimizer.deactivate else None



    def fit(self, data_dev):

        # Generate, splitting & preprocessing
        if isinstance(self.config.featurizer.feature_types, list):
            self.selected_feature = self.optimaldata.run(data_dev)

            self.datagenerator.featurizer.setting(feature_types=self.selected_feature)
            self.datapreprocessor = self.optimaldata.dataprep_fitted[self.selected_feature]

            self.train = self.optimaldata.train[self.selected_feature + "_clean"]
            self.test = self.optimaldata.test[self.selected_feature + "_clean"]

        elif isinstance(self.config.featurizer.feature_types, str):
            self.selected_feature = self.config.featurizer.feature_types
            self.data_dev = self.datagenerator.generate(data_dev)
            self.train, self.test = self.splitter.fit(self.data_dev.values())
            self.train, _, _ = self.datapreprocessor.fit_transform(self.train)
            self.test, _, _ = self.datapreprocessor.transform(self.test)

        # Feature selection
        self.train = self.feature_selector.fit_transform(self.train)

        # Build & optimize
        model_map = _get_model_map(
            task_type=None, add_model=self.model_dev.add_model, n_jobs=self.n_jobs
        )
        
        # If select_model in model_dev is a list or None, perform cross validation to choose best
        # Then optimize hyperparams of the best algorithm
        if isinstance(self.model_dev.select_model, list) or not self.model_dev.select_model:
            # Get best model by cross validation
            self.model_dev.fit(self.train)

            if self.optimizer:
                report = deepcopy(self.model_dev.report)

                # Optimize hyperparams of the best model
                self.optimizer.setting(select_model=self.model_dev.select_model)
                best_params, _ = self.optimizer.optimize(self.train)

                # Update the best model & params & get the cross validation report
                new_model = model_map[self.model_dev.select_model].set_params(**best_params)
                self.model_dev.setting(
                    add_model={f"{self.model_dev.select_model}_opt": new_model},
                    select_model=f"{self.model_dev.select_model}_opt",
                    compare=False,
                )
                self.model_dev.fit(self.train)

                self.model_dev.report = (
                    pd.merge(
                        report,
                        self.model_dev.report,
                        on=["scoring", "cv_cycle"],
                        suffixes=("_1", "_2"),
                    )
                    .set_index(["scoring", "cv_cycle"])
                    .sort_index(axis=1)
                    .reset_index()
                )

        # If select_model in model_dev is a str, directly optimize hyperparams of the chosen algorithm
        elif isinstance(self.model_dev.select_model, str):
            if self.optimizer:
                self.optimizer.setting(select_model=self.model_dev.select_model)
                best_params, _ = self.optimizer.optimize(self.train)

                if self.model_dev.select_model not in model_map:
                    raise ValueError(
                        f"Selected model '{self.model_dev.select_model}' is not available."
                    )
                new_model = model_map[self.model_dev.select_model].set_params(**best_params)
                self.model_dev.setting(add_model={self.model_dev.select_model: new_model})
                
            self.model_dev.fit(self.train)

        return self
    
    def predict(self, data_pred):
        """
        Predict activity values for new data.
        
        Parameters:
        -----------
        data_pred : pd.DataFrame
            The input data for prediction. It should include the SMILES & ID columns.
        
        """
        self.data_pred = self.datagenerator.generate(data_pred)
        self.data_pred, _, _ = self.datapreprocessor.transform(self.data_pred)
        self.data_pred = self.feature_selector.transform(self.train)
        pred_result = self.model_dev.predict(self.data_pred)

        return pred_result
    
        ### CHUA CHAY THU func predict

        ### CHUA ADD CONFORMAL PREDICT & AD ###
        ### CAN CODE THEM def analysis(self): gom external_val, statistical (non_parametric or parametric or all) + roc curve
        ### def run_all(self, data_dev, data_pred): chay het fit, predict, analysis




        




