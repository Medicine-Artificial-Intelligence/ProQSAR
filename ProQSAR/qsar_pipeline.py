import os
import logging
import pandas as pd
from ProQSAR.ModelDeveloper.model_developer_utils import _get_model_map
from ProQSAR.data_generator import DataGenerator
from ProQSAR.data_preprocessor import DataPreprocessor
from ProQSAR.optimal_dataset import OptimalDataset
from ProQSAR.Config.validation_config import CrossValidationConfig
from ProQSAR.Config.config import Config
from ProQSAR.ModelDeveloper.model_validation import ModelValidation
from copy import deepcopy
from typing import Optional, Iterable, Union


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
        random_state: int = 42,
        config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.activity_col = activity_col
        self.id_col = id_col
        self.save_data = save_data
        self.save_dir = save_dir
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.config = config or Config()
        self.smiles_col = smiles_col

        self.optimaldata = OptimalDataset(
            activity_col,
            id_col,
            self.smiles_col,
            mol_col,
            save_data,
            save_dir,
            n_jobs,
            config,
            **kwargs,
        )

        self.datagenerator = DataGenerator(
            activity_col,
            id_col,
            self.smiles_col,
            mol_col,
            n_jobs=self.n_jobs,
            config=self.config,
        )
        self.datapreprocessor = DataPreprocessor(
            activity_col, id_col, config=self.config
        )

        self.splitter = self.config.splitter.set_params(
            activity_col=activity_col,
            smiles_col=(
                self.smiles_col
                if self.config.standardizer.deactivate
                else f"standardized_{self.smiles_col}"
            ),
        )

        self.feature_selector = self.config.feature_selector.set_params(
            activity_col=activity_col,
            id_col=id_col,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            **kwargs,
        )

        self.model_dev = self.config.model_dev.set_params(
            activity_col=activity_col,
            id_col=id_col,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            **kwargs,
        )

        self.optimizer = (
            self.config.optimizer.set_params(
                activity_col=activity_col,
                id_col=id_col,
                n_jobs=self.n_jobs,
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
                scoring=self.scoring_target,
            )
            if not self.config.optimizer.deactivate
            else None
        )

        self.conf_pred = (
            self.config.conf_pred.set_params(
                activity_col=activity_col,
                id_col=id_col,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
            if not self.config.conf_pred.deactivate
            else None
        )

        self.ad = (
            self.config.ad.set_params(activity_col=activity_col, id_col=id_col)
            if not self.config.ad.deactivate
            else None
        )

    def fit(self, data_dev):

        # Generate, splitting & preprocessing
        if isinstance(self.config.featurizer.feature_types, list):
            self.selected_feature = self.optimaldata.run(data_dev)

            self.datagenerator.featurizer.set_params(
                feature_types=self.selected_feature
            )
            self.datapreprocessor = self.optimaldata.dataprep_fitted[
                self.selected_feature
            ]

            self.data_dev = self.optimaldata.data_features[self.selected_feature]
            self.train = self.optimaldata.train[self.selected_feature + "_clean"]
            self.test = self.optimaldata.test[self.selected_feature + "_clean"]

        elif isinstance(self.config.featurizer.feature_types, str):
            self.selected_feature = self.config.featurizer.feature_types
            self.data_dev = self.datagenerator.generate(data_dev)

            self.train, self.test = self.splitter.fit(self.data_dev)
            self.train = self.train.drop(
                columns=(
                    self.smiles_col
                    if self.config.standardizer.deactivate
                    else f"standardized_{self.smiles_col}"
                )
            )
            self.test = self.test.drop(
                columns=(
                    self.smiles_col
                    if self.config.standardizer.deactivate
                    else f"standardized_{self.smiles_col}"
                )
            )
            self.train, _, _ = self.datapreprocessor.fit_transform(self.train)
            self.test, _, _ = self.datapreprocessor.transform(self.test)

        # Feature selection
        self.train = self.feature_selector.fit_transform(self.train)
        self.test = self.feature_selector.transform(self.test)

        # Model development & optimization
        model_map = _get_model_map(
            task_type=None, add_model=self.model_dev.add_model, n_jobs=self.n_jobs
        )

        # If select_model in model_dev is a list or None, perform cross validation to choose best
        # Then optimize hyperparams of the best algorithm
        if (
            isinstance(self.model_dev.select_model, list)
            or not self.model_dev.select_model
        ):
            # Get best model by cross validation
            self.model_dev.fit(self.train)

            if self.optimizer:
                report = deepcopy(self.model_dev.report)

                # Optimize hyperparams of the best model
                self.optimizer.set_params(select_model=self.model_dev.select_model)
                best_params, _ = self.optimizer.optimize(self.train)

                # Update the best model & params & get the cross validation report
                new_model = model_map[self.model_dev.select_model].set_params(
                    **best_params
                )
                self.model_dev.set_params(
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
                self.optimizer.set_params(select_model=self.model_dev.select_model)
                best_params, _ = self.optimizer.optimize(self.train)

                if self.model_dev.select_model not in model_map:
                    raise ValueError(
                        f"Selected model '{self.model_dev.select_model}' is not available."
                    )
                new_model = model_map[self.model_dev.select_model].set_params(
                    **best_params
                )
                self.model_dev.set_params(
                    add_model={self.model_dev.select_model: new_model}
                )

            self.model_dev.fit(self.train)

        # Conformal predictor
        if self.conf_pred:
            self.conf_pred.set_params(model=self.model_dev)
            self.conf_pred.fit(self.train)

        # Applicability domain
        if self.ad:
            self.ad.fit(self.train)

        return self

    def predict(self, data_pred, alpha: Optional[Union[float, Iterable[float]]] = None):
        """
        Predict activity values for new data.

        Parameters:
        -----------
        data_pred : pd.DataFrame
            The input data for prediction. It should include the SMILES & ID columns.

        """
        # Generate feature, preprocess data & apply feature selection
        self.data_pred = self.datagenerator.generate(data_pred)
        self.data_pred = self.data_pred.drop(
            columns=(
                self.smiles_col
                if self.config.standardizer.deactivate
                else f"standardized_{self.smiles_col}"
            ),
            errors='ignore'
        )

        self.data_pred, _, _ = self.datapreprocessor.transform(self.data_pred)
        self.data_pred = self.feature_selector.transform(self.data_pred)

        # Conformal Predictor
        if self.conf_pred:
            pred_result = self.conf_pred.predict(self.data_pred, alpha=alpha)
        else:
            pred_result = self.model_dev.predict(self.data_pred)

        # Predict using AD (if enable)
        if self.ad:
            ad_result = self.ad.predict(self.data_pred)
            pred_result = pd.merge(pred_result, ad_result, on=self.id_col)

        return pred_result
    
    def validate(self):
        # Cross validation report
        self.cv_report = self.model_dev.report

        # External validation report
        select_model = self.cv_report.columns[
            ~self.cv_report.columns.isin(['scoring', 'cv_cycle'])
        ]

        self.ev_report = ModelValidation.external_validation_report(
            data_train=self.train,
            data_test=self.test,
            activity_col=self.activity_col,
            id_col=self.id_col,
            select_model=select_model,
            add_model=self.model_dev.add_model,
            scoring_list=
        )

        ### CAN CODE THEM def analysis(self): gom external_val, statistical (non_parametric or parametric or all) + roc curve
        ### def run_all(self, data_dev, data_pred): chay het fit, predict, analysis
