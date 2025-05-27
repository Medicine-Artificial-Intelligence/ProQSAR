import os
import pickle
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from ProQSAR.ModelDeveloper.model_developer_utils import (
    _get_model_map,
    _match_cv_ev_metrics,
)
from ProQSAR.data_generator import DataGenerator
from ProQSAR.data_preprocessor import DataPreprocessor
from ProQSAR.optimal_dataset import OptimalDataset
from ProQSAR.Config.config import Config
from ProQSAR.Config.debug import setup_logging
from ProQSAR.ModelDeveloper.model_validation import ModelValidation
from ProQSAR.Analysis.statistical_analysis import StatisticalAnalysis
from copy import deepcopy
from typing import Optional, Iterable, Union


logger = setup_logging()


class ProQSAR:
    def __init__(
        self,
        activity_col: str,
        id_col: str,
        smiles_col: str = "SMILES",
        mol_col: str = "mol",
        project_name: str = "Project",
        n_jobs: int = 1,
        random_state: int = 42,
        scoring_target: str = None,
        scoring_list: Optional[Union[list, str]] = None,
        n_splits: int = 5,
        n_repeats: int = 5,
        config=None,
        log_file: Optional[str] = "logging.log",
        log_level: str = "INFO",
    ):
        self.activity_col = activity_col
        self.id_col = id_col
        self.smiles_col = smiles_col
        self.mol_col = mol_col
        self.project_name = project_name
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.scoring_target = scoring_target
        self.scoring_list = scoring_list
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.config = config or Config()

        self.logger = setup_logging(log_level, f"{project_name}/{log_file}")

        self.shape_summary = {}

        self.optimaldata = OptimalDataset(
            activity_col,
            id_col,
            self.smiles_col,
            mol_col,
            save_dir=self.project_name,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            config=self.config,
            scoring_target=self.scoring_target,
            scoring_list=self.scoring_list,
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            save_cv_report=True,
            cv_report_name="cv_report_datasets",
        )

        self.datagenerator = DataGenerator(
            activity_col,
            id_col,
            smiles_col,
            mol_col,
            n_jobs=1,
            save_dir=self.project_name,
            config=self.config,
        )
        self.datapreprocessor = DataPreprocessor(
            activity_col, id_col, save_dir=self.project_name, config=self.config
        )

        self.splitter = self.config.splitter.set_params(
            activity_col=activity_col,
            smiles_col=(
                self.smiles_col
                if self.config.standardizer.deactivate
                else f"standardized_{self.smiles_col}"
            ),
            mol_col=(
                self.mol_col
                if self.config.standardizer.deactivate
                else f"standardized_mol"
            ),
            save_dir=self.project_name,
            random_state=self.random_state,
        )

        self.feature_selector = self.config.feature_selector.set_params(
            activity_col=activity_col,
            id_col=id_col,
            save_trans_data=True,
            save_dir=self.project_name,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            scoring_target=self.scoring_target,
            scoring_list=self.scoring_list,
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            save_cv_report=True,
            cv_report_name="cv_report_feature_selectors",
        )

        self.model_dev = self.config.model_dev.set_params(
            activity_col=activity_col,
            id_col=id_col,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            scoring_target=self.scoring_target,
            scoring_list=self.scoring_list,
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            save_cv_report=False,
        )

        self.optimizer = (
            self.config.optimizer.set_params(
                activity_col=activity_col,
                id_col=id_col,
                n_jobs=self.n_jobs,
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
                scoring=self.scoring_target,
                random_state=self.random_state,
                study_name=self.project_name,
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
        self.logger.info("----------FITTING----------")

        # Generating features, splitting & preprocessing
        if (
            isinstance(self.config.featurizer.feature_types, list)
            and not self.config.featurizer.deactivate
        ):
            self.logger.info(
                f"Finding optimal dataset among {self.config.featurizer.feature_types}."
            )
            self.optimaldata.datagenerator.set_params(data_name="data_dev")
            self.selected_feature = self.optimaldata.run(data_dev)
            self.logger.info(
                f"----------Optimal dataset: {self.selected_feature}----------"
            )

            self.datagenerator.featurizer.set_params(
                feature_types=self.selected_feature
            )
            self.datapreprocessor = self.optimaldata.dataprep_fitted[
                self.selected_feature
            ]

            self.data_dev = self.optimaldata.data_features[self.selected_feature]
            self.train = self.optimaldata.train[self.selected_feature + "_preprocessed"]
            self.test = self.optimaldata.test[self.selected_feature + "_preprocessed"]
            self.shape_summary = self.optimaldata.shape_summary

        elif (
            isinstance(self.config.featurizer.feature_types, str)
            or self.config.featurizer.deactivate
        ):

            if self.config.featurizer.deactivate:
                self.selected_feature = "original"
            else:
                self.selected_feature = self.config.featurizer.feature_types

            # Generate features
            self.data_dev = self.datagenerator.set_params(
                data_name="data_dev"
            ).generate(data_dev)

            # Train test split
            self.splitter.set_params(data_name=self.selected_feature)
            self.train, self.test = self.splitter.fit(self.data_dev)

            ## Record data shape transformation
            self._record_shape("original", self.selected_feature, "train", self.train)
            self._record_shape("original", self.selected_feature, "test", self.train)

            # Train data preprocessing
            self.datapreprocessor.set_params(data_name=f"train_{self.selected_feature}")
            self.train = self.datapreprocessor.fit_transform(self.train)

            ## Record data shape transformation
            for step, transformer in self.datapreprocessor.pipeline.steps:
                self._record_shape(
                    step, self.selected_feature, "train", transformer.transformed_data
                )

            # Test data preprocessing
            self.datapreprocessor.set_params(data_name=f"test_{self.selected_feature}")
            self.test = self.datapreprocessor.transform(self.test)

            ## Record data shape transformation
            for step, transformer in self.datapreprocessor.pipeline.steps:
                self._record_shape(
                    step, self.selected_feature, "test", transformer.transformed_data
                )

        # Feature selection - train
        self.feature_selector.set_params(
            trans_data_name=f"train_{self.selected_feature}_fs"
        )
        self.train = self.feature_selector.fit_transform(self.train)

        self.feature_selector.set_params(
            trans_data_name=f"test_{self.selected_feature}_fs"
        )
        self.test = self.feature_selector.transform(self.test)

        ## Record data shape transformation after feature selection
        self._record_shape(
            f"feature_selector ({self.feature_selector.select_method})",
            self.selected_feature,
            "train",
            self.train,
        )
        self._record_shape(
            f"feature_selector ({self.feature_selector.select_method})",
            self.selected_feature,
            "test",
            self.test,
        )

        # Model development & optimization
        add_model = deepcopy(self.model_dev.add_model)
        model_map = _get_model_map(
            task_type=None, add_model=add_model, n_jobs=self.n_jobs
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
                add_model.update({f"{self.model_dev.select_model}_opt": new_model})
                self.model_dev.set_params(
                    add_model=add_model,
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
                add_model.update({self.model_dev.select_model: new_model})
                self.model_dev.set_params(add_model=add_model)

            self.model_dev.fit(self.train)

        # Conformal predictor
        if self.conf_pred:
            self.conf_pred.set_params(model=self.model_dev)
            self.conf_pred.fit(self.train)

        # Applicability domain
        if self.ad:
            self.ad.fit(self.train)

        # Save pipeline
        save_path = f"{self.project_name}/proqsar.pkl"
        with open(save_path, "wb") as file:
            pickle.dump(self, file)
        self.logger.info(
            f"ProQSAR: Automated QSAR pipeline saved at: {save_path}."
        )

        return self

    def _predict_wo_prep(
        self,
        data,
        alpha: Optional[Union[float, Iterable[float]]] = None,
        save_name: str = "pred",
    ):

        # Conformal Predictor
        if self.conf_pred:
            pred_result = self.conf_pred.predict(data, alpha=alpha)
        else:
            pred_result = self.model_dev.predict(data)

        # Predict using AD (if enable)
        if self.ad:
            ad_result = self.ad.predict(data)
            pred_result = pd.merge(pred_result, ad_result, on=self.id_col)

        # Save pred_result
        save_path = f"{self.project_name}/PredResult"
        os.makedirs(save_path, exist_ok=True)
        pred_result.to_csv(f"{save_path}/{save_name}.csv", index=False)

        return pred_result

    def predict(self, data_pred, alpha: Optional[Union[float, Iterable[float]]] = None):
        """
        Predict activity values for new data.

        Parameters:
        -----------
        data_pred : pd.DataFrame
            The input data for prediction. It should include the SMILES & ID columns.

        """
        self.logger.info("----------PREDICTING----------")

        # Generate feature, preprocess data & apply feature selection
        self.data_pred = self.datagenerator.set_params(data_name="data_pred").generate(
            data_pred
        )
        self.data_pred = self.data_pred.drop(
            columns=self.splitter.smiles_col, errors="ignore"
        )

        self.datapreprocessor.set_params(data_name=f"data_pred_{self.selected_feature}")
        self.data_pred = self.datapreprocessor.transform(self.data_pred)

        self.feature_selector.set_params(
            trans_data_name=f"data_pred_{self.selected_feature}_fs"
        )
        self.data_pred = self.feature_selector.transform(self.data_pred)

        # Predict data_pred
        pred_result = self._predict_wo_prep(
            self.data_pred, alpha, save_name="data_pred"
        )

        return pred_result

    def validate(self):
        self.logger.info("----------VALIDATING----------")

        # Cross validation report
        self.cv_report = self.model_dev.report
        self.cv_report.to_csv(f"{self.project_name}/cv_report_model.csv", index=False)

        # External validation report
        select_model = self.cv_report.columns[
            ~self.cv_report.columns.isin(["scoring", "cv_cycle"])
        ]

        self.ev_report = ModelValidation.external_validation_report(
            data_train=self.train,
            data_test=self.test,
            activity_col=self.activity_col,
            id_col=self.id_col,
            select_model=select_model,
            add_model=self.model_dev.add_model,
            scoring_list=_match_cv_ev_metrics(self.cv_report["scoring"].unique()),
            n_jobs=self.n_jobs,
            save_csv=True,
            csv_name="ev_report_model",
            save_dir=self.project_name,
        )
        # Generate ROC-AUC / PR curve (for classification) or scatter plot (for regresssion)
        if self.model_dev.task_type == "C":
            ModelValidation.make_curve(
                data_train=self.train,
                data_test=self.test,
                activity_col=self.activity_col,
                id_col=self.id_col,
                select_model=select_model,
                add_model=self.model_dev.add_model,
                save_dir=self.project_name,
                n_jobs=self.n_jobs,
            )
        else:
            ModelValidation.make_scatter_plot(
                data_train=self.train,
                data_test=self.test,
                activity_col=self.activity_col,
                id_col=self.id_col,
                select_model=select_model,
                add_model=self.model_dev.add_model,
                scoring_df=self.ev_report,
                save_dir=self.project_name,
                n_jobs=self.n_jobs,
            )

        return self.cv_report, self.ev_report

    def analysis(self):
        self.logger.info("----------ANALYSING----------")

        if self.optimaldata.report is not None:
            self.logger.info(f"----------Dataset----------")
            StatisticalAnalysis.analysis(
                report_df=self.optimaldata.report,
                scoring_list=None,
                method_list=None,
                check_assumptions=True,
                method="all",
                save_dir=f"{self.project_name}/OptimalDataStat",
            )
        if self.feature_selector.report is not None:
            self.logger.info(f"----------FeatureSelector----------")
            StatisticalAnalysis.analysis(
                report_df=self.feature_selector.report,
                scoring_list=None,
                method_list=None,
                check_assumptions=True,
                method="all",
                save_dir=f"{self.project_name}/FeatureSelectorStat",
            )
        if self.model_dev.report is not None:
            self.logger.info(f"----------ModelDev----------")
            StatisticalAnalysis.analysis(
                report_df=self.model_dev.report,
                scoring_list=None,
                method_list=None,
                check_assumptions=True,
                method="all",
                save_dir=f"{self.project_name}/ModelDevStat",
            )

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

        shape_summary_df = pd.DataFrame(records)
        shape_summary_df.to_csv(f"{self.project_name}/shape_summary.csv", index=False)

        return shape_summary_df

    def run_all(
        self, 
        data_dev: pd.DataFrame, 
        data_pred: Optional[pd.DataFrame] = None, 
        alpha: Optional[Union[float, Iterable[float]]] = None
    ):
        self.fit(data_dev)
        self._predict_wo_prep(self.test, alpha, save_name="test_pred")
        self.validate()
        self.analysis()
        self.get_shape_summary_df()

        if data_pred is not None:
            self.predict(data_pred, alpha)
