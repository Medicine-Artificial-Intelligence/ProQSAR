import os
import pickle
import time
import datetime
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
from typing import Optional, Iterable, Union, Any


class ProQSAR:
    def __init__(
        self,
        activity_col: str = "pChEMBL",
        id_col: str = "ID",
        smiles_col: str = "SMILES",
        mol_col: str = "mol",
        project_name: str = "Project",
        n_jobs: int = 1,
        random_state: int = 42,
        scoring_target: Optional[str] = None,
        scoring_list: Optional[Union[list, str]] = None,
        n_splits: int = 5,
        n_repeats: int = 5,
        keep_all_train: bool = False,
        keep_all_test: bool = False,
        keep_all_pred: bool = False,
        config=None,
        log_file: Optional[str] = "logging.log",
        log_level: str = "INFO",
    ):
        # Basic settings
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
        self.keep_all_test = keep_all_test
        self.keep_all_pred = keep_all_pred

        # Configuration and directories
        self.config = config or Config()
        self.save_dir = f"Project/{self.project_name}"
        os.makedirs(self.save_dir, exist_ok=True)

        # Logging setup
        self.logger = setup_logging(log_level, f"{self.save_dir}/{log_file}")
        self.shape_summary: dict[str, Any] = {}

        self.optimaldata = OptimalDataset(
            activity_col,
            id_col,
            self.smiles_col,
            mol_col,
            save_dir=self.save_dir,
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
            n_jobs=self.n_jobs,
            save_dir=self.save_dir,
            config=self.config,
        )

        self.datapreprocessor = DataPreprocessor(
            activity_col, id_col, save_dir=self.save_dir, config=self.config
        )
        if keep_all_train:
            self.datapreprocessor.duplicate.set_params(rows=False)
            self.datapreprocessor.multiv_outlier.set_params(deactivate=True)

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
            save_dir=self.save_dir,
            random_state=self.random_state,
        )

        self.feature_selector = self.config.feature_selector.set_params(
            activity_col=activity_col,
            id_col=id_col,
            save_trans_data=True,
            save_dir=self.save_dir,
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

    def fit(self, data_dev: pd.DataFrame):
        """
        Fit the complete QSAR pipeline on the development dataset.
        """

        start_time = time.perf_counter()
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
            self.test = self.optimaldata.test[self.selected_feature]
            # self.test = self.optimaldata.test[self.selected_feature + "_preprocessed"]
            self.shape_summary = self.optimaldata.shape_summary

        elif (
            isinstance(self.config.featurizer.feature_types, str)
            or self.config.featurizer.deactivate
        ):

            self.selected_feature = (
                "original" if self.config.featurizer.deactivate else self.config.featurizer.feature_types
            )

            # Generate features
            self.data_dev = self.datagenerator.set_params(
                data_name="data_dev"
            ).generate(data_dev)

            # Train test split
            self.splitter.set_params(data_name=self.selected_feature)
            self.train, self.test = self.splitter.fit(self.data_dev)

            ## Record data shape transformation
            self._record_shape("original", self.selected_feature, "train", self.train)
            # self._record_shape("original", self.selected_feature, "test", self.test)

            # Train data preprocessing
            self.datapreprocessor.set_params(data_name=f"train_{self.selected_feature}")
            self.datapreprocessor.fit(self.train)
            self.train = self.datapreprocessor.transform(self.train)

            ## Record data shape transformation
            for step, transformer in self.datapreprocessor.pipeline.steps:
                self._record_shape(
                    step, self.selected_feature, "train", transformer.transformed_data
                )

            # Test data preprocessing
            # self.datapreprocessor.set_params(data_name=f"test_{self.selected_feature}")
            # self.test = self.datapreprocessor.transform(self.test)

            ## Record data shape transformation
            # for step, transformer in self.datapreprocessor.pipeline.steps:
            #    self._record_shape(
            #        step, self.selected_feature, "test", transformer.transformed_data
            #    )

        # Save the fitted data preprocessor
        save_path = f"{self.save_dir}/proqsar.pkl"
        self.save_pipeline(save_path)
    

        # Feature selection - train
        self.feature_selector.set_params(
            trans_data_name=f"train_{self.selected_feature}_feature_selector", 
        )
        self.train = self.feature_selector.fit_transform(self.train)

        # Save the fitted feature selector
        self.save_pipeline(save_path)

        ## Record data shape transformation after feature selection
        self._record_shape(
            f"feature_selector ({self.feature_selector.select_method})",
            self.selected_feature,
            "train",
            self.train,
        )
        # self._record_shape(
        #    f"feature_selector ({self.feature_selector.select_method})",
        #    self.selected_feature,
        #    "test",
        #    self.test,
        # )

        # Model Development & Optimization
        self.model_dev.fit(self.train)
        self.select_model = deepcopy(self.model_dev.select_model)

        # Save the model development
        self.save_pipeline(save_path)

        # If select_model in model_dev is a list or None, perform cross validation to choose best
        # If select_model in model_dev is a str, directly use this algorithm to build model
        # If optimizer: optimize hyperparams of the best/selected algorithm & compare to base model
        # Which better will be used

        if self.optimizer:
            self._optimize_model(select_model=self.select_model)
            self.save_pipeline(save_path)


        # Conformal predictor
        if self.conf_pred:
            self.conf_pred.set_params(model=self.model_dev)
            self.conf_pred.fit(self.train)

        # Applicability domain
        if self.ad:
            self.ad.fit(self.train)

        # Save pipeline
        self.save_pipeline(save_path)
        self.logger.info(f"ProQSAR: Pipeline saved at {save_path}.")

        # End the timer
        elapsed = datetime.timedelta(seconds=time.perf_counter() - start_time)
        self.logger.info(f"----- FIT COMPLETE in {elapsed} -----")

        return self

    def _optimize_model(self, select_model: str) -> None:
        """
        Selects the current model, runs hyperparameter optimization,
        and updates the model if the optimized version improves CV score.
        """
        # Get the current report and add_model from model_dev
        add_model = deepcopy(self.model_dev.add_model)
        model_map = _get_model_map(
            task_type=None, add_model=add_model, n_jobs=self.n_jobs
        )
        # Capture baseline report and score
        base_report = deepcopy(self.model_dev.report)

        # Optimize hyperparams of the best/selected algorithm & compare to base model
        #  Whichever performs better (higher mean CV score) will be used.

        # Hyperparameter optimization on current best model
        self.optimizer.set_params(select_model=select_model)
        opt_best_params, opt_best_score = self.optimizer.optimize(self.train)

        # Compare optimized score to baseline
        if base_report is not None:
            base_best_score = (
                base_report
                .query(f"scoring == @self.model_dev.scoring_target")
                .set_index("cv_cycle")
                .at["mean", select_model]
                )

            if opt_best_score > base_best_score:
                self.logger.info(
                    f"Optimized params improved mean CV score "
                    f"({opt_best_score:.4f} > {base_best_score:.4f}); using optimized model."
                )
                # Build optimized model and refit
                optimized = model_map[select_model].set_params(**opt_best_params)
                add_model[f"{select_model}_opt"] = optimized
                self.model_dev.set_params(
                    add_model=add_model,
                    select_model=f"{select_model}_opt",
                    cross_validate=True,
                )
                self.model_dev.fit(self.train)

                # Merge baseline and optimized reports for comparison
                self.model_dev.report = (
                    pd.merge(
                        base_report,
                        self.model_dev.report,
                        on=["scoring", "cv_cycle"],
                        suffixes=("_1", "_2"),
                    )
                    .set_index(["scoring", "cv_cycle"])
                    .sort_index(axis=1)
                    .reset_index()
                )

            else:
                # Revert to base if optimization didn’t help
                self.logger.info(
                    f"Optimized params did not improve ({opt_best_score:.4f} ≤ {base_best_score:.4f}); "
                    f"keeping base model."
                )
        else:
            optimized = model_map[select_model].set_params(**opt_best_params)
            add_model[f"{select_model}_opt"] = optimized
            self.model_dev.set_params(
                add_model=add_model,
                select_model=f"{select_model}_opt",
                #cross_validate=True,
            )
            self.model_dev.fit(self.train)


    def optimize(self):
        """
        Standalone hyperparameter optimization on the existing trained model.
        """
        self.logger.info("----------OPTIMIZING HYPERPARAMETERS----------")
        start_time = time.perf_counter()

        # Ensure optimizer is active
        self.optimizer = self.config.optimizer.set_params(deactivate=False)

        # Get the current report and add_model from model_dev
        self._optimize_model(select_model=self.select_model)

        # Refit conformal & AD
        if self.conf_pred:
            self.conf_pred.set_params(model=self.model_dev)
            self.conf_pred.fit(self.train)

        if self.ad:
            self.ad.fit(self.train)

        elapsed = datetime.timedelta(seconds=time.perf_counter() - start_time)
        self.logger.info(f"----- OPTIMIZATION COMPLETE in {elapsed} -----")

        return self


    def save_pipeline(self, path: str):
        """
        Save the ProQSAR pipeline to a pickle file.
        Parameters:
        -----------
        path : str
            The path where the ProQSAR pipeline will be saved.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(self, file)


    def load_pipeline(self, path: str):
        """
        Load a ProQSAR pipeline from a pickle file.
        Parameters:
        -----------
        path : str
            The path to the pickle file containing the ProQSAR pipeline.
        """
        with open(path, "rb") as file:
            loaded_pipeline = pickle.load(file)

        # Update the current instance with the loaded pipeline's attributes
        self.__dict__.update(loaded_pipeline.__dict__)
        self.logger.info(f"ProQSAR: Pipeline loaded from {path}.")
        return self
    
    
    def _apply_generator(
        self,
        data: pd.DataFrame,
        data_name: str = "test",
        record_shape=True
    ) -> pd.DataFrame:
        """
        Take the raw data and run it through the DataGenerator pipeline.
        Returns the generated features DataFrame.
        """
        # Copy the data to avoid modifying the original DataFrame
        df = deepcopy(data)
        self.datagenerator.set_params(data_name=data_name, save_dir=self.save_dir)
        df = self.datagenerator.generate(df)

        # Drop SMILES and mol columns if they exist
        df = df.drop(
            columns=[self.splitter.smiles_col, self.splitter.mol_col], errors="ignore"
        )
        # Record the shape of the original data
        if record_shape:
            self._record_shape("original", self.selected_feature, data_name, df)

        return df

    def _apply_prep(
        self,
        data: pd.DataFrame,
        data_name: str = "test",
        keep_all_records: bool = False,
        record_shape: bool = False,
    ) -> pd.DataFrame:
        """
        Take the raw data and run it through the already‐fitted
        DataPreprocessor and FeatureSelector pipelines. Returns the fully transformed data DataFrame.
        """
        if keep_all_records:
            self.datapreprocessor.duplicate.set_params(rows=False)
            self.datapreprocessor.multiv_outlier.set_params(deactivate=True)

        # Copy the data to avoid modifying the original DataFrame
        df = deepcopy(data)
        (
            self._record_shape("original", self.selected_feature, data_name, df)
            if record_shape
            else None
        )

        # Data preprocessing
        self.datapreprocessor.set_params(
            data_name=f"{data_name}_{self.selected_feature}", save_dir=self.save_dir
        )
        df = self.datapreprocessor.transform(df)

        if record_shape:
            for step, transformer in self.datapreprocessor.pipeline.steps:
                self._record_shape(
                    step, self.selected_feature, data_name, transformer.transformed_data
                )
        # Feature selection
        self.feature_selector.set_params(
            trans_data_name=f"{data_name}_{self.selected_feature}_feature_selector",
            save_dir=self.save_dir,
        )
        df = self.feature_selector.transform(df)
        if record_shape:
            self._record_shape(
                f"feature_selector ({self.feature_selector.select_method})",
                self.selected_feature,
                data_name,
                df,
            )
        return df

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
        save_path = f"{self.save_dir}/PredResult"
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
        data_pred = self._apply_generator(
            data_pred,
            data_name="data_pred",
            record_shape=True,
        ) 

        # Preprocess data_pred
        data_pred = self._apply_prep(
            data_pred,
            data_name="data_pred",
            keep_all_records=self.keep_all_pred,
            record_shape=True,
        )

        # Predict data_pred
        pred_result = self._predict_wo_prep(data_pred, alpha, save_name="data_pred")

        return pred_result

    def validate(self, external_test_data: Optional[pd.DataFrame] = None):
        self.logger.info("----------VALIDATING----------")

        # Cross validation report
        if self.model_dev.report is not None:
            self.cv_report = self.model_dev.report
        else:
            self.cv_report = ModelValidation.cross_validation_report(
                data=self.train,
                activity_col=self.activity_col,
                id_col=self.id_col,
                add_model=self.model_dev.add_model,
                select_model=self.select_model,
                scoring_list=self.scoring_list,
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
        self.cv_report.to_csv(f"{self.save_dir}/cv_report_model.csv", index=False)

        # External validation report
        # Select models for external validation report
        select_model = self.cv_report.columns[
            ~self.cv_report.columns.isin(["scoring", "cv_cycle"])
        ]
        # Prepare test data for external validation
        if external_test_data is not None:
            self.logger.info(
                "External test data provided. Using it for external validation."
            )
            test_data = external_test_data
        elif hasattr(self, "test") and self.test is not None:
            self.logger.info(
                "External validation will be performed on the test set from the train-test split."
            )
            test_data = self.test
        else:
            raise ValueError(
                "No external test data provided and no test set available from the train-test split."
            )

        self.test_prep = self._apply_prep(
            test_data,
            data_name="test",
            keep_all_records=self.keep_all_test,
            record_shape=True,
        )

        self.ev_report = ModelValidation.external_validation_report(
            data_train=self.train,
            data_test=self.test_prep,
            activity_col=self.activity_col,
            id_col=self.id_col,
            select_model=select_model,
            add_model=self.model_dev.add_model,
            scoring_list=_match_cv_ev_metrics(self.cv_report["scoring"].unique()),
            n_jobs=self.n_jobs,
            save_csv=True,
            csv_name="ev_report_model",
            save_dir=self.save_dir,
        )
        # Generate ROC-AUC / PR curve (for classification) or scatter plot (for regresssion)
        if self.model_dev.task_type == "C":
            ModelValidation.make_curve(
                data_train=self.train,
                data_test=self.test_prep,
                activity_col=self.activity_col,
                id_col=self.id_col,
                select_model=select_model,
                add_model=self.model_dev.add_model,
                save_dir=self.save_dir,
                n_jobs=self.n_jobs,
            )
        else:
            ModelValidation.make_scatter_plot(
                data_train=self.train,
                data_test=self.test_prep,
                activity_col=self.activity_col,
                id_col=self.id_col,
                select_model=select_model,
                add_model=self.model_dev.add_model,
                scoring_df=self.ev_report,
                save_dir=self.save_dir,
                n_jobs=self.n_jobs,
            )

        return self.cv_report, self.ev_report

    def analysis(self):
        self.logger.info("----------ANALYSING----------")

        if self.optimaldata.report is not None:
            self.logger.info("----------OptimalData----------")
            StatisticalAnalysis.analysis(
                report_df=self.optimaldata.report,
                scoring_list=None,
                method_list=None,
                check_assumptions=True,
                method="all",
                save_dir=f"{self.save_dir}/OptimalDataStat",
            )
        if self.feature_selector.report is not None:
            self.logger.info("----------FeatureSelector----------")
            StatisticalAnalysis.analysis(
                report_df=self.feature_selector.report,
                scoring_list=None,
                method_list=None,
                check_assumptions=True,
                method="all",
                save_dir=f"{self.save_dir}/FeatureSelectorStat",
            )
        if self.model_dev.report is not None:
            self.logger.info("----------ModelDev----------")
            StatisticalAnalysis.analysis(
                report_df=self.model_dev.report,
                scoring_list=None,
                method_list=None,
                check_assumptions=True,
                method="all",
                save_dir=f"{self.save_dir}/ModelDevStat",
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
        shape_summary_df.to_csv(f"{self.save_dir}/shape_summary.csv", index=False)

        return shape_summary_df

    def run_all(
        self,
        data_dev: pd.DataFrame,
        data_pred: Optional[pd.DataFrame] = None,
        data_test: Optional[pd.DataFrame] = None,
        alpha: Optional[Union[float, Iterable[float]]] = None,
    ):
        """Run the entire ProQSAR pipeline from fitting to prediction.
        Parameters:
        -----------
        data_dev : pd.DataFrame
            The development dataset for fitting the model.
        data_pred : Optional[pd.DataFrame]
            The prediction dataset. If provided, predictions will be made on this data.
        alpha : Optional[Union[float, Iterable[float]]]
            Significance level for the conformal predictor. If None, predictions will be made without confidence intervals.
        """
        # Start the timer
        start = time.perf_counter()

        self.logger.info(
            f"----------STARTING PROQSAR PIPELINE AT {datetime.datetime.now()}----------"
        )

        # Fit the model with the development data
        self.fit(data_dev)
        self.validate(external_test_data=data_test)
        self.analysis()
        self._predict_wo_prep(self.test_prep, alpha, save_name="test_pred")

        if data_pred is not None:
            self.predict(data_pred, alpha)

        self.get_shape_summary_df()
        
        self.logger.info(
            f"----------PROQSAR PIPELINE COMPLETED AT {datetime.datetime.now()}----------"
        )

        # End the timer
        end = time.perf_counter()
        # Calculate elapsed time
        self.logger.info(f"Elapsed time: {datetime.timedelta(seconds=(end-start))}")
