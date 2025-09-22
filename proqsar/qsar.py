import os
import pickle
import time
import datetime
import pandas as pd
import matplotlib

from proqsar.Model.ModelDeveloper.model_developer_utils import (
    _get_model_map,
    _match_cv_ev_metrics,
)
from proqsar.data_generator import DataGenerator
from proqsar.data_preprocessor import DataPreprocessor
from proqsar.optimal_dataset import OptimalDataset
from proqsar.Config.config import Config
from proqsar.Config.debug import setup_logging
from proqsar.Model.ModelDeveloper.model_validation import ModelValidation
from proqsar.Evaluation.statistical_analysis import StatisticalAnalysis
from copy import deepcopy
from typing import Optional, Iterable, Union, Any, Tuple
from typing import Dict, List

matplotlib.use("Agg")


class ProQSAR:
    """
    Top-level orchestrator for an end-to-end QSAR modelling pipeline using ProQSAR components.

    The ProQSAR class wires together DataGenerator, DataPreprocessor, OptimalDataset,
    ModelDeveloper, Optimizer, Conformal Predictor and Applicability Domain components
    (as provided through a Config object) to provide a high-level API:
      - `fit(data_dev)`: finds the optimal feature set (if applicable), preprocesses,
        selects features, trains models and optionally optimizes them.
      - `predict(data_pred)`: generate predictions for new raw input.
      - `validate(external_test_data)`: create CV and external validation reports.
      - `analysis()`: run StatisticalAnalysis on collected CV reports.
      - `run_all(...)`: run the whole pipeline (fit → validate → analysis → predict).

    Notes
    -----
    This class expects a ProQSAR `Config` object to supply components such as:
      - config.featurizer
      - config.splitter
      - config.feature_selector
      - config.model_dev
      - config.optimizer
      - config.conf_pred
      - config.ad
    If `config` is None a default `Config()` is instantiated.

    :param activity_col: Column name for target/activity values (default: "pChEMBL").
    :type activity_col: str
    :param id_col: Column name for identifiers (default: "ID").
    :type id_col: str
    :param smiles_col: Column name containing SMILES strings (default: "SMILES").
    :type smiles_col: str
    :param mol_col: Column name used for molecule objects (default: "mol").
    :type mol_col: str
    :param project_name: Name of the project directory placed under "Project/" (default: "Project").
    :type project_name: str
    :param n_jobs: Number of parallel jobs where supported (default: 1).
    :type n_jobs: int
    :param random_state: RNG seed for reproducibility (default: 42).
    :type random_state: int
    :param scoring_target: Primary metric used for selecting models/feature sets (default: None).
    :type scoring_target: Optional[str]
    :param scoring_list: List (or single) metric(s) to compute during CV (default: None).
    :type scoring_list: Optional[Union[list, str]]
    :param n_splits: Number of CV splits (default: 5).
    :type n_splits: int
    :param n_repeats: Number of CV repeats (default: 5).
    :type n_repeats: int
    :param keep_all_train: If True, tweak preprocessing to retain all training records (default: False).
    :type keep_all_train: bool
    :param keep_all_test: If True, keep all test records during preprocessing (default: False).
    :type keep_all_test: bool
    :param keep_all_pred: If True, keep all prediction records during preprocessing (default: False).
    :type keep_all_pred: bool
    :param config: Optional ProQSAR Config instance (if None, a default Config() is used).
    :type config: Optional[Config]
    :param log_file: Name of the log file saved inside the project directory (default: "logging.log").
    :type log_file: Optional[str]
    :param log_level: Logging level name passed to the debug setup (default: "INFO").
    :type log_level: str
    """

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
        self.shape_summary: Dict[str, Any] = {}

        # Components
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
            # adjust duplicate/removal behavior to keep training rows / avoid dropping via multiv_outlier
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
                else "standardized_mol"
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

    def fit(self, data_dev: pd.DataFrame) -> "ProQSAR":
        """
        Fit the full ProQSAR pipeline on the development dataset.

        This performs feature generation (or dataset selection), splitting,
        preprocessing, feature selection, model development and optional
        hyperparameter optimization. Conformal predictor and applicability
        domain components are also fitted if configured.

        :param data_dev: Development dataset (raw) containing SMILES, id and activity columns.
        :type data_dev: pd.DataFrame
        :returns: self (fitted pipeline)
        :rtype: ProQSAR
        """
        start_time = time.perf_counter()
        self.logger.info("----------FITTING----------")

        # Generating features, splitting & preprocessing
        if (
            isinstance(self.config.featurizer.feature_types, list)
            and not self.config.featurizer.deactivate
        ):
            # Search optimal feature set
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
            self.shape_summary = self.optimaldata.shape_summary

        elif (
            isinstance(self.config.featurizer.feature_types, str)
            or self.config.featurizer.deactivate
        ):
            self.selected_feature = (
                "original"
                if self.config.featurizer.deactivate
                else self.config.featurizer.feature_types
            )

            # Generate features
            self.data_dev = self.datagenerator.set_params(
                data_name="data_dev"
            ).generate(data_dev)

            # Train test split
            self.splitter.set_params(data_name=self.selected_feature)
            self.train, self.test = self.splitter.fit(self.data_dev)

            # Record data shape transformation
            self._record_shape("original", self.selected_feature, "train", self.train)

            # Train data preprocessing
            self.datapreprocessor.set_params(data_name=f"train_{self.selected_feature}")
            self.datapreprocessor.fit(self.train)
            self.train = self.datapreprocessor.transform(self.train)

            # Record shapes after each preprocessing step
            for step, transformer in self.datapreprocessor.pipeline.steps:
                self._record_shape(
                    step, self.selected_feature, "train", transformer.transformed_data
                )

        # Save the fitted pipeline state
        save_path = f"{self.save_dir}/proqsar.pkl"
        self.save_pipeline(save_path)

        # Feature selection on the training data
        self.feature_selector.set_params(
            trans_data_name=f"train_{self.selected_feature}_feature_selector",
        )
        self.train = self.feature_selector.fit_transform(self.train)

        # Save the pipeline again
        self.save_pipeline(save_path)

        # Record shapes after feature selection
        self._record_shape(
            f"feature_selector ({self.feature_selector.select_method})",
            self.selected_feature,
            "train",
            self.train,
        )

        # Model development
        self.model_dev.fit(self.train)
        self.select_model = deepcopy(self.model_dev.select_model)

        # Save the model development object
        self.save_pipeline(save_path)

        # Optional optimizer
        if self.optimizer:
            self._optimize_model(select_model=self.select_model)

        # Conformal predictor
        if self.conf_pred:
            self.conf_pred.set_params(model=self.model_dev)
            self.conf_pred.fit(self.train)

        # Applicability domain
        if self.ad:
            self.ad.fit(self.train)

        # Final save
        self.save_pipeline(save_path)
        self.logger.info(f"ProQSAR: Pipeline saved at {save_path}.")

        elapsed = datetime.timedelta(seconds=time.perf_counter() - start_time)
        self.logger.info(f"----- FIT COMPLETE in {elapsed} -----")

        return self

    def _optimize_model(self, select_model: str) -> None:
        """
        Internal helper that executes hyperparameter optimization for a selected model.

        The method compares optimized performance against the baseline CV mean score.
        If the optimized parameters improve the score the optimized model is used.

        :param select_model: Name of the model to optimize (as present in model_dev).
        :type select_model: str
        :returns: None
        """
        add_model = deepcopy(self.model_dev.add_model)
        model_map = _get_model_map(
            task_type=None, add_model=add_model, n_jobs=self.n_jobs
        )

        base_report = deepcopy(self.model_dev.report)

        # Run optimizer
        self.optimizer.set_params(select_model=select_model)
        opt_best_params, opt_best_score = self.optimizer.optimize(self.train)

        if base_report is not None:
            base_best_score = (
                base_report.query("scoring == @self.model_dev.scoring_target")
                .set_index("cv_cycle")
                .at["mean", select_model]
            )

            if opt_best_score > base_best_score:
                self.logger.info(
                    f"Optimized params improved mean CV score "
                    f"({opt_best_score:.4f} > {base_best_score:.4f}); using optimized model."
                )
                optimized = model_map[select_model].set_params(**opt_best_params)
                add_model[f"{select_model}_opt"] = optimized
                self.model_dev.set_params(
                    add_model=add_model,
                    select_model=f"{select_model}_opt",
                    cross_validate=True,
                )
                self.model_dev.fit(self.train)

                # merge reports
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
                self.logger.info(
                    f"Optimized params did not improve ({opt_best_score:.4f} ≤ "
                    + f"{base_best_score:.4f}); keeping base model."
                )
        else:
            optimized = model_map[select_model].set_params(**opt_best_params)
            add_model[f"{select_model}_opt"] = optimized
            self.model_dev.set_params(
                add_model=add_model,
                select_model=f"{select_model}_opt",
            )
            self.model_dev.fit(self.train)

    def optimize(self) -> "ProQSAR":
        """
        Public method to run optimizer (if configured) for the current selected model.

        After optimization, conformal predictor and AD (if configured) are refit.

        :returns: self (after optimization)
        :rtype: ProQSAR
        """
        self.logger.info("----------OPTIMIZING HYPERPARAMETERS----------")
        start_time = time.perf_counter()

        # Ensure optimizer is active
        self.optimizer = self.config.optimizer.set_params(deactivate=False)

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

    def save_pipeline(self, path: str) -> None:
        """
        Persist the entire ProQSAR pipeline instance to disk using pickle.

        :param path: File path where the pipeline will be saved.
        :type path: str
        :returns: None
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(self, file)

    def load_pipeline(self, path: str) -> "ProQSAR":
        """
        Load a previously saved ProQSAR pipeline and update the current instance.

        :param path: Path to the pickled ProQSAR object.
        :type path: str
        :returns: self updated with loaded pipeline state
        :rtype: ProQSAR
        """
        with open(path, "rb") as file:
            loaded_pipeline = pickle.load(file)

        self.__dict__.update(loaded_pipeline.__dict__)
        self.logger.info(f"ProQSAR: Pipeline loaded from {path}.")
        return self

    def _apply_generator(
        self, data: pd.DataFrame, data_name: str = "test", record_shape: bool = True
    ) -> pd.DataFrame:
        """
        Run DataGenerator on raw input and return a features DataFrame (SMILES/mol removed).

        :param data: Raw input DataFrame containing SMILES and id columns.
        :type data: pd.DataFrame
        :param data_name: Name used for metadata / saved files (default "test").
        :type data_name: str
        :param record_shape: If True, record the shape of the produced DataFrame in shape_summary.
        :type record_shape: bool
        :returns: Generated features DataFrame with SMILES/mol columns removed.
        :rtype: pd.DataFrame
        """
        df = deepcopy(data)
        self.datagenerator.set_params(data_name=data_name, save_dir=self.save_dir)
        df = self.datagenerator.generate(df)

        # Drop SMILES and mol columns if they exist
        df = df.drop(
            columns=[self.splitter.smiles_col, self.splitter.mol_col], errors="ignore"
        )

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
        Apply the fitted DataPreprocessor and FeatureSelector on provided data.

        :param data: DataFrame to preprocess and transform.
        :type data: pd.DataFrame
        :param data_name: Label used in metadata and saved outputs (default "test").
        :type data_name: str
        :param keep_all_records: If True, adjust preprocessors to keep all records (useful for predictions).
        :type keep_all_records: bool
        :param record_shape: If True, record shapes at each preprocessing step.
        :type record_shape: bool
        :returns: Fully transformed DataFrame after preprocessing and feature selection.
        :rtype: pd.DataFrame
        """
        if keep_all_records:
            self.datapreprocessor.duplicate.set_params(rows=False)
            self.datapreprocessor.multiv_outlier.set_params(deactivate=True)

        df = deepcopy(data)
        if record_shape:
            self._record_shape("original", self.selected_feature, data_name, df)

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
        data: pd.DataFrame,
        alpha: Optional[Union[float, Iterable[float]]] = None,
        save_name: str = "pred",
    ) -> pd.DataFrame:
        """
        Produce predictions assuming `data` is already preprocessed.

        Optionally uses ConformalPredictor and ApplicabilityDomain if configured,
        and saves the result CSV in save_dir/PredResult.

        :param data: Preprocessed DataFrame ready for prediction.
        :type data: pd.DataFrame
        :param alpha: Significance level(s) for conformal predictions (if used).
        :type alpha: Optional[Union[float, Iterable[float]]]
        :param save_name: File name (without directory) to use when saving predictions.
        :type save_name: str
        :returns: DataFrame containing prediction results.
        :rtype: pd.DataFrame
        """
        # Conformal Predictor (if configured)
        if self.conf_pred:
            pred_result = self.conf_pred.predict(data, alpha=alpha)
        else:
            pred_result = self.model_dev.predict(data)

        # Applicability Domain merge (if configured)
        if self.ad:
            ad_result = self.ad.predict(data)
            pred_result = pd.merge(pred_result, ad_result, on=self.id_col)

        # Save predictions
        save_path = f"{self.save_dir}/PredResult"
        os.makedirs(save_path, exist_ok=True)
        pred_result.to_csv(f"{save_path}/{save_name}.csv", index=False)

        return pred_result

    def predict(
        self,
        data_pred: pd.DataFrame,
        alpha: Optional[Union[float, Iterable[float]]] = None,
    ) -> pd.DataFrame:
        """
        Generate predictions on raw input data.

        This runs feature generation, preprocessing & feature selection for the
        input data and then predicts using the fitted model (or ConformalPredictor).

        :param data_pred: Raw dataset containing SMILES & ID columns for prediction.
        :type data_pred: pd.DataFrame
        :param alpha: Significance level(s) for conformal predictions (if configured).
        :type alpha: Optional[Union[float, Iterable[float]]]
        :returns: DataFrame with prediction results.
        :rtype: pd.DataFrame
        """
        self.logger.info("----------PREDICTING----------")

        # Generate features
        data_pred = self._apply_generator(
            data_pred,
            data_name="data_pred",
            record_shape=True,
        )

        # Preprocess and select features
        data_pred = self._apply_prep(
            data_pred,
            data_name="data_pred",
            keep_all_records=self.keep_all_pred,
            record_shape=True,
        )

        # Predict
        pred_result = self._predict_wo_prep(data_pred, alpha, save_name="data_pred")
        return pred_result

    def validate(
        self, external_test_data: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run validation for the fitted pipeline.

        Produces:
          - cross-validation report (cv_report)
          - external validation report (ev_report), using external_test_data if provided,
            otherwise the pipeline's test set.

        :param external_test_data: Optional external test dataset for validation.
        :type external_test_data: Optional[pd.DataFrame]
        :returns: Tuple of (cv_report, ev_report)
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        :raises ValueError: If neither external_test_data nor a pipeline test set is available.
        """
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

        # Prepare test data
        if external_test_data is not None:
            self.logger.info(
                "External test data provided. Using it for external validation."
            )
            test_data = self._apply_generator(
                external_test_data, data_name="test", record_shape=True
            )
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
            select_model=self.cv_report.columns[
                ~self.cv_report.columns.isin(["scoring", "cv_cycle"])
            ],
            add_model=self.model_dev.add_model,
            scoring_list=_match_cv_ev_metrics(self.cv_report["scoring"].unique()),
            n_jobs=self.n_jobs,
            save_csv=True,
            csv_name="ev_report_model",
            save_dir=self.save_dir,
        )

        # Diagnostic plots
        if self.model_dev.task_type == "C":
            ModelValidation.make_curve(
                data_train=self.train,
                data_test=self.test_prep,
                activity_col=self.activity_col,
                id_col=self.id_col,
                select_model=self.cv_report.columns[
                    ~self.cv_report.columns.isin(["scoring", "cv_cycle"])
                ],
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
                select_model=self.cv_report.columns[
                    ~self.cv_report.columns.isin(["scoring", "cv_cycle"])
                ],
                add_model=self.model_dev.add_model,
                scoring_df=self.ev_report,
                save_dir=self.save_dir,
                n_jobs=self.n_jobs,
            )

        return self.cv_report, self.ev_report

    def analysis(self) -> None:
        """
        Run StatisticalAnalysis on collected CV reports (optimaldata, feature_selector, model_dev).

        Generated figures and CSVs are written under configured project folders.
        """
        self.logger.info("----------ANALYSING----------")

        if getattr(self.optimaldata, "report", None) is not None:
            self.logger.info("----------OptimalData----------")
            StatisticalAnalysis.analysis(
                report_df=self.optimaldata.report,
                scoring_list=None,
                method_list=None,
                check_assumptions=True,
                method="all",
                save_dir=f"{self.save_dir}/OptimalDataStat",
            )
        if getattr(self.feature_selector, "report", None) is not None:
            self.logger.info("----------FeatureSelector----------")
            StatisticalAnalysis.analysis(
                report_df=self.feature_selector.report,
                scoring_list=None,
                method_list=None,
                check_assumptions=True,
                method="all",
                save_dir=f"{self.save_dir}/FeatureSelectorStat",
            )
        if getattr(self.model_dev, "report", None) is not None:
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
    ) -> None:
        """
        Record shapes at different pipeline stages in a nested dictionary.

        :param stage_name: Name of the pipeline stage (e.g., 'duplicate', 'lowvar').
        :type stage_name: str
        :param feature_set_name: Feature set identifier (e.g., 'ECFP4').
        :type feature_set_name: str
        :param data_name: The dataset label (e.g., 'train', 'test', 'data_pred').
        :type data_name: str
        :param data: DataFrame, a (n_rows, n_cols) tuple, or None. If DataFrame is provided its .shape is recorded.
        :type data: Optional[Union[pd.DataFrame, tuple]]
        :returns: None
        """
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
        """
        Convert the recorded shape_summary into a tidy pandas DataFrame and save it.

        :returns: DataFrame summarizing shapes across pipeline stages and datasets.
        :rtype: pd.DataFrame
        """
        records: List[Dict[str, Any]] = []
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
    ) -> None:
        """
        Convenience wrapper that runs the full pipeline: fit, validate, analysis, and predict.

        :param data_dev: Development dataset used to fit the pipeline.
        :type data_dev: pd.DataFrame
        :param data_pred: Optional dataset to run final predictions on after fitting.
        :type data_pred: Optional[pd.DataFrame]
        :param data_test: Optional dataset used for external validation.
        :type data_test: Optional[pd.DataFrame]
        :param alpha: Optional significance level(s) for conformal predictions.
        :type alpha: Optional[Union[float, Iterable[float]]]
        :returns: None
        """
        start = time.perf_counter()
        self.logger.info(
            f"----------STARTING PROQSAR PIPELINE AT {datetime.datetime.now()}----------"
        )

        # Fit, validate, analyze
        self.fit(data_dev)
        self.validate(external_test_data=data_test)
        self.analysis()

        # Save predictions for test set
        self._predict_wo_prep(self.test_prep, alpha, save_name="test_pred")

        # Predict for additional dataset if provided
        if data_pred is not None:
            self.predict(data_pred, alpha)

        # Save shape summary
        self.get_shape_summary_df()

        self.logger.info(
            f"----------PROQSAR PIPELINE COMPLETED AT {datetime.datetime.now()}----------"
        )
        end = time.perf_counter()
        self.logger.info(f"Elapsed time: {datetime.timedelta(seconds=(end - start))}")
