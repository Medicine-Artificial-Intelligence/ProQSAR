from typing import Any, Dict, Iterable, Optional, Tuple, Union
from proqsar.Data.Standardizer.smiles_standardizer import SMILESStandardizer
from proqsar.Data.Featurizer.feature_generator import FeatureGenerator
from proqsar.Preprocessor.Clean.duplicate_handler import DuplicateHandler
from proqsar.Preprocessor.Clean.missing_handler import MissingHandler
from proqsar.Preprocessor.Clean.low_variance_handler import LowVarianceHandler
from proqsar.Preprocessor.Outlier.univariate_outliers import UnivariateOutliersHandler
from proqsar.Preprocessor.Outlier.kbin_handler import KBinHandler
from proqsar.Preprocessor.Outlier.multivariate_outliers import (
    MultivariateOutliersHandler,
)
from proqsar.Preprocessor.Clean.rescaler import Rescaler
from proqsar.Data.Splitter.data_splitter import Splitter
from proqsar.Model.FeatureSelector.feature_selector import FeatureSelector
from proqsar.Model.ModelDeveloper.model_developer import ModelDeveloper
from proqsar.Model.Optimizer.optimizer import Optimizer
from proqsar.Evaluation.applicability_domain import ApplicabilityDomain
from proqsar.Evaluation.conformal_predictor import ConformalPredictor

ParamLike = Optional[Union[Dict[str, Any], Iterable[Tuple[str, Any]], Any]]


class Config:
    """
    Configuration factory for constructing ProQSAR pipeline components.

    Each attribute on this object will be an instance of the corresponding
    pipeline class (e.g., `SMILESStandardizer`, `FeatureGenerator`, etc.).
    The constructor accepts either:
      - None (use the class default),
      - a dict of parameters to pass to `.set_params(**params)`,
      - a list/tuple of key-value pairs convertible to dict,
      - or an already-instantiated object of the target class.

    This pattern keeps pipeline assembly concise while allowing full
    customization and dependency-injection (passing custom instances).

    Attributes
    ----------
    standardizer, featurizer, splitter, duplicate, missing, lowvar, univ_outlier,
    kbin, multiv_outlier, rescaler, feature_selector, model_dev, optimizer, ad,
    conf_pred : Any
        Instances of the corresponding ProQSAR component classes.

    Example
    -------
    cfg = Config(
        featurizer={"feature_types": ["ECFP6", "rdkdes"]},
        optimizer={"n_trials": 50},
    )

    :param standardizer: Optional parameterization or instance for SMILES standardizer.
    :type standardizer: dict | iterable | object | None
    :param featurizer: Optional parameterization or instance for feature generator.
    :type featurizer: dict | iterable | object | None
    :param splitter: Optional parameterization or instance for data splitter.
    :type splitter: dict | iterable | object | None
    :param duplicate: Optional parameterization or instance for duplicate handler.
    :type duplicate: dict | iterable | object | None
    :param missing: Optional parameterization or instance for missing handler.
    :type missing: dict | iterable | object | None
    :param lowvar: Optional parameterization or instance for low variance handler.
    :type lowvar: dict | iterable | object | None
    :param univ_outlier: Optional parameterization or instance for univariate outlier handler.
    :type univ_outlier: dict | iterable | object | None
    :param kbin: Optional parameterization or instance for K-bin handler.
    :type kbin: dict | iterable | object | None
    :param multiv_outlier: Optional parameterization or instance for multivariate outlier handler.
    :type multiv_outlier: dict | iterable | object | None
    :param rescaler: Optional parameterization or instance for rescaler.
    :type rescaler: dict | iterable | object | None
    :param feature_selector: Optional parameterization or instance for feature selector.
    :type feature_selector: dict | iterable | object | None
    :param model_dev: Optional parameterization or instance for model developer.
    :type model_dev: dict | iterable | object | None
    :param optimizer: Optional parameterization or instance for optimizer.
    :type optimizer: dict | iterable | object | None
    :param ad: Optional parameterization or instance for applicability domain.
    :type ad: dict | iterable | object | None
    :param conf_pred: Optional parameterization or instance for conformal predictor.
    :type conf_pred: dict | iterable | object | None
    """

    def __init__(
        self,
        standardizer: ParamLike = None,
        featurizer: ParamLike = None,
        splitter: ParamLike = None,
        duplicate: ParamLike = None,
        missing: ParamLike = None,
        lowvar: ParamLike = None,
        univ_outlier: ParamLike = None,
        kbin: ParamLike = None,
        multiv_outlier: ParamLike = None,
        rescaler: ParamLike = None,
        feature_selector: ParamLike = None,
        model_dev: ParamLike = None,
        optimizer: ParamLike = None,
        ad: ParamLike = None,
        conf_pred: ParamLike = None,
    ):
        """
        Initialize the Config object by creating/assigning pipeline components.

        Parameters
        ----------
        standardizer : dict | iterable | object | None
            Parameter specification or instance for `SMILESStandardizer`. If None,
            a `SMILESStandardizer()` is created.
        featurizer : dict | iterable | object | None
            Parameter specification or instance for `FeatureGenerator`. If None,
            a `FeatureGenerator()` is created.
        splitter : dict | iterable | object | None
            Parameter specification or instance for `Splitter`. If None,
            a `Splitter()` is created.
        duplicate : dict | iterable | object | None
            Parameter specification or instance for `DuplicateHandler`. If None,
            a `DuplicateHandler()` is created.
        missing : dict | iterable | object | None
            Parameter specification or instance for `MissingHandler`. If None,
            a `MissingHandler()` is created.
        lowvar : dict | iterable | object | None
            Parameter specification or instance for `LowVarianceHandler`. If None,
            a `LowVarianceHandler()` is created.
        univ_outlier : dict | iterable | object | None
            Parameter specification or instance for `UnivariateOutliersHandler`. If None,
            a `UnivariateOutliersHandler()` is created.
        kbin : dict | iterable | object | None
            Parameter specification or instance for `KBinHandler`. If None,
            a `KBinHandler()` is created.
        multiv_outlier : dict | iterable | object | None
            Parameter specification or instance for `MultivariateOutliersHandler`. If None,
            a `MultivariateOutliersHandler()` is created.
        rescaler : dict | iterable | object | None
            Parameter specification or instance for `Rescaler`. If None,
            a `Rescaler()` is created.
        feature_selector : dict | iterable | object | None
            Parameter specification or instance for `FeatureSelector`. If None,
            a `FeatureSelector()` is created.
        model_dev : dict | iterable | object | None
            Parameter specification or instance for `ModelDeveloper`. If None,
            a `ModelDeveloper()` is created.
        optimizer : dict | iterable | object | None
            Parameter specification or instance for `Optimizer`. If None,
            an `Optimizer()` is created.
        ad : dict | iterable | object | None
            Parameter specification or instance for `ApplicabilityDomain`. If None,
            an `ApplicabilityDomain()` is created.
        conf_pred : dict | iterable | object | None
            Parameter specification or instance for `ConformalPredictor`. If None,
            a `ConformalPredictor()` is created.
        """
        self.standardizer = self._create_instance(standardizer, SMILESStandardizer)
        self.featurizer = self._create_instance(featurizer, FeatureGenerator)
        self.splitter = self._create_instance(splitter, Splitter)
        self.duplicate = self._create_instance(duplicate, DuplicateHandler)
        self.missing = self._create_instance(missing, MissingHandler)
        self.lowvar = self._create_instance(lowvar, LowVarianceHandler)
        self.univ_outlier = self._create_instance(
            univ_outlier, UnivariateOutliersHandler
        )
        self.kbin = self._create_instance(kbin, KBinHandler)
        self.multiv_outlier = self._create_instance(
            multiv_outlier, MultivariateOutliersHandler
        )
        self.rescaler = self._create_instance(rescaler, Rescaler)
        self.feature_selector = self._create_instance(feature_selector, FeatureSelector)
        self.model_dev = self._create_instance(model_dev, ModelDeveloper)
        self.optimizer = self._create_instance(optimizer, Optimizer)
        self.ad = self._create_instance(ad, ApplicabilityDomain)
        self.conf_pred = self._create_instance(conf_pred, ConformalPredictor)

    def _create_instance(
        self,
        param: Optional[Union[Dict[str, Any], Iterable[Tuple[str, Any]], Any]],
        cls: type,
    ) -> Any:
        """
        Helper to create or return an instance for a specific pipeline component.

        Behavior (preserves the original logic):
          - If `param` is None: instantiate `cls()` and return it.
          - If `param` is a dict: instantiate `cls()` and call `.set_params(**param)`.
          - If `param` is a tuple/list: convert to dict via `dict(param)` and call `.set_params(...)`.
          - Otherwise: return `param` assuming it is already an instance.

        Parameters
        ----------
        param : dict | iterable | object | None
            Parameterization or instance for the target class. Accepted types:
            - None: instantiate the class with no args,
            - dict: instantiate and call `.set_params(**param)`,
            - list/tuple of key-value pairs: converted to dict and passed to `.set_params`,
            - otherwise: assumed to be a pre-instantiated object and returned unchanged.
        cls : type
            The class to instantiate when `param` is None or a parameter mapping.
            This should be the class object (e.g., `SMILESStandardizer`).

        Returns
        -------
        object
            An instance of `cls` (parameterized if dict/list provided) or the original `param`.
        """
        if param is None:
            return cls()
        elif isinstance(param, dict):
            return cls().set_params(**param)
        elif isinstance(param, (tuple, list)):
            return cls().set_params(**dict(param))
        else:
            return param
