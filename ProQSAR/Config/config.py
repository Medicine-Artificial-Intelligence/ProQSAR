from typing import Any, Dict, Iterable, Optional, Tuple, Union
from ProQSAR.Standardizer.smiles_standardizer import SMILESStandardizer
from ProQSAR.Featurizer.feature_generator import FeatureGenerator
from ProQSAR.Preprocessor.duplicate_handler import DuplicateHandler
from ProQSAR.Preprocessor.missing_handler import MissingHandler
from ProQSAR.Preprocessor.low_variance_handler import LowVarianceHandler
from ProQSAR.Outlier.univariate_outliers import UnivariateOutliersHandler
from ProQSAR.Outlier.kbin_handler import KBinHandler
from ProQSAR.Outlier.multivariate_outliers import MultivariateOutliersHandler
from ProQSAR.Rescaler.rescaler import Rescaler
from ProQSAR.Splitter.data_splitter import Splitter
from ProQSAR.FeatureSelector.feature_selector import FeatureSelector
from ProQSAR.ModelDeveloper.model_developer import ModelDeveloper
from ProQSAR.Optimizer.optimizer import Optimizer
from ProQSAR.ApplicabilityDomain.applicability_domain import ApplicabilityDomain
from ProQSAR.Uncertainty.conformal_predictor import ConformalPredictor

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
        standardizer, featurizer, splitter, duplicate, missing, lowvar,
        univ_outlier, kbin, multiv_outlier, rescaler, feature_selector,
        model_dev, optimizer, ad, conf_pred : optional
            Per-component parameter specification or pre-constructed instance.
            See class docstring for accepted input types and behavior.
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
        param : None | dict | iterable-of-pairs | object
            Parameterization or instance for the target class.
        cls : type
            The class to instantiate when `param` is None or a parameter mapping.

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
