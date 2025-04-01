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


class Config:

    def __init__(
        self,
        standardizer=None,
        featurizer=None,
        splitter=None,
        duplicate=None,
        missing=None,
        lowvar=None,
        univ_outlier=None,
        kbin=None,
        multiv_outlier=None,
        rescaler=None,
        feature_selector=None,
        model_dev=None,
        optimizer=None,
        ad=None,
        conf_pred=None,
    ):
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

    def _create_instance(self, param, cls):
        """
        Helper function that creates an instance for component class `cls`
        using its setting() method.
        If 'param' is:
        - None: returns cls.setting() (default instance)
        - a dict: returns cls.setting(**param)
        - a tuple or list: converts it to dict and returns cls.setting(**dict(param))
        - Otherwise, assumes it's already an instance.
        """

        if param is None:
            return cls()
        elif isinstance(param, dict):
            return cls().set_params(**param)
        elif isinstance(param, (tuple, list)):
            return cls().set_params(**dict(param))
        else:
            return param
