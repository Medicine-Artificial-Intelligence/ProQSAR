import os
import logging
import pandas as pd
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
from ProQSAR.ModelDeveloper.model_validation import ModelValidation
from typing import Union, Optional

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
            ):
        self.standardizer = self._create_instance(standardizer, SMILESStandardizer)
        self.featurizer = self._create_instance(featurizer, FeatureGenerator)
        self.splitter = self._create_instance(splitter, Splitter)
        self.duplicate = self._create_instance(duplicate, DuplicateHandler)
        self.missing = self._create_instance(missing, MissingHandler)
        self.lowvar = self._create_instance(lowvar, LowVarianceHandler)
        self.univ_outlier = self._create_instance(univ_outlier, UnivariateOutliersHandler)
        self.kbin = self._create_instance(kbin, KBinHandler)
        self.multiv_outlier = self._create_instance(multiv_outlier, MultivariateOutliersHandler)
        self.rescaler = self._create_instance(rescaler, Rescaler)
        self.feature_selector = self._create_instance(feature_selector, FeatureSelector)
        self.model_dev = self._create_instance(model_dev, ModelDeveloper)


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
            return cls().setting(**param)
        elif isinstance(param, (tuple, list)):
            return cls().setting(**dict(param))
        else:
            return param
        


    """
    def get_config(self):
        return {
            "SMILESStandardizer": self.standardizer.__dict__,
            "FeatureGenerator": self.featurizer.__dict__,
            "Splitter": self.splitter.__dict__,
            "DuplicateHandler": self.duplicate.get_params(),
            "MissingHandler": self.missing.get_params(),
            "LowVarianceHandler": self.lowvar.get_params(),
            "UnivariateOutliersHandler": self.univ_outlier.get_params(),
            "KBinHandler": self.kbin_instance.get_params(),
            "MultivariateOutliersHandler": self.multi_instance.get_params(),
            "Rescaler": self.rescaler_instance.get_params(),
            "FeatureSelector": self.fs_instance.get_params(),
            "ModelDeveloper": self.model_instance.get_params()
        }
    """