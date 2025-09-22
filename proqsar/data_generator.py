import pandas as pd
from typing import Optional, Any, Union, Dict
from sklearn.base import BaseEstimator
from proqsar.Config.config import Config


class DataGenerator(BaseEstimator):
    """
    High-level helper that standardizes SMILES strings and computes molecular
    features using the ProQSAR pipeline configuration.

    The DataGenerator composes two components from a ProQSAR `Config`:
      * ``standardizer``: used to standardize SMILES and produce RDKit Mol objects
        (or other standardized representations).
      * ``featurizer``: used to compute fingerprints/descriptors from (standardized)
        molecules.

    Typical usage::

        dg = DataGenerator(activity_col="activity", id_col="id", smiles_col="SMILES")
        feature_df = dg.generate(raw_dataframe)

    Sphinx-style parameters and attributes are documented on the constructor and
    class-level sections respectively.
    """

    def __init__(
        self,
        activity_col: str,
        id_col: str,
        smiles_col: str,
        mol_col: str = "mol",
        n_jobs: int = 1,
        save_dir: Optional[str] = "Project/DataGenerator",
        data_name: Optional[str] = None,
        config=None,
    ):
        """
        Initialize a DataGenerator.

        The constructor prepares `standardizer` and `featurizer` instances from
        the provided ``config``. If ``config`` is ``None`` a default ``Config()``
        is used. The featurizer's expected molecule and smiles column names are
        set depending on whether the standardizer is active.

        :param activity_col: Column name containing the target/activity values in the input data.
        :type activity_col: str
        :param id_col: Column name containing the identifier for each sample.
        :type id_col: str
        :param smiles_col: Column name containing raw SMILES strings in the input data.
        :type smiles_col: str
        :param mol_col: Column name to be used for RDKit molecule objects in downstream steps.
                        Default is ``"mol"``. If the standardizer is active the featurizer will
                        expect ``"standardized_mol"`` instead.
        :type mol_col: str
        :param n_jobs: Number of parallel jobs to use where supported (forwarded to the
                       standardizer/featurizer). Default is ``1``.
        :type n_jobs: int
        :param save_dir: Directory used for any intermediate or output saving by the featurizer.
                         Default: ``"Project/DataGenerator"``.
        :type save_dir: Optional[str]
        :param data_name: Optional base name used by the featurizer when saving feature files.
        :type data_name: Optional[str]
        :param config: ProQSAR ``Config`` object. If ``None``, a default ``Config()`` is created.
        :type config: Optional[Config]

        :ivar standardizer: The configured standardizer instance (from Config). Its parameters are
                            updated to use the provided ``smiles_col`` and ``n_jobs``.
        :vartype standardizer: object
        :ivar featurizer: The configured featurizer instance (from Config). Its parameters are
                          updated to use the appropriate molecule column name (standardized vs raw),
                          activity/id column names, smiles column name, number of jobs and save_dir.
        :vartype featurizer: object
        """
        self.activity_col = activity_col
        self.id_col = id_col
        self.smiles_col = smiles_col
        self.mol_col = mol_col
        self.n_jobs = n_jobs
        self.save_dir = save_dir
        self.data_name = data_name
        self.config = config or Config()

        # Configure standardizer and featurizer using the config object.
        # If standardizer is active, featurizer should use standardized columns.
        self.standardizer = self.config.standardizer.set_params(
            smiles_col=self.smiles_col, n_jobs=self.n_jobs
        )
        self.featurizer = self.config.featurizer.set_params(
            mol_col=(
                self.mol_col if self.standardizer.deactivate else "standardized_mol"
            ),
            activity_col=self.activity_col,
            id_col=self.id_col,
            smiles_col=(
                self.smiles_col
                if self.standardizer.deactivate
                else ("standardized_" + self.smiles_col)
            ),
            n_jobs=self.n_jobs,
            save_dir=self.save_dir,
        )

    def generate(
        self, data: Union[pd.DataFrame, list, dict]
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Standardize SMILES and generate molecular features.

        Steps performed:
          1. Calls the standardizer's ``standardize_dict_smiles`` on the provided
             ``data`` to obtain standardized records (returned as a list/dict
             compatible with a DataFrame).
          2. Converts the standardizer output into a DataFrame and passes it to
             the featurizer's ``generate_features`` method.
          3. If the featurizer returns a single feature set (single key), that
             DataFrame is returned directly; otherwise a dict mapping feature
             type -> DataFrame is returned.

        :param data: The input dataset containing SMILES and (optionally) activity/id
                     columns. The exact accepted format depends on the standardizer used;
                     most commonly a pandas DataFrame is provided.
        :type data: pd.DataFrame | list | dict
        :returns: A single feature DataFrame if only one feature type was generated,
                  otherwise a dictionary mapping feature-type names to DataFrames.
        :rtype: pd.DataFrame or dict[str, pd.DataFrame]
        :raises: Any exceptions raised by the underlying standardizer/featurizer are propagated.
        """
        standardized_data = pd.DataFrame(
            self.standardizer.standardize_dict_smiles(data)
        )

        data_features = self.featurizer.set_params(
            data_name=self.data_name
        ).generate_features(standardized_data)

        if len(data_features.keys()) == 1:
            return list(data_features.values())[0]
        else:
            return data_features

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Return estimator parameters in a dictionary, matching sklearn's convention.

        When ``deep=True``, if a parameter value contains a ``get_params`` method it
        will be expanded with a ``component__param`` key naming convention (sklearn-style).

        :param deep: If True, return parameters of nested objects as well (default True).
        :type deep: bool
        :returns: Dictionary of parameter names mapped to their values.
        :rtype: dict
        """
        out: Dict[str, Any] = {}
        for key in self.__dict__:
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                for sub_key, sub_value in deep_items:
                    out[f"{key}__{sub_key}"] = sub_value
            out[key] = value

        return out
