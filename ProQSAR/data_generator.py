import pandas as pd
from sklearn.base import BaseEstimator
from ProQSAR.Config.config import Config


class DataGenerator(BaseEstimator):
    def __init__(
        self,
        activity_col: str,
        id_col: str,
        smiles_col: str,
        mol_col: str = "mol",
        n_jobs: int = -1,
        config=None,
    ):

        self.activity_col = activity_col
        self.id_col = id_col
        self.smiles_col = smiles_col
        self.mol_col = mol_col
        self.n_jobs = n_jobs
        self.config = config or Config()

        self.standardizer = self.config.standardizer.set_params(
            smiles_col=smiles_col, n_jobs=n_jobs
        )
        self.featurizer = self.config.featurizer.set_params(
            mol_col=mol_col if self.standardizer.deactivate else "standardized_mol",
            activity_col=activity_col,
            id_col=id_col,
            n_jobs=n_jobs,
        )

    def generate(self, data):
        standardized_data = pd.DataFrame(
            self.standardizer.standardize_dict_smiles(data)
        )
        data_features = self.featurizer.generate_features(standardized_data)

        if not self.standardizer.deactivate:
            for df in data_features.values():
                df["standardized_" + self.smiles_col] = standardized_data[
                    "standardized_" + self.smiles_col
                ]

        if len(data_features.keys()) == 1:
            return list(data_features.values())[0]
        else:
            return data_features

    def get_params(self, deep=True) -> dict:
        """Return all hyperparameters as a dictionary."""
        out = {}
        for key in self.__dict__:
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                for sub_key, sub_value in deep_items:
                    out[f"{key}__{sub_key}"] = sub_value
            out[key] = value

        return out
