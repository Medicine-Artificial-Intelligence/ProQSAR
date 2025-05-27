import pandas as pd
from typing import Optional
from sklearn.base import BaseEstimator
from ProQSAR.Config.config import Config


class DataGenerator(BaseEstimator):
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

        self.activity_col = activity_col
        self.id_col = id_col
        self.smiles_col = smiles_col
        self.mol_col = mol_col
        self.n_jobs = n_jobs
        self.save_dir = save_dir
        self.data_name = data_name
        self.config = config or Config()

        self.standardizer = self.config.standardizer.set_params(
            smiles_col=self.smiles_col, n_jobs=self.n_jobs
        )
        self.featurizer = self.config.featurizer.set_params(
            mol_col=self.mol_col if self.standardizer.deactivate else "standardized_mol",
            activity_col=self.activity_col,
            id_col=self.id_col,
            smiles_col=self.smiles_col if self.standardizer.deactivate else ("standardized_" + self.smiles_col),
            n_jobs=self.n_jobs,
            save_dir=self.save_dir,
        )

    def generate(self, data):
        self.standardized_data = pd.DataFrame(
            self.standardizer.standardize_dict_smiles(data)
        )
        data_features = self.featurizer.set_params(
            data_name=self.data_name
        ).generate_features(self.standardized_data)

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
