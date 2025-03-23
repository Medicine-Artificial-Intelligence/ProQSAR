import pandas as pd
from ProQSAR.config import Config


class DataGenerator:
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
        self.config = config or Config()

        self.standardizer = self.config.standardizer.setting(
            smiles_col=smiles_col, n_jobs=n_jobs
        )
        self.featurizer = self.config.featurizer.setting(
            mol_col=mol_col if self.standardizer.deactivate else "standardized_mol",
            activity_col=activity_col,
            id_col=id_col,
            n_jobs=n_jobs,
        )

    def generate(self, data):
        standardized_data = pd.DataFrame(
            self.standardizer.standardize_dict_smiles(data)
        )
        features = self.featurizer.generate_features(standardized_data)

        for df in features.values():
            df["standardized_" + self.smiles_col] = standardized_data[
                "standardized_" + self.smiles_col
            ]

        return features

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

    def __repr__(self):
        """Return a string representation of the estimator."""
        class_name = self.__class__.__name__
        params = self.get_params(deep=False)
        param_str = ", ".join(f"{key}={repr(value)}" for key, value in params.items())
        return f"{class_name}({param_str})"
