import pandas as pd
from typing import Union
from ProQSAR.Standardizer.smiles_standardizer import SMILESStandardizer
from ProQSAR.Featurizer.feature_generator import FeatureGenerator


class DataGenerator:
    def __init__(
            self, 
            activity_col: str, 
            id_col: str, 
            smiles_col: str, 
            **kwargs
        ):
        
        self.activity_col = activity_col
        self.id_col = id_col
        self.smiles_col = smiles_col

        self.smilesstandardizer = SMILESStandardizer()

        self.featuregenerator = FeatureGenerator(
            mol_col="standardized_mol",
            activity_col=activity_col,
            id_col=id_col,
            **kwargs

        )

    def generate(self, data):
        standardized_data = pd.DataFrame(
            self.smilesstandardizer.standardize_dict_smiles(data, key=self.smiles_col)
        )
        features = self.featuregenerator.generate_features(standardized_data)

        for df in features.values():
            df["standardized_" + self.smiles_col] = standardized_data[
                "standardized_" + self.smiles_col
            ]

        return features


    # CO NEN XOA KO???
    def set_params(self, stage, **kwargs):
        component = getattr(self, stage.lower(), None)
        if component and isinstance(component, (SMILESStandardizer, FeatureGenerator)):
            for key, value in kwargs.items():
                if hasattr(component, key):
                    setattr(component, key, value)
                else:
                    raise AttributeError(
                        f"{stage} does not have a parameter named '{key}'"
                    )
        else:
            raise ValueError(f"Invalid stage name: {stage}")


    def get_params(self):
        """
        Get parameters of the DataGenerator class as a dictionary.

        Parameters
        ----------
        deep : bool, optional, default=True
            If True, will include parameters of the underlying components (SMILESStandardizer, FeatureGenerator).

        Returns
        -------
        params : dict
            Dictionary containing parameters and their values.
        """
        params = {
            "datapreprocessor__smiles_col": self.smiles_col,
            "datagenerator__activity_col": self.activity_col,
            "datagenerator__id_col": self.id_col,
        }

        # Include parameters from SMILESStandardizer
        smilesstandardizer_params = self.smilesstandardizer.__dict__
        params.update(
            {
                "smilesstandardizer__" + key: value
                for key, value in smilesstandardizer_params.items()
            }
        )

        # Include parameters from FeatureGenerator
        featuregenerator_params = self.featuregenerator.__dict__
        params.update(
            {
                "featuregenerator__" + key: value
                for key, value in featuregenerator_params.items()
            }
        )

        return params
