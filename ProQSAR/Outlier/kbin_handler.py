import os
import pickle
import pandas as pd
from typing import Optional
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.exceptions import NotFittedError
from ProQSAR.Outlier.univariate_outliers import UnivariateOutliersHandler


class KBinHandler:
    """
    A class to handle the discretization of features in a DataFrame using KBinsDiscretizer
    after filtering out univariate outliers.

    Attributes:
        id_col (Optional[str]): The column name of the ID feature.
        activity_col (Optional[str]): The column name of the activity feature.
        n_bins (int): Number of bins to produce.
        encode (str): Method used to encode the transformed result.
        strategy (str): Strategy used to define the widths of the bins.
        save_dir (Optional[str]): Directory to save the fitted models.
    """

    def __init__(
        self,
        id_col: Optional[str] = None,
        activity_col: Optional[str] = None,
        n_bins: int = 3,
        encode: str = "ordinal",
        strategy: str = "quantile",
        save_dir: Optional[str] = None,
    ) -> None:

        self.id_col = id_col
        self.activity_col = activity_col
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.save_dir = save_dir
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    def fit(self, data: pd.DataFrame):
        """
        Fits the KBinsDiscretizer to the 'bad' features identified by the UnivariateOutliersHandler.

        Parameters:
            data (pd.DataFrame): The input data on which to fit the KBinsDiscretizer.
        """

        _, self.bad = UnivariateOutliersHandler._feature_quality(
            data, id_col=self.id_col, activity_col=self.activity_col
        )

        self.kbin = None
        if self.bad:
            self.kbin = KBinsDiscretizer(
                n_bins=self.n_bins, encode=self.encode, strategy=self.strategy
            ).fit(data[self.bad])

        if self.save_dir:
            with open(f"{self.save_dir}/bad_features.pkl", "wb") as file:
                pickle.dump(self.bad, file)
            with open(f"{self.save_dir}/kbin.pkl", "wb") as file:
                pickle.dump(self.kbin, file)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the 'bad' features of the input data using the fitted KBinsDiscretizer.

        Parameters:
            data (pd.DataFrame): The input data to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame with 'bad' features binned.
        """
        transformed_data = data.copy()

        if not transformed_data[self.bad].empty:
            new_bad_data = pd.DataFrame(self.kbin.transform(transformed_data[self.bad]))
            new_bad_data.columns = [
                "Kbin" + str(i) for i in range(1, len(new_bad_data.columns) + 1)
            ]
            transformed_data.drop(columns=self.bad, inplace=True)
            transformed_data = pd.concat([transformed_data, new_bad_data], axis=1)

        return transformed_data

    @staticmethod
    def static_transform(data: pd.DataFrame, save_dir: str) -> pd.DataFrame:
        if os.path.exists(f"{save_dir}/bad_features.pkl"):
            with open(f"{save_dir}/bad_features.pkl", "rb") as file:
                bad = pickle.load(file)
        else:
            raise NotFittedError(
                "The KBinHandler instance is not fitted yet. Call 'fit' before using this method."
            )

        transformed_data = data.copy()
        if os.path.exists(f"{save_dir}/kbin.pkl") and not transformed_data[bad].empty:
            with open(f"{save_dir}/kbin.pkl", "rb") as file:
                kbin = pickle.load(file)
            new_bad_data = pd.DataFrame(kbin.transform(transformed_data[bad]))
            new_bad_data.columns = [
                "Kbin" + str(i) for i in range(1, len(new_bad_data.columns) + 1)
            ]
            transformed_data.drop(columns=bad, inplace=True)
            transformed_data = pd.concat([transformed_data, new_bad_data], axis=1)

        return transformed_data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the KBinsDiscretizer and then transforms the 'bad' features in one step.

        Parameters:
            data (pd.DataFrame): The input data to fit and transform.

        Returns:
            pd.DataFrame: Transformed DataFrame with 'bad' features binned.
        """
        self.fit(data)
        return self.transform(data)
