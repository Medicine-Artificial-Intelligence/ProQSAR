import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    KBinsDiscretizer,
)
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from Data_preprocess import Data_preprocess  #
import warnings

warnings.filterwarnings(action="ignore")
sns.set("notebook")


class Univariate_Outliers(Data_preprocess):
    """
    - Check quality features
    - Remove univariate outliers: using Interquartile Range (IQR)
    - Find the suitable data transform  method to minimize the amount of outliers to be deleted.
      . Imputation
      . Winsorzation
      . Transformation
    - KBIN Discretizer

    Input:
    ------
    Data_train: pandas.DataFrame
        Data for training model after preprocessing.
    Data_test: pandas.DataFrame
        Data for external validation after preprocessing
    acitivity_col: str
        Name of acitivity columns (pIC50, pChEMBL Value)
    handling_method: string ('Winsorization', 'Imputation', 'Transformation')
        Handling outliers method
    transform_method: string ('Uniform Transformer', 'Gaussian Transformer', 'Power Transformer')
        Handling outliers method

    Returns:
    --------
    Data_train, Data_test: pandas.DataFrame
        Data after handling outliers

    """

    def __init__(
        self,
        data_train,
        data_test,
        activity_col,
        handling_method="Transformation",
        transform_method="Uniform Transformer",
        Kbin_handling="Y",
        variance_threshold="Y",
    ):
        self.activity_col = activity_col
        self.handling_method = handling_method
        self.transform_method = transform_method
        self.Kbin_handling = Kbin_handling
        self.variance_threshold = variance_threshold

        self.data_train_0 = data_train
        self.data_test_0 = data_test

        self.data_train = self.data_train_0.copy()
        self.data_test = self.data_test_0.copy()

        self.scl1 = PowerTransformer()
        self.scl2 = QuantileTransformer(output_distribution="normal")
        self.scl3 = QuantileTransformer(output_distribution="uniform")

    ############# 1. Check number of outliers would be removed ####################################
    def Check_IQR(self):
        self.df_train = self.data_train.copy()
        self.df_test = self.data_test.copy()

        for col_name in (
            self.df_train.drop([self.activity_col], axis=1)
            .select_dtypes("float")
            .columns
        ):
            q1 = self.df_train[col_name].quantile(0.25)
            q3 = self.df_train[col_name].quantile(0.75)
            iqr = q3 - q1x
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            self.df_train = self.df_train[
                (self.df_train[col_name] <= high) & (self.df_train[col_name] >= low)
            ]
            self.df_test = self.df_test[
                (self.df_test[col_name] <= high) & (self.df_test[col_name] >= low)
            ]
        print(
            "Total data remove on Train",
            self.data_train_0.shape[0] - self.df_train.shape[0],
        )
        print(
            "Total data remove on Test",
            self.data_test_0.shape[0] - self.df_test.shape[0],
        )
        self.data_train_clean = self.df_train
        self.data_test_clean = self.df_test

    # Check good or bad features. Good features mean that there would be no outliers
    def Quality_features(self):
        self.good = []
        self.bad = []
        self.df_train = self.data_train.copy()
        self.df_test = self.data_test.copy()
        for col_name in (
            self.df_train.drop([self.activity_col], axis=1)
            .select_dtypes("float64")
            .columns
        ):
            q1 = self.df_train[col_name].quantile(0.25)
            q3 = self.df_train[col_name].quantile(0.75)
            iqr = q3 - q1
            remove = (
                self.data_train.shape[0]
                - (
                    self.df_train[
                        (self.df_train[col_name] <= (q3 + 1.5 * iqr))
                        & (self.df_train[col_name] >= (q1 - 1.5 * iqr))
                    ]
                ).shape[0]
            )
            if remove == 0:
                self.good.append(col_name)
            else:
                self.bad.append(col_name)
        print(f"Number of good features: {len(self.good)}")
        print(f"Number of bad features with data remove > 0: {len(self.bad)}")
        print("*" * 75)

    def Check_univariate_outliers(self):
        self.Check_IQR()
        self.Quality_features()

    ############# 2. Method Selection ###########################################################
    def Method(self):
        if self.handling_method == "Winsorization":
            self.Winsorization()
        elif self.handling_method == "Imputation":
            self.Impute_nan()
        elif self.handling_method == "Transformation":
            self.Transformation()
        else:
            return None

    ############# 3. Winsorization ###########################################################
    def Winsorization(self):
        print("Handling with Winsorization method")
        self.df_train = self.data_train.copy()
        self.df_test = self.data_test.copy()
        for col_name in (
            self.df_train.drop([self.activity_col], axis=1)
            .select_dtypes(include="float64")
            .columns
        ):
            q1 = self.df_train[col_name].quantile(0.25)
            q3 = self.df_train[col_name].quantile(0.75)
            iqr = q3 - q1
            self.df_train.loc[
                (self.df_train[col_name] <= (q1 - 1.5 * iqr)), col_name
            ] = (q1 - 1.5 * iqr)
            self.df_train.loc[
                (self.df_train[col_name] >= (q3 + 1.5 * iqr)), col_name
            ] = (q3 + 1.5 * iqr)
            # for test
            self.df_test.loc[(self.df_test[col_name] <= (q1 - 1.5 * iqr)), col_name] = (
                q1 - 1.5 * iqr
            )
            self.df_test.loc[(self.df_test[col_name] >= (q3 + 1.5 * iqr)), col_name] = (
                q3 + 1.5 * iqr
            )
        self.data_train = self.df_train
        self.data_test = self.df_test
        self.Check_univariate_outliers()

    ############# 4. Imputation - Outlier == NaN ###########################################################

    def Impute_nan(self):
        print("Handling with Imputation method")
        for col_name in (
            self.data_train.drop([self.activity_col], axis=1)
            .select_dtypes(include="float64")
            .columns
        ):
            q1 = self.data_train[col_name].quantile(0.25)
            q3 = self.data_train[col_name].quantile(0.75)
            iqr = q3 - q1
            self.data_train.loc[
                (self.data_train[col_name] < (q1 - 1.5 * iqr)), col_name
            ] = np.nan
            self.data_train.loc[
                (self.data_train[col_name] > (q3 + 1.5 * iqr)), col_name
            ] = np.nan
            # for test
            self.data_test.loc[
                (self.data_test[col_name] < (q1 - 1.5 * iqr)), col_name
            ] = np.nan
            self.data_test.loc[
                (self.data_test[col_name] > (q3 + 1.5 * iqr)), col_name
            ] = np.nan
        self.Missing_value_cleaning()  # Gọi lại hàm ở Class preprocess để chọn phương pháp impute phù hợp
        self.Check_univariate_outliers()
        self.Method()

    ############# 5. Transformation ###########################################################

    def Transformation(self):

        if self.transform_method == "Power Transformer":
            self.scl = self.scl1
            print("Power Transformer technique")
        elif self.transform_method == "Gaussian Transformer":
            self.scl = self.scl2
            print("Gaussian Transformer technique")
        elif self.transform_method == "Uniform Transformer":
            self.scl = self.scl3
            print("Uniform Transformer technique")
        else:
            return None

        # Train
        self.data_train_good = self.data_train.drop(self.bad, axis=1)
        self.data_train_bad = self.data_train[self.bad]
        self.real_bad = self.bad.copy()
        if len(self.real_bad) == 0:
            self.data_train = self.data_train
            self.data_test = self.data_test

        else:
            self.scl.fit(self.data_train_bad)

            # with open(SAVE_PREFIX + 'scl.pkl','wb') as f:
            # pickle.dump(self.scl,f)

            self.bad_new = pd.DataFrame(
                self.scl.transform(self.data_train_bad), columns=self.bad
            )
            self.data_train = pd.concat([self.data_train_good, self.bad_new], axis=1)

            # test

            self.data_test_good = self.data_test.drop(self.bad, axis=1)
            self.data_test_bad = self.data_test[self.bad]

            self.bad_new = pd.DataFrame(
                self.scl.transform(self.data_test_bad), columns=self.bad
            )
            self.data_test = pd.concat([self.data_test_good, self.bad_new], axis=1)

            self.Check_univariate_outliers()

            if self.Kbin_handling == "Y":
                self.KBin()
            else:
                pass

    def KBin(self):
        print("Handling with KBin method")
        # Train
        self.data_train_good = self.data_train.drop(self.bad, axis=1)
        self.data_train_bad = self.data_train[self.bad]
        bad_col_kbin = self.data_train_bad.columns

        print("//////", self.bad, len(self.bad))
        if len(self.bad) != 0:
            while True:
                try:
                    # self.n_bins = int(input("Please input number of bins"))
                    self.n_bins = 3
                    # self.encode = input("Please input type of encode")
                    self.encode = "ordinal"
                    # self.strategy = input("Please input type of strategy")
                    self.strategy = "quantile"
                    kst = KBinsDiscretizer(
                        n_bins=3, encode=self.encode, strategy=self.strategy
                    )
                    break
                except:
                    print("Error")
            kst.fit(self.data_train_bad)
            self.bad_new = pd.DataFrame(kst.transform(self.data_train_bad)).astype(
                "int64"
            )
            self.bad_new.columns = [
                "Kbin" + str(i) for i in range(1, len(self.bad_new.columns) + 1)
            ]
            self.data_train_clean = pd.concat(
                [self.data_train_good, self.bad_new], axis=1
            )
            self.data_train = self.data_train_clean

            # test

            self.data_test_good = self.data_test.drop(self.bad, axis=1)
            self.data_test_bad = self.data_test[self.bad]
            self.bad_new = pd.DataFrame(kst.transform(self.data_test_bad)).astype(
                "int64"
            )
            self.bad_new.columns = [
                "Kbin" + str(i) for i in range(1, len(self.bad_new.columns) + 1)
            ]

            self.data_test_clean = pd.concat(
                [self.data_test_good, self.bad_new], axis=1
            )
            self.data_test = self.data_test_clean

            self.Check_univariate_outliers()
            if self.variance_threshold == "Y":
                self.remove_low_variance(thresh=0.05)

    def fit(self):
        print("Remove by IQR without handling")
        self.Check_univariate_outliers()
        self.Method()
