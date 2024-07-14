import pickle
import os
import numpy as np
import pandas as pd


class predict:
    def __init__(self, materials_path, data, activity_col, ID):
        self.SAVE_PREFIX = materials_path

        self.data = data
        self.data_pre = self.data.copy()
        self.activity_col = activity_col
        self.ID = ID

    def pickleload(self, fname):
        with open(self.SAVE_PREFIX + fname, "rb") as f:
            return pickle.load(f)

    def loadfile(self, fname):
        with open(self.SAVE_PREFIX + fname, "r") as f:
            return eval(f.read())

    def prepare_data_pred(self):
        self.data_pred = self.data.drop(self.data.columns[0], axis=1)
        # self.data_pre = self.data_pre

    def Check_NaN(self, data):
        index = []
        for key, value in enumerate(data):
            if type(value) == float or type(value) == int:
                continue
            else:
                index.append(key)
        data[index] = np.nan

    def dupcol(self):
        DUP = self.loadfile("DUP_COL.txt")
        self.data_pre.drop(DUP, axis=1, inplace=True)
        self.data_pre.reset_index(drop=True, inplace=True)

    def Missing_value_cleaning(self):
        drop_cols = self.loadfile("drop_cols.txt")
        self.data_pre.drop(drop_cols, axis=1, inplace=True)

        # null_data_train = self.data_pre[self.data_pre.isnull().any(axis=1)]

        # # impute
        # imp = self.pickleload("imp.pkl")

        # Data_train=pd.DataFrame(imp.transform(self.data_pre))
        # Data_train.columns=self.data_pre.columns
        # Data_train.index=self.data_pre.index

        self.data_pre = self.data_pre

    def variance_threshold(self):
        FEATURES = self.loadfile("Variance_Cols.txt")
        self.data_pre = self.data_pre.loc[:, FEATURES]

    def Nomial(self):
        DINHTINH = self.loadfile("Nomial_Col.txt")
        self.data_pre[DINHTINH] = self.data_pre[DINHTINH].astype("int64")

    def features_selection(self):
        self.X_pre = self.data_pre.drop(self.activity_col, axis=1)
        self.y_pre = self.data_pre[self.activity_col]
        # Load model
        self.select = self.pickleload("select_transform.pkl")
        self.X_pre = self.select.transform(self.X_pre)
        return self.X_pre

    def model_predict(self):
        self.num_molecules = self.data_pre.shape[0]
        self.model = self.pickleload("model.pkl")
        self.structures_active = sum(self.model.predict(self.X_pre))
        print("The number of active structures:", self.structures_active)
        print(
            "Percentage of active structures:",
            self.structures_active * 100 / len(self.X_pre),
        )
        index = np.array(range(1, self.num_molecules + 1))
        self.proba = self.model.predict_proba(self.X_pre)[:, 1] * 100
        self.y_pre = self.model.predict(self.X_pre)
        self.Report = {
            "Index": self.data[self.ID].values,
            "Probability": self.proba,
            "Predict": self.y_pre,
        }

        self.report = pd.DataFrame(self.Report)
        return self.report

    def predict(self):
        self.prepare_data_pred()
        self.data_pre.apply(self.Check_NaN)
        self.dupcol()
        self.Missing_value_cleaning()
        self.variance_threshold()
        self.Nomial()
        self.features_selection()
        self.model_predict()
        return self.report
