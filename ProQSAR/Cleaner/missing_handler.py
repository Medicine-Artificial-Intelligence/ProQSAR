import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from typing import Tuple

class MissingHandler:
    def __init__(
        self,
        id_col: str,
        activity_col: str,
        missing_thresh: float = 40.0,
        imputation_strategy: str = 'mean',  
        n_neighbors: int = 5
        ):
        self.id_col = id_col
        self.activity_col = activity_col
        self.missing_thresh = missing_thresh
        self.imputation_strategy = imputation_strategy
        self.n_neighbors = n_neighbors

    def calculate_missing_percent(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the percentage of missing values for each column in the DataFrame.

        Parameters:
        data : pd.DataFrame
            The DataFrame to analyze.

        Returns:
        pd.DataFrame
            A DataFrame containing the percentage of missing values for each column.
            The DataFrame has two columns:
            - 'ColumnName': The name of the columns in the original DataFrame.
            - 'MissingPercent': The percentage of missing values in each column.
        """
        missing_percent = (data.isnull().sum() / len(data)) * 100
        return pd.DataFrame({
            'ColumnName': data.columns,
            'MissingPercent': missing_percent
            })
    
    def fit(self, data: pd.DataFrame):
        
        # Exclude id_col and activity_col
        self.columns_to_exclude = [self.id_col, self.activity_col]
        data_to_impute = data.drop(columns=self.columns_to_exclude)

        # Drops columns with a missing value percentage higher than the threshold
        missing_percent_df = self.calculate_missing_percent(data_to_impute)
        self.drop_cols = missing_percent_df[
            missing_percent_df['MissingPercent'] > self.missing_thresh
            ]['ColumnName'].tolist()
        data_to_impute.drop(columns=self.drop_cols, inplace=True)

        # Determine binary columns and non-binary columns
        self.binary_cols = [
            col 
            for col in data_to_impute.columns
            if data_to_impute[col].nunique(dropna=True) <= 2 
            and data_to_impute[col].dropna().isin([0, 1]).all()
            ]
        data_binary = data_to_impute[self.binary_cols]
        data_non_binary = data_to_impute.drop(columns=self.binary_cols)

        # Fit imputation transformer for binary columns
        if self.binary_cols:
            self.binary_imputer = SimpleImputer(strategy='most_frequent')
            self.binary_imputer.fit(data_binary) 

        # Fit imputation transformer for non-binary columns
        if self.imputation_strategy == "mean":
            self.non_binary_imputer = SimpleImputer(strategy="mean")
        elif self.imputation_strategy == "median":
            self.non_binary_imputer = SimpleImputer(strategy="median")
        elif self.imputation_strategy == "mode":
            self.non_binary_imputer = SimpleImputer(strategy="most_frequent")
        elif self.imputation_strategy == "knn":
            self.non_binary_imputer = KNNImputer(n_neighbors=self.n_neighbors)
        elif self.imputation_strategy == "mice":
            estimator = BayesianRidge()  # Default estimator, can be changed
            self.non_binary_imputer = IterativeImputer(estimator=estimator, random_state=42)
            
        self.non_binary_imputer.fit(data_non_binary)

        return self

    def transform(self, data: pd.DataFrame):
        # Exclude id_col and activity_col
        data_to_impute = data.drop(columns=self.columns_to_exclude)

        # Drop columns based on the training data threshold
        data_to_impute.drop(columns=self.drop_cols, inplace=True, errors='ignore')

        # Separate binary columns and non-binary columns
        data_binary = data_to_impute[self.binary_cols]
        data_non_binary = data_to_impute.drop(columns=self.binary_cols, errors='ignore')

        # Apply imputation transformer for binary columns
        if self.binary_cols:
            imputed_data_binary = pd.DataFrame(
                self.binary_imputer.transform(data_binary), 
                columns=data_binary.columns
                )
        else:
            imputed_data_binary = data_binary

        # Apply imputation transformer for non-binary columns
        imputed_data_non_binary = pd.DataFrame(
            self.non_binary_imputer.transform(data_non_binary), 
            columns=data_non_binary.columns
            )

        # Concatenate imputed binary, non-binary columns and excluded columns
        imputed_data = pd.concat([
            data[self.columns_to_exclude], 
            imputed_data_binary,
            imputed_data_non_binary
            ], axis=1)
    
    def fit_transform(self, data: pd.DataFrame):
        self.fit(data)
        return self.transform(data)