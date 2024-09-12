import logging
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Literal
from ProQSAR.Partitioner.scaffold import scaffold_split
from ProQSAR.Partitioner.stratified_scaffold import stratified_scaffold
from sklearn.model_selection import train_test_split


class Partitioner:
    @staticmethod
    def Check_NaN(data: pd.DataFrame) -> pd.DataFrame:
        """
        Check for NaN values in a given data.

        Parameters
        ----------
        data : iterable
            The data to check for NaN values.

        Returns
        -------
        None
            If NaN values are found, they are replaced with np.nan in the
            original data.
        """
        index = []
        for key, value in enumerate(data):
            try:
                float(value)
                if np.isnan(value):
                    index.append(key)
                else:
                    continue
            except Exception as e:
                logging.error(f"Missing value is not converted to float: {e}")
                index.append(key)
        if len(index) != 0:
            data[index] = np.nan
        return data

    @staticmethod
    def target_bin(
        data: pd.DataFrame,
        activity_col: str,
        thresh: float,
        input_target_style: str = "pIC50",
    ) -> pd.DataFrame:
        """
        Binary classification of a given column in a DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the column to be modified.
        activity_col : str
            Name of the column to be modified.
        thresh : float
            Threshold value for the classification.
        input_target_style : str
            Input style of the target data. Default is 'pIC50'.
            If 'pIC50', the target is assumed to be in the pIC50 format
            and the classification will be done accordingly.
            If not 'pIC50', the target is assumed to be in the IC50 format
            and the classification will be done accordingly.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the modified column.
        """
        if input_target_style != "pIC50":
            thresh = thresh
            t1 = data[activity_col] < thresh
            data.loc[t1, activity_col] = 1
            t2 = data[activity_col] >= thresh
            data.loc[t2, activity_col] = 0
            data[activity_col] = data[activity_col].astype("int64")
        else:
            thresh = thresh
            t1 = data[activity_col] < thresh
            data.loc[t1, activity_col] = 0
            t2 = data[activity_col] >= thresh
            data.loc[t2, activity_col] = 1
            data[activity_col] = data[activity_col].astype("int64")
        return data

    def data_partitioner(
        self,
        data: pd.DataFrame,
        activity_col: str = "pIC50",
        smiles_col: str = "smiles",
        task_type: str = "C",
        target_thresh: Optional[float] = 6,
        option_partitioner: Literal[
            "random", "scaffold", "stratified_scaffold"
        ] = "random",
        random_seed: int = 42,
        test_size: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Partitions the dataset into training and test sets based on various partitioning strategies.

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset to be partitioned.
        activity_col : str
            Column name in the dataset that contains activity or target values.
        smiles_col : str, optional, default='smiles'
            Column name containing SMILES representations of molecules.
        task_type : str, optional, default='C'
            Task type, either 'C' for classification or 'R' for regression.
        target_thresh : float, optional
            Threshold for binarizing the target values if the task is classification.
        option_partitioner : str, optional, default='random'
            Partitioning method to use. Options are:
            - 'random': Random partitioning using train_test_split.
            - 'scaffold': Scaffold-based partitioning.
            - 'stratified_scaffold': Stratified scaffold partitioning (only for classification tasks).
        random_seed : int, optional, default=42
            Seed for random number generators to ensure reproducibility.
        test_size : float, optional, default=0.2
            Proportion of the dataset to be used for testing.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing the training and testing datasets after partitioning.

        Raises:
        -------
        ValueError:
            If the classification task type is selected, but the activity column does not contain exactly 2 unique
            classes.
        """
        df = data.drop([smiles_col], axis=1)
        df.apply(Partitioner.Check_NaN)
        data = pd.concat([data[smiles_col], df], axis=1)
        if task_type.title() == "C":
            if len(data[activity_col].unique()) != 2:
                data = Partitioner.target_bin(
                    data, activity_col, target_thresh, input_target_style=activity_col
                )
            stratify = data[activity_col]

        elif task_type.title() == "R":
            stratify = None

        if option_partitioner == "random":
            data_train, data_test = train_test_split(
                data, test_size=0.2, random_state=random_seed, stratify=stratify
            )

        elif option_partitioner == "scaffold":
            data_train, data_test = scaffold_split(
                data, smiles_col, test_size=0.2, random_state=random_seed
            )

        elif option_partitioner == "stratified_scaffold" and task_type.title() == "C":
            number_split = int(1 / test_size)
            data_train, data_test = stratified_scaffold(
                data,
                smiles_col,
                activity_col,
                n_splits=number_split,
                scaff_based="median",
                random_state=random_seed,
                shuffle=True,
            )

        X_train = data_train.drop([activity_col, smiles_col], axis=1)
        y_train = data_train[[activity_col]]
        X_test = data_test.drop([activity_col, smiles_col], axis=1)
        y_test = data_test[[activity_col]]

        # index:
        idx = X_train.T.index

        # Train:
        df_X_train = pd.DataFrame(X_train, columns=idx)
        df_y_train = pd.DataFrame(y_train, columns=[activity_col])
        data_train = pd.concat(
            [df_y_train, data_train[[smiles_col]], df_X_train], axis=1
        )

        # test
        df_X_test = pd.DataFrame(X_test, columns=idx)
        df_y_test = pd.DataFrame(y_test, columns=[activity_col])
        data_test = pd.concat([df_y_test, data_test[[smiles_col]], df_X_test], axis=1)

        return data_train, data_test
