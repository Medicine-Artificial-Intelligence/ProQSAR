import os
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.exceptions import NotFittedError
from typing import Optional


class LowVarianceHandler(BaseEstimator, TransformerMixin):
    """
    A class to handle low variance feature selection from a dataset.

    Attributes:
    id_col (Optional[str]): The column name representing the ID column.
    activity_col (Optional[str]): The column name representing the activity column.
    var_thresh (float): The variance threshold for feature selection.
    save_method (bool): Whether to save the fitted model to disk.
    visualize (bool): Whether to visualize variance analysis.
    save_image (bool): Whether to save the variance analysis image.
    image_name (str): The name of the image file for variance analysis.
    save_dir (str): The directory to save the model and transformed data.
    save_trans_data (bool): Whether to save the transformed data.
    trans_data_name (str): The name of the transformed data file.
    selected_columns (Optional[list]): The selected columns after fitting.
    """

    def __init__(
        self,
        activity_col: Optional[str] = None,
        id_col: Optional[str] = None,
        var_thresh: float = 0.05,
        save_method: bool = False,
        visualize: bool = False,
        save_image: bool = False,
        image_name: str = "variance_analysis.png",
        save_dir: str = "Project/LowVarianceHandler",
        save_trans_data: bool = False,
        trans_data_name: str = "trans_data",
        deactivate: bool = False,
    ):
        """
        Initialize the LowVarianceHandler.

        Parameters:
        - activity_col (str): The column name for the activity column.
        - id_col (str): The column name for the ID column.
        - var_thresh (float): The variance threshold. Default is 0.05.
        - save_method (bool): Whether to save the fitted missing data handler.
        - visualize (bool): Whether to visualize the variance threshold analysis.
        Default is True.
        - save_image (bool): Whether to save the plot as an image. Default is True.
        - image_name (str): The path to save the image if save_image is True.
        Default is 'variance_analysis.png'.
        - save_dir (str): The directory to save the image and selected columns file.
        Default is "Project/VarianceHandler".
        - save_trans_data (bool): Whether to save the transformed data.
        - trans_data_name (str): File name for saved transformed data.
        """
        self.activity_col = activity_col
        self.id_col = id_col
        self.var_thresh = var_thresh
        self.save_method = save_method
        self.visualize = visualize
        self.save_image = save_image
        self.image_name = image_name
        self.save_dir = save_dir
        self.save_trans_data = save_trans_data
        self.trans_data_name = trans_data_name
        self.deactivate = deactivate
        self.selected_columns = None

    @staticmethod
    def variance_threshold_analysis(
        data: pd.DataFrame,
        activity_col: Optional[str] = None,
        id_col: Optional[str] = None,
        set_style: str = "whitegrid",
        save_image: bool = False,
        image_name: str = "variance_analysis.png",
        save_dir: str = "Project/VarianceHandler",
    ) -> None:
        """
        Perform variance threshold analysis on the non-binary features of a dataset
        and plot the results.

        Parameters:
        - data (pd.DataFrame): The input data.
        - id_col (str): The column name for the ID column.
        - activity_col (str): The column name for the activity column.
        - set_style (str): The style of the seaborn plot. Default is "whitegrid".
        - save_image (bool): Whether to save the plot as an image. Default is False.
        - image_name (str): The path to save the image if save_image is True.
            Default is 'variance_analysis.png'.
        - save_dir (str): The directory to save the image if save_image is True.
            Default is "Project/VarianceHandler".
        """
        try:
            columns_to_exclude = [activity_col, id_col]
            temp_data = data.drop(columns=columns_to_exclude)
            binary_cols = [
                col
                for col in temp_data.columns
                if temp_data[col].dropna().isin([0, 1]).all()
            ]
            non_binary_cols = [
                col for col in temp_data.columns if col not in binary_cols
            ]

            if non_binary_cols:
                X_non_binary = temp_data[non_binary_cols]
                thresholds = np.arange(0.0, 1, 0.05)
                results = []

                for t in thresholds:
                    transform = VarianceThreshold(threshold=t)
                    try:
                        X_selected = transform.fit_transform(X_non_binary)
                        n_features = X_selected.shape[1] + len(binary_cols)
                    except ValueError:
                        n_features = len(binary_cols)
                    results.append(n_features)

                sns.set_theme(style=set_style)
                plt.figure(figsize=(14, 8))
                plt.plot(thresholds, results, marker=".")  # Added marker
                plt.title("Variance Analysis", fontsize=24, weight="semibold")
                plt.xlabel("Variance Threshold", fontsize=16)
                plt.ylabel("Number of Features", fontsize=16)
                plt.grid(True)  # Added grid

                # Add annotations for key points
                for i, txt in enumerate(results):
                    plt.annotate(
                        txt,
                        (thresholds[i], results[i]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=10,
                    )
                plt.show()

                if save_image:
                    if save_dir and not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(os.path.join(save_dir, image_name))
                    logging.info(
                        f"Variance threshold analysis figure save at: {save_dir}/{image_name}"
                    )
            else:
                logging.warning("No non-binary columns to apply variance threshold.")

        except Exception as e:
            logging.error(f"Error in variance threshold analysis: {e}")
            raise

    @staticmethod
    def select_features_by_variance(
        data: pd.DataFrame,
        activity_col: Optional[str] = None,
        id_col: Optional[str] = None,
        var_thresh: float = 0.05,
    ) -> list:
        """
        Select features from data based on variance threshold.

        Parameters:
        - data (pd.DataFrame): The input data.
        - activity_col (str): The column name for the activity column.
        - id_col (str): The column name for the ID column.
        - var_thresh (float): The variance threshold.
        Features with variance below this threshold will be removed.
        - deactivate (bool): Flag to deactivate the process.

        Returns:
        - list: The list of selected columns.
        """
        try:
            columns_to_exclude = [id_col, activity_col]
            temp_data = data.drop(columns=columns_to_exclude)
            binary_cols = [
                col
                for col in temp_data.columns
                if temp_data[col].dropna().isin([0, 1]).all()
            ]
            non_binary_cols = [
                col for col in temp_data.columns if col not in binary_cols
            ]

            selected_features = []
            if non_binary_cols:
                selector = VarianceThreshold(var_thresh)
                try:
                    selector.fit(data[non_binary_cols])
                    features = selector.get_support(indices=True)
                    selected_features = data[non_binary_cols].columns[features].tolist()
                except ValueError:
                    pass
            else:
                logging.warning("No non-binary columns to apply variance threshold.")

            return columns_to_exclude + binary_cols + selected_features

        except Exception as e:
            logging.error(f"Error in feature selection by variance: {e}")
            return []

    def fit(self, data: pd.DataFrame, y=None):
        """
        Fits the variance-related preprocessing steps on the data.

        Parameters:
        - data (pd.DataFrame): The input data.
        """
        if self.deactivate:
            logging.info("LowVarianceHandler is deactivated. Skipping fit.")
            return self

        try:
            if self.visualize:
                LowVarianceHandler.variance_threshold_analysis(
                    data=data,
                    id_col=self.id_col,
                    activity_col=self.activity_col,
                    save_image=self.save_image,
                    image_name=self.image_name,
                    save_dir=self.save_dir,
                )

            self.selected_columns = LowVarianceHandler.select_features_by_variance(
                data=data,
                activity_col=self.activity_col,
                id_col=self.id_col,
                var_thresh=self.var_thresh,
            )

            if self.save_method:
                if self.save_dir and not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir, exist_ok=True)
                with open(f"{self.save_dir}/low_variance_handler.pkl", "wb") as file:
                    pickle.dump(self, file)
                logging.info(
                    f"LowVarianceHandler method saved at: {self.save_dir}/low_variance_handler.pkl"
                )

            logging.info("LowVarianceHandler fitted successfully.")

        except Exception as e:
            logging.error(f"Error in fitting LowVarianceHandler: {e}")

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by selecting the pre-determined features.

        Parameters:
        - data (pd.DataFrame): The input data.

        Returns:
        - pd.DataFrame: The transformed data with selected features.
        """
        if self.deactivate:
            logging.info(
                "LowVarianceHandler is deactivated. Returning unmodified data."
            )
            return data

        try:
            if self.selected_columns is None:
                raise NotFittedError(
                    "LowVarianceHandler is not fitted yet. call 'fit' before using this method."
                )

            transformed_data = data[self.selected_columns]

            if self.save_trans_data:
                if self.save_dir and not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir, exist_ok=True)
                if os.path.exists(f"{self.save_dir}/{self.trans_data_name}.csv"):
                    base, ext = os.path.splitext(self.trans_data_name)
                    counter = 1
                    new_filename = f"{base} ({counter}){ext}"

                    while os.path.exists(f"{self.save_dir}/{new_filename}.csv"):
                        counter += 1
                        new_filename = f"{base} ({counter}){ext}"

                    csv_name = new_filename

                else:
                    csv_name = self.trans_data_name

                transformed_data.to_csv(f"{self.save_dir}/{csv_name}.csv")
                logging.info(
                    f"Transformed data saved at: {self.save_dir}/{csv_name}.csv"
                )

            return transformed_data

        except Exception as e:
            logging.error(f"Error in transforming data: {e}")
            raise

    def fit_transform(self, data: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit the handler and transform the data.

        Parameters:
        - data (pd.DataFrame): The input data.

        Returns:
        - pd.DataFrame: The transformed data with selected features.
        """
        if self.deactivate:
            logging.info(
                "LowVarianceHandler is deactivated. Returning unmodified data."
            )
            return data

        self.fit(data)
        return self.transform(data)

    def setting(self, **kwargs):
        valid_keys = self.__dict__.keys()
        for key in kwargs:
            if key not in valid_keys:
                raise KeyError(f"'{key}' is not a valid attribute of LowVarianceHandler.")
        self.__dict__.update(**kwargs)

        return self