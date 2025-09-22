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
    Preprocessing transformer that removes low-variance features.

    Features with variance below a specified threshold are dropped.
    Supports visualization, saving fitted objects, and saving transformed data.
    """

    def __init__(
        self,
        activity_col: Optional[str] = None,
        id_col: Optional[str] = None,
        var_thresh: float = 0.05,
        save_method: bool = False,
        visualize: bool = False,
        save_image: bool = False,
        image_name: str = "variance_analysis",
        save_dir: str = "Project/LowVarianceHandler",
        save_trans_data: bool = False,
        trans_data_name: str = "trans_data",
        deactivate: bool = False,
    ):
        """
        Initialize the LowVarianceHandler.

        :param activity_col: Column name of the activity/target variable.
        :type activity_col: Optional[str]
        :param id_col: Column name of the identifier column.
        :type id_col: Optional[str]
        :param var_thresh: Variance threshold for feature selection. Default is 0.05.
        :type var_thresh: float
        :param save_method: Save the fitted handler object to disk if True.
        :type save_method: bool
        :param visualize: If True, generate a variance analysis plot during fitting.
        :type visualize: bool
        :param save_image: If True, save the variance analysis plot as an image.
        :type save_image: bool
        :param image_name: Base filename for the saved plot.
        :type image_name: str
        :param save_dir: Directory to save outputs (plots, models, transformed data).
        :type save_dir: str
        :param save_trans_data: If True, save the transformed DataFrame as CSV.
        :type save_trans_data: bool
        :param trans_data_name: Base filename for the saved transformed data.
        :type trans_data_name: str
        :param deactivate: If True, disables this transformer (returns input unchanged).
        :type deactivate: bool
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
        image_name: str = "variance_analysis",
        save_dir: str = "Project/VarianceHandler",
    ) -> None:
        """
        Perform variance threshold analysis on non-binary features
        and plot retained feature counts as threshold increases.

        :param data: Input DataFrame.
        :type data: pandas.DataFrame
        :param activity_col: Activity column to exclude from analysis.
        :type activity_col: Optional[str]
        :param id_col: ID column to exclude from analysis.
        :type id_col: Optional[str]
        :param set_style: Seaborn plot style (default "whitegrid").
        :type set_style: str
        :param save_image: Whether to save the plot as an image.
        :type save_image: bool
        :param image_name: Base filename for saved image.
        :type image_name: str
        :param save_dir: Directory to save plot if ``save_image=True``.
        :type save_dir: str
        :return: None
        :rtype: None
        :raises Exception: If variance analysis fails.
        """
        try:
            columns_to_exclude = [activity_col, id_col]
            temp_data = data.drop(columns=columns_to_exclude, errors="ignore")
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
                plt.plot(thresholds, results, marker=".")
                plt.title("Variance Analysis", fontsize=24, weight="semibold")
                plt.xlabel("Variance Threshold", fontsize=16)
                plt.ylabel("Number of Features", fontsize=16)
                plt.grid(True)

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
                    plt.savefig(os.path.join(save_dir, f"{image_name}.pdf"))
                    logging.info(
                        f"LowVarianceHandler: Variance analysis saved at {save_dir}/{image_name}.pdf"
                    )
            else:
                logging.info(
                    "LowVarianceHandler: No non-binary columns to apply variance threshold."
                )

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
        Select features that pass the variance threshold.

        :param data: Input DataFrame.
        :type data: pandas.DataFrame
        :param activity_col: Activity column to exclude from selection.
        :type activity_col: Optional[str]
        :param id_col: ID column to exclude from selection.
        :type id_col: Optional[str]
        :param var_thresh: Minimum variance required to retain a feature.
        :type var_thresh: float
        :return: List of selected feature names.
        :rtype: list
        :raises Exception: If variance selection fails.
        """
        try:
            columns_to_exclude = [id_col, activity_col]
            temp_data = data.drop(columns=columns_to_exclude, errors="ignore")
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
                logging.info(
                    f"LowVarianceHandler: Applying variance threshold {var_thresh}"
                )
                selector = VarianceThreshold(var_thresh)
                try:
                    selector.fit(data[non_binary_cols])
                    features = selector.get_support(indices=True)
                    selected_features = data[non_binary_cols].columns[features].tolist()
                except ValueError:
                    pass
            else:
                logging.warning(
                    "LowVarianceHandler: No non-binary columns to apply variance threshold."
                )

            return binary_cols + selected_features

        except Exception as e:
            logging.error(f"Error in feature selection by variance: {e}")
            return []

    def fit(self, data: pd.DataFrame, y=None) -> "LowVarianceHandler":
        """
        Fit the handler by determining which features exceed the variance threshold.

        :param data: Input DataFrame to fit on.
        :type data: pandas.DataFrame
        :param y: Ignored. Present for sklearn compatibility.
        :type y: Optional[pandas.Series]
        :return: The fitted LowVarianceHandler instance.
        :rtype: LowVarianceHandler
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
                    f"LowVarianceHandler: Fitted object saved at {self.save_dir}/low_variance_handler.pkl"
                )

        except Exception as e:
            logging.error(f"Error in fitting LowVarianceHandler: {e}")

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by keeping only selected features.

        :param data: Input DataFrame to transform.
        :type data: pandas.DataFrame
        :return: Transformed DataFrame with only retained features.
        :rtype: pandas.DataFrame
        :raises NotFittedError: If called before ``fit``.
        :raises Exception: For unexpected errors during transformation.
        """
        if self.deactivate:
            self.transformed_data = data
            logging.info(
                "LowVarianceHandler is deactivated. Returning unmodified data."
            )
            return data

        try:
            if self.selected_columns is None:
                raise NotFittedError(
                    "LowVarianceHandler is not fitted yet. Call 'fit' before using this method."
                )
            transformed_data = pd.concat(
                [
                    data.filter(items=[self.id_col, self.activity_col]),
                    data[self.selected_columns],
                ],
                axis=1,
            )

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

                transformed_data.to_csv(f"{self.save_dir}/{csv_name}.csv", index=False)
                logging.info(
                    f"LowVarianceHandler: Transformed data saved at {self.save_dir}/{csv_name}.csv"
                )
            self.transformed_data = transformed_data

            return transformed_data

        except Exception as e:
            logging.error(f"Error in transforming data: {e}")
            raise

    def fit_transform(self, data: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit the handler and transform the data.

        :param data: Input DataFrame.
        :type data: pandas.DataFrame
        :param y: Ignored. Present for sklearn compatibility.
        :type y: Optional[pandas.Series]
        :return: Transformed DataFrame with selected features retained.
        :rtype: pandas.DataFrame
        """
        if self.deactivate:
            logging.info(
                "LowVarianceHandler is deactivated. Returning unmodified data."
            )
            return data

        self.fit(data)
        return self.transform(data)
