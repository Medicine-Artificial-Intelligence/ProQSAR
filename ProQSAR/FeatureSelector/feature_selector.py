import os
import pickle
import logging
import pandas as pd
from sklearn.exceptions import NotFittedError
from typing import Optional, Union, List

from ProQSAR.FeatureSelector.feature_selector_utils import (
    _get_method_map,
    evaluate_feature_selectors,
)
from ProQSAR.ModelDeveloper.model_developer_utils import (
    _get_task_type,
    _get_cv_strategy,
)
from ProQSAR.validation_config import CrossValidationConfig


class FeatureSelector(CrossValidationConfig):
    """
    A class for selecting features from a dataset based on specified criteria.

    Attributes:
        activity_col (str): Column name for the target variable.
        id_col (str): Column name for the unique identifier.
        select_method (str): Feature selection method. 
        add_method (Optional[dict]): Additional feature selection methods.
        scoring (Optional[str]): Scoring metric for model evaluation.
        n_splits (int): Number of splits for cross-validation.
        n_repeats (int): Number of repeats for cross-validation.
        save_method (bool): Whether to save the fitted feature selector.
        save_trans_data (bool): Whether to save the transformed data.
        trans_data_name (str): File name for saved transformed data.
        save_dir (Optional[str]): Directory to save outputs.
        save_cv_report (bool): Whether to save a CV report.
        cv_report_name (str): Name for the CV report file.
        visualize (Optional[str]): Visualization options.
        save_fig (bool): Whether to save figures.
        fig_prefix (str): Prefix for saved figure files.
        n_jobs (int): Number of jobs to run in parallel.
        deactivate (bool): Flag to deactivate the feature selection process.

    Methods:
        fit(data: pd.DataFrame) -> object:
            Fits the feature selector to the provided data.
        transform(data: pd.DataFrame) -> pd.DataFrame:
            Transforms the data based on the fitted selector.
        fit_transform(data: pd.DataFrame) -> pd.DataFrame:
            Fits and transforms the data in one step.
    """

    def __init__(
        self,
        activity_col: str = "activity",
        id_col: str = "id",
        select_method: Optional[Union[str, List[str]]] = None,
        add_method: Optional[dict] = None,
        compare: bool = True,
        save_method: bool = False,
        save_trans_data: bool = False,
        trans_data_name: str = "trans_data",
        save_dir: Optional[str] = "Project/FeatureSelector",
        n_jobs: int = -1,
        deactivate: bool = False,
        **kwargs
    ):
        """
        Initializes the FeatureSelector with the specified parameters.

        Parameters are set based on selection and cross-validation criteria.
        """
        super().__init__(**kwargs)
        self.activity_col = activity_col
        self.id_col = id_col
        self.select_method = select_method
        self.add_method = add_method
        self.compare = compare
        self.save_method = save_method
        self.save_trans_data = save_trans_data
        self.trans_data_name = trans_data_name
        self.save_dir = save_dir
        self.n_jobs = n_jobs
        self.deactivate = deactivate
        
        self.feature_selector = None
        self.task_type = None
        self.cv = None
        self.report = None

    def fit(self, data: pd.DataFrame) -> object:
        """
        Fits the feature selector to the data.

        Determines the best feature selection method based on cross-validation or
        uses a specified method, then fits the data.

        Parameters:
            data (pd.DataFrame): Input data containing features and target column.

        Returns:
            object: Fitted feature selector object.

        Raises:
            ValueError: If an unrecognized selection method is specified.
        """
        if self.deactivate:
            logging.info("FeatureSelector is deactivated. Skipping fit.")
            return self

        try:
            logging.info("Starting feature selection fitting process.")
            X_data = data.drop([self.activity_col, self.id_col], axis=1)
            y_data = data[self.activity_col]

            self.task_type = _get_task_type(data, self.activity_col)
            self.method_map = _get_method_map(
                self.task_type, self.add_method, self.n_jobs
            )
            self.cv = _get_cv_strategy(
                self.task_type, n_splits=self.n_splits, n_repeats=self.n_repeats
            )
            # Set scorings
            self.scoring_target = (
                self.scoring_target or "f1" if self.task_type == "C" else "r2"
            )
            if self.scoring_list:
                if isinstance(self.scoring_list, str):
                    self.scoring_list = [self.scoring_list]

                if self.scoring_target not in self.scoring_list:
                    self.scoring_list.append(self.scoring_target)

            if isinstance(self.select_method, list) or not self.select_method:
                if self.compare:
                    self.report = evaluate_feature_selectors(
                        data=data,
                        activity_col=self.activity_col,
                        id_col=self.id_col,
                        select_method=self.select_method,
                        add_method=self.add_method,
                        scoring_list=self.scoring_list,
                        n_splits=self.n_splits,
                        n_repeats=self.n_repeats,
                        visualize=self.visualize,
                        save_fig=self.save_fig,
                        fig_prefix=self.fig_prefix,
                        save_csv=self.save_cv_report,
                        csv_name=self.cv_report_name,
                        save_dir=self.save_dir,
                        n_jobs=self.n_jobs,
                    )

                    self.select_method = (
                        self.report.set_index(["scoring", "cv_cycle"])
                        .loc[(f"{self.scoring_target}", "mean")]
                        .idxmax()
                    )
                    if self.select_method == "NoFS":
                        self.deactivate = True
                        logging.info(
                            "Skipping feature selection is considered to be the optimal method."
                        )
                        return self
                    else:
                        self.feature_selector = self.method_map[self.select_method].fit(
                            X=X_data, y=y_data
                        )
                else:
                    raise AttributeError(
                        "'select_method' is entered as a list."
                        "To evaluate and use the best method among the entered methods, turn 'compare = True'."
                        "Otherwise, select_method must be a string as the name of the method."
                    )
            elif isinstance(self.select_method, str):
                if self.select_method not in self.method_map:
                    raise ValueError(f"Method '{self.select_method}' not recognized.")
                else:
                    self.feature_selector = self.method_map[self.select_method].fit(
                        X=X_data, y=y_data
                    )
                    self.report = evaluate_feature_selectors(
                        data=data,
                        activity_col=self.activity_col,
                        id_col=self.id_col,
                        select_method=None if self.compare else self.select_method,
                        add_method=self.add_method,
                        scoring_list=self.scoring_list,
                        n_splits=self.n_splits,
                        n_repeats=self.n_repeats,
                        visualize=self.visualize,
                        save_fig=self.save_fig,
                        fig_prefix=self.fig_prefix,
                        save_csv=self.save_cv_report,
                        csv_name=self.cv_report_name,
                        save_dir=self.save_dir,
                        n_jobs=self.n_jobs,
                    )

            logging.info(
                f"Feature selection method '{self.select_method}' applied successfully."
            )

            if self.save_method:
                os.makedirs(self.save_dir, exist_ok=True)
                with open(f"{self.save_dir}/feature_selector.pkl", "wb") as file:
                    pickle.dump(self, file)
                logging.info("Feature selector model saved successfully.")

            return self

        except Exception as e:
            logging.error(f"Error in fit method: {e}")
            raise

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms data using the fitted feature selector.

        Parameters:
            data (pd.DataFrame): Input data containing features and target column.

        Returns:
            pd.DataFrame: Transformed data with selected features.

        Raises:
            NotFittedError: If the feature selector is not fitted yet.
        """
        if self.deactivate:
            logging.info("FeatureSelector is deactivated. Returning unmodified data.")
            return data

        try:
            if self.feature_selector is None:
                raise NotFittedError(
                    "FeatureSelector is not fitted yet. Call 'fit' before using this method."
                )

            X_data = data.drop([self.activity_col, self.id_col], axis=1)
            selected_features = self.feature_selector.transform(X_data)

            transformed_data = pd.DataFrame(
                selected_features,
                columns=X_data.columns[self.feature_selector.get_support()],
            )
            transformed_data[[self.id_col, self.activity_col]] = data[
                [self.id_col, self.activity_col]
            ].values

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
                logging.info(f"Transformed data saved at {self.save_dir}.")

            return transformed_data

        except Exception as e:
            logging.error(f"Error in transform method: {e}")
            raise

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fits and transforms the data in one step.

        Parameters:
            data (pd.DataFrame): Input data containing features and target column.

        Returns:
            pd.DataFrame: Transformed data with selected features.
        """
        if self.deactivate:
            logging.info("FeatureSelector is deactivated. Returning unmodified data.")
            return data

        self.fit(data)
        return self.transform(data)
    
    def setting(self, **kwargs):
        valid_keys = self.__dict__.keys()
        for key in kwargs:
            if key not in valid_keys:
                raise KeyError(f"'{key}' is not a valid attribute of FeatureSelector.")
        self.__dict__.update(**kwargs)

        return self