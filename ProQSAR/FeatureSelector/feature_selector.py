import os
import pickle
import logging
import pandas as pd
from sklearn.exceptions import NotFittedError
from typing import Optional, Union, List
from sklearn.base import BaseEstimator
from ProQSAR.FeatureSelector.feature_selector_utils import (
    _get_method_map,
    evaluate_feature_selectors,
)
from ProQSAR.ModelDeveloper.model_developer_utils import (
    _get_task_type,
    _get_cv_strategy,
)
from ProQSAR.Config.validation_config import CrossValidationConfig


class FeatureSelector(CrossValidationConfig, BaseEstimator):
    """
    FeatureSelector is a pipeline component that wraps a collection of
    feature-selection strategies and provides an estimator-like interface.

    Key behaviors:
      - If `select_method` is a list or None and `cross_validate` is True,
        the class will evaluate the candidate selectors using repeated CV and
        choose the best-performing selector according to `scoring_target`.
      - If `select_method` is a string, the corresponding selector will be
        fitted on the provided data.
      - The fitted selector can be used to transform new data with `transform`.
      - The object supports `fit`, `transform`, `fit_transform`, and a
        `set_params` method for simple parameter injection.

    Parameters (constructor)
    ------------------------
    activity_col : str
        Column name for the target variable (default: "activity").
    id_col : str
        Column name for the record identifier (default: "id").
    select_method : Optional[str | List[str]]
        Name of the method to use, or a list of methods to compare.
    add_method : Optional[dict]
        Extra methods to add to the default method map (name -> selector instance).
    cross_validate : bool
        If True, evaluate candidates with CV (default True).
    save_method : bool
        If True, save the fitted FeatureSelector object as a pickle (default False).
    save_trans_data : bool
        If True, save produced transformed data to CSV when transform is called.
    trans_data_name : str
        Base filename for transformed data.
    save_dir : Optional[str]
        Directory used to save pickles / transformed data (default: Project/FeatureSelector).
    n_jobs : int
        Number of parallel jobs passed to underlying estimators (default 1).
    random_state : Optional[int]
        Random seed for stochastic methods (default 42).
    deactivate : bool
        If True, the selector is deactivated and fit/transform will skip processing.
    **kwargs
        Forwarded to CrossValidationConfig for CV-related settings (n_splits, n_repeats, scoring, etc.)

    Attributes
    ----------
    feature_selector
        The fitted selector instance (after calling `fit`).
    report
        CV report DataFrame generated when comparing methods (if cross-validated).
    task_type
        Task type inferred from data ('C' or 'R').
    cv
        Cross-validation strategy object created via _get_cv_strategy.
    """

    def __init__(
        self,
        activity_col: str = "activity",
        id_col: str = "id",
        select_method: Optional[Union[str, List[str]]] = None,
        add_method: Optional[dict] = None,
        cross_validate: bool = True,
        save_method: bool = False,
        save_trans_data: bool = False,
        trans_data_name: str = "trans_data",
        save_dir: Optional[str] = "Project/FeatureSelector",
        n_jobs: int = 1,
        random_state: Optional[int] = 42,
        deactivate: bool = False,
        **kwargs,
    ):
        """
        Initialize FeatureSelector and forward CV-related kwargs to
        CrossValidationConfig.__init__ (keeps original behavior).
        """
        CrossValidationConfig.__init__(self, **kwargs)
        self.activity_col = activity_col
        self.id_col = id_col
        self.select_method = select_method
        self.add_method = add_method
        self.cross_validate = cross_validate
        self.save_method = save_method
        self.save_trans_data = save_trans_data
        self.trans_data_name = trans_data_name
        self.save_dir = save_dir
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.deactivate = deactivate

    def fit(self, data: pd.DataFrame) -> object:
        """
        Fit the feature selector(s) on `data`.

        Behavior:
          - If `deactivate` is True, the method returns immediately.
          - Infers task type and CV strategy.
          - If `select_method` is a list or None and `cross_validate` is True,
            evaluate candidate selectors using `evaluate_feature_selectors`
            and pick the best method. Fit that selector to the whole dataset.
          - If `select_method` is a single string, fit the corresponding selector.
          - Optionally save the FeatureSelector instance to disk (pickle) when
            `save_method` is True.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame including features, activity_col, and id_col.

        Returns
        -------
        self : FeatureSelector
            The fitted FeatureSelector object.

        Raises
        ------
        Exception
            Any unexpected exception is logged and re-raised.
        """

        if self.deactivate:
            logging.info("FeatureSelector is deactivated. Skipping fit.")
            return self

        try:
            X_data = data.drop([self.activity_col, self.id_col], axis=1)
            y_data = data[self.activity_col]

            self.task_type = _get_task_type(data, self.activity_col)
            method_map = _get_method_map(
                self.task_type,
                self.add_method,
                self.n_jobs,
                random_state=self.random_state,
            )
            self.cv = _get_cv_strategy(
                self.task_type,
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
                random_state=self.random_state,
            )
            # Set scorings
            if self.scoring_target is None:
                self.scoring_target = "f1" if self.task_type == "C" else "r2"

            if self.scoring_list:
                if isinstance(self.scoring_list, str):
                    self.scoring_list = [self.scoring_list]

                if self.scoring_target not in self.scoring_list:
                    self.scoring_list.append(self.scoring_target)

            self.feature_selector = None
            self.report = None
            if isinstance(self.select_method, list) or not self.select_method:
                if self.cross_validate:
                    logging.info(
                        "FeatureSelector: Selecting the optimal feature selection method "
                        f"among {self.select_method or list(method_map.keys())}, "
                        f"scoring target: '{self.scoring_target}'."
                    )
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
                        random_state=self.random_state,
                    )

                    self.select_method = (
                        self.report.set_index(["scoring", "cv_cycle"])
                        .loc[(f"{self.scoring_target}", "mean")]
                        .idxmax()
                    )
                    if self.select_method == "NoFS":
                        self.deactivate = True
                        logging.info(
                            "FeatureSelector: Skipping feature selection is considered to be the optimal method."
                        )
                        return self
                    else:
                        logging.info(f"FeatureSelector: Using '{self.select_method}'.")
                        self.feature_selector = method_map[self.select_method].fit(
                            X=X_data, y=y_data
                        )
                else:
                    raise AttributeError(
                        "'select_method' is entered as a list."
                        "To evaluate and use the best method among the entered methods, turn 'compare = True'."
                        "Otherwise, select_method must be a string as the name of the method."
                    )
            elif isinstance(self.select_method, str):
                if self.select_method not in method_map:
                    raise ValueError(
                        f"FeatureSelector: Method '{self.select_method}' not recognized."
                    )
                else:
                    logging.info(f"FeatureSelector: Using method: {self.select_method}")

                    self.feature_selector = method_map[self.select_method].fit(
                        X=X_data, y=y_data
                    )

                    if self.cross_validate:
                        logging.info(
                            "FeatureSelector: Cross-validation is enabled, generating report."
                        )
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
                            random_state=self.random_state,
                        )
            else:
                raise AttributeError(
                    f"'select_method' is entered as a {type(self.select_method)}"
                    "Please input a string or a list or None."
                )

            if self.save_method:
                os.makedirs(self.save_dir, exist_ok=True)
                with open(f"{self.save_dir}/feature_selector.pkl", "wb") as file:
                    pickle.dump(self, file)
                logging.info(
                    f"FeatureSelector saved at: {self.save_dir}/feature_selector.pkl."
                )

            return self

        except Exception as e:
            logging.error(f"Error in fit method: {e}")
            raise

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform `data` by selecting features using the fitted selector.

        Behavior:
          - If `deactivate` is True, returns the input data unchanged.
          - Raises NotFittedError if `fit` has not been called.
          - Produces a DataFrame containing selected feature columns and preserves
            the id and activity columns (if present).
          - Optionally saves the transformed data to CSV when `save_trans_data` is True.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with selected features and original id/activity columns.

        Raises
        ------
        NotFittedError
            If the internal feature selector has not been fitted yet.
        Exception
            Any unexpected exception is logged and re-raised.
        """
        if self.deactivate:
            logging.info("FeatureSelector is deactivated. Returning unmodified data.")
            return data

        try:
            if self.feature_selector is None:
                raise NotFittedError(
                    "FeatureSelector is not fitted yet. Call 'fit' before using this method."
                )

            X_data = data.drop(
                [self.activity_col, self.id_col], axis=1, errors="ignore"
            )
            selected_features = self.feature_selector.transform(X_data)

            transformed_data = pd.DataFrame(
                selected_features,
                columns=X_data.columns[self.feature_selector.get_support()],
            )

            cols = [
                col for col in [self.id_col, self.activity_col] if col in data.columns
            ]
            transformed_data[cols] = data[cols].values

            if self.activity_col in transformed_data.columns:
                transformed_data[self.activity_col] = (
                    transformed_data[self.activity_col].astype(int)
                    if self.task_type == "C"
                    else transformed_data[self.activity_col].astype(float)
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
                    f"FeatureSelector: Transformed data saved at: {self.save_dir}/{csv_name}.csv."
                )

            return transformed_data

        except Exception as e:
            logging.error(f"Error in transform method: {e}")
            raise

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience method that runs `fit` followed by `transform`.

        If `deactivate` is True, returns the input data unchanged.
        """
        if self.deactivate:
            logging.info("FeatureSelector is deactivated. Returning unmodified data.")
            return data

        self.fit(data)
        return self.transform(data)

    def set_params(self, **kwargs):
        """
        Simple parameter setter that updates attributes if they exist.

        Raises KeyError for unknown keys. Returns self to allow fluent chaining.

        Parameters
        ----------
        **kwargs : dict
            Attribute names and their new values.

        Returns
        -------
        self : FeatureSelector
        """
        valid_keys = self.__dict__.keys()
        for key in kwargs:
            if key not in valid_keys:
                raise KeyError(f"'{key}' is not a valid attribute.")
        self.__dict__.update(**kwargs)

        return self
