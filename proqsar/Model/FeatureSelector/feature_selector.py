import os
import pickle
import logging
import pandas as pd
from sklearn.exceptions import NotFittedError
from typing import Optional, Union, List
from sklearn.base import BaseEstimator
from .feature_selector_utils import (
    _get_method_map,
    evaluate_feature_selectors,
)
from ..ModelDeveloper.model_developer_utils import (
    _get_task_type,
    _get_cv_strategy,
)
from proqsar.Config.validation_config import CrossValidationConfig


class FeatureSelector(CrossValidationConfig, BaseEstimator):
    """
    Pipeline component for feature selection.

    This class wraps multiple feature-selection strategies and provides an
    estimator-like interface, making it compatible with scikit-learn pipelines.

    Key behaviors:
      - If ``select_method`` is a list (or None) and ``cross_validate=True``,
        evaluates candidate selectors with repeated CV and selects the best one
        based on ``scoring_target``.
      - If ``select_method`` is a string, directly fits the corresponding selector.
      - Provides ``fit``, ``transform``, ``fit_transform`` and ``set_params`` methods.
      - Supports saving fitted models and transformed datasets.

    :param activity_col: Column name for the target variable. Default is ``"activity"``.
    :type activity_col: str
    :param id_col: Column name for record identifiers. Default is ``"id"``.
    :type id_col: str
    :param select_method: Method name or list of method names. If None, all methods are compared.
    :type select_method: Optional[Union[str, List[str]]]
    :param add_method: Extra methods to add to the method map (name → selector instance).
    :type add_method: Optional[dict]
    :param cross_validate: If True, compare candidate methods with CV. Default is ``True``.
    :type cross_validate: bool
    :param save_method: If True, save the fitted FeatureSelector object as pickle. Default is ``False``.
    :type save_method: bool
    :param save_trans_data: If True, save transformed datasets to CSV. Default is ``False``.
    :type save_trans_data: bool
    :param trans_data_name: Base filename for transformed datasets. Default is ``"trans_data"``.
    :type trans_data_name: str
    :param save_dir: Directory for saving models and transformed data. Default is ``"Project/FeatureSelector"``.
    :type save_dir: Optional[str]
    :param n_jobs: Number of parallel jobs for supported estimators. Default is ``1``.
    :type n_jobs: int
    :param random_state: Random seed for reproducibility. Default is ``42``.
    :type random_state: Optional[int]
    :param deactivate: If True, disables feature selection (fit/transform skipped). Default is ``False``.
    :type deactivate: bool
    :param kwargs: Additional arguments forwarded to :class:`CrossValidationConfig`.

    **Example**

    .. code-block:: python

        import pandas as pd
        from proqsar.FeatureSelector.feature_selector import FeatureSelector

        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "feature2": [5, 4, 3, 2, 1],
            "activity": [0, 1, 0, 1, 0]
        })

        selector = FeatureSelector(
            activity_col="activity",
            id_col="id",
            select_method=["Anova", "MutualInformation"],
            cross_validate=True
        )

        selector.fit(df)
        df_transformed = selector.transform(df)

        print(df_transformed.head())
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
        Initialize FeatureSelector.

        :param activity_col: Column name for the target variable. Default is ``"activity"``.
        :type activity_col: str
        :param id_col: Column name for record identifiers. Default is ``"id"``.
        :type id_col: str
        :param select_method: Method name or list of method names. Default is ``None``.
        :type select_method: Optional[Union[str, List[str]]]
        :param add_method: Extra feature-selection methods to add. Default is ``None``.
        :type add_method: Optional[dict]
        :param cross_validate: If True, evaluate candidate methods with CV. Default is ``True``.
        :type cross_validate: bool
        :param save_method: If True, save fitted object to pickle. Default is ``False``.
        :type save_method: bool
        :param save_trans_data: If True, save transformed datasets to CSV. Default is ``False``.
        :type save_trans_data: bool
        :param trans_data_name: Base filename for transformed datasets. Default is ``"trans_data"``.
        :type trans_data_name: str
        :param save_dir: Directory for saving models and transformed data. Default is ``"Project/FeatureSelector"``.
        :type save_dir: Optional[str]
        :param n_jobs: Number of parallel jobs. Default is ``1``.
        :type n_jobs: int
        :param random_state: Random seed. Default is ``42``.
        :type random_state: Optional[int]
        :param deactivate: If True, skip feature selection. Default is ``False``.
        :type deactivate: bool
        :param kwargs: Extra CV-related arguments passed to :class:`CrossValidationConfig`.
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

    def fit(self, data: pd.DataFrame) -> "FeatureSelector":
        """
        Fit feature selector(s) on the dataset.

        :param data: Input DataFrame containing features, id column, and activity column.
        :type data: pd.DataFrame

        :return: Self, with fitted selector and optional CV report.
        :rtype: FeatureSelector

        :raises ValueError: If ``select_method`` is invalid or not recognized.
        :raises AttributeError: If a list of methods is provided without ``cross_validate=True``.
        :raises Exception: For unexpected runtime errors.
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
        Transform dataset using the fitted selector.

        :param data: Input DataFrame to transform.
        :type data: pd.DataFrame

        :return: Transformed DataFrame with selected features and preserved id/activity columns.
        :rtype: pd.DataFrame

        :raises NotFittedError: If ``fit`` has not been called before.
        :raises Exception: For unexpected runtime errors.
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
        Fit to the dataset, then transform it.

        :param data: Input DataFrame.
        :type data: pd.DataFrame

        :return: Transformed dataset.
        :rtype: pd.DataFrame
        """
        if self.deactivate:
            logging.info("FeatureSelector is deactivated. Returning unmodified data.")
            return data

        self.fit(data)
        return self.transform(data)

    def set_params(self, **kwargs) -> "FeatureSelector":
        """
        Update attributes with provided keyword arguments.

        :param kwargs: Mapping of attribute names to values.
        :type kwargs: dict

        :return: Updated FeatureSelector object.
        :rtype: FeatureSelector

        :raises KeyError: If an invalid attribute name is provided.
        """
        valid_keys = self.__dict__.keys()
        for key in kwargs:
            if key not in valid_keys:
                raise KeyError(f"'{key}' is not a valid attribute.")
        self.__dict__.update(**kwargs)

        return self
