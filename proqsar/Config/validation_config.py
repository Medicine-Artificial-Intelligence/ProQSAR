from typing import Optional, Union


class CrossValidationConfig:
    """
    Configuration container for cross-validation and reporting options.

    This lightweight datalike object centralizes common parameters used when
    running repeated cross-validation and generating reports/figures across the
    project (e.g., ModelDeveloper, FeatureSelector).

    :param scoring_target: Primary metric name used for model selection
                           (e.g. "f1", "accuracy", "r2"). If ``None``, callers
                           may set a sensible default based on task type.
    :type scoring_target: Optional[str]
    :param scoring_list: A single metric name or a list of metric names to compute
                         during CV. If a string is provided it will be treated as
                         a single-element list by callers.
    :type scoring_list: Optional[Union[list, str]]
    :param n_splits: Number of folds per repeat (K in K-fold).
    :type n_splits: int
    :param n_repeats: Number of repeated CV iterations (for RepeatedKFold-like schemes).
    :type n_repeats: int
    :param save_cv_report: If True, save an aggregated CV report (CSV) to disk.
    :type save_cv_report: bool
    :param cv_report_name: Base filename (without extension) for the saved CV report.
    :type cv_report_name: str
    :param visualize: Visualization type requested (e.g., "box", "violin") or None to disable plotting.
    :type visualize: Optional[str]
    :param save_fig: If True, save CV figures to disk using ``fig_prefix``.
    :type save_fig: bool
    :param fig_prefix: Filename prefix for saved CV figures.
    :type fig_prefix: str
    """

    def __init__(
        self,
        scoring_target: Optional[str] = None,
        scoring_list: Optional[Union[list, str]] = None,
        n_splits: int = 5,
        n_repeats: int = 5,
        save_cv_report: bool = False,
        cv_report_name: str = "cv_report",
        visualize: Optional[str] = None,
        save_fig: bool = False,
        fig_prefix: str = "cv_graph",
    ):
        """
        Initialize a CrossValidationConfig instance.

        :param scoring_target: Primary metric name used for model selection
                               (e.g. "f1", "accuracy", "r2"). Default ``None``.
        :type scoring_target: Optional[str]
        :param scoring_list: A single metric name or a list of metric names to compute
                             during CV. Default ``None``.
        :type scoring_list: Optional[Union[list, str]]
        :param n_splits: Number of folds per repeat (default: 5).
        :type n_splits: int
        :param n_repeats: Number of repeated CV iterations (default: 5).
        :type n_repeats: int
        :param save_cv_report: If True, save an aggregated CV report (CSV) to disk (default: False).
        :type save_cv_report: bool
        :param cv_report_name: Base filename (without extension) for the saved CV report (default: "cv_report").
        :type cv_report_name: str
        :param visualize: Visualization type requested (e.g., "box", "violin")
        or None to disable plotting (default: None).
        :type visualize: Optional[str]
        :param save_fig: If True, save CV figures to disk using ``fig_prefix`` (default: False).
        :type save_fig: bool
        :param fig_prefix: Filename prefix for saved CV figures (default: "cv_graph").
        :type fig_prefix: str

        :return: None
        :rtype: None
        """
        self.scoring_target = scoring_target
        self.scoring_list = scoring_list
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.save_cv_report = save_cv_report
        self.cv_report_name = cv_report_name
        self.visualize = visualize
        self.save_fig = save_fig
        self.fig_prefix = fig_prefix
