from typing import Optional, Union


class CrossValidationConfig:
    """
    Configuration object for cross-validation and reporting.

    This small datalike object centralizes the common parameters used when
    running repeated cross-validation and generating reports/figures.

    Parameters
    ----------
    scoring_target : optional str
        The primary metric name used for model selection (e.g. "f1_score", "accuracy", "r2").
    scoring_list : optional list or str
        A single metric name or a list of metric names to compute during CV.
        If a string is passed, it is treated as a single metric.
    n_splits : int, default=5
        Number of folds for each CV repeat (e.g., K in K-fold CV).
    n_repeats : int, default=5
        Number of repeated CV iterations (if using RepeatedKFold-like schemes).
    save_cv_report : bool, default=False
        If True, save an aggregated CV report (CSV) to disk.
    cv_report_name : str, default="cv_report"
        Base filename (without extension) for the saved CV report.
    visualize : optional str
        Type of visualization requested (e.g., "boxplot", "violin"); None disables.
    save_fig : bool, default=False
        If True, save CV figures to disk using fig_prefix.
    fig_prefix : str, default="cv_graph"
        Filename prefix for saved CV figures.
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
        self.scoring_target = scoring_target
        self.scoring_list = scoring_list
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.save_cv_report = save_cv_report
        self.cv_report_name = cv_report_name
        self.visualize = visualize
        self.save_fig = save_fig
        self.fig_prefix = fig_prefix
