from typing import Optional, Union


class CrossValidationConfig:

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
