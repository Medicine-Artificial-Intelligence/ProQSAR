from __future__ import annotations

import datetime
import logging
from typing import Any, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd


class Inference:
    """
    Inference-focused runner that prepares inputs, calls a prediction pipeline, and writes
    results back in-place by default.

    The runner stores light metadata after each run:
      - last_input_df: full input DataFrame after prediction (deep-copied when possible)
      - last_preds: DataFrame or Series-like predictions captured from the pipeline
      - last_run_time, last_n, last_prediction_summary

    The pretty `__repr__` produces a concise box showing inference statistics:
      - prediction mean/std/quantiles
      - Applicability Domain (AD) counts if present in the input frame
      - largest Prediction Interval (PI) range if PI lower/upper columns exist
      - top / bottom K predicted items (shows SMILES if available)

    :param pipeline: Object exposing required attributes and method:
                     `id_col`, `smiles_col`, `activity_col`, and a callable
                     `predict(df, alpha=...)` which returns a DataFrame, Series/array-like,
                     or mapping of prediction values.
    :type pipeline: object
    :param inplace: If True and the provided input is a pandas DataFrame, mutate it in-place.
                    If False a copy is used and returned. Default: True.
    :type inplace: bool
    :param alpha: Default alpha forwarded to `pipeline.predict`. Default: 0.05.
    :type alpha: float
    :param logger: Optional logger to use for exceptions and debug messages. If None,
                   the module logger is used.
    :type logger: logging.Logger | None
    """

    def __init__(
        self,
        pipeline: Any,
        *,
        inplace: bool = True,
        alpha: float = 0.05,
        logger: Optional[logging.Logger] = None,
    ):
        self.pipeline = pipeline
        self.inplace = bool(inplace)
        self.alpha = float(alpha)
        self.logger = logger or logging.getLogger(__name__)

        # runtime metadata
        self.last_run_time: Optional[datetime.datetime] = None
        self.last_n: Optional[int] = None
        self.last_prediction_summary: Optional[Mapping[str, float]] = None
        self.last_preds: Optional[pd.DataFrame] = None
        # full input frame after predictions (preferred for stats)
        self.last_input_df: Optional[pd.DataFrame] = None

        # optional metadata or user-attached fields
        self.project_name = getattr(self.pipeline, "project_name", None)
        self.save_dir = getattr(self.pipeline, "save_dir", None)
        self.selected_feature = getattr(self.pipeline, "selected_feature", None)

        # validate pipeline shape
        self._validate_pipeline()

    # -------------------------
    # Public API (kept small)
    # -------------------------
    def run(
        self,
        data: Union[pd.DataFrame, dict, list],
        *,
        smiles_key: Optional[str] = None,
        id_key: Optional[str] = None,
        ground_truth: Optional[str] = None,
        alpha: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Prepare `data` for the pipeline, call `pipeline.predict(...)` and write predictions
        back into the DataFrame.

        After a successful run `self.last_input_df` will contain a deep copy of the input
        DataFrame including any prediction columns added by the pipeline.

        :param data: Input table to run inference on. Accepts a pandas DataFrame, a list of
                     dicts, or a dict-of-lists. If a DataFrame is passed and `inplace=True`,
                     the passed object is mutated.
        :type data: pandas.DataFrame | dict | list
        :param smiles_key: Column name in `data` that contains SMILES strings. Required.
        :type smiles_key: str
        :param id_key: Column name in `data` to use as the identifier. If None, the DataFrame
                       index is used when unique; otherwise sequential integers are generated.
        :type id_key: str or None
        :param ground_truth: Column name in `data` that holds ground-truth activity values.
                             If None, the runner will create the pipeline activity column
                             (filled with pandas.NA if absent).
        :type ground_truth: str or None
        :param alpha: Optional override for the alpha parameter passed to the pipeline.
                      If None, the runner's default `self.alpha` is used.
        :type alpha: float or None

        :returns: The DataFrame that was used for inference. If `inplace=True` and the
                  caller provided a DataFrame, the same object (mutated) is returned.
        :rtype: pandas.DataFrame

        :raises ValueError: If `smiles_key` is None.
        :raises KeyError: If required input columns (smiles_key, id_key, or ground_truth) are not found.
        :raises RuntimeError: If `pipeline.predict(...)` raises an exception.
        """
        if smiles_key is None:
            raise ValueError("`smiles_key` is required.")
        call_alpha = self.alpha if alpha is None else float(alpha)
        original_df_ref = data if isinstance(data, pd.DataFrame) else None
        df = self._prepare_dataframe(data)
        self._prepare_columns(
            df, smiles_key=smiles_key, id_key=id_key, ground_truth=ground_truth
        )
        preds = self._call_predict(df, call_alpha)
        # apply predictions (mutates df)
        self._apply_predictions_inplace(
            df,
            preds,
            getattr(self.pipeline, "id_col"),
            getattr(self.pipeline, "activity_col"),
        )
        # after predictions, save full frame for reporting and debugging
        try:
            self.last_input_df = df.copy(deep=True)
        except Exception:
            # best-effort fallback
            self.last_input_df = df.copy(deep=False)
        self._update_run_metadata(df, getattr(self.pipeline, "activity_col"), preds)
        return data if original_df_ref is not None else df

    # -------------------------
    # Small helpers
    # -------------------------
    def _validate_pipeline(self) -> None:
        """
        Ensure the pipeline provides the minimum required attributes and method.

        :raises ValueError: If `pipeline` is None.
        :raises AttributeError: If `pipeline` lacks `.predict` or the column attribute names.
        """
        if self.pipeline is None:
            raise ValueError("`pipeline` must be provided.")
        if not hasattr(self.pipeline, "predict"):
            raise AttributeError("`pipeline` must implement `.predict(df, alpha=...)`.")
        for attr in ("id_col", "smiles_col", "activity_col"):
            if not hasattr(self.pipeline, attr):
                raise AttributeError(f"`pipeline` must expose attribute '{attr}'.")

    def _prepare_dataframe(self, data: Union[pd.DataFrame, dict, list]) -> pd.DataFrame:
        """
        Return a DataFrame to operate on according to the `inplace` policy.

        :param data: Input data
        :type data: pandas.DataFrame | dict | list
        :returns: DataFrame view (same object if inplace and input was DataFrame)
        :rtype: pandas.DataFrame
        """
        if isinstance(data, pd.DataFrame):
            return data if self.inplace else data.copy(deep=True)
        return pd.DataFrame(data)

    def _prepare_columns(
        self,
        df: pd.DataFrame,
        *,
        smiles_key: str,
        id_key: Optional[str],
        ground_truth: Optional[str],
    ) -> None:
        """
        Validate required columns exist and write canonical pipeline columns into `df`.

        This mutates `df` in-place.

        :param df: DataFrame to modify.
        :type df: pandas.DataFrame
        :param smiles_key: Input column containing SMILES.
        :type smiles_key: str
        :param id_key: Optional input id column name.
        :type id_key: str or None
        :param ground_truth: Optional ground truth column name.
        :type ground_truth: str or None

        :raises KeyError: If smiles_key or id_key or ground_truth (when provided) are missing.
        """
        if smiles_key not in df.columns:
            raise KeyError(
                f"smiles_key='{smiles_key}' not found in input columns: {list(df.columns)}"
            )
        df[getattr(self.pipeline, "smiles_col")] = df[smiles_key].astype(str)
        id_col = getattr(self.pipeline, "id_col")
        if id_key is None:
            try:
                idx = df.index
                df[id_col] = idx.astype(object) if idx.is_unique else np.arange(len(df))
            except Exception:
                df[id_col] = np.arange(len(df))
        else:
            if id_key not in df.columns:
                raise KeyError(
                    f"id_key='{id_key}' not found in input columns: {list(df.columns)}"
                )
            df[id_col] = df[id_key].values
        activity_col = getattr(self.pipeline, "activity_col")
        if ground_truth is None:
            if activity_col not in df.columns:
                df[activity_col] = pd.NA
        else:
            if ground_truth not in df.columns:
                raise KeyError(
                    f"ground_truth='{ground_truth}' not found in input columns: {list(df.columns)}"
                )
            df[activity_col] = df[ground_truth].values

    def _call_predict(self, df: pd.DataFrame, alpha: float) -> Any:
        """
        Call `pipeline.predict` with error wrapping.

        :param df: Prepared DataFrame to pass to the pipeline.
        :type df: pandas.DataFrame
        :param alpha: Alpha forwarded to pipeline.
        :type alpha: float
        :returns: Pipeline predictions (type depends on pipeline implementation).
        :rtype: Any
        :raises RuntimeError: If the pipeline raises an exception.
        """
        try:
            return self.pipeline.predict(df, alpha=alpha)
        except Exception as exc:
            self.logger.exception("pipeline.predict raised an exception")
            raise RuntimeError(
                f"pipeline.predict(...) raised an exception: {exc}"
            ) from exc

    def _apply_predictions_inplace(
        self, target_df: pd.DataFrame, preds: Any, id_col: str, activity_col: str
    ) -> None:
        """
        Mutate `target_df` by assigning prediction outputs.

        Supports preds as:
          - pandas.DataFrame (aligns on id_col if present)
          - mapping / dict-like (converted to DataFrame)
          - Series / array-like (1D assigned to activity_col; 2D assigned to activity_col/_i suffixes)

        :param target_df: DataFrame to mutate.
        :type target_df: pandas.DataFrame
        :param preds: Predictions returned by the pipeline.
        :type preds: Any
        :param id_col: Identifier column name used for alignment.
        :type id_col: str
        :param activity_col: Pipeline activity/prediction column name to write to.
        :type activity_col: str

        :raises TypeError: If preds type is unsupported.
        :raises ValueError: If row counts mismatch for row-wise assignment.
        """
        if isinstance(preds, pd.DataFrame):
            self._assign_from_dataframe_preds(target_df, preds, id_col)
            self.last_preds = preds.copy()
            return
        if isinstance(preds, dict):
            preds_df = pd.DataFrame(preds)
            self._assign_from_dataframe_preds(target_df, preds_df, id_col)
            self.last_preds = preds_df.copy()
            return
        if isinstance(preds, (pd.Series, list, tuple, np.ndarray)):
            self._assign_from_array_preds(target_df, preds, activity_col)
            try:
                self.last_preds = pd.DataFrame(
                    {activity_col: np.asarray(preds).ravel()}
                )
            except Exception:
                self.last_preds = None
            return
        raise TypeError(f"Unsupported prediction return type: {type(preds)}")

    def _assign_from_dataframe_preds(
        self, target_df: pd.DataFrame, preds_df: pd.DataFrame, id_col: str
    ) -> None:
        """
        Assign columns coming from a prediction DataFrame. Aligns on `id_col` when present.

        :param target_df: target DataFrame to mutate
        :type target_df: pandas.DataFrame
        :param preds_df: predictions DataFrame
        :type preds_df: pandas.DataFrame
        :param id_col: identifier column name
        :type id_col: str
        """
        preds_df = preds_df.copy()
        if id_col in preds_df.columns:
            t_indexed = target_df.set_index(id_col)
            p_indexed = preds_df.set_index(id_col)
            joined = t_indexed.join(p_indexed, how="left", rsuffix="_tmp")
            for col in p_indexed.columns:
                if col == id_col:
                    continue
                vals = joined[col].where(~joined[col].isna(), t_indexed.get(col))
                target_df[col] = vals.values
        else:
            if len(preds_df) != len(target_df):
                raise ValueError(
                    f"Predictions DataFrame has {len(preds_df)} rows but input has {len(target_df)}."
                )
            for col in preds_df.columns:
                target_df[col] = preds_df[col].values

    def _assign_from_array_preds(
        self, target_df: pd.DataFrame, preds: Any, activity_col: str
    ) -> None:
        """
        Assign predictions from array-like outputs into `activity_col` (or suffixed columns).

        :param target_df: target DataFrame to mutate
        :type target_df: pandas.DataFrame
        :param preds: array-like predictions
        :type preds: array-like
        :param activity_col: base activity column name
        :type activity_col: str
        """
        arr = np.asarray(preds)
        if arr.ndim == 0:
            target_df[activity_col] = arr.item()
            return
        if arr.ndim == 1:
            if arr.shape[0] != len(target_df):
                raise ValueError(
                    f"Prediction length ({arr.shape[0]}) != input length ({len(target_df)})."
                )
            target_df[activity_col] = arr
            return
        if arr.ndim == 2:
            if arr.shape[0] != len(target_df):
                raise ValueError(
                    f"Prediction length ({arr.shape[0]}) != input length ({len(target_df)})."
                )
            for i in range(arr.shape[1]):
                colname = activity_col if i == 0 else f"{activity_col}_{i}"
                target_df[colname] = arr[:, i]
                return
        raise ValueError("Predictions with ndim > 2 are not supported.")

    def _update_run_metadata(
        self, df: pd.DataFrame, activity_col: str, preds: Any
    ) -> None:
        """
        Update simple run metadata after a prediction run.

        :param df: The full DataFrame after predictions (may include new columns).
        :type df: pandas.DataFrame
        :param activity_col: Pipeline activity column name.
        :type activity_col: str
        """
        self.last_run_time = datetime.datetime.now()
        self.last_n = len(df)
        try:
            numeric = pd.to_numeric(df[activity_col], errors="coerce")
            self.last_prediction_summary = {
                "mean": float(np.nanmean(numeric)),
                "std": float(np.nanstd(numeric)),
                "nan_frac": float(np.isnan(numeric).mean()),
            }
        except Exception:
            self.last_prediction_summary = None

    # -------------------------
    # Inference-focused repr helpers
    # -------------------------
    def _find_ad_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Case-insensitive search for a boolean Applicability Domain column.

        Recognised names include common variants: 'in_ad', 'AD', 'Applicability domain', etc.

        :param df: DataFrame to search
        :type df: pandas.DataFrame
        :returns: column name if found, otherwise None
        :rtype: str or None
        """
        candidates = [
            "in_ad",
            "ad",
            "inside_ad",
            "is_in_ad",
            "in_applicability_domain",
            "applicability domain",
            "applicability_domain",
            "applicabilitydomain",
            "applicability_domain_flag",
            "AD",
        ]
        lowered = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in lowered:
                return lowered[cand.lower()]
        for col in df.columns:
            low = col.lower().replace("_", " ")
            if "applicab" in low and "domain" in low:
                return col
        return None

    def _find_pi_columns(self, df: pd.DataFrame) -> Optional[Tuple[str, str]]:
        """
        Look for two columns representing lower/upper Prediction Interval.

        :param df: DataFrame to search
        :type df: pandas.DataFrame
        :returns: tuple (lower_col, upper_col) or None
        :rtype: tuple[str, str] | None
        """
        lower_candidates = [
            "pi_lower",
            "pred_lower",
            "lower",
            "ci_lower",
            "prediction_interval_lower",
            "pi.l",
        ]
        upper_candidates = [
            "pi_upper",
            "pred_upper",
            "upper",
            "ci_upper",
            "prediction_interval_upper",
            "pi.u",
        ]
        lowered = {c.lower(): c for c in df.columns}
        for low in lower_candidates:
            for up in upper_candidates:
                if low in lowered and up in lowered:
                    return (lowered[low], lowered[up])
        base = getattr(self.pipeline, "activity_col", "")
        if base:
            low_name = f"{base}_lower".lower()
            up_name = f"{base}_upper".lower()
            if low_name in lowered and up_name in lowered:
                return (lowered[low_name], lowered[up_name])
        return None

    def _inference_statistics(
        self, df: pd.DataFrame, activity_col: str, id_col: str, top_k: int = 3
    ) -> dict:
        """
        Compute lightweight inference statistics (numerical summary, AD counts, PI largest range,
        and top/bottom predictions).

        :param df: DataFrame to compute stats from (preferably the full input frame).
        :type df: pandas.DataFrame
        :param activity_col: activity/prediction column name.
        :type activity_col: str
        :param id_col: identifier column name for referencing examples.
        :type id_col: str
        :param top_k: how many top/bottom examples to keep. Default: 3.
        :type top_k: int
        :returns: dict with keys like 'mean','std','in_ad','out_ad','largest_pi_range','top_k','bottom_k',...
        :rtype: dict
        """
        stats = {
            "n": len(df),
            "last_run": (
                self.last_run_time.isoformat(timespec="seconds")
                if self.last_run_time
                else None
            ),
        }
        numeric = pd.to_numeric(
            df.get(activity_col, pd.Series(dtype=float)), errors="coerce"
        )
        stats.update(
            {
                "mean": float(np.nanmean(numeric)) if numeric.size else None,
                "std": float(np.nanstd(numeric)) if numeric.size else None,
                "nan_frac": float(np.isnan(numeric).mean()) if numeric.size else None,
                "q10": float(np.nanpercentile(numeric, 10)) if numeric.size else None,
                "q50": float(np.nanpercentile(numeric, 50)) if numeric.size else None,
                "q90": float(np.nanpercentile(numeric, 90)) if numeric.size else None,
            }
        )

        # AD counts (Applicability domain)
        ad_col = self._find_ad_column(df)
        if ad_col is not None:
            try:
                ser = (
                    df[ad_col]
                    .astype(object)
                    .map(lambda x: bool(x) if pd.notna(x) else False)
                )
                in_ad = int(ser.sum())
                out_ad = int(len(ser) - in_ad)
                stats["ad_col"] = ad_col
                stats["in_ad"] = in_ad
                stats["out_ad"] = out_ad
                stats["in_ad_frac"] = float(in_ad / max(1, len(df)))
            except Exception:
                stats["ad_col"] = ad_col
                stats["in_ad"] = None
                stats["out_ad"] = None
                stats["in_ad_frac"] = None
        else:
            stats["ad_col"] = None
            stats["in_ad"] = None
            stats["out_ad"] = None
            stats["in_ad_frac"] = None

        # PI largest range (Prediction Interval)
        pi_cols = self._find_pi_columns(df)
        if pi_cols is not None:
            low, up = pi_cols
            low_num = pd.to_numeric(df[low], errors="coerce")
            up_num = pd.to_numeric(df[up], errors="coerce")
            range_vals = up_num - low_num
            if not range_vals.dropna().empty:
                largest_idx = int(range_vals.idxmax())
                stats["pi_cols"] = (low, up)
                stats["largest_pi_range"] = float(range_vals.max())
                stats["largest_pi_id"] = df.iloc[largest_idx].get(id_col)
                stats["largest_pi_low"] = (
                    float(low_num.iloc[largest_idx])
                    if pd.notna(low_num.iloc[largest_idx])
                    else None
                )
                stats["largest_pi_up"] = (
                    float(up_num.iloc[largest_idx])
                    if pd.notna(up_num.iloc[largest_idx])
                    else None
                )
            else:
                stats["pi_cols"] = (low, up)
                stats["largest_pi_range"] = None
        else:
            stats["pi_cols"] = None
            stats["largest_pi_range"] = None

        # top / bottom K predicted: prefer smiles column for display; fall back to id.
        smiles_col = getattr(self.pipeline, "smiles_col", None)
        try:
            display_key = (
                smiles_col if (smiles_col and smiles_col in df.columns) else id_col
            )
            ranked = df[[display_key, activity_col]].copy()
            ranked[activity_col] = pd.to_numeric(ranked[activity_col], errors="coerce")
            ranked_sorted = ranked.sort_values(by=activity_col, ascending=False).dropna(
                subset=[activity_col]
            )

            def _pair_from_row(row):
                key = row[display_key]
                val = row[activity_col]
                try:
                    return (str(key), float(val))
                except Exception:
                    return (str(key), None)

            stats["top_k"] = [
                _pair_from_row(r)
                for r in ranked_sorted.head(top_k).itertuples(index=False, name=None)
            ]
            stats["bottom_k"] = [
                _pair_from_row(r)
                for r in ranked_sorted.tail(top_k).itertuples(index=False, name=None)
            ]
        except Exception:
            stats["top_k"] = None
            stats["bottom_k"] = None

        return stats

    def _repr_basic_info(self) -> dict:
        """
        Collect a few basic attributes used by __repr__ (keeps __repr__ small).

        :returns: dict with keys 'proj', 'savedir', 'selected'
        :rtype: dict
        """
        proj = getattr(self, "project_name", None) or getattr(
            self.pipeline, "project_name", "<unknown>"
        )
        savedir = getattr(self, "save_dir", None) or getattr(
            self.pipeline, "save_dir", "<unknown>"
        )
        selected = getattr(self, "selected_feature", None) or getattr(
            self.pipeline, "selected_feature", None
        )
        return {"proj": proj, "savedir": savedir, "selected": selected}

    def __repr__(self) -> str:
        """
        Pretty boxed representation focused on inference statistics.

        Uses `self.last_input_df` (preferred) or `self.last_preds` to compute statistics.
        """
        info = self._repr_basic_info()
        activity_col = getattr(self.pipeline, "activity_col")
        id_col = getattr(self.pipeline, "id_col")

        # prefer last_input_df saved after run
        df_for_stats = (
            self.last_input_df if isinstance(self.last_input_df, pd.DataFrame) else None
        )
        if df_for_stats is None and isinstance(self.last_preds, pd.DataFrame):
            df_for_stats = self.last_preds.copy()
        if df_for_stats is None:
            df_for_stats = pd.DataFrame(columns=[id_col, activity_col])

        stats = self._inference_statistics(
            df_for_stats, activity_col=activity_col, id_col=id_col
        )

        # build boxed pretty repr focused on inference
        box_width = 72
        pip_name = getattr(self.pipeline, "__class__", type(self.pipeline)).__name__
        lines = [
            "┌" + "─" * box_width + "┐",
            (f"│ Inference ({pip_name})").ljust(box_width) + " │",
            "├" + "─" * box_width + "┤",
            ("│ Project: " + str(info["proj"])).ljust(box_width) + " │",
            ("│ Save Dir: " + str(info["savedir"])).ljust(box_width) + " │",
            ("│ Selected feature: " + repr(info["selected"])).ljust(box_width) + " │",
            ("│ Last run (rows): " + str(stats.get("n"))).ljust(box_width) + " │",
        ]

        # AD info (if available) — label as Applicability domain
        if stats.get("ad_col"):
            in_ad = stats.get("in_ad")
            out_ad = stats.get("out_ad")
            in_frac = stats.get("in_ad_frac")
            lines.append(
                ("│ Applicability domain column: " + str(stats["ad_col"])).ljust(
                    box_width
                )
                + " │"
            )
            lines.append(
                (
                    "│ AD: in="
                    + str(in_ad)
                    + f" ({in_frac:.2%})"
                    + "  out="
                    + str(out_ad)
                ).ljust(box_width)
                + " │"
            )

        # prediction summary
        mean = stats.get("mean")
        std = stats.get("std")
        nan_frac = stats.get("nan_frac")
        q10, q50, q90 = stats.get("q10"), stats.get("q50"), stats.get("q90")
        lines.append(
            (
                "│ Predictions — mean: "
                + (f"{mean:.3f}" if mean is not None else "n/a")
                + "  std: "
                + (f"{std:.3f}" if std is not None else "n/a")
                + "  nan%: "
                + (f"{nan_frac:.2%}" if nan_frac is not None else "n/a")
            ).ljust(box_width)
            + " │"
        )
        lines.append(
            (
                "│ Quantiles (10/50/90): "
                + (f"{q10:.3f}" if q10 is not None else "n/a")
                + " / "
                + (f"{q50:.3f}" if q50 is not None else "n/a")
                + " / "
                + (f"{q90:.3f}" if q90 is not None else "n/a")
            ).ljust(box_width)
            + " │"
        )

        # PI largest range (Prediction Interval) — include alpha used
        if stats.get("pi_cols"):
            lines.append(
                (
                    "│ Prediction Interval (alpha="
                    + f"{self.alpha:.3f}"
                    + "): largest range = "
                    + (
                        f"{stats['largest_pi_range']:.3f}"
                        if stats.get("largest_pi_range") is not None
                        else "n/a"
                    )
                    + " (id="
                    + str(stats.get("largest_pi_id"))
                    + ")"
                ).ljust(box_width)
                + " │"
            )

        # top / bottom examples (show smiles:value or id:value)
        if stats.get("top_k"):
            top_str = (
                ", ".join(
                    [
                        f"{t[0]}:{t[1]:.3f}" if t[1] is not None else f"{t[0]}:n/a"
                        for t in stats["top_k"]
                    ]
                )
                if stats["top_k"]
                else "n/a"
            )
            lines.append(("│ Top predictions: " + top_str).ljust(box_width) + " │")
        if stats.get("bottom_k"):
            bot_str = (
                ", ".join(
                    [
                        f"{t[0]}:{t[1]:.3f}" if t[1] is not None else f"{t[0]}:n/a"
                        for t in stats["bottom_k"]
                    ]
                )
                if stats["bottom_k"]
                else "n/a"
            )
            lines.append(("│ Bottom predictions: " + bot_str).ljust(box_width) + " │")

        lines += ["└" + "─" * box_width + "┘"]
        return "\n".join(lines)

    # -------------------------
    # Utilities
    # -------------------------
    def get_last_prediction_frame(self) -> Optional[pd.DataFrame]:
        """
        Return a copy of the last predictions frame (if available).

        :returns: DataFrame copy of predictions or None
        :rtype: pandas.DataFrame | None
        """
        return None if self.last_preds is None else self.last_preds.copy()

    def summary(self) -> dict:
        """
        Return a small summary of the last run suitable for programmatic consumption.

        :returns: Dict with keys 'last_run_time', 'n', and 'prediction_summary'
        :rtype: dict
        """
        return {
            "last_run_time": self.last_run_time,
            "n": self.last_n,
            "prediction_summary": self.last_prediction_summary,
        }
