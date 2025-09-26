import pandas as pd
import numpy as np
import pickle
import os
import logging
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin


class DuplicateHandler(BaseEstimator, TransformerMixin):
    """
    A preprocessing transformer to detect and remove duplicate
    columns and rows in a pandas DataFrame.

    - Duplicate columns are removed (if cols=True) using exact column equality.
    - Duplicate rows are consolidated (if rows=True) based on all *feature* columns
      (i.e., all columns except `id_col`, `activity_col`, and any removed dup columns).
      The consolidation strategy for the activity column is controlled by `keep`:

        * 'first'  : keep the first occurrence as-is
        * 'last'   : keep the last occurrence as-is
        * 'random' : keep a random occurrence (requires `random_state` for determinism)
        * 'min'    : keep the row with minimum activity
        * 'max'    : keep the row with maximum activity
        * 'mean'   : collapse duplicates and set activity to the mean
        * 'median' : collapse duplicates and set activity to the median

      For 'mean' / 'median', the first row of the group is retained and its activity
      value is replaced by the aggregated statistic.

    Supports saving the fitted handler and transformed data for reproducibility.
    """

    def __init__(
        self,
        activity_col: Optional[str] = None,
        id_col: Optional[str] = None,
        cols: bool = True,
        rows: bool = True,
        keep: str = "mean",
        random_state: Optional[int] = 42,
        save_method: bool = False,
        save_dir: str = "Project/DuplicateHandler",
        save_trans_data: bool = False,
        trans_data_name: str = "trans_data",
        deactivate: bool = False,
    ):
        """
        Initialize the DuplicateHandler with configuration.

        :param activity_col: Column name for the activity or target variable.
        :type activity_col: Optional[str]
        :param id_col: Column name for the identifier column.
        :type id_col: Optional[str]
        :param cols: Whether to remove duplicate columns.
        :type cols: bool
        :param rows: Whether to remove duplicate rows.
        :type rows: bool
        :param keep: Strategy to resolve duplicates for the activity column.
                     One of {'first','last','random','min','max','mean','median'}.
        :type keep: str
        :param random_state: Seed for random selection when keep=='random'.
        :type random_state: Optional[int]
        :param save_method: Save the fitted DuplicateHandler object to disk if True.
        :type save_method: bool
        :param save_dir: Directory to save the handler and transformed data.
        :type save_dir: str
        :param save_trans_data: Save transformed data as CSV if True.
        :type save_trans_data: bool
        :param trans_data_name: Base filename for saved transformed data.
        :type trans_data_name: str
        :param deactivate: If True, the transformer is a no-op and returns input unchanged.
        :type deactivate: bool
        """
        self.id_col = id_col
        self.activity_col = activity_col
        self.cols = cols
        self.rows = rows
        self.keep = keep
        self.random_state = random_state
        self.save_method = save_method
        self.save_dir = save_dir
        self.save_trans_data = save_trans_data
        self.trans_data_name = trans_data_name
        self.deactivate = deactivate
        self.dup_cols = None

    def _normalized_keep(self) -> str:
        k = (self.keep or "").strip().lower()
        valid = {"first", "last", "random", "min", "max", "mean", "median"}
        if k not in valid:
            raise ValueError(
                f"Invalid keep='{self.keep}'. Must be one of {sorted(valid)}."
            )
        return k

    def _require_activity(self, for_ops: list[str]):
        k = self._normalized_keep()
        if k in for_ops and not self.activity_col:
            raise ValueError(f"keep='{self.keep}' requires 'activity_col' to be set.")

    def _keep_first_last(
        self, df: pd.DataFrame, feature_cols: list[str], keep_mode: str
    ) -> pd.DataFrame:
        # Rows that will be removed when using drop_duplicates with keep_mode
        dropped_idx = df.index[
            df.duplicated(subset=feature_cols, keep=keep_mode)
        ].tolist()
        if dropped_idx:
            logging.info(
                f"DuplicateHandler: Using keep='{keep_mode}', "
                f"Dropped {len(dropped_idx)} duplicate row(s) (original indices): {dropped_idx}."
            )
        else:
            logging.info(
                f"DuplicateHandler: No duplicate rows dropped (keep='{keep_mode}')."
            )
        return df.drop_duplicates(subset=feature_cols, keep=keep_mode).reset_index(
            drop=True
        )

    def _keep_random(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        rng = np.random.RandomState(self.random_state)
        groups = df.groupby(feature_cols, dropna=False, sort=False)
        kept_indices = []
        for _, g in groups:
            # g.index is the original indices for this group
            pick = rng.randint(len(g))
            kept_indices.append(g.index[pick])
        kept_set = set(kept_indices)
        all_set = set(df.index.tolist())
        dropped = sorted(all_set - kept_set)
        if dropped:
            logging.info(
                "DuplicateHandler: Using keep='random', "
                f"Dropped {len(dropped)} duplicate row(s) (original indices): {dropped}"
            )
        else:
            logging.info("DuplicateHandler: No duplicate rows dropped keep='(random)'.")
        # Keep rows in the same order as they appear in the original df, filtered by kept_set
        kept_in_order = [idx for idx in df.index if idx in kept_set]
        return df.loc[kept_in_order].reset_index(drop=True)

    def _to_numeric_activity(self, s: pd.Series) -> pd.Series:
        out = pd.to_numeric(s, errors="coerce")
        if out.isna().all():
            raise ValueError(
                f"Activity column '{self.activity_col}' is non-numeric in a duplicate group."
            )
        return out

    def _keep_min_max(
        self, df: pd.DataFrame, feature_cols: list[str], keep_mode: str
    ) -> pd.DataFrame:
        # requires activity_col
        if self.activity_col not in df.columns:
            raise ValueError(
                f"activity_col='{self.activity_col}' not found in DataFrame."
            )
        kept_indices = []
        dropped_indices = []
        for _, g in df.groupby(feature_cols, dropna=False, sort=False):
            if len(g) == 1:
                kept_indices.append(g.index[0])
                continue
            vals = self._to_numeric_activity(g[self.activity_col])
            chosen_idx = vals.idxmin() if keep_mode == "min" else vals.idxmax()
            kept_indices.append(chosen_idx)
            dropped_indices.extend([i for i in g.index if i != chosen_idx])
        if dropped_indices:
            logging.info(
                f"DuplicateHandler: Using keep='{keep_mode}', "
                f"Dropped {len(dropped_indices)} duplicate row(s) (original indices): {sorted(dropped_indices)}"
            )
        else:
            logging.info(
                f"DuplicateHandler: No duplicate rows dropped (keep='{keep_mode}')."
            )
        # Preserve original order of kept rows
        kept_in_order = [idx for idx in df.index if idx in set(kept_indices)]
        return df.loc[kept_in_order].reset_index(drop=True)

    def _keep_mean_median(
        self, df: pd.DataFrame, feature_cols: list[str], keep_mode: str
    ) -> pd.DataFrame:
        # requires activity_col
        if self.activity_col not in df.columns:
            raise ValueError(
                f"activity_col='{self.activity_col}' not found in DataFrame."
            )

        kept_indices = []
        agg_map = {}  # map original index -> aggregated activity value
        dropped_indices = []

        # group by feature columns and compute aggregated activity for groups > 1
        for _, g in df.groupby(feature_cols, dropna=False, sort=False):
            if len(g) == 1:
                kept_indices.append(g.index[0])
                continue
            vals = self._to_numeric_activity(g[self.activity_col])
            agg_val = vals.mean() if keep_mode == "mean" else vals.median()
            rep_idx = g.index[0]  # keep the first row as the representative
            kept_indices.append(rep_idx)
            agg_map[rep_idx] = agg_val
            dropped_indices.extend([i for i in g.index if i != rep_idx])

        if dropped_indices:
            logging.info(
                f"DuplicateHandler: Using keep='{keep_mode}', "
                f"Dropped {len(dropped_indices)} duplicate row(s) (original indices): {sorted(dropped_indices)}"
            )
        else:
            logging.info(
                f"DuplicateHandler: No duplicate rows dropped (keep='{keep_mode}')."
            )

        # Select representative rows in original order, copy to preserve dtypes
        kept_in_order = [idx for idx in df.index if idx in set(kept_indices)]
        out_df = df.loc[kept_in_order].copy().reset_index(drop=True)

        # Assign aggregated activity values to the corresponding retained rows
        # Map original index -> new position (since we reset_index above)
        origidx_to_newpos = {
            orig_idx: new_pos for new_pos, orig_idx in enumerate(kept_in_order)
        }
        for orig_idx, agg_val in agg_map.items():
            new_pos = origidx_to_newpos[orig_idx]
            out_df.at[new_pos, self.activity_col] = agg_val

        return out_df

    def _resolve_duplicate_groups(
        self, df: pd.DataFrame, feature_cols: list[str], keep_mode: str
    ) -> pd.DataFrame:
        km = (keep_mode or "").strip().lower()
        if km in {"first", "last"}:
            return self._keep_first_last(df, feature_cols, km)
        if km == "random":
            return self._keep_random(df, feature_cols)
        if km in {"min", "max"}:
            self._require_activity(for_ops=["min", "max"])
            return self._keep_min_max(df, feature_cols, km)
        if km in {"mean", "median"}:
            self._require_activity(for_ops=["mean", "median"])
            return self._keep_mean_median(df, feature_cols, km)
        raise ValueError(f"Unknown keep mode: {keep_mode}")

    def fit(self, data: pd.DataFrame, y=None) -> "DuplicateHandler":
        """
        Fit the handler by identifying duplicate columns.

        :param data: Input DataFrame to inspect for duplicate columns.
        :type data: pandas.DataFrame
        :param y: Ignored. Present for sklearn compatibility.
        :type y: Optional[pandas.Series]
        :return: The fitted DuplicateHandler instance.
        :rtype: DuplicateHandler
        :raises Exception: If an unexpected error occurs during fitting.
        """
        if self.deactivate:
            logging.info("DuplicateHandler is deactivated. Skipping fit.")
            return self

        try:
            temp_data = data.drop(
                columns=[self.id_col, self.activity_col], errors="ignore"
            )
            # Identify duplicate columns (exact equality across all rows)
            self.dup_cols = temp_data.columns[temp_data.T.duplicated()].tolist()
            logging.info(
                f"DuplicateHandler: Identified duplicate columns: {self.dup_cols}"
            )

            if self.save_method:
                if self.save_dir and not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir, exist_ok=True)
                with open(f"{self.save_dir}/duplicate_handler.pkl", "wb") as file:
                    pickle.dump(self, file)
                logging.info(
                    f"DuplicateHandler saved at: {self.save_dir}/duplicate_handler.pkl"
                )

        except Exception as e:
            logging.error(f"An error occurred while fitting: {e}")
            raise

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the DataFrame by removing duplicate rows and columns.

        :param data: Input DataFrame to transform.
        :type data: pandas.DataFrame
        :return: Transformed DataFrame with duplicates removed.
        :rtype: pandas.DataFrame
        :raises ValueError: If a required column is missing.
        :raises Exception: For any unexpected error during transformation.
        """
        if self.deactivate:
            self.transformed_data = data
            logging.info("DuplicateHandler is deactivated. Returning unmodified data.")
            return data

        try:
            df = data.copy()
            # 1) Drop duplicate columns detected in fit (if requested)
            drop_cols = [] if not self.cols else (self.dup_cols or [])
            if drop_cols:
                df = df.drop(columns=drop_cols, errors="ignore")
                logging.info(
                    f"DuplicateHandler: Dropped {len(drop_cols)} duplicate columns: {drop_cols}."
                )
            else:
                logging.info(
                    "DuplicateHandler: Column consolidation disabled. No Columns dropped."
                )

            # 2) Consolidate duplicate rows (if requested)
            if not self.rows:
                out = df.reset_index(drop=True)
                logging.info(
                    "DuplicateHandler: Row consolidation disabled. No rows dropped."
                )
            else:
                # Feature columns define equality of rows (exclude id/activity)
                feature_cols = [
                    c for c in df.columns if c not in {self.id_col, self.activity_col}
                ]

                if len(feature_cols) == 0:
                    logging.warning(
                        "No feature columns left to determine duplicate rows; skipping row consolidation."
                    )
                    out = df.reset_index(drop=True)
                    logging.info("DuplicateHandler: No rows dropped.")

                else:
                    keep_mode = (self.keep or "").strip().lower()
                    out = self._resolve_duplicate_groups(df, feature_cols, keep_mode)

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

                out.to_csv(f"{self.save_dir}/{csv_name}.csv", index=False)
                logging.info(
                    f"DuplicateHandler: Transformed data saved at: "
                    f"{self.save_dir}/{csv_name}.csv"
                )

            self.transformed_data = out

        except KeyError as e:
            logging.error(f"Column missing in the dataframe: {e}")
            raise ValueError(f"Column {e} not found in the dataframe.")

        except Exception as e:
            logging.error(f"An error occurred while transforming the data: {e}")
            raise

        return out

    def fit_transform(self, data: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit the handler and then transform the data.

        :param data: Input DataFrame to fit and transform.
        :type data: pandas.DataFrame
        :param y: Ignored. Present for sklearn compatibility.
        :type y: Optional[pandas.Series]
        :return: Transformed DataFrame with duplicates removed.
        :rtype: pandas.DataFrame
        """
        if self.deactivate:
            logging.info("DuplicateHandler is deactivated. Returning unmodified data.")
            return data

        self.fit(data)
        return self.transform(data)
