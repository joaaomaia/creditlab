"""Ever-Cure Matrix estimator for credit risk analysis.

This module learns the probability of rollforward and rollback (cure) within
an observation horizon. It mirrors the API of ``TransitionMatrixLearner`` for
consistency.

Example
-------
>>> from ever_cure_matrix import EverCureMatrixLearner
>>> learner = EverCureMatrixLearner(buckets=[0, 15, 30, 60, 90])
>>> learner.fit(df_panel,
...             id_col="id_contrato",
...             time_col="data_ref",
...             bucket_col="dias_atraso")
>>> learner.plot_heatmaps()
"""
from __future__ import annotations

from typing import Dict, List, Optional
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger("creditlab.tm")
logger.setLevel(logging.INFO)

__all__ = ["EverCureMatrixLearner"]


class EverCureMatrixLearner:
    """Learn ever-cure matrices and plot seaborn heatmaps."""

    def __init__(self, *, buckets: List[int], assert_monotonic: bool = False) -> None:
        self.buckets = sorted(buckets)
        self.n = len(self.buckets)
        self.assert_monotonic = assert_monotonic
        self.horizon = 12
        self.group_col: Optional[str] = None
        # matrices with roll columns already appended
        self._counts_global: np.ndarray | None = None
        self._perc_global: np.ndarray | None = None
        self._counts_by_gh: Dict[str, np.ndarray] = {}
        self._perc_by_gh: Dict[str, np.ndarray] = {}
        self._counts_by_stage: Dict[int, np.ndarray] = {}
        self._perc_by_stage: Dict[int, np.ndarray] = {}
        self.logger = logger.getChild(self.__class__.__name__)

    # ------------------------------------------------------------------
    def fit(
        self,
        panel: pd.DataFrame,
        *,
        id_col: str,
        time_col: str,
        bucket_col: str,
        horizon: int = 12,
        group_col: str | None = None,
    ) -> "EverCureMatrixLearner":
        """Calculate ever-cure matrices from panel data."""
        self.horizon = int(horizon)
        self.group_col = group_col
        df = self._prepare_panel(panel, id_col, time_col, bucket_col, group_col)
        self._build_matrices(df)
        return self

    # ------------------------------------------------------------------
    def _prepare_panel(
        self,
        panel: pd.DataFrame,
        id_col: str,
        time_col: str,
        bucket_col: str,
        group_col: str | None,
    ) -> pd.DataFrame:
        cols = [id_col, time_col, bucket_col] + ([group_col] if group_col else [])
        df = panel[cols].copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values([id_col, time_col])

        parts = []
        for _, grp in df.groupby(id_col):
            s = grp[bucket_col].reset_index(drop=True)
            rev = s.iloc[::-1]
            fmax = rev.rolling(self.horizon, min_periods=1).max().iloc[::-1].shift(-1)
            fmin = rev.rolling(self.horizon, min_periods=1).min().iloc[::-1].shift(-1)
            sub = grp.copy()
            sub["future_max"] = fmax.values
            sub["future_min"] = fmin.values
            parts.append(sub)
        df = pd.concat(parts)
        df = df.dropna(subset=["future_max", "future_min"])

        df["bucket_idx"] = np.searchsorted(self.buckets, df[bucket_col], side="right") - 1
        df["future_max_idx"] = np.searchsorted(
            self.buckets, df["future_max"], side="right"
        ) - 1
        df["future_min_idx"] = np.searchsorted(
            self.buckets, df["future_min"], side="right"
        ) - 1
        return df

    # ------------------------------------------------------------------
    def _event_matrix(self, df: pd.DataFrame) -> np.ndarray:
        mat = np.zeros((self.n, self.n), dtype=float)
        for row in df.itertuples(index=False):
            i = int(row.bucket_idx)
            max_idx = int(row.future_max_idx)
            min_idx = int(row.future_min_idx)
            if max_idx > i:
                mat[i, i + 1 : max_idx + 1] += 1
            if min_idx < i:
                start = max(min_idx, 0)
                mat[i, start:i] += 1
        return mat

    # ------------------------------------------------------------------
    def _percent_matrix(self, counts: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        perc = np.full_like(counts, np.nan, dtype=float)
        row_counts = (
            df.groupby("bucket_idx").size().reindex(range(self.n), fill_value=0).to_numpy()
        )
        for i in range(self.n):
            if row_counts[i] > 0:
                perc[i] = counts[i] / row_counts[i] * 100.0
        return perc

    # ------------------------------------------------------------------
    def _add_roll_cols(self, mat: np.ndarray) -> np.ndarray:
        ext = np.zeros((self.n, self.n + 2), dtype=float)
        ext[:, : self.n] = mat
        for i in range(self.n):
            ext[i, self.n] = np.nansum(mat[i, :i])
            ext[i, self.n + 1] = np.nansum(mat[i, i + 1 :])
        return ext

    # ------------------------------------------------------------------
    def _assert_monotone(self, mat: np.ndarray) -> None:
        for i in range(self.n):
            for j in range(i + 1, self.n - 1):
                if not np.isnan(mat[i, j]) and not np.isnan(mat[i, j + 1]):
                    assert mat[i, j] >= mat[i, j + 1]
            for j in range(i - 1, 0, -1):
                if not np.isnan(mat[i, j]) and not np.isnan(mat[i, j - 1]):
                    assert mat[i, j] >= mat[i, j - 1]

    # ------------------------------------------------------------------
    def _build_matrices(self, df: pd.DataFrame) -> None:
        counts = self._event_matrix(df)
        perc = self._percent_matrix(counts, df)
        if self.assert_monotonic:
            self._assert_monotone(perc)
        self._counts_global = self._add_roll_cols(counts)
        self._perc_global = self._add_roll_cols(perc)

        if self.group_col:
            for gh, grp in df.groupby(self.group_col):
                c = self._event_matrix(grp)
                p = self._percent_matrix(c, grp)
                self._counts_by_gh[gh] = self._add_roll_cols(c)
                self._perc_by_gh[gh] = self._add_roll_cols(p)

        for idx in range(self.n):
            stage_val = self.buckets[idx]
            grp = df[df["bucket_idx"] == idx]
            if grp.empty:
                continue
            c = self._event_matrix(grp)
            p = self._percent_matrix(c, grp)
            self._counts_by_stage[stage_val] = self._add_roll_cols(c)
            self._perc_by_stage[stage_val] = self._add_roll_cols(p)

    # ------------------------------------------------------------------
    def get_matrix(
        self, kind: str = "percent", gh: str | None = None, stage: int | None = None
    ) -> np.ndarray:
        if kind not in {"percent", "counts"}:
            raise ValueError("kind must be 'percent' or 'counts'")
        src_global = self._perc_global if kind == "percent" else self._counts_global
        src_by_gh = self._perc_by_gh if kind == "percent" else self._counts_by_gh
        src_by_stage = self._perc_by_stage if kind == "percent" else self._counts_by_stage

        if gh is None and stage is None:
            if src_global is None:
                raise RuntimeError("fit() not called yet")
            return src_global
        if gh is not None:
            return src_by_gh[gh]
        if stage is not None:
            return src_by_stage[stage]
        raise ValueError("Specify either gh or stage (or neither for global)")

    # ------------------------------------------------------------------
    def plot_heatmaps(
        self, modes: List[str] | None = None, thr_pct: float = 0.5, decimals: int = 1
    ) -> List[plt.Figure]:
        if modes is None:
            modes = ["global"]

        xt = [str(b) for b in self.buckets] + ["Rollback %", "Rollforward %"]
        yt = [str(b) for b in self.buckets]
        cmap = sns.color_palette("Blues", as_cmap=True)

        figs: List[plt.Figure] = []

        def _plot(mat_p: np.ndarray, mat_c: np.ndarray, title: str) -> None:
            df = pd.DataFrame(mat_p, index=yt, columns=xt)
            annot = np.empty_like(df, dtype=object)
            mask = (df < thr_pct) | df.isna()
            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    val = df.iat[i, j]
                    cnt = mat_c[i, j]
                    if np.isnan(val) or val < thr_pct:
                        annot[i, j] = ""
                    else:
                        annot[i, j] = f"{val:.{decimals}f}%\n{int(cnt):,}".replace(",", " ")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                df,
                mask=mask,
                annot=annot,
                fmt="",
                cmap=cmap,
                ax=ax,
                xticklabels=xt,
                yticklabels=yt,
            )
            ax.set_xlabel("Target bucket")
            ax.set_ylabel("Current bucket")
            ax.set_title(title)
            figs.append(fig)

        if "global" in modes:
            _plot(
                self.get_matrix("percent"),
                self.get_matrix("counts"),
                f"Ever-Cure Matrix – global (h={self.horizon})",
            )

        if "grupo_homogeneo" in modes and self.group_col:
            for gh in sorted(self._perc_by_gh):
                _plot(
                    self.get_matrix("percent", gh=gh),
                    self.get_matrix("counts", gh=gh),
                    f"Ever-Cure – {gh} (h={self.horizon})",
                )

        if "stage" in modes:
            for stage in sorted(self._perc_by_stage):
                _plot(
                    self.get_matrix("percent", stage=stage),
                    self.get_matrix("counts", stage=stage),
                    f"Ever-Cure – bucket {stage} (h={self.horizon})",
                )

        return figs


if __name__ == "__main__":
    # Quick demo with random panel
    ids = np.repeat(np.arange(5), 24)
    dates = pd.date_range("2020-01-01", periods=24, freq="M").tolist() * 5
    delays = np.random.choice([0, 15, 30, 60, 90], size=len(ids))
    demo = pd.DataFrame({"id": ids, "ref": dates, "bucket": delays})

    ecm = EverCureMatrixLearner(buckets=[0, 15, 30, 60, 90])
    ecm.fit(demo, id_col="id", time_col="ref", bucket_col="bucket")
    ecm.plot_heatmaps()
    plt.show()
