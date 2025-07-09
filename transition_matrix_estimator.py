"""Estimate empirical transition matrices from panel data and plot them.

Usage example
--------------
>>> from transition_matrix_estimator import TransitionMatrixLearner
>>> learner = TransitionMatrixLearner(buckets=[0,15,30,60,90])
>>> learner.fit(df_panel,
...              id_col="id_contrato",
...              time_col="data_ref",
...              bucket_col="dias_atraso",
...              group_col="grupo_homogeneo")
>>> learner.plot_heatmaps(["global", "grupo_homogeneo", "stage"])

The method will create one seaborn heatmap per requested modality and return the
list of matplotlib Figure objects (useful for saving in notebooks).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = ["TransitionMatrixLearner"]


class TransitionMatrixLearner:
    """Learn transition matrices and provide quick seaborn visualisation."""

    def __init__(self, *, buckets: List[int], alpha: float = 1.0):
        self.buckets = sorted(buckets)
        self.alpha = alpha  # Laplace smoothing
        self.n = len(self.buckets)
        self._mat_global: np.ndarray | None = None
        self._mat_by_gh: Dict[str, np.ndarray] = {}
        self._mat_by_stage: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    def fit(
        self,
        panel: pd.DataFrame,
        *,
        id_col: str,
        time_col: str,
        bucket_col: str,
        group_col: str | None = None,
    ) -> "TransitionMatrixLearner":
        """Count transitions (t -> t+1 month) per modality and normalise."""
        panel = panel[[id_col, time_col, bucket_col] + ([group_col] if group_col else [])].copy()
        panel[time_col] = pd.to_datetime(panel[time_col])
        panel = panel.sort_values([id_col, time_col])

        def bucket_idx(val: int) -> int:
            idx = np.searchsorted(self.buckets, val, side="right") - 1
            return max(0, min(idx, self.n - 1))

        # create shifted df to align t and t+1
        shifted = panel.copy()
        shifted[time_col] += pd.DateOffset(months=1)
        merged = panel.merge(
            shifted,
            on=[id_col, time_col],
            suffixes=("_t", "_t1"),
            how="inner",
        )

        # global matrix
        self._mat_global = self._count_to_matrix(
            merged[bucket_col + "_t"], merged[bucket_col + "_t1"]
        )

        # by GH
        if group_col:
            for gh, grp in merged.groupby(group_col + "_t"):
                self._mat_by_gh[gh] = self._count_to_matrix(
                    grp[bucket_col + "_t"], grp[bucket_col + "_t1"]
                )

        # by stage (current bucket)
        for idx, grp in merged.groupby(bucket_col + "_t"):
            self._mat_by_stage[int(idx)] = self._count_to_matrix(
                grp[bucket_col + "_t"], grp[bucket_col + "_t1"]
            )
        return self

    # ------------------------------------------------------------------
    def _count_to_matrix(self, col_from: pd.Series, col_to: pd.Series) -> np.ndarray:
        mat = np.zeros((self.n, self.n), dtype=float) + self.alpha  # Laplace
        for src, dst in zip(col_from, col_to):
            i = np.searchsorted(self.buckets, src, side="right") - 1
            j = np.searchsorted(self.buckets, dst, side="right") - 1
            mat[i, j] += 1
        mat /= mat.sum(axis=1, keepdims=True)
        return mat

    # ------------------------------------------------------------------
    def get_matrix(self, *, gh: str | None = None, stage: int | None = None) -> np.ndarray:
        if gh is None and stage is None:
            if self._mat_global is None:
                raise RuntimeError("fit() not called yet")
            return self._mat_global
        if gh is not None:
            return self._mat_by_gh[gh]
        if stage is not None:
            return self._mat_by_stage[stage]
        raise ValueError("Specify either gh or stage (or neither for global)")

    # ------------------------------------------------------------------

    def plot_heatmaps(self, modes: List[str] | None = None) -> List[plt.Figure]:
        """
        Plot heatmaps for requested modalities.

        Parameters
        ----------
        modes : list[str] | None
            Options: "global", "grupo_homogeneo", "stage". Default = ["global"].

        Returns
        -------
        list[matplotlib.figure.Figure]
        """
        if modes is None:
            modes = ["global"]

        figs: List[plt.Figure] = []
        cmap = sns.color_palette("Blues", as_cmap=True)
        xt = yt = [str(b) for b in self.buckets]

        def _prep(mat: np.ndarray, thr: float = 0.5) -> pd.DataFrame:
            """
            Converte a matriz em percentuais e substitui por NaN
            todos os valores abaixo do limiar 'thr' (em pontos-percentuais).
            """
            perc = mat * 100
            perc[perc < thr] = np.nan        # “apaga” zeros (e ≈0) para não aparecerem
            return pd.DataFrame(perc, index=yt, columns=xt)


        # 1. Global
        if "global" in modes:
            df = _prep(self._mat_global)
            mask = df.isna()
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                df,
                mask=df.isna(),   # células < thr ficam brancas
                annot=True,
                fmt=".0f",
                cmap=cmap,
                ax=ax,
                xticklabels=xt,
                yticklabels=yt,
            )
            ax.set_title("Matriz de Transição Global (%)")
            ax.set_xlabel("Bucket Atraso - Próxima Safra")
            ax.set_ylabel("Bucket Atraso - Safra Atual")
            figs.append(fig)

        # 2. Grupo homogêneo
        if "grupo_homogeneo" in modes:
            for gh, mat in self._mat_by_gh.items():
                df = _prep(mat)
                mask = df.isna()
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(
                    df,
                    mask=df.isna(),   # células < thr ficam brancas
                    annot=True,
                    fmt=".0f",
                    cmap=cmap,
                    ax=ax,
                    xticklabels=xt,
                    yticklabels=yt,
                )
                ax.set_title(f"Transition Matrix – {gh} (%)")
                ax.set_xlabel("Next bucket")
                ax.set_ylabel("Current bucket")
                figs.append(fig)

        # 3. Stage atual
        if "stage" in modes:
            for stage, mat in self._mat_by_stage.items():
                df = _prep(mat)
                mask = df.isna()
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(
                    df,
                    mask=df.isna(),   # células < thr ficam brancas
                    annot=True,
                    fmt=".0f",
                    cmap=cmap,
                    ax=ax,
                    xticklabels=xt,
                    yticklabels=yt,
                )
                ax.set_title(f"Transition Matrix – current bucket {stage} (%)")
                ax.set_xlabel("Next bucket")
                ax.set_ylabel("Current bucket")
                figs.append(fig)

        return figs



# # ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     # Tiny example with random data (for ad‑hoc run)
#     ids = np.repeat(np.arange(5), 6)
#     dates = pd.date_range("2020-01-01", periods=6, freq="M").tolist() * 5
#     delays = np.random.choice([0, 15, 30, 60, 90], size=len(ids))
#     df_demo = pd.DataFrame({"id_contrato": ids, "data_ref": dates, "dias_atraso": delays})

#     learner = TransitionMatrixLearner(buckets=[0, 15, 30, 60, 90])
#     learner.fit(df_demo)
#     print("Global matrix:\n", learner.get_matrix())
