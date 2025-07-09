# credit_data_sampler.py
"""Sampler util to rebalance per‑safra target prevalence in the creditlab panel.

Changelog
---------
- v0.2.0  Adiciona ``preserve_gh_order`` para manter a ordenação de risco dos
  grupos homogêneos durante o balanceamento.

Typical use
-----------
>>> from credit_data_sampler import TargetSampler
>>> sampler = TargetSampler(target_ratio=0.10)
>>> panel_balanced = sampler.fit_transform(df_panel,
...                                         target_col="ever90m12",
...                                         safra_col="safra",
...                                         random_state=42)

The sampler keeps **all positive (target==1)** rows in each safra and downsamples
negative rows so that the final prevalence approximates `target_ratio`.
If a safra already exceeds the target ratio, no action is taken (to avoid losing
positives). You can flip the `keep_positives` flag if you need symmetrical
behaviour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, List
import warnings
import logging

__all__ = ["TargetSampler"]


class TargetSampler:
    """Rebalance panel DataFrame to hit a desired per‑safra target ratio.

    Parameters
    ----------
    target_ratio : float, default 0.10
        Desired prevalence (0–1) of *positives* (target == 1) em cada safra.
    keep_positives : bool, default True
        If True, **nunca** remove linhas positivas; apenas downsamples negativas.
        Se False, aplica amostragem simétrica (pode remover positivos ou negativos
        conforme necessário).
    per_group : bool, default False
        Se True, aplica balanceamento dentro de cada ``grupo_homogeneo`` e safra.
    preserve_rank : bool, default True
        Mantém a ordenação de risco dos grupos homogêneos após o sampling.
        ``per_group`` força o balanceamento individual; ``preserve_rank`` apenas
        garante monotonicidade nas bad-rates.
    max_oversample : float, default 2.0
        Fator máximo de replicação permitido para negativos quando
        ``preserve_rank`` está ativo.
    tolerance_pp : float, default 2.0
        Desvio tolerado (em pontos‑percentuais) entre ``target_ratio`` e o valor
        atingido quando o oversampling é limitado.
    min_pos : int, default 5
        Número mínimo de positivos por grupo antes do downsample (com oversampling).
    """

    def __init__(
        self,
        target_ratio: float = 0.10,
        *,
        keep_positives: bool = True,
        per_group: bool = False,
        preserve_rank: bool | None = None,
        max_oversample: float = 2.0,
        tolerance_pp: float = 2.0,
        min_pos: int = 5,
        preserve_gh_order: bool | None = None,
    ):
        if not 0 < target_ratio < 1:
            raise ValueError("target_ratio must be between 0 and 1 (exclusive)")
        self.target_ratio = target_ratio
        self.keep_positives = keep_positives
        self.per_group = per_group
        if preserve_rank is None and preserve_gh_order is not None:
            preserve_rank = preserve_gh_order
        self.preserve_rank = True if preserve_rank is None else bool(preserve_rank)
        self.max_oversample = float(max_oversample)
        self.tolerance_pp = float(tolerance_pp)
        self.preserve_gh_order = self.preserve_rank  # backward compat
        self.min_pos = min_pos

    # ------------------------------------------------------------------
    def _jitter(self, df: pd.DataFrame, target_col: str, rng: np.random.Generator) -> None:
        num_cols = df.select_dtypes(include=["float", "int"]).columns.difference([target_col])
        for col in num_cols:
            std = df[col].std() if df[col].std() > 0 else 1
            df[col] += rng.normal(0, std * 0.01, size=len(df))

    def _sample_negatives(
        self,
        neg: pd.DataFrame,
        new_size: int,
        target_col: str,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        cur = len(neg)
        if cur >= new_size:
            idx = rng.choice(neg.index, size=new_size, replace=False)
            return neg.loc[idx]
        extra_idx = rng.choice(neg.index, size=new_size - cur, replace=True)
        extra = neg.loc[extra_idx].copy()
        self._jitter(extra, target_col, rng)
        return pd.concat([neg, extra], ignore_index=True)

    def _balance_group_monotone(
        self,
        grp: pd.DataFrame,
        *,
        target_col: str,
        prev_rate: float | None,
        rng: np.random.Generator,
    ) -> Tuple[pd.DataFrame, float]:
        pos = grp[grp[target_col] == 1]
        neg = grp[grp[target_col] == 0]

        n_pos = len(pos)
        n_neg = len(neg)
        if n_pos < self.min_pos and n_pos > 0:
            extra = pos.sample(self.min_pos - n_pos, replace=True, random_state=rng.integers(0, 1_000_000))
            self._jitter(extra, target_col, rng)
            pos = pd.concat([pos, extra], ignore_index=True)
            n_pos = len(pos)

        if n_pos == 0 or n_neg == 0:
            rate = n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0.0
            return grp, rate

        desired_neg = int(np.round(n_pos * (1 - self.target_ratio) / self.target_ratio))
        max_neg = int(n_neg * self.max_oversample)
        if prev_rate is not None:
            req_neg = int(np.ceil(n_pos * (1 - prev_rate) / prev_rate)) if prev_rate > 0 else max_neg
            desired_neg = max(desired_neg, req_neg)

        capped = False
        if desired_neg > max_neg:
            capped = True
            desired_neg = max_neg

        neg_bal = self._sample_negatives(neg, desired_neg, target_col, rng)
        grp_bal = pd.concat([pos, neg_bal], ignore_index=True)
        rate = len(pos) / len(grp_bal)
        if capped and abs(rate - self.target_ratio) > self.tolerance_pp / 100:
            import logging
            logging.warning(
                "Target ratio deviates %.1f pp in group", (rate - self.target_ratio) * 100
            )
        return grp_bal, rate


    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def fit_transform(
        self,
        panel: pd.DataFrame,
        *,
        target_col: str = "ever90m12",
        safra_col: str = "safra",
        group_col: str = "grupo_homogeneo",
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """Return a new DataFrame with rebalanced target prevalence por safra."""
        rng = np.random.default_rng(random_state)
        pieces: List[pd.DataFrame] = []

        if self.preserve_rank:
            for safra, df_safra in panel.groupby(safra_col, sort=False):
                order = (
                    df_safra.groupby(group_col)[target_col]
                    .mean()
                    .sort_values(ascending=False)
                    .index
                )
                prev = None
                bal_groups: List[pd.DataFrame] = []
                for gh in order:
                    grp = df_safra[df_safra[group_col] == gh]
                    bal, prev = self._balance_group_monotone(
                        grp,
                        target_col=target_col,
                        prev_rate=prev,
                        rng=rng,
                    )
                    bal_groups.append(bal)
                cat = pd.CategoricalDtype(order, ordered=True)
                for df_b in bal_groups:
                    df_b[group_col] = df_b[group_col].astype(cat)
                df_bal = pd.concat(bal_groups, ignore_index=True).sort_values(group_col)
                pieces.append(df_bal)
        else:
            group_keys = [safra_col]
            if self.per_group:
                group_keys.append(group_col)

            for _, grp in panel.groupby(group_keys, sort=False):
                pos = grp[grp[target_col] == 1]
                neg = grp[grp[target_col] == 0]

                n_pos = len(pos)
                n_neg = len(neg)

                if n_pos < self.min_pos and n_pos > 0:
                    extra = pos.sample(
                        self.min_pos - n_pos,
                        replace=True,
                        random_state=rng.integers(0, 1_000_000),
                    )
                    self._jitter(extra, target_col, rng)
                    pos = pd.concat([pos, extra], ignore_index=True)
                    n_pos = len(pos)

                if n_pos == 0 or n_neg == 0:
                    pieces.append(grp)
                    continue

                if self.keep_positives:
                    new_neg_n = int(n_pos * (1 - self.target_ratio) / self.target_ratio)
                    neg = self._sample_negatives(neg, new_neg_n, target_col, rng)
                    grp_bal = pd.concat([pos, neg], ignore_index=True)
                else:
                    total_desired = int(n_pos + n_neg)
                    n_pos_desired = int(total_desired * self.target_ratio)
                    n_neg_desired = total_desired - n_pos_desired
                    if n_pos > n_pos_desired:
                        pos_idx = rng.choice(pos.index, size=n_pos_desired, replace=False)
                        pos = pos.loc[pos_idx]
                    if n_neg > n_neg_desired:
                        neg_idx = rng.choice(neg.index, size=n_neg_desired, replace=False)
                        neg = neg.loc[neg_idx]
                    grp_bal = pd.concat([pos, neg], ignore_index=True)

                pieces.append(grp_bal)

        balanced = pd.concat(pieces, ignore_index=True)
        # reorder rows by original order of safras to keep panel temporality
        balanced = balanced.sort_values([safra_col, "id_contrato", "data_ref"], kind="mergesort")
        balanced.reset_index(drop=True, inplace=True)
        return balanced


# ----------------------------------------------------------------------
# Quick sanity check ----------------------------------------------------
# ----------------------------------------------------------------------
# if __name__ == "__main__":
#     # pequeno teste rápido utilizando gerador sintético (se disponível)
#     try:
#         from credit_data_synthesizer import default_group_profiles, CreditDataSynthesizer

#         synth = CreditDataSynthesizer(
#             group_profiles=default_group_profiles(3),
#             contracts_per_group=3_000,
#             n_safras=12,
#             random_seed=0,
#         )
#         _, panel, _ = synth.generate()
#         sampler = TargetSampler(0.10)
#         balanced = sampler.fit_transform(panel, target_col="ever90m12")

#         prev = balanced.groupby("safra")["ever90m12"].mean()
#         print("Prev per safra (balanced):")
#         print(prev.head())
#     except ImportError:
#         # GitHub CI sem dependência do synthesizer ‑ ignora
#         pass
