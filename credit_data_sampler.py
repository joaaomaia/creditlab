# credit_data_sampler.py
"""Sampler util to rebalance per‑safra target prevalence in the creditlab panel.

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
    min_pos : int, default 5
        Número mínimo de positivos por grupo antes do downsample (com oversampling).
    """

    def __init__(
        self,
        target_ratio: float = 0.10,
        *,
        keep_positives: bool = True,
        per_group: bool = False,
        min_pos: int = 5,
    ):
        if not 0 < target_ratio < 1:
            raise ValueError("target_ratio must be between 0 and 1 (exclusive)")
        self.target_ratio = target_ratio
        self.keep_positives = keep_positives
        self.per_group = per_group
        self.min_pos = min_pos

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
        group_keys = [safra_col]
        if self.per_group:
            group_keys.append(group_col)

        for keys, grp in panel.groupby(group_keys, sort=False):
            pos = grp[grp[target_col] == 1]
            neg = grp[grp[target_col] == 0]

            n_pos = len(pos)
            n_neg = len(neg)
            if n_pos < self.min_pos and n_pos > 0:
                extra = pos.sample(self.min_pos - n_pos, replace=True, random_state=rng.integers(0, 1_000_000))
                num_cols = extra.select_dtypes(include=["float", "int"]).columns.difference([target_col])
                for col in num_cols:
                    std = extra[col].std() if extra[col].std() > 0 else 1
                    extra[col] += rng.normal(0, std * 0.01, size=len(extra))
                pos = pd.concat([pos, extra], ignore_index=True)
                n_pos = len(pos)

            if n_pos == 0 or n_neg == 0:
                pieces.append(grp)
                continue

            # razão atual
            r = n_pos / (n_pos + n_neg)
            if self.keep_positives:
                new_neg_n = int(n_pos * (1 - self.target_ratio) / self.target_ratio)
                if n_neg >= new_neg_n:
                    sampled_neg_idx = rng.choice(neg.index, size=new_neg_n, replace=False)
                    neg = neg.loc[sampled_neg_idx]
                else:
                    extra_idx = rng.choice(neg.index, size=new_neg_n - n_neg, replace=True)
                    extra = neg.loc[extra_idx].copy()
                    num_cols = extra.select_dtypes(include=["float", "int"]).columns.difference([target_col])
                    for col in num_cols:
                        std = extra[col].std() if extra[col].std() > 0 else 1
                        extra[col] += rng.normal(0, std * 0.01, size=len(extra))
                    neg = pd.concat([neg, extra], ignore_index=True)
                grp_bal = pd.concat([pos, neg], ignore_index=True)
            else:
                # amostragem simétrica (pode reduzir qualquer lado)
                total_desired = int((n_pos + n_neg))  # preserva tamanho
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
