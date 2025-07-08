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
    """

    def __init__(self, target_ratio: float = 0.10, *, keep_positives: bool = True):
        if not 0 < target_ratio < 1:
            raise ValueError("target_ratio must be between 0 and 1 (exclusive)")
        self.target_ratio = target_ratio
        self.keep_positives = keep_positives

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def fit_transform(
        self,
        panel: pd.DataFrame,
        *,
        target_col: str = "ever90m12",
        safra_col: str = "safra",
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """Return a new DataFrame with rebalanced target prevalence por safra."""
        rng = np.random.default_rng(random_state)
        pieces: List[pd.DataFrame] = []

        for safra, grp in panel.groupby(safra_col, sort=False):
            pos = grp[grp[target_col] == 1]
            neg = grp[grp[target_col] == 0]

            n_pos = len(pos)
            n_neg = len(neg)
            if n_pos == 0 or n_neg == 0:
                pieces.append(grp)
                continue

            # razão atual
            r = n_pos / (n_pos + n_neg)
            if self.keep_positives:
                if r >= self.target_ratio:
                    # já está >= alvo → mantém como está
                    pieces.append(grp)
                    continue
                # precisamos reduzir negativos
                new_neg_n = int(n_pos * (1 - self.target_ratio) / self.target_ratio)
                sampled_neg_idx = rng.choice(neg.index, size=new_neg_n, replace=False)
                grp_bal = pd.concat([pos, neg.loc[sampled_neg_idx]], ignore_index=True)
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
