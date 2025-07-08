# credit_data_synthesizer.py
"""Synthetic credit data generator for mentorship use.

Features
========
- Supports multiple *GroupProfiles* (homogeneous risk segments).
- Generates an initial snapshot (safra 0) and evolves it into a monthly panel.
- Handles: days in arrears logic, refinancing (good behaviour), renegotiation (bad behaviour),
  latent risk score, kernel‑based stratification (quantile bucketing), macro‑stress shocks,
  and moving‑window targets (ever90m12, over90m12).
- Outputs three pandas DataFrames: `snapshot`, `panel`, `trace`.

Usage
-----
>>> from credit_data_synthesizer import CreditDataSynthesizer, default_group_profiles
>>> synth = CreditDataSynthesizer(default_group_profiles, contracts_per_group=10_000, n_safras=24)
>>> snapshot, panel, trace = synth.generate()

The code is kept self‑contained (only depends on `pandas` and `numpy`).
"""

from __future__ import annotations

import itertools
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

###############################################################################
# Helper dataclass describing a risk segment
###############################################################################

@dataclass
class GroupProfile:
    """Blueprint for a homogeneous risk group."""

    name: str
    pd12m_base: float  # baseline 12‑month PD
    curing_rate: float  # monthly probability to cure when in arrears 1‑89d
    p_reneg_base: float  # monthly renegotiation probability (exogenous shock)
    p_refin_good: float  # probability of refinancing once eligible

    # Distributions for contract & customer attributes. Each entry maps a column
    # name to a callable `f(size) -> ndarray`.
    dist_generators: Dict[str, Any] = field(default_factory=dict)

    def sample_contracts(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        """Return a DataFrame with *n* synthetic contracts following group stats."""
        cols = {
            'id_contrato': np.arange(n, dtype='int64'),
            'id_cliente': rng.integers(1e9, 2e9, size=n, dtype='int64'),
            'grupo_homogeneo': self.name,
            'nivel_refinanciamento': np.zeros(n, dtype='int8'),
            # start date randomised within last year for variety
            'data_inicio_contrato': pd.to_datetime('today') - pd.to_timedelta(rng.integers(0, 365, n), unit='D'),
            'dias_atraso': rng.choice([0, 15, 30, 60, 90], p=self._delay_probs(), size=n),
        }

        # Generate feature columns
        for col, gen in self.dist_generators.items():
            cols[col] = gen(n)

        return pd.DataFrame(cols)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _delay_probs(self) -> List[float]:
        """Crude distribution of initial arrears."""
        # Worse groups have heavier right tail; scale by baseline PD
        tail = min(self.pd12m_base * 2, 0.5)
        p0 = max(0.5 - tail, 0.1)
        return [p0, tail * 0.25, tail * 0.25, tail * 0.15, tail * 0.10]


###############################################################################
# Main synthesizer
###############################################################################

class CreditDataSynthesizer:
    """Factory for synthetic credit‑risk datasets."""

    def __init__(
        self,
        group_profiles: List[GroupProfile],
        contracts_per_group: int = 10_000,
        n_safras: int = 24,
        random_seed: int = 42,
        target_tolerance_pp: float = 1.0,
        min_parcels_refin: int = 4,
        stress_shocks: Dict[int, float] | None = None,  # {safra_idx: pd_delta}
        kernel_trick: bool = True,
    ) -> None:
        self.group_profiles = group_profiles
        self.contracts_per_group = contracts_per_group
        self.n_safras = n_safras
        self.rng = np.random.default_rng(random_seed)
        self.target_tolerance_pp = target_tolerance_pp / 100  # convert to decimal
        self.min_parcels_refin = min_parcels_refin
        self.stress_shocks = stress_shocks or {}
        self.kernel_trick = kernel_trick

        # Placeholders for outputs
        self.snapshot: pd.DataFrame | None = None
        self.panel: pd.DataFrame | None = None
        self.trace: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate snapshot, panel and renegotiation trace."""
        self._generate_snapshot()
        self._generate_panel()
        self._apply_targets()
        return self.snapshot.copy(), self.panel.copy(), self.trace.copy()

    # ------------------------------------------------------------------
    # Private building blocks
    # ------------------------------------------------------------------

    def _generate_snapshot(self) -> None:
        """Create initial snapshot (safra 0) for all group profiles."""
        records = []
        for g_idx, gp in enumerate(self.group_profiles):
            df = gp.sample_contracts(self.contracts_per_group, self.rng)
            # Kernel‑trick: quantile bucketing on renda & idade para estratificar
            if self.kernel_trick and {'renda_mensal', 'idade_cliente'} <= set(df.columns):
                q_renda = pd.qcut(df['renda_mensal'], q=4, labels=False)
                q_idade = pd.qcut(df['idade_cliente'], q=4, labels=False)
                df['subcluster'] = (q_renda.astype('int8') << 2) + q_idade.astype('int8')
            records.append(df)
        snapshot = pd.concat(records, ignore_index=True)
        snapshot['safra_idx'] = 0
        snapshot['data_ref'] = pd.Timestamp('today').normalize()  # T0
        # latent risk score ∈ [0,1] drawn ~Beta(a,b) scaled by group PD
        snapshot['latent_risk_score'] = self._sample_latent_risk(snapshot)
        self.snapshot = snapshot
        # Init trace & internal counters
        self.trace = pd.DataFrame(columns=['id_contrato_old', 'id_contrato_new', 'data_evento', 'motivo'])
        self._next_contract_id = snapshot['id_contrato'].max() + 1

    # .................................................................
    def _generate_panel(self) -> None:
        """Iteratively evolve contracts month‑by‑month."""
        if self.snapshot is None:
            raise RuntimeError('Call _generate_snapshot first.')

        current = self.snapshot.copy()
        panel = [current]

        for m in range(1, self.n_safras):
            current = self._evolve_one_month(current, m)
            panel.append(current)

        self.panel = pd.concat(panel, ignore_index=True)

    # .................................................................
    def _evolve_one_month(self, df_prev: pd.DataFrame, m: int) -> pd.DataFrame:
        """Return DataFrame for safra *m* evolved from previous month."""
        df = df_prev.copy()
        gp_map = {gp.name: gp for gp in self.group_profiles}

        # 1) Update data_ref & safra
        df['safra_idx'] = m
        df['data_ref'] = df['data_ref'] + pd.DateOffset(months=1)

        # 2) Days in arrears Markov step & cure
        for g_name, sub in df.groupby('grupo_homogeneo'):
            gp = gp_map[g_name]
            mask = df['grupo_homogeneo'] == g_name
            days = df.loc[mask, 'dias_atraso'].values
            cured = (days > 0) & (self.rng.random(len(days)) < gp.curing_rate)
            days[cured] = 0
            # progression probabilities (simplistic)
            progress = (days < 90) & (self.rng.random(len(days)) < gp.pd12m_base / 12)
            days[progress] += 30  # jump 0→30→60→90
            df.loc[mask, 'dias_atraso'] = np.clip(days, 0, 120)

        # 3) Refinancing rule (good behaviour)
        elig_refin = (df['dias_atraso'] == 0) & (df['nivel_refinanciamento'] < 3)
        elig_refin &= self.rng.random(len(df)) < df['grupo_homogeneo'].map(lambda g: gp_map[g].p_refin_good)
        df.loc[elig_refin, 'nivel_refinanciamento'] += 1
        # (could increase saldo_devedor etc.)

        # 4) Renegotiation events (bad)
        reneg_mask = (df['dias_atraso'] >= 90) | (
            self.rng.random(len(df)) < df['grupo_homogeneo'].map(lambda g: gp_map[g].p_reneg_base)
        )
        df_reneg = df[reneg_mask]
        if not df_reneg.empty:
            new_ids = np.arange(self._next_contract_id, self._next_contract_id + len(df_reneg))
            self._next_contract_id += len(df_reneg)
            # record trace
            self.trace = pd.concat([
                self.trace,
                pd.DataFrame({
                    'id_contrato_old': df_reneg['id_contrato'].values,
                    'id_contrato_new': new_ids,
                    'data_evento': df_reneg['data_ref'].values,
                    'motivo': 'renegociacao',
                })
            ])
            # reset attributes for new contracts
            df.loc[reneg_mask, 'id_contrato'] = new_ids
            df.loc[reneg_mask, 'dias_atraso'] = 0
            df.loc[reneg_mask, 'nivel_refinanciamento'] = 0

        # 5) Macro‑stress shock adjustment to baseline PD in this safra
        shock = self.stress_shocks.get(m, 0.0)
        if shock:
            df['pd12m_effective'] = df['grupo_homogeneo'].map(lambda g: gp_map[g].pd12m_base * (1 + shock))
        else:
            df['pd12m_effective'] = df['grupo_homogeneo'].map(lambda g: gp_map[g].pd12m_base)

        # 6) Latent risk drift (random walk bounded [0,1])
        drift = self.rng.normal(0, 0.02, size=len(df))
        df['latent_risk_score'] = np.clip(df['latent_risk_score'] + drift, 0, 1)

        return df

    # .................................................................
    def _apply_targets(self) -> None:
        """Label ever90m12 and over90m12 using sliding window."""
        if self.panel is None:
            raise RuntimeError('Call _generate_panel first.')
        panel = self.panel.sort_values(['id_contrato', 'safra_idx']).copy()

        panel['ever90m12'] = 0
        panel['over90m12'] = 0

        # We compute forward‑looking flags contract‑wise
        for cid, sub in panel.groupby('id_contrato'):
            arrears = sub['dias_atraso'].values
            reneg_flags = sub['id_contrato'].duplicated().astype(int).values  # after reneg a new id, so dup False.
            ever = np.zeros(len(sub), dtype=int)
            over = np.zeros(len(sub), dtype=int)
            for i in range(len(sub)):
                window = arrears[i:i+12]
                if (window >= 90).any() or (reneg_flags[i:i+12] == 1).any():
                    ever[i] = 1
                # over: must hit ≥90 and never drop below 30 afterwards
                if (window >= 90).any() and (window < 30).sum() == 0:
                    over[i] = 1
            panel.loc[sub.index, 'ever90m12'] = ever
            panel.loc[sub.index, 'over90m12'] = over

        self.panel = panel

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _sample_latent_risk(self, df: pd.DataFrame) -> np.ndarray:
        """Sample latent risk ~ Beta scaled by group PD."""
        scores = np.empty(len(df))
        for g, sub in df.groupby('grupo_homogeneo'):
            gp = next(p for p in self.group_profiles if p.name == g)
            a = max(0.5, 5 * gp.pd12m_base)  # higher risk → heavier right tail
            b = 5
            scores[sub.index] = self.rng.beta(a, b, size=len(sub))
        return scores

###############################################################################
# Convenience: default demo profiles
###############################################################################

def _normal_int(mu: float, sigma: float, low: int, high: int):
    return lambda n: np.clip(np.round(np.random.default_rng().normal(mu, sigma, size=n)), low, high).astype(int)

def default_group_profiles() -> List[GroupProfile]:
    """Return three sample group profiles."""
    profiles = [
        GroupProfile(
            name='G_pior',
            pd12m_base=0.12,
            curing_rate=0.05,
            p_reneg_base=0.02,
            p_refin_good=0.05,
            dist_generators={
                'idade_cliente': _normal_int(35, 8, 18, 70),
                'renda_mensal': lambda n: np.random.default_rng().lognormal(mean=8, sigma=0.5, size=n),
            },
        ),
        GroupProfile(
            name='G_medio',
            pd12m_base=0.08,
            curing_rate=0.07,
            p_reneg_base=0.015,
            p_refin_good=0.1,
            dist_generators={
                'idade_cliente': _normal_int(40, 7, 18, 70),
                'renda_mensal': lambda n: np.random.default_rng().lognormal(mean=9, sigma=0.4, size=n),
            },
        ),
        GroupProfile(
            name='G_bom',
            pd12m_base=0.05,
            curing_rate=0.10,
            p_reneg_base=0.01,
            p_refin_good=0.2,
            dist_generators={
                'idade_cliente': _normal_int(45, 6, 25, 70),
                'renda_mensal': lambda n: np.random.default_rng().lognormal(mean=10, sigma=0.3, size=n),
            },
        ),
    ]
    return profiles

# ###############################################################################
# # Script usage guard
# ###############################################################################

# if __name__ == "__main__":
#     synth = CreditDataSynthesizer(default_group_profiles(), contracts_per_group=1_000, n_safras=12)
#     snap, panel, trace = synth.generate()
#     print(snap.head())
#     print(panel.head())
#     print(trace.head())
