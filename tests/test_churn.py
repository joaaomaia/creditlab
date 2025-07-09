import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from credit_data_synthesizer import CreditDataSynthesizer, default_group_profiles
from credit_data_sampler import TargetSampler


def make_synth():
    return CreditDataSynthesizer(
        group_profiles=default_group_profiles(2),
        contracts_per_group=40,
        n_safras=12,
        random_seed=0,
        kernel_trick=False,
    )


def test_start_leq_dataref():
    synth = make_synth()
    _, panel, _ = synth.generate()
    assert (panel["data_inicio_contrato"] <= panel["data_ref"]).all()


def test_one_active_contract_per_client():
    synth = make_synth()
    _, panel, _ = synth.generate()
    assert not panel.duplicated(["id_cliente", "safra"]).any()


def test_preserve_rank_after_sampling():
    synth = make_synth()
    _, panel, _ = synth.generate()
    sampler = TargetSampler(target_ratio=0.08, preserve_rank=True, max_oversample=10)
    bal = sampler.fit_transform(panel, target_col="ever90m12", safra_col="safra", group_col="grupo_homogeneo", random_state=1)
    for _, df in bal.groupby("safra"):
        rates = df.groupby("grupo_homogeneo")["ever90m12"].mean().values
        assert np.all(np.diff(rates) <= 1e-3)


def test_churn_rates():
    synth = CreditDataSynthesizer(
        group_profiles=default_group_profiles(1),
        contracts_per_group=50,
        n_safras=13,
        random_seed=1,
        kernel_trick=False,
        new_contract_rate=0.05,
        closure_rate=0.03,
    )
    _, panel, _ = synth.generate()
    closed = synth.closed.groupby("safra")["id_contrato"].nunique()
    open_counts = panel.groupby("safra")["id_contrato"].nunique()
    prev_open = open_counts.shift(1).fillna(open_counts.iloc[0])
    new_counts = open_counts - (prev_open - closed.reindex(open_counts.index, fill_value=0))
    rate_new = (new_counts.iloc[1:] / prev_open.iloc[:-1]).mean()
    rate_close = (closed.reindex(open_counts.index, fill_value=0).iloc[1:] / prev_open.iloc[:-1]).mean()
    assert abs(rate_new - synth.new_contract_rate) < synth.new_contract_rate * 0.22
    assert abs(rate_close - synth.closure_rate) < synth.closure_rate * 0.22
