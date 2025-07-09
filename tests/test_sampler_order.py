import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from credit_data_synthesizer import CreditDataSynthesizer, default_group_profiles
from credit_data_sampler import TargetSampler


def make_panel():
    synth = CreditDataSynthesizer(
        group_profiles=default_group_profiles(3),
        contracts_per_group=60,
        n_safras=6,
        random_seed=3,
        kernel_trick=False,
    )
    _, panel, _ = synth.generate()
    return panel


def test_gh_monotone_after_sampling():
    panel = make_panel()
    sampler = TargetSampler(target_ratio=0.10, preserve_rank=True, max_oversample=10)
    bal = sampler.fit_transform(
        panel,
        target_col="ever90m12",
        safra_col="safra",
        group_col="grupo_homogeneo",
        random_state=1,
    )
    for safra, df in bal.groupby("safra"):
        rates = df.groupby("grupo_homogeneo")["ever90m12"].mean().sort_index().values
        assert np.all(np.diff(rates) <= 0)


def test_volume_change_bounds():
    panel = make_panel()
    sampler = TargetSampler(target_ratio=0.10, preserve_rank=True, max_oversample=10)
    bal = sampler.fit_transform(
        panel,
        target_col="ever90m12",
        safra_col="safra",
        group_col="grupo_homogeneo",
        random_state=2,
    )
    orig = panel.groupby("safra").size()
    new = bal.groupby("safra").size()
    lower = (orig * 0.5).astype(int)
    upper = (orig * 2.5).astype(int)
    assert ((new >= lower) & (new <= upper)).all()
