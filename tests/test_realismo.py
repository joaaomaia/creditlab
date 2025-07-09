import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
from credit_data_synthesizer import (
    CreditDataSynthesizer,
    default_group_profiles,
    GroupProfile,
)
from credit_data_sampler import TargetSampler


def test_transition_matrix_rowsum():
    profiles = default_group_profiles(3)
    for gp in profiles:
        assert np.allclose(gp.transition_matrix.sum(axis=1), 1.0, atol=1e-3)


def test_targets_include_reneg():
    tm = np.zeros((5, 5))
    tm[:, 3] = 1.0
    gp = GroupProfile(name="G", pd_base=0.12, refin_prob=0.0, reneg_prob_exog=1.0, transition_matrix=tm)
    synth = CreditDataSynthesizer(group_profiles=[gp], contracts_per_group=1, n_safras=3, random_seed=0, kernel_trick=False)
    snap, panel, trace = synth.generate()
    assert len(trace) > 0
    assert panel["ever90m12"].max() == 1


def test_per_group_positive_presence():
    synth = CreditDataSynthesizer(group_profiles=default_group_profiles(3), contracts_per_group=20, n_safras=8, random_seed=1, kernel_trick=False)
    snap, panel, trace = synth.generate()
    counts = panel.groupby("grupo_homogeneo")["ever90m12"].sum()
    assert (counts > 0).all()


def test_sampler_group_balance():
    synth = CreditDataSynthesizer(group_profiles=default_group_profiles(2), contracts_per_group=40, n_safras=6, random_seed=2, kernel_trick=False)
    snap, panel, trace = synth.generate()
    sampler = TargetSampler(target_ratio=0.10, per_group=True, max_oversample=10)
    balanced = sampler.fit_transform(panel, target_col="ever90m12", safra_col="safra", group_col="grupo_homogeneo", random_state=0)
    prev = balanced.groupby(["safra", "grupo_homogeneo"])["ever90m12"].mean()
    assert (np.abs(prev - 0.10) < 0.015).all()

