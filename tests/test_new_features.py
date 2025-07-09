import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import logging
import numpy as np
from credit_data_synthesizer import CreditDataSynthesizer, GroupProfile, default_group_profiles, DEFAULT_BUCKETS


def test_refin_probability():
    tm = np.eye(len(DEFAULT_BUCKETS))
    gp0 = GroupProfile(name="G", pd_base=0.06, p_accept_refin=0.0, reneg_prob_exog=0.0, transition_matrix=tm)
    synth0 = CreditDataSynthesizer(group_profiles=[gp0], contracts_per_group=40, n_safras=6, random_seed=0, kernel_trick=False, force_event_rate=False)
    _, panel0, _ = synth0.generate()
    assert panel0["nivel_refinanciamento"].sum() == 0

    gp1 = GroupProfile(name="G", pd_base=0.06, p_accept_refin=1.0, reneg_prob_exog=0.0, transition_matrix=tm)
    synth1 = CreditDataSynthesizer(group_profiles=[gp1], contracts_per_group=40, n_safras=6, random_seed=1, kernel_trick=False, force_event_rate=False)
    _, panel1, _ = synth1.generate()
    assert panel1["nivel_refinanciamento"].sum() > 0


def test_reneg_stage1():
    n = len(DEFAULT_BUCKETS)
    tm = np.zeros((n, n))
    idx15 = DEFAULT_BUCKETS.index(15)
    tm[:, idx15] = 1.0
    gp = GroupProfile(name="G", pd_base=0.12, p_accept_refin=0.0, reneg_prob_exog=0.2, transition_matrix=tm)
    synth = CreditDataSynthesizer(group_profiles=[gp], contracts_per_group=40, n_safras=6, random_seed=2, kernel_trick=False, force_event_rate=False)
    _, _, trace = synth.generate()
    assert len(trace) > 0


def test_multi_contracts():
    synth = CreditDataSynthesizer(
        group_profiles=default_group_profiles(2),
        contracts_per_group=30,
        n_safras=4,
        random_seed=3,
        kernel_trick=False,
        force_event_rate=False,
        max_simultaneous_contracts=3,
    )
    snap, _, _ = synth.generate()
    pairs = snap.assign(m=snap["data_inicio_contrato"].dt.strftime("%Y%m"))
    assert len(pairs) == len(pairs.drop_duplicates(["id_cliente", "m"]))


def test_post_sampling_ratio(caplog):
    caplog.set_level(logging.WARNING)
    synth = CreditDataSynthesizer(
        group_profiles=default_group_profiles(2),
        contracts_per_group=200,
        n_safras=6,
        random_seed=4,
        kernel_trick=False,
        target_ratio=0.33,
        tol_pp=0.5,
    )
    _, panel, _ = synth.generate()
    real = panel["ever90m12"].mean()
    assert abs(real - 0.33) <= 0.005
