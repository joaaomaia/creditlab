import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from credit_data_synthesizer import CreditDataSynthesizer, default_group_profiles, GroupProfile

CUSTOM_BUCKETS = [0,15,30,60,90,120,180,240,360]


def test_bucket_membership():
    synth = CreditDataSynthesizer(
        group_profiles=default_group_profiles(2),
        contracts_per_group=50,
        n_safras=6,
        random_seed=0,
        buckets=CUSTOM_BUCKETS,
        kernel_trick=False,
    )
    snap, panel, _ = synth.generate()
    assert set(panel["dias_atraso"].unique()).issubset(CUSTOM_BUCKETS)


def test_matrix_size():
    synth = CreditDataSynthesizer(
        group_profiles=default_group_profiles(2),
        contracts_per_group=10,
        n_safras=2,
        random_seed=1,
        buckets=CUSTOM_BUCKETS,
        kernel_trick=False,
    )
    for gp in synth.group_profiles:
        assert gp.transition_matrix.shape == (len(CUSTOM_BUCKETS), len(CUSTOM_BUCKETS))


def test_ever360():
    n = len(CUSTOM_BUCKETS)
    tm = np.zeros((n, n))
    tm[:, CUSTOM_BUCKETS.index(360)] = 1.0
    gp = GroupProfile(name="G", pd_base=0.12, refin_prob=0.0, reneg_prob_exog=0.0, transition_matrix=tm)
    synth = CreditDataSynthesizer(
        group_profiles=[gp],
        contracts_per_group=1,
        n_safras=20,
        random_seed=2,
        buckets=CUSTOM_BUCKETS,
        kernel_trick=False,
    )
    _, panel, _ = synth.generate()
    assert panel["ever360m18"].max() == 1

