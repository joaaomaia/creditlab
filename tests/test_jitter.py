import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from credit_data_synthesizer import (
    CreditDataSynthesizer,
    default_group_profiles,
    bucket_index,
)
from transition_matrix_estimator import TransitionMatrixLearner


def _nearest(val, buckets):
    arr = np.array(buckets)
    return int(arr[np.argmin(np.abs(arr - val))])


def test_jitter_range():
    synth = CreditDataSynthesizer(
        group_profiles=default_group_profiles(2),
        contracts_per_group=40,
        n_safras=4,
        random_seed=0,
        kernel_trick=False,
        force_event_rate=False,
        jitter=True,
    )
    _, panel, _ = synth.generate()
    buckets = synth.buckets
    arr = panel["dias_atraso"].to_numpy()
    nearest = np.array([_nearest(x, buckets) for x in arr])
    diff = np.abs(arr - nearest)
    allowed = np.array([max(synth.jitter_min, int(round(b * synth.jitter_pct))) for b in nearest])
    assert (diff <= allowed).all()


def test_target_consistency():
    synth = CreditDataSynthesizer(
        group_profiles=default_group_profiles(1),
        contracts_per_group=30,
        n_safras=6,
        random_seed=1,
        kernel_trick=False,
        force_event_rate=False,
        jitter=True,
    )
    _, panel, _ = synth.generate()
    orig = panel[["ever90m12", "over90m12", "ever360m18"]].sum()
    buckets = synth.buckets
    rebucket = panel.copy()
    rebucket["dias_atraso"] = [
        buckets[bucket_index(x, buckets)] for x in rebucket["dias_atraso"]
    ]
    tmp = CreditDataSynthesizer(
        group_profiles=default_group_profiles(1),
        contracts_per_group=1,
        n_safras=1,
        kernel_trick=False,
        force_event_rate=False,
    )
    tmp._panel = rebucket
    tmp._trace = synth.trace.copy()
    tmp.buckets = buckets
    tmp._compute_targets()
    new = tmp._panel[["ever90m12", "over90m12", "ever360m18"]].sum()
    assert orig.equals(new)


def test_heatmap_flat_rows():
    synth = CreditDataSynthesizer(
        group_profiles=default_group_profiles(2),
        contracts_per_group=200,
        n_safras=5,
        random_seed=2,
        kernel_trick=False,
        force_event_rate=False,
        jitter=True,
    )
    _, panel, _ = synth.generate()
    learner = TransitionMatrixLearner(buckets=synth.buckets, drop_empty=False, min_count=1)
    learner.fit(panel, id_col="id_contrato", time_col="data_ref", bucket_col="dias_atraso")
    mat = learner.get_matrix()
    assert (mat.sum(axis=1) > 0).all()
