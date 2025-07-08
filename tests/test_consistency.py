from pathlib import Path
import pandas as pd
import numpy as np
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from credit_data_synthesizer import CreditDataSynthesizer, default_group_profiles


def make_synth():
    profiles = default_group_profiles(2)
    return CreditDataSynthesizer(
        group_profiles=profiles,
        contracts_per_group=50,
        n_safras=6,
        random_seed=1,
        buckets=[0,30,60,120,150,360],
        kernel_trick=False,
    )


def test_start_before_safra():
    synth = make_synth()
    snap, panel, _ = synth.generate()
    assert (snap["data_inicio_contrato"] <= snap["data_ref"]).all()


def test_start_unique_per_client_safra():
    synth = make_synth()
    snap, panel, _ = synth.generate()
    start_month = pd.to_datetime(snap["data_inicio_contrato"]).dt.strftime("%Y%m")
    pairs = list(zip(snap["id_cliente"], start_month))
    assert len(pairs) == len(set(pairs))


def test_birthdate_unique():
    synth = make_synth()
    snap, panel, _ = synth.generate()
    counts = snap.groupby("id_cliente")["data_nascimento"].nunique()
    assert (counts == 1).all()
    assert synth.clients["id_cliente"].is_unique


def test_targets_dynamic():
    synth = make_synth()
    snap, panel, _ = synth.generate()
    assert panel["ever90m12"].sum() > 0
    assert panel["ever360m18"].sum() > 0
