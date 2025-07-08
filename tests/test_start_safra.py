import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from credit_data_synthesizer import CreditDataSynthesizer, default_group_profiles

N_GROUPS = 2
CONTRACTS = 50
N_SAFRAS = 36
START_SAFRA = "202001"


def make_synth():
    profiles = default_group_profiles(N_GROUPS)
    return CreditDataSynthesizer(
        group_profiles=profiles,
        contracts_per_group=CONTRACTS,
        n_safras=N_SAFRAS,
        random_seed=24,
        start_safra=START_SAFRA,
    )


def test_custom_start_safra():
    synth = make_synth()
    snap, panel, trace = synth.generate()
    assert snap["data_ref"].min() == pd.Timestamp("2020-01-01")


def test_panel_month_count():
    synth = make_synth()
    snap, panel, trace = synth.generate()
    assert panel["safra"].nunique() == N_SAFRAS


def test_first_and_last_dates():
    synth = make_synth()
    snap, panel, trace = synth.generate()
    first = pd.to_datetime(panel["safra"].min(), format="%Y%m")
    last = pd.to_datetime(panel["safra"].max(), format="%Y%m")
    diff = (last.year - first.year) * 12 + last.month - first.month
    assert diff == N_SAFRAS - 1
