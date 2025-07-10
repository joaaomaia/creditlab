import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from credit_data_synthesizer import CreditDataSynthesizer, default_group_profiles


def test_bad_rate_target():
    synth = CreditDataSynthesizer(
        group_profiles=default_group_profiles(2),
        contracts_per_group=200,
        n_safras=6,
        random_seed=0,
        force_event_rate=True,
        sampler_kwargs={"max_iter": 4},
    )
    _, panel, _ = synth.generate()
    diff = (panel.groupby("safra")["ever90m12"].mean() - 0.10).abs().max()
    assert diff < 0.005
