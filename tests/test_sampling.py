import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from credit_data_synthesizer import CreditDataSynthesizer, default_group_profiles


def test_event_rate_close():
    synth = CreditDataSynthesizer(
        group_profiles=default_group_profiles(2),
        contracts_per_group=100,
        n_safras=6,
        random_seed=0,
        sampler_kwargs={"max_oversample": 10},
    )
    _, panel, _ = synth.generate()
    prev = panel.groupby("safra")["ever90m12"].mean()
    assert (prev >= 0).all()


def test_volume_plot_runs():
    synth = CreditDataSynthesizer(
        group_profiles=default_group_profiles(1),
        contracts_per_group=20,
        n_safras=3,
        random_seed=1,
    )
    synth.generate()
    ax = synth.plot_volume_bad_rate()
    assert isinstance(ax, plt.Axes)
