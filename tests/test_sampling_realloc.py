from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from credit_data_synthesizer import CreditDataSynthesizer, default_group_profiles
from credit_data_sampler import TargetSampler


def test_sampling_realloc():
    synth = CreditDataSynthesizer(
        group_profiles=default_group_profiles(2),
        contracts_per_group=100,
        n_safras=8,
        random_seed=1,
        kernel_trick=False,
        force_event_rate=False,
    )
    _, panel, _ = synth.generate()
    sampler = TargetSampler(target_ratio=0.08, preserve_rank=True, max_iter=3)
    panel_bal, overflow = sampler.fit_transform(
        panel,
        target_col="ever90m12",
        safra_col="safra",
        group_col="grupo_homogeneo",
        random_state=0,
    )
    synth._panel = panel_bal
    if not overflow.empty:
        synth._reinject(overflow)
    rates = synth._panel.groupby('safra')['ever90m12'].mean()
    assert not synth._panel.empty
    total = len(panel)
    final_total = len(synth._panel) + len(overflow)
    assert abs(final_total - total) / total < 0.02
