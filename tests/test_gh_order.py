import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from credit_data_synthesizer import CreditDataSynthesizer, default_group_profiles


def test_gh_order():
    synth = CreditDataSynthesizer(
        group_profiles=default_group_profiles(3),
        contracts_per_group=100,
        n_safras=5,
        random_seed=1,
    )
    _, panel, _ = synth.generate()
    order = [f"GH{i+1}" for i in range(3)]
    rates = panel.groupby(["safra", "grupo_homogeneo"])["ever90m12"].mean().unstack()
    for _, row in rates.iterrows():
        assert row.reindex(order).is_monotonic_decreasing
