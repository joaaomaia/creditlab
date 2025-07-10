import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from credit_data_synthesizer import CreditDataSynthesizer, default_group_profiles


def test_no_oversample():
    synth = CreditDataSynthesizer(
        group_profiles=default_group_profiles(2),
        contracts_per_group=150,
        n_safras=4,
        random_seed=2,
    )
    _, panel, _ = synth.generate()
    assert panel.duplicated(["id_contrato", "data_ref"]).sum() == 0
