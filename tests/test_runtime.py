import sys
from pathlib import Path
import pytest
sys.path.append(str(Path(__file__).resolve().parents[1]))

from credit_data_synthesizer import CreditDataSynthesizer, default_group_profiles


@pytest.mark.timeout(2)
def test_runtime():
    synth = CreditDataSynthesizer(
        group_profiles=default_group_profiles(1),
        contracts_per_group=50,
        n_safras=3,
        random_seed=0,
    )
    synth.generate()

