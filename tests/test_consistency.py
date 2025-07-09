from pathlib import Path
import sys
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from credit_data_synthesizer import CreditDataSynthesizer, default_group_profiles


@pytest.fixture(scope="module")
def synth() -> CreditDataSynthesizer:
    profiles = default_group_profiles(2)
    return CreditDataSynthesizer(
        group_profiles=profiles,
        contracts_per_group=50,
        n_safras=6,
        random_seed=1,
        buckets=[0, 30, 60, 120, 150, 360],
        kernel_trick=False,
    )


@pytest.fixture(scope="module")
def panel(synth: CreditDataSynthesizer) -> pd.DataFrame:
    _, panel, _ = synth.generate()
    return panel


def test_start_before_safra(panel: pd.DataFrame):
    assert (panel["data_inicio_contrato"] <= panel["data_ref"]).all()


def test_unique_client_start_month(panel: pd.DataFrame):
    snap = panel[panel["data_ref"] == panel["data_ref"].min()].copy()
    pairs = snap.assign(m=snap["data_inicio_contrato"].dt.strftime("%Y%m"))
    assert not pairs.duplicated(["id_cliente", "m"]).any()


def test_unique_birthdate(synth: CreditDataSynthesizer):
    counts = synth.clients.groupby("id_cliente")["data_nascimento"].nunique()
    assert counts.max() == 1


def test_targets_dynamic(panel: pd.DataFrame):
    assert panel["ever90m12"].sum() > 0
    assert panel["ever360m18"].sum() > 0
