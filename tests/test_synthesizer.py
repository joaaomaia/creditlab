import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import pandas as pd
from credit_data_synthesizer import CreditDataSynthesizer, default_group_profiles

N_GROUPS = 3
CONTRACTS = 100
N_SAFRAS = 12


def make_synth():
    profiles = default_group_profiles(N_GROUPS)
    return CreditDataSynthesizer(
        group_profiles=profiles,
        contracts_per_group=CONTRACTS,
        n_safras=N_SAFRAS,
        random_seed=42,
    )


def test_snapshot_size():
    synth = make_synth()
    snap, panel, trace = synth.generate()
    first = panel["data_ref"].min()
    assert len(snap) == len(panel[panel["data_ref"] == first])


def test_panel_safras_count():
    synth = make_synth()
    snap, panel, trace = synth.generate()
    assert panel["safra"].nunique() == N_SAFRAS


def test_unique_ids():
    synth = make_synth()
    snap, panel, trace = synth.generate()
    assert snap["id_contrato"].is_unique
    assert panel.groupby("id_contrato")["id_cliente"].nunique().max() == 1


def test_features_presence():
    synth = make_synth()
    snap, panel, trace = synth.generate()
    required = {
        "dias_atraso",
        "nivel_refinanciamento",
        "saldo_devedor",
        "valor_parcela",
        "prazo_meses",
        "taxa_juros_anual_pct",
    }
    assert required.issubset(panel.columns)


def test_targets_flagged():
    synth = make_synth()
    snap, panel, trace = synth.generate()
    assert snap["ever90m12"].sum() > 0
