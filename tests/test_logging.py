from pathlib import Path
import logging
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from credit_data_synthesizer import CreditDataSynthesizer, default_group_profiles


def test_log_emits(monkeypatch):
    calls = []
    def fake_log(self, level, msg, *args, **kwargs):
        calls.append(msg % args if args else msg)
    monkeypatch.setattr(logging.Logger, "log", fake_log)
    synth = CreditDataSynthesizer(
        group_profiles=default_group_profiles(1),
        contracts_per_group=20,
        n_safras=3,
        random_seed=0,
        verbose=True,
    )
    synth.generate()
    assert len(calls) > 0


def test_summary_funcs():
    synth = CreditDataSynthesizer(
        group_profiles=default_group_profiles(1),
        contracts_per_group=10,
        n_safras=2,
        random_seed=1,
    )
    synth.generate()
    sf = synth.summary_by_safra()
    gh = synth.summary_by_gh()
    assert list(sf.columns) == ["vol", "bad"]
    assert list(gh.columns) == ["vol", "bad"]
    assert len(sf) == synth.panel["safra"].nunique()
    assert len(gh) == synth.panel["grupo_homogeneo"].nunique()

