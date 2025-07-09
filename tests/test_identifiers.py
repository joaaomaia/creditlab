from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from credit_data_synthesizer import CreditDataSynthesizer, default_group_profiles


def make_data():
    synth = CreditDataSynthesizer(
        group_profiles=default_group_profiles(2),
        contracts_per_group=50,
        n_safras=6,
        random_seed=0,
        kernel_trick=False,
    )
    return synth.generate()


def test_id_format():
    snap, panel, _ = make_data()
    assert snap['id_contrato'].between(10_000_000, 99_999_999).all()
    assert panel['id_contrato'].between(10_000_000, 99_999_999).all()


def test_unique_ids():
    snap, panel, _ = make_data()
    assert snap['id_contrato'].is_unique
    assert panel.groupby('id_contrato')['id_cliente'].nunique().max() == 1
