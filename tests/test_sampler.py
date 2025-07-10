import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from credit_data_sampler import TargetSampler


def make_toy_panel():
    n = 40
    data = []
    for m, safra in enumerate(["201601", "201602"]):
        pos = np.zeros(n, dtype=int)
        pos[: n // 2] = 1
        np.random.default_rng(m).shuffle(pos)
        df = pd.DataFrame({
            "id_contrato": np.arange(m * n, (m + 1) * n),
            "data_ref": pd.Timestamp(f"2016-{m+1:02d}-01"),
            "safra": safra,
            "grupo_homogeneo": "GH1",
            "ever90m12": pos,
        })
        data.append(df)
    return pd.concat(data, ignore_index=True)


def test_sampler_respects_return_info():
    df = make_toy_panel()
    sampler = TargetSampler(0.10, strategy="hybrid")
    bal, ovf, info = sampler.fit_transform(df, return_info=True)
    assert "max_pp" in info
