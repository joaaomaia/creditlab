import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from transition_matrix_estimator import TransitionMatrixLearner


def test_rebin_moves_counts():
    counts = np.array([
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
    ], dtype=float)
    learner = TransitionMatrixLearner(
        buckets=[0, 30, 60, 90], auto_rebin=True, min_count=2
    )
    total = counts.sum()
    learner._clean_matrix(counts)
    assert counts.sum() == total


def test_drop_updates_labels():
    import pandas as pd
    df = pd.DataFrame(
        {
            "id": [1, 1],
            "data_ref": pd.to_datetime(["2021-01-01", "2021-02-01"]),
            "delay": [0, 0],
        }
    )
    learner = TransitionMatrixLearner(
        buckets=[0, 30, 60], drop_empty=True, min_count=1
    )
    learner.fit(df, id_col="id", time_col="data_ref", bucket_col="delay")
    mat = learner.get_matrix()
    assert len(learner.cleaned_buckets) == mat.shape[0]


def test_no_flat_rows():
    import pandas as pd
    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3, 3],
            "data_ref": pd.to_datetime(
                [
                    "2021-01-01",
                    "2021-02-01",
                    "2021-01-01",
                    "2021-02-01",
                    "2021-01-01",
                    "2021-02-01",
                    "2021-03-01",
                ]
            ),
            "delay": [0, 30, 0, 0, 60, 90, 90],
        }
    )
    learner = TransitionMatrixLearner(
        buckets=[0, 30, 60, 90], drop_empty=True, min_count=1
    )
    learner.fit(df, id_col="id", time_col="data_ref", bucket_col="delay")
    mat = learner.get_matrix()
    assert (mat.std(axis=1) > 0).all()
