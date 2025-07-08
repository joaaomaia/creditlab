import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from credit_data_synthesizer import (
    generate_transition_matrix,
    DEFAULT_BUCKETS,
    _risk_score,
)


def test_matrix_rowsum():
    mat = generate_transition_matrix(DEFAULT_BUCKETS, sev=0.5)
    assert np.allclose(mat.sum(axis=1), 1.0)


def test_min_diag():
    mat = generate_transition_matrix(DEFAULT_BUCKETS, sev=1.0)
    assert (np.diag(mat) >= 0.01).all()


def test_shape():
    buckets = [0, 15, 30, 60, 90, 120]
    base = np.eye(len(buckets))
    mat = generate_transition_matrix(buckets, sev=0.3, base_mat=base)
    assert mat.shape == (len(buckets), len(buckets))


def test_monotone_risk():
    mat1 = generate_transition_matrix(DEFAULT_BUCKETS, sev=0.2)
    mat2 = generate_transition_matrix(DEFAULT_BUCKETS, sev=0.8)
    assert _risk_score(mat2, DEFAULT_BUCKETS) > _risk_score(mat1, DEFAULT_BUCKETS)
