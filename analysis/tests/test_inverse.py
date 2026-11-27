"""Offline tests for inverse-readout helpers (experiment 014)."""

import numpy as np

from tc.inverse import (
    gauge_dim,
    kl,
    null_basis,
    prob_jacobian,
    rank_nullity,
    softmax,
    topk_indices,
)


def toy(v=40, d=8, seed=0):
    rng = np.random.default_rng(seed)
    W_U = rng.standard_normal((v, d))
    logits = rng.standard_normal(v)
    p = softmax(logits)
    return W_U, p


def test_softmax_and_topk():
    p = softmax(np.array([1.0, 2.0, 3.0]))
    assert abs(p.sum() - 1.0) < 1e-12
    assert p[2] > p[1] > p[0]
    assert list(topk_indices(np.array([0.1, 0.7, 0.2]), 2)) == [1, 2]


def test_rank_nullity_full_rank_rows():
    rng = np.random.default_rng(1)
    J = rng.standard_normal((5, 8))
    rn = rank_nullity(J)
    assert rn["rank"] == 5
    assert rn["null_dim"] == 3


def test_prob_jacobian_null_basis():
    W_U, p = toy()
    idx = topk_indices(p, 5)
    J = prob_jacobian(W_U, p, idx)
    B = null_basis(J)
    assert B.shape[1] == W_U.shape[1]
    assert B.shape[0] == W_U.shape[1] - rank_nullity(J)["rank"]
    assert np.abs(J @ B.T).max() < 1e-8


def test_prob_jacobian_null_dim_matches_d_minus_m():
    W_U, p = toy(v=60, d=8)
    for m in (3, 5, 7):
        J = prob_jacobian(W_U, p, topk_indices(p, m))
        assert rank_nullity(J)["null_dim"] == 8 - m


def test_kl():
    p = np.array([0.2, 0.3, 0.5])
    assert kl(p, p) < 1e-12
    q = np.array([0.5, 0.3, 0.2])
    assert kl(p, q) > 0


def test_gauge_dim():
    assert gauge_dim("rmsnorm") == 1
    assert gauge_dim("layernorm") == 2
    assert gauge_dim("other") == 0


def test_simplex_interp_and_js_cosine():
    from tc.inverse import cosine, js, simplex_interp

    a = np.array([0.2, 0.8])
    b = np.array([0.6, 0.4])
    mid = simplex_interp(a, b, 0.5)
    assert abs(mid.sum() - 1.0) < 1e-12
    assert np.allclose(mid, [0.4, 0.6])
    assert js(a, a) < 1e-12
    assert js(a, b) > 0
    assert abs(cosine([1, 0], [1, 0]) - 1.0) < 1e-12
    assert cosine([1, 0], [0, 1]) < 1e-9
