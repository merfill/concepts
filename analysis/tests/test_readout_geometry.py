"""Offline tests for the readout-geometry helpers (experiment 022)."""

import numpy as np

from tc.readout_geometry import (
    captured_fraction,
    layer_alignment,
    project,
    rdm_corr,
    top_directions_from_cov,
)
from tc.slot_relations import hellinger_rdm


def test_rdm_corr_identity_and_noise():
    rng = np.random.default_rng(0)
    D = rng.random((8, 8))
    D = D + D.T
    assert np.isclose(rdm_corr(D, D), 1.0)
    noise = rng.random((8, 8))
    noise = noise + noise.T
    assert rdm_corr(D, noise) < 1.0


def test_top_directions_from_cov():
    rng = np.random.default_rng(0)
    d, k = 10, 3
    V_true, _ = np.linalg.qr(rng.normal(size=(d, k)))
    s = np.asarray([5.0, 3.0, 1.0])
    U, _ = np.linalg.qr(rng.normal(size=(20, k)))
    W = (U * s) @ V_true.T
    WtW = W.T @ W
    V_rec, eigvals = top_directions_from_cov(WtW, k=3)
    assert np.allclose(np.sort(eigvals)[::-1][:3], s ** 2, atol=1e-8)
    sv = np.linalg.svd(V_true.T @ V_rec, compute_uv=False)
    assert np.all(sv > 0.999)
    assert np.isclose(captured_fraction(eigvals, 3), 1.0)


def test_captured_fraction_zero():
    assert np.isclose(captured_fraction(np.asarray([4.0, 1.0, 0.0]), 1), 0.8)


def test_project_shape():
    X = np.arange(12.0).reshape(3, 4)
    V = np.eye(4)[:, :2]
    Z = project(X, V)
    assert Z.shape == (3, 2)
    assert np.allclose(Z, X[:, :2])


def test_layer_alignment_smoke():
    rng = np.random.default_rng(0)
    H = rng.normal(size=(6, 3, 4))
    P = rng.dirichlet(np.ones(5), size=6)
    D_out = hellinger_rdm(P)
    values = layer_alignment(H, D_out)
    assert len(values) == 3
    assert all(np.isfinite(v) for v in values)
