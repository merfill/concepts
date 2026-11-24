"""Unit tests for the transition-operator instrument (experiment 005).

No model is downloaded: the estimator is validated on a synthetic linear map,
and CKA/operator-RDM on synthetic representations.  These lock the definitions
before the operator is measured on a transformer.
"""

import numpy as np

from tc.operator import (
    finite_diff_operator,
    linear_cka,
    operator_distance,
    operator_effective_dimension,
    operator_rdm,
    random_probe_basis,
)


def test_probe_basis_orthonormal():
    P = random_probe_basis(10, 4, seed=0)
    assert P.shape == (10, 4)
    assert np.allclose(P.T @ P, np.eye(4), atol=1e-10)


def test_finite_diff_recovers_linear_operator():
    rng = np.random.default_rng(1)
    d = 6
    A = rng.standard_normal((d, d))
    fn = lambda h: A @ h
    P = random_probe_basis(d, 3, seed=2)
    h0 = rng.standard_normal(d)
    phi = finite_diff_operator(fn, h0, P, eps=1e-6)
    assert np.allclose(phi, A @ P, atol=1e-5)


def test_cka_self_and_symmetry():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((12, 5))
    Y = rng.standard_normal((12, 5))
    assert abs(linear_cka(X, X) - 1.0) < 1e-10
    assert abs(linear_cka(X, Y) - linear_cka(Y, X)) < 1e-10


def test_cka_invariant_to_feature_rotation():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((15, 6))
    Y = rng.standard_normal((15, 6))
    Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    base = linear_cka(X, Y)
    rotated = linear_cka(X @ Q, Y @ Q)
    assert abs(base - rotated) < 1e-8


def test_operator_distance_self_zero():
    rng = np.random.default_rng(5)
    phi = rng.standard_normal((8, 3))
    assert operator_distance(phi, phi, "cka") < 1e-10
    assert operator_distance(phi, phi, "cosine") < 1e-10


def test_operator_rdm_properties():
    rng = np.random.default_rng(6)
    phis = [rng.standard_normal((7, 3)) for _ in range(5)]
    M = operator_rdm(phis)
    assert np.allclose(M, M.T)
    assert np.allclose(np.diag(M), 0.0)


def test_effective_dimension_flat_vs_concentrated():
    rng = np.random.default_rng(7)
    flat = [rng.standard_normal((6, 2)) for _ in range(12)]
    base = rng.standard_normal((6, 2))
    scaled = rng.standard_normal(12) * 3.0 + 5.0  # rank-1 family: all scalar multiples of base
    concentrated = [s * base for s in scaled]
    assert operator_effective_dimension(concentrated) < operator_effective_dimension(flat)
