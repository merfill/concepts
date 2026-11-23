"""Unit tests for the Phi implementation (no model download required).

These are the 'step 0 / unit' checks from the document: Phi must be invariant to
transformations that are mathematically irrelevant (rotation of the whole value
basis, relabeling of positions) and must be a proper probability vector.
"""

import numpy as np

from tc.metrics import hellinger
from tc.phi import phi_a, phi_pool, phi_spec


def random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return q


def test_phi_spec_is_distribution():
    rng = np.random.default_rng(0)
    p = phi_spec(rng.standard_normal((7, 5)))
    assert abs(p.sum() - 1.0) < 1e-9
    assert np.all(p >= -1e-12)


def test_phi_spec_rotation_invariance():
    rng = np.random.default_rng(1)
    V = rng.standard_normal((7, 5))
    Q = random_orthogonal(5, rng)
    assert np.allclose(phi_spec(V), phi_spec(V @ Q), atol=1e-9)


def test_phi_spec_permutation_invariance():
    rng = np.random.default_rng(2)
    V = rng.standard_normal((7, 5))
    perm = rng.permutation(7)
    assert np.allclose(phi_spec(V), phi_spec(V[perm]), atol=1e-10)


def test_phi_spec_padding():
    rng = np.random.default_rng(3)
    p = phi_spec(rng.standard_normal((4, 6)), d=10)
    assert p.size == 10
    assert abs(p.sum() - 1.0) < 1e-9


def test_phi_a_permutation_invariance():
    rng = np.random.default_rng(4)
    A = rng.random((6, 6))
    A /= A.sum(axis=1, keepdims=True)
    perm = rng.permutation(6)
    P = np.eye(6)[perm]
    assert np.allclose(phi_a(A), phi_a(P @ A @ P.T), atol=1e-10)


def test_phi_a_transpose_same_spectrum():
    # Documents the corrected claim: spectrum(A) == spectrum(A^T).
    rng = np.random.default_rng(5)
    A = rng.random((6, 6))
    A /= A.sum(axis=1, keepdims=True)
    assert np.allclose(phi_a(A), phi_a(A.T), atol=1e-10)


def test_phi_pool_unit_norm():
    rng = np.random.default_rng(6)
    u = phi_pool(rng.standard_normal((4, 3)))
    assert abs(np.linalg.norm(u) - 1.0) < 1e-9


def test_hellinger_basic():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert abs(hellinger(p, q) - 1.0) < 1e-9
    assert hellinger(p, p) < 1e-9
    assert hellinger(p, p, ) >= 0.0
