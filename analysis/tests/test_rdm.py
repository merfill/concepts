"""Unit tests for the RDM/RSA instrument (experiment 004, no model required).

The instrument must (a) recover a planted relational structure, (b) report no
structure under relabeling, and (c) separate structure stability from subspace
orientation.  These lock the definitions before any model is run.
"""

import numpy as np

from tc.rdm import (
    effective_rank,
    orientation_stability,
    principal_angles,
    rdm_js,
    rdm_length,
    rdm_vectors,
    rsa,
    rsa_partial,
    rsa_permutation,
    rsa,
    subspace_basis,
    zscore_columns,
)


def _rand_dist(rng, n):
    p = rng.random(n)
    return p / p.sum()


def test_rdm_js_symmetric_zero_diag():
    rng = np.random.default_rng(0)
    dists = [_rand_dist(rng, 25) for _ in range(6)]
    M = rdm_js(dists, k=8)
    assert np.allclose(M, M.T)
    assert np.allclose(np.diag(M), 0.0)
    assert np.all(M >= -1e-12)


def test_rdm_vectors_euclidean_basic():
    V = [np.array([0.0, 0.0]), np.array([3.0, 4.0]), np.array([3.0, 4.0])]
    M = rdm_vectors(V)
    assert abs(M[0, 1] - 5.0) < 1e-12
    assert M[1, 2] < 1e-12


def test_zscore_columns_unit_variance():
    M = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    Z = zscore_columns(M)
    assert np.allclose(Z.mean(axis=0), 0.0)
    assert np.allclose(Z.std(axis=0), 1.0)


def test_rsa_identity_and_orthogonal():
    rng = np.random.default_rng(1)
    D = rdm_vectors([rng.standard_normal(4) for _ in range(8)])
    assert abs(rsa(D, D) - 1.0) < 1e-9


def test_permutation_signal_vs_null():
    rng = np.random.default_rng(2)
    V = rng.standard_normal((16, 3))
    A = rdm_vectors(V)
    B = A + 0.05 * rng.random(A.shape)  # noisy copy -> high RSA
    obs, p_sig, _ = rsa_permutation(A, B, n_perm=300, seed=0)
    assert obs > 0.9 and p_sig < 0.01

    C = rdm_vectors([rng.standard_normal(3) for _ in range(16)])
    obs_null, p_null, _ = rsa_permutation(A, C, n_perm=300, seed=0)
    assert p_null > 0.01


def test_partial_rsa_removes_shared_confound():
    rng = np.random.default_rng(3)
    conf = rdm_length(rng.integers(0, 10, size=14))
    A = conf + rdm_vectors([rng.standard_normal(2) for _ in range(14)])
    B = conf + rdm_vectors([rng.standard_normal(2) for _ in range(14)])
    raw = rsa(A, B)
    part = rsa_partial(A, B, [conf])
    assert raw > part


def test_subspace_basis_and_principal_angles():
    rng = np.random.default_rng(4)
    basis = np.linalg.qr(rng.standard_normal((10, 3)))[0]
    scores = rng.standard_normal((12, 3))
    X = scores @ basis.T
    U = subspace_basis(X, k=3)
    cos, mean = principal_angles(basis, U)
    assert mean > 0.99

    other = np.linalg.qr(rng.standard_normal((10, 3)))[0]
    _, mean_other = principal_angles(basis, other)
    assert mean_other < 0.9


def test_orientation_stability_shared_vs_random():
    rng = np.random.default_rng(5)
    group_basis = np.linalg.qr(rng.standard_normal((8, 2)))[0]
    groups = np.repeat(np.arange(4), 4)
    shards = np.tile([0, 0, 1, 1], 4)

    # both shards share the same within-group subspace -> high agreement
    X = rng.standard_normal((16, 2)) @ group_basis.T
    X += np.repeat(rng.standard_normal((4, 8)), 4, axis=0)
    _, shared = orientation_stability(X, groups, shards, k=2)
    assert shared > 0.7

    # random noise -> lower agreement (chance level of 2-D in 8-D)
    Y = rng.standard_normal((16, 8))
    _, noise = orientation_stability(Y, groups, shards, k=2)
    assert noise < shared


def test_effective_rank_flat_vs_concentrated():
    flat = effective_rank(np.ones(10))
    conc = effective_rank(np.array([10.0, 0, 0, 0, 0]))
    assert conc < flat
