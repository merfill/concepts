"""Offline tests for the causal-ablation helpers (experiment 021)."""

import numpy as np

from tc.causal_ablation import (
    bootstrap_diff_ci,
    bootstrap_mean_ci,
    center_by_groups,
    direction_auc,
    fisher_subspace,
    form_subspace,
    gradient_summary,
    orthonormal_columns,
    project_out,
    random_subspace,
    removed_fraction,
)


def test_center_by_groups():
    X = np.asarray([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]])
    Xc, means = center_by_groups(X, [[0, 1], [2]])
    assert np.allclose(Xc[0], [-1.0, -1.0])
    assert np.allclose(Xc[2], [0.0, 0.0])
    assert len(means) == 2


def test_fisher_subspace_separates_and_ablation_removes():
    rng = np.random.default_rng(0)
    X0 = rng.normal(size=(60, 12))
    X1 = X0 + np.r_[3.0, np.zeros(11)]
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(60), np.ones(60)]
    W = fisher_subspace(X, y, k=2)
    assert W.shape[0] == 12
    assert direction_auc(X, y, W) > 0.95
    Xr = project_out(X, W)
    assert np.abs(Xr @ W).max() < 1e-10
    gap = Xr[y == 1].mean(0) @ W[:, 0] - Xr[y == 0].mean(0) @ W[:, 0]
    assert abs(gap) < 1e-10


def test_random_subspace_orthonormal_and_seeded():
    W = random_subspace(10, 3, seed=0)
    assert W.shape == (10, 3)
    assert np.allclose(W.T @ W, np.eye(3))
    W2 = random_subspace(10, 3, seed=0)
    assert np.allclose(W, W2)
    W3 = random_subspace(10, 3, seed=1)
    assert not np.allclose(W, W3)


def test_form_subspace_removes_covariate():
    rng = np.random.default_rng(0)
    C = rng.normal(size=(80, 2))
    C -= C.mean(axis=0)
    B = np.asarray([[1.0, 2.0, 0.0, -1.0], [0.0, 1.0, 3.0, 0.5]])
    X = C @ B + 0.01 * rng.normal(size=(80, 4))
    W = form_subspace(X, C, k=2)
    assert W.shape == (4, 2)
    Xr = project_out(X, W)
    for j in range(2):
        for c in range(2):
            assert abs(np.corrcoef(Xr[:, j], C[:, c])[0, 1]) < 1e-6


def test_removed_fraction():
    X = np.asarray([[3.0, 4.0], [0.0, 2.0]])
    w = np.asarray([[1.0], [0.0]])
    frac = removed_fraction(X, w)
    assert np.isclose(frac[0], 3.0 / 5.0)
    assert np.isclose(frac[1], 0.0)


def test_bootstrap_intervals():
    lo, hi, mean = bootstrap_mean_ci([0.6, 0.7, 0.8], n_boot=200, seed=0)
    assert lo <= mean <= hi
    lo2, hi2, mean2 = bootstrap_diff_ci([0.8, 0.9, 0.7], [0.5, 0.6, 0.6],
                                       n_boot=200, seed=0)
    assert lo2 <= mean2 <= hi2
    assert mean2 > 0


def test_gradient_summary_smoke():
    labels = np.asarray(["self", "sibling", "unrelated_same",
                         "self", "sibling", "unrelated_same"], dtype=object)
    groups = [[0, 1, 2], [3, 4, 5]]
    coords = np.asarray([0.0, 0.2, 5.0, 10.0, 10.2, 15.0])
    D = np.abs(coords[:, None] - coords[None, :])
    res = gradient_summary(D, labels, groups, "sibling", "unrelated_same",
                           n_boot=100, seed=0)
    assert np.isclose(res["mean"], 1.0)
    assert len(res["per_anchor"]) == 2
    assert res["ci95"][0] <= res["ci95"][1]
