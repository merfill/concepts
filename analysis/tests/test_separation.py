"""Offline tests for the form/meaning subspace separation (experiment 017)."""

import numpy as np

from tc.form_meaning import build_items, labels
from tc.separation import (
    abs_diff_rdm,
    cosine_rdm,
    fisher_directions,
    fisher_ratio,
    label_rdm,
    mean_principal_angle,
    principal_angles,
    project,
    residualize,
    scatter,
)


def _orth(d, k, rng):
    q, _ = np.linalg.qr(rng.standard_normal((d, k)))
    return q


def _make_clean(d=30, n_classes=5, n_variants=4, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    M = _orth(d, 2, rng)
    F0 = _orth(d, 3, rng)
    F = F0 - M @ (M.T @ F0)
    F, _ = np.linalg.qr(F)
    m_coef = rng.standard_normal((n_classes, 2))
    f_coef = rng.standard_normal((n_variants, 3))
    xs, ym, yf = [], [], []
    for c in range(n_classes):
        for v in range(n_variants):
            x = M @ m_coef[c] + F @ f_coef[v] + noise * rng.standard_normal(d)
            xs.append(x)
            ym.append(c)
            yf.append(v)
    return np.asarray(xs), np.asarray(ym), np.asarray(yf), M, F


def test_scatter_shapes_and_symmetry():
    X, ym, _, _, _ = _make_clean()
    sc = scatter(X, ym)
    for key in ("within", "between"):
        assert sc[key].shape == (X.shape[1], X.shape[1])
        assert np.allclose(sc[key], sc[key].T)
    assert sc["within_trace"] > 0
    assert sc["between_trace"] > 0
    assert sc["ratio"] > 0


def test_fisher_recovers_meaning_and_form_subspaces():
    X, ym, _, M, F = _make_clean()
    dirs = fisher_directions(X, ym, n_meaning=2, n_form=3)
    assert mean_principal_angle(dirs["W_meaning"], M) < 10.0
    assert mean_principal_angle(dirs["W_form"], F) < 10.0
    assert mean_principal_angle(dirs["W_meaning"], dirs["W_form"]) > 75.0


def test_projection_keeps_meaning_and_drops_form():
    X, ym, _, _, _ = _make_clean()
    dirs = fisher_directions(X, ym, n_meaning=2, n_form=3)
    r0 = fisher_ratio(X, ym)
    Xm = project(X, dirs["W_meaning"], dirs["mean"])
    Xf = project(X, dirs["W_form"], dirs["mean"])
    assert fisher_ratio(Xm, ym) > r0
    assert fisher_ratio(Xf, ym) < r0


def test_principal_angles_identical_and_orthogonal():
    rng = np.random.default_rng(3)
    A = _orth(10, 3, rng)
    assert np.allclose(principal_angles(A, A), 0.0, atol=1e-6)
    B = _orth(10, 3, rng)
    B = B - A @ (A.T @ B)
    B, _ = np.linalg.qr(B)
    assert np.allclose(principal_angles(A, B), 90.0, atol=1e-6)


def test_residualize_removes_linear_effect():
    rng = np.random.default_rng(4)
    cov = rng.standard_normal(50)
    X = np.column_stack([2.0 * cov + 0.1 * rng.standard_normal(50),
                         -cov + 0.1 * rng.standard_normal(50)])
    R = residualize(X, [cov])
    assert abs(R[:, 0] @ cov) < 0.05
    assert abs(R[:, 1] @ cov) < 0.05


def test_rdms():
    y = np.array([0, 0, 1, 1])
    M = label_rdm(y)
    assert M[0, 1] == 0 and M[0, 2] == 1
    d = abs_diff_rdm(np.array([0.0, 1.0, 3.0]))
    assert d[0, 2] == 3.0
    X = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    C = cosine_rdm(X)
    assert abs(C[0, 1]) < 1e-12
    assert abs(C[0, 2] - 1.0) < 1e-12


def test_build_items_and_labels():
    items = build_items()
    assert len(items) == 16 * 3 * 3
    ym, yf = labels(items)
    assert len(np.unique(ym)) == 16
    assert len(np.unique(yf)) == 9
    for s in np.unique(ym):
        assert int(np.sum(ym == s)) == 9
