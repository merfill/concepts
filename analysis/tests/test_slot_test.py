"""Offline tests for the slot test (experiment 019)."""

import numpy as np

from tc.concept_centroids import hellinger_gram
from tc.slot_test import (
    FRAME,
    build_items,
    centroid_rdm,
    cluster_centroid_gram,
    cluster_labels,
    gradient_from_coords,
    labels,
    rdm_controls,
    residualized_gradient,
    semantic_gradient,
    slot_metrics,
)


def _relabelled_synth(n_triples=4, n_fillers=3, V=25, noise=0.005, seed=0):
    rng = np.random.default_rng(seed)
    P, yt, yr = [], [], []
    for t in range(n_triples):
        base = rng.dirichlet(0.3 * np.ones(V))
        opposite = rng.dirichlet(0.3 * np.ones(V))
        for r in range(3):
            base_r = opposite if r == 2 else base + (0.02 if r == 1 else 0.0)
            for _ in range(n_fillers):
                p = np.clip(base_r + noise * rng.random(V), 1e-8, None)
                P.append(p / p.sum())
                yt.append(t)
                yr.append(r)
    return np.asarray(P), np.asarray(yt), np.asarray(yr)


def test_build_items():
    items = build_items()
    assert len(items) == 10 * 3 * 3
    assert items[0]["text"] == "When we say hot, we mean"
    yt, yr, slots = labels(items)
    assert len(np.unique(yt)) == 10
    assert len(np.unique(yr)) == 3
    assert len(np.unique(slots)) == 3
    assert len(np.unique(cluster_labels(items))) == 30


def test_centroid_gram_and_rdm():
    P, yt, yr = _relabelled_synth()
    G = hellinger_gram(P)
    labels_ = yt * 3 + yr
    uniq, M = cluster_centroid_gram(G, labels_)
    assert M.shape == (len(uniq), len(uniq))
    assert np.allclose(M, M.T)
    R = centroid_rdm(M)
    assert np.allclose(np.diag(R), 0.0)
    assert np.all(R >= 0)


def test_semantic_gradient_synthetic():
    P, yt, yr = _relabelled_synth()
    G = hellinger_gram(P)
    res = semantic_gradient(G, yt, yr)
    assert res["fraction_related_closer"] == 1.0
    assert res["d_target_related"] < res["d_target_opposite"]


def test_residualized_gradient_smoke():
    P, yt, yr = _relabelled_synth()
    G = hellinger_gram(P)
    cov = np.column_stack([np.arange(P.shape[0]), np.ones(P.shape[0])])
    res = residualized_gradient(G, yt, yr, cov)
    assert 0.0 <= res["fraction_related_closer"] <= 1.0


def test_rdm_controls_smoke():
    P, yt, yr = _relabelled_synth()
    G = hellinger_gram(P)
    lengths = np.arange(P.shape[0], dtype=float)
    res = rdm_controls(G, lengths, yt, yr)
    for key in res:
        assert np.isfinite(res[key])


def test_slot_metrics_smoke():
    items = build_items()
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(20), size=len(items))
    lengths = np.arange(len(items), dtype=float)
    cov = np.column_stack([lengths, np.ones(len(items))])
    m = slot_metrics(items, P, lengths, cov)
    assert set([
        "gradient", "gradient_residualized", "role_accuracy", "triple_accuracy",
        "role_accuracy_resid", "triple_accuracy_resid", "rdm_controls", "G",
    ]).issubset(m)
    assert m["G"].shape == (len(items), len(items))
