"""Offline tests for concept centroids (experiment 018, step 1)."""

import numpy as np

from tc.concept_centroids import (
    CONCEPTS,
    anchor_diagnostic,
    build_items,
    centered_gram,
    content_words,
    cross_frame_transfer,
    cv_fisher,
    cv_fisher_covariates,
    cv_fisher_residualized,
    cv_raw,
    frequency_length_covariates,
    hellinger_centroid_accuracy,
    hellinger_gram,
    lexical_overlap_stats,
    permutation_null,
    permutation_null_residualized,
    rdm_partial,
    tfidf_centroid_accuracy,
    tfidf_fisher_cv,
    train_pca,
)


def _synth(n_concepts=5, n_slots=4, V=30, noise=0.005, seed=0):
    rng = np.random.default_rng(seed)
    bases = rng.dirichlet(0.3 * np.ones(V), size=n_concepts)
    P, y, slots = [], [], []
    for c in range(n_concepts):
        for s in range(n_slots):
            p = np.clip(bases[c] + noise * rng.random(V), 1e-8, None)
            P.append(p / p.sum())
            y.append(c)
            slots.append(s)
    return np.asarray(P), np.asarray(y), np.asarray(slots)


def test_build_items():
    items = build_items()
    assert len(items) == 20 * 5
    assert len(CONCEPTS) == 20
    assert len({it["concept"] for it in items}) == 20
    assert len({it["slot"] for it in items}) == 5
    for ci in range(20):
        assert sum(1 for it in items if it["concept"] == ci) == 5
    assert items[0]["text"] == "You can describe a dog as"


def test_hellinger_gram():
    P, _, _ = _synth()
    G = hellinger_gram(P)
    assert G.shape == (P.shape[0], P.shape[0])
    assert np.allclose(np.diag(G), 1.0, atol=1e-10)
    assert np.allclose(G, G.T)
    X = np.sqrt(P)
    assert np.allclose(G, X @ X.T)


def test_centered_gram_train_mean_zero():
    P, _, slots = _synth()
    G = hellinger_gram(P)
    train = np.where(slots != 0)[0]
    Gc = centered_gram(G, train)
    row_means = Gc[np.ix_(train, train)].mean(axis=1)
    assert np.allclose(row_means, 0.0, atol=1e-10)


def test_train_pca_shapes():
    P, _, slots = _synth()
    G = hellinger_gram(P)
    train = np.where(slots != 0)[0]
    test = np.where(slots == 0)[0]
    Ztr, Zte = train_pca(centered_gram(G, train), train, test)
    assert Ztr.shape[0] == train.size
    assert Zte.shape[0] == test.size
    assert Ztr.shape[1] == Zte.shape[1]


def test_cv_recovers_synthetic_concepts():
    P, y, slots = _synth()
    G = hellinger_gram(P)
    acc = cv_fisher(G, y, slots, 3)["accuracy"]
    assert acc > 0.8
    raw = cv_raw(G, y, slots, 3)["accuracy"]
    assert 0.0 <= raw <= 1.0


def test_baselines_run():
    P, y, slots = _synth()
    items = build_items()[: P.shape[0]]  # not aligned, but only smoke for tfidf below
    hell = hellinger_centroid_accuracy(P, y, slots)
    assert 0.0 <= hell <= 1.0
    texts = [it["text"] for it in items]
    tfidf = tfidf_centroid_accuracy(texts, y, slots)
    assert 0.0 <= tfidf <= 1.0


def test_null_drops_to_chance():
    P, y, slots = _synth()
    G = hellinger_gram(P)
    res = permutation_null(G, y, slots, 3, n_perm=30, seed=0)
    assert res["observed"] > 0.8
    assert res["mean"] < 0.5
    assert 0.0 <= res["p"] <= 1.0


def test_content_words():
    assert content_words("a pair of jeans") == {"pair", "jeans"}
    assert "the" not in content_words("the striped zebra")


def test_anchor_diagnostic():
    items = [
        {"concept": 0, "concept_name": "zebra", "paraphrase": "a striped zebra"},
        {"concept": 0, "concept_name": "zebra", "paraphrase": "a zebra with stripes"},
        {"concept": 1, "concept_name": "lion", "paraphrase": "a big cat"},
        {"concept": 1, "concept_name": "lion", "paraphrase": "a roaring big cat"},
    ]
    diag = {d["concept"]: d for d in anchor_diagnostic(items)}
    assert "zebra" in diag["zebra"]["anchors"]


def test_tfidf_fisher_on_lexical_anchors():
    texts, y, slots = [], [], []
    for c in range(5):
        for s in range(4):
            texts.append(f"unique{c} alpha beta gamma")
            y.append(c)
            slots.append(s)
    acc = tfidf_fisher_cv(texts, np.asarray(y), np.asarray(slots), 3)
    assert acc > 0.9


def test_residualized_and_rdm_smoke():
    P, y, slots = _synth()
    G = hellinger_gram(P)
    texts = [f"concept{c} slot{s}" for c, s in zip(y, slots)]
    acc = cv_fisher_residualized(G, texts, y, slots, 3)
    assert 0.0 <= acc <= 1.0
    rdm = rdm_partial(P, texts, y)
    for key in ("rho_model_concept", "rho_model_lexical",
                "partial_model_concept_given_lexical"):
        assert key in rdm
        assert np.isfinite(rdm[key])


def test_lexical_overlap_stats():
    st = lexical_overlap_stats(build_items())
    assert st["within_mean"] < 0.05
    assert abs(st["within_mean"] - st["between_mean"]) < 0.05


def test_residualized_null_runs():
    P, y, slots = _synth()
    G = hellinger_gram(P)
    texts = [f"concept{c} slot{s}" for c, s in zip(y, slots)]
    res = permutation_null_residualized(G, texts, y, slots, 3, n_perm=10, seed=0)
    assert 0.0 <= res["p"] <= 1.0
    assert res["observed"] >= 0.0


def test_build_items_frame():
    items = build_items("{x} can be described as")
    assert items[0]["text"] == "a dog can be described as"


def test_frequency_length_covariates():
    cov = frequency_length_covariates(["a dog", "a long yellow tropical produce"], [2, 5])
    assert cov.shape == (2, 2)
    assert cov[0, 0] == 2 and cov[1, 0] == 5
    assert np.all(np.isfinite(cov))


def test_cv_fisher_covariates_removes_label_signal():
    P, y, slots = _synth()
    G = hellinger_gram(P)
    acc0 = cv_fisher(G, y, slots, 3)["accuracy"]
    onehot = np.eye(np.unique(y).size)[y]
    acc1 = cv_fisher_covariates(G, y, slots, onehot, 3)
    assert acc0 > 0.8
    assert acc1 < acc0 - 0.3


def _synth_frames(n_concepts=5, n_slots=4, V=30, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    bases = rng.dirichlet(0.3 * np.ones(V), size=n_concepts)

    def frame(seed2):
        r = np.random.default_rng(seed2)
        P, y, slots = [], [], []
        for c in range(n_concepts):
            for s in range(n_slots):
                p = np.clip(bases[c] + noise * r.random(V), 1e-8, None)
                P.append(p / p.sum())
                y.append(c)
                slots.append(s)
        return np.asarray(P), np.asarray(y), np.asarray(slots)

    Pa, ya, _ = frame(1)
    Pb, yb, _ = frame(2)
    P = np.vstack([Pa, Pb])
    y = np.concatenate([ya, yb])
    frames = np.asarray([0] * len(ya) + [1] * len(yb))
    return P, y, frames


def test_cross_frame_transfer():
    P, y, frames = _synth_frames()
    G = hellinger_gram(P)
    res = cross_frame_transfer(G, y, frames, 3)
    assert res["accuracy_A_to_B"] > 0.8
    assert res["accuracy_B_to_A"] > 0.8
    assert res["rdm_corr_A_to_B"] > 0.8
