"""Offline tests for the WordNet slot-relation experiment (020)."""

import numpy as np

from tc.slot_relations import (
    CONTROL_RELATION,
    HARD_CONTROL_RELATION,
    anchor_groups,
    analyze_distribution,
    build_stimuli,
    clean_name,
    hellinger_rdm,
    increment_stats,
    pair_divergence,
    pairwise_rdm,
    profile_correlation,
    read_tsv,
    relation_gradient,
    relation_labels,
    relation_names,
    relation_pair_distance,
    relation_profile,
    relation_separation,
    residualize_coords,
    select_relation_words,
    separation_permutation,
    state_layer_metrics,
    stimuli_text,
    write_tsv,
)


def make_items(spec, pos="noun"):
    return [
        {"pos": pos, "anchor_id": a, "anchor": anc, "relation": r, "word": w,
         "zipf": 5.0, "n_chars": len(w), "n_tokens_slot": 1}
        for a, anc, r, w in spec
    ]


def _two_anchor_items():
    spec = []
    for a, anchor in enumerate(["dog", "car"]):
        spec += [
            (a, anchor, "self", f"{anchor}_self"),
            (a, anchor, "sibling", f"{anchor}_s1"),
            (a, anchor, "sibling", f"{anchor}_s2"),
            (a, anchor, "unrelated", f"{anchor}_u1"),
            (a, anchor, "unrelated", f"{anchor}_u2"),
            (a, anchor, "unrelated", f"{anchor}_u3"),
        ]
    return make_items(spec)


def _clustered_distance(items, close=0.1, far=5.0, offset=10.0):
    coords = []
    for a, _ in enumerate(["dog", "car"]):
        coords += [offset * a + 0.0, offset * a + close, offset * a + 2 * close,
                   offset * a + far, offset * a + far, offset * a + far]
    coords = np.asarray(coords)
    return np.abs(coords[:, None] - coords[None, :])


def test_clean_name():
    assert clean_name("Domestic_Animal") == "domestic animal"
    assert clean_name(" dog ") == "dog"


def test_select_relation_words_filters_and_caps():
    cands = ["cat", "horse", "mouse", "dog", "cat", "zyzzyva_x"]
    got = select_relation_words(cands, anchor="dog", max_n=2)
    assert got == ["cat", "horse"]


def test_build_stimuli_injected():
    def relatives(word, pos):
        if pos == "noun" and word == "dog":
            return {"self": ["dog"], "sibling": ["cat", "horse", "mouse"],
                    "hypernym": ["canine"], "hyponym": []}
        return None

    def pool(pos, min_zipf):
        return {"table": (4.9, "noun.artifact"), "window": (4.8, "noun.artifact"),
                "cat": (5.0, "noun.animal"), "horse": (5.0, "noun.animal"),
                "bird": (4.9, "noun.animal"), "fish": (4.8, "noun.animal")}

    def lexname(word, pos):
        return "noun.animal" if word == "dog" else None

    items = build_stimuli(noun_anchors=["dog"], adj_anchors=[], max_per_relation=2,
                          n_controls=2, relatives_fn=relatives, pool_fn=pool,
                          lexname_fn=lexname)
    rels = [it["relation"] for it in items]
    assert rels.count("self") == 1
    assert rels.count("sibling") == 2
    assert rels.count("unrelated") == 2
    assert rels.count("unrelated_same") == 2
    controls = [it["word"] for it in items if it["relation"] == "unrelated"]
    assert set(controls) == {"table", "window"}
    hard = [it["word"] for it in items if it["relation"] == "unrelated_same"]
    assert set(hard) == {"bird", "fish"}
    assert all(it["pos"] == "noun" for it in items)
    again = build_stimuli(noun_anchors=["dog"], adj_anchors=[], max_per_relation=2,
                          n_controls=2, relatives_fn=relatives, pool_fn=pool,
                          lexname_fn=lexname)
    assert [it["word"] for it in items] == [it["word"] for it in again]


def test_anchor_groups_and_labels():
    items = _two_anchor_items()
    groups = anchor_groups(items)
    labels = relation_labels(items)
    assert len(groups) == 2
    assert sum(len(g) for g in groups) == len(items)
    assert set(labels) == {"self", "sibling", "unrelated"}
    assert relation_names("noun", items)[:3] == ["self", "sibling", "hypernym"]


def test_relation_profile_symmetric():
    items = _two_anchor_items()
    D = _clustered_distance(items)
    names = ["self", "sibling", "unrelated"]
    prof = relation_profile(D, relation_labels(items), anchor_groups(items), names)
    M = np.asarray(prof["mean"])
    assert M.shape == (3, 3)
    assert np.allclose(M, M.T)
    assert M[1, 2] > M[1, 1]
    assert np.isclose(M[1, 1], 0.1)


def test_relation_pair_distance():
    items = _two_anchor_items()
    D = _clustered_distance(items)
    close = relation_pair_distance(D, relation_labels(items), anchor_groups(items),
                                   "self", "sibling")
    far = relation_pair_distance(D, relation_labels(items), anchor_groups(items),
                                 "self", "unrelated")
    assert close["count"] == 4
    assert close["mean"] < far["mean"]


def test_relation_separation_and_permutation():
    items = _two_anchor_items()
    D = _clustered_distance(items)
    labels = relation_labels(items)
    groups = anchor_groups(items)
    auc = relation_separation(D, labels, groups, ["sibling"])["sibling"]
    assert auc > 0.9
    perm = separation_permutation(D, labels, groups, "sibling", n_perm=20, seed=0)
    assert 0.0 < perm["p"] <= 1.0
    assert perm["auc"] == auc


def test_relation_gradient_singleton():
    items = make_items([
        (0, "a", "self", "a_self"), (0, "a", "hypernym", "a_hyp"),
        (0, "a", "unrelated", "a_u1"), (0, "a", "unrelated", "a_u2"),
        (1, "b", "self", "b_self"), (1, "b", "hypernym", "b_hyp"),
        (1, "b", "unrelated", "b_u1"), (1, "b", "unrelated", "b_u2"),
    ])
    coords = np.asarray([0.0, 0.2, 5.0, 5.0, 10.0, 10.2, 15.0, 15.0])
    D = np.abs(coords[:, None] - coords[None, :])
    labels = relation_labels(items)
    groups = anchor_groups(items)
    names = ["self", "hypernym", "unrelated"]
    grad = relation_gradient(D, labels, groups, names)["hypernym"]
    assert grad == 1.0


def test_relation_gradient_hard_control():
    items = make_items([
        (0, "a", "self", "a_self"), (0, "a", "sibling", "a_sib"),
        (0, "a", CONTROL_RELATION, "a_easy"),
        (0, "a", HARD_CONTROL_RELATION, "a_hard"),
    ])
    coords = np.asarray([0.0, 0.2, 5.0, 0.1])
    D = np.abs(coords[:, None] - coords[None, :])
    labels = relation_labels(items)
    groups = anchor_groups(items)
    names = ["self", "sibling", CONTROL_RELATION, HARD_CONTROL_RELATION]
    easy = relation_gradient(D, labels, groups, names)["sibling"]
    hard = relation_gradient(D, labels, groups, names,
                             control=HARD_CONTROL_RELATION)["sibling"]
    assert easy == 1.0
    assert hard == 0.0


def test_hellinger_and_pairwise_rdm():
    P = np.asarray([[0.5, 0.5], [0.5, 0.5], [1.0, 0.0]])
    D = hellinger_rdm(P)
    assert np.allclose(np.diag(D), 0.0)
    assert np.allclose(D, D.T)
    assert np.isclose(D[0, 1], 0.0)
    assert np.isclose(D[0, 2], np.sqrt(1.0 - np.sqrt(0.5)))

    X = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    C = pairwise_rdm(X, "cosine")
    assert np.isclose(C[0, 1], 0.0)
    assert np.isclose(C[0, 2], 1.0)
    E = pairwise_rdm(X, "euclid")
    assert np.isclose(E[0, 2], np.sqrt(2.0))


def test_pair_divergence_identical_is_zero():
    P = np.asarray([[0.3, 0.7], [0.3, 0.7], [0.9, 0.1]])
    items = make_items([(0, "a", "self", "x"), (0, "a", "sibling", "y"),
                        (0, "a", "unrelated", "z")])
    labels = relation_labels(items)
    groups = anchor_groups(items)
    js = pair_divergence(P, labels, groups, "self", "sibling", "js")["mean"]
    assert np.isclose(js, 0.0)


def test_residualize_coords_removes_covariate():
    rng = np.random.default_rng(0)
    c = rng.normal(size=30)
    X = np.column_stack([3.0 * c + 0.01 * rng.normal(size=30),
                         -2.0 * c + 0.01 * rng.normal(size=30)])
    R = residualize_coords(X, c[:, None])
    for j in range(R.shape[1]):
        assert abs(np.corrcoef(R[:, j], c)[0, 1]) < 1e-6


def test_state_layer_metrics_and_increment():
    rng = np.random.default_rng(0)
    items = _two_anchor_items()
    H = rng.normal(size=(len(items), 3, 5))
    names = ["self", "sibling", "unrelated"]
    layers = state_layer_metrics(H, relation_labels(items), anchor_groups(items),
                                 names, "cosine")
    assert len(layers) == 3
    assert layers[0]["layer"] == 0
    assert set(layers[0]["separation"]) == set(names)

    inc = increment_stats(np.transpose(H, (1, 0, 2)))
    assert len(inc["norm_mean"]) == 2
    assert all(v >= 0 for v in inc["norm_mean"])


def test_profile_correlation_identity():
    items = _two_anchor_items()
    D = _clustered_distance(items)
    prof = relation_profile(D, relation_labels(items), anchor_groups(items),
                            ["self", "sibling", "unrelated"])
    assert np.isclose(profile_correlation(prof, prof), 1.0)


def test_analyze_distribution_smoke():
    rng = np.random.default_rng(0)
    items = _two_anchor_items()
    P = rng.dirichlet(0.5 * np.ones(12), size=len(items))
    res = analyze_distribution(P, items, n_perm=10)
    assert res["n_items"] == len(items)
    assert "raw" in res and "residualized" in res
    assert "sibling" in res["raw"]["separation"]
    assert {"self", "sibling", "unrelated"}.issubset(set(res["names"]))


def test_tsv_roundtrip(tmp_path):
    items = _two_anchor_items()
    path = tmp_path / "stim.tsv"
    write_tsv(items, path)
    back = read_tsv(path)
    cols = ["pos", "anchor_id", "anchor", "relation", "word", "zipf", "n_chars"]
    assert len(back) == len(items)
    for got, exp in zip(back, items):
        assert {k: got[k] for k in cols} == {k: exp[k] for k in cols}


def test_stimuli_text():
    items = make_items([(0, "dog", "self", "dog")])
    assert stimuli_text(items, "When we say {x}, we mean") == \
        ["When we say dog, we mean"]
