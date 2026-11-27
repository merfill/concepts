"""Unit tests for the hierarchy calibration instrument (experiment 012).

Model-free only: rendering, normalisation, and the ``Q`` / ``MI`` /
frame-agreement instruments are locked on synthetic data before any transformer
is loaded (same philosophy as ``test_rdm.py``).
"""

import math

import numpy as np
import pytest

from tc.hierarchy import (
    article,
    entropy,
    frame_agreement,
    mutual_information,
    normalize_probs,
    permutation_pvalue,
    render,
)


def test_article():
    assert article("dog") == "a"
    assert article("animal") == "an"


def test_render_slots_and_capitalisation():
    assert render("{X} is a {Y}.", x="dog", y="animal") == "Dog is a animal."
    assert render("Is {aX} a {Y}?", x="apple", y="fruit") == "Is an apple a fruit?"
    assert render("{X}, such as", x="dog") == "Dog, such as"


def test_normalize_probs_sums_to_one_and_fills_missing():
    raw = {"a": 1.0, "b": 3.0}
    dist = normalize_probs(raw, ["a", "b", "c"])
    assert dist["a"] == pytest.approx(0.25)
    assert dist["b"] == pytest.approx(0.75)
    assert dist["c"] == 0.0
    assert sum(dist.values()) == pytest.approx(1.0)


def test_normalize_probs_zero_total_is_flat_zero():
    dist = normalize_probs({"a": 0.0}, ["a", "b"])
    assert all(v == 0.0 for v in dist.values())


def test_entropy_uniform_and_point_mass():
    assert entropy({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}) == pytest.approx(math.log(4))
    assert entropy({"a": 1.0, "b": 0.0}) == pytest.approx(0.0)


def test_mutual_information_identical_reads_is_zero():
    d = {"a": 0.5, "b": 0.5}
    assert mutual_information([d, d, d]) == pytest.approx(0.0, abs=1e-12)


def test_mutual_information_unique_support_is_log_n():
    dists = [{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}]
    assert mutual_information(dists) == pytest.approx(math.log(2), abs=1e-12)


def test_mutual_information_none_above_identical():
    rng = np.random.default_rng(0)
    base = rng.random(6)
    base /= base.sum()
    identical = [dict(zip("abcdef", base)) for _ in range(5)]
    spread = [normalize_probs(dict(zip("abcdef", rng.random(6)))) for _ in range(5)]
    assert mutual_information(spread) > mutual_information(identical)


def test_frame_agreement_identical_is_one():
    a = [0.1, 0.2, 0.3, 0.4]
    assert frame_agreement(a, a) == pytest.approx(1.0)


def test_permutation_pvalue_uses_plus_one():
    assert permutation_pvalue(10.0, [1, 2, 3]) == pytest.approx(1.0 / 4.0)
    assert permutation_pvalue(0.0, [1, 2, 3]) == pytest.approx(1.0)


def test_triad_concepts_flattens_levels():
    from tc.hierarchy import triad_concepts

    triads = [{"super": "animal", "basic": "dog", "sub": "poodle"}]
    assert triad_concepts(triads) == [
        ("animal", "super", 0),
        ("dog", "basic", 0),
        ("poodle", "sub", 0),
    ]


def test_load_triads_filters_split(tmp_path):
    from tc.hierarchy import load_triads

    path = tmp_path / "triads.tsv"
    path.write_text(
        "super\tbasic\tsub\tsplit\n"
        "animal\tdog\tpoodle\tcalibration\n"
        "vehicle\tcar\tsedan\tmain\n",
        encoding="utf-8",
    )
    calib = load_triads(path, split="calibration")
    assert len(calib) == 1 and calib[0]["basic"] == "dog"
    assert len(load_triads(path)) == 2


def _triads():
    return [
        {"super": "animal", "basic": "dog", "sub": "poodle", "split": "main"},
        {"super": "vehicle", "basic": "car", "sub": "sedan", "split": "main"},
        {"super": "plant", "basic": "tree", "sub": "oak", "split": "main"},
        {"super": "furniture", "basic": "chair", "sub": "armchair", "split": "main"},
        {"super": "tool", "basic": "saw", "sub": "hacksaw", "split": "main"},
    ]


def test_build_candidates_labels_and_step():
    from tc.hierarchy import build_candidates

    cands = build_candidates(_triads(), 0, n_sib=2, n_unrel=2, seed=0)
    assert set(cands) == {("animal", "super"), ("dog", "basic")}
    sup = cands[("animal", "super")]
    bas = cands[("dog", "basic")]
    assert sup["dog"] == "desc"
    assert sup["poodle"] == "desc_wrong_level"
    assert bas["poodle"] == "desc"
    assert bas["animal"] == "anc"
    # no candidate is both the child and a distractor
    assert sup["dog"] != "sib_branch"


def test_auc_perfect_and_chance():
    from tc.hierarchy import auc

    assert auc([0.9, 0.8, 0.1, 0.2], [True, True, False, False]) == pytest.approx(1.0)
    assert auc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]) == pytest.approx(0.5)


def test_descendant_metrics_mass_and_top():
    from tc.hierarchy import descendant_metrics

    labels = {"dog": "desc", "car": "sib_branch", "oak": "unrelated"}
    dist = {"dog": 0.6, "car": 0.3, "oak": 0.1}
    out = descendant_metrics(dist, labels)
    assert out["desc_mass"] == pytest.approx(0.6)
    assert out["top"] == "dog" and out["top_is_desc"]
    assert out["auc"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# WordNet fragment graph core (no nltk: synthetic fragment)
# --------------------------------------------------------------------------- #

def _fragment():
    from tc.hierarchy import build_fragment

    edges = [
        ("animal", "dog"), ("animal", "cat"), ("animal", "horse"),
        ("dog", "poodle"), ("dog", "beagle"), ("cat", "siamese"),
        ("vehicle", "car"), ("vehicle", "truck"), ("car", "sedan"),
    ]
    words = {n: n for n in {"animal", "dog", "cat", "horse", "poodle",
                            "beagle", "siamese", "vehicle", "car", "truck",
                            "sedan"}}
    return build_fragment(edges, words, ["animal", "vehicle"])


def test_build_fragment_depth_and_root():
    frag = _fragment()
    assert frag["nodes"]["animal"]["depth"] == 0
    assert frag["nodes"]["dog"]["depth"] == 1
    assert frag["nodes"]["poodle"]["depth"] == 2
    assert frag["nodes"]["car"]["root"] == "vehicle"
    assert frag["nodes"]["sedan"]["depth"] == 2


def test_fragment_descendants_and_ancestors():
    from tc.hierarchy import fragment_ancestors, fragment_descendants

    frag = _fragment()
    assert fragment_descendants(frag, "animal") == {
        "dog", "cat", "horse", "poodle", "beagle", "siamese"}
    assert fragment_ancestors(frag, "poodle") == {"dog", "animal"}


def test_fragment_concepts_excludes_roots_and_leaves():
    from tc.hierarchy import fragment_concepts

    concepts = fragment_concepts(_fragment())
    assert "animal" not in concepts and "vehicle" not in concepts
    assert "dog" in concepts
    # leaves have no children, so they cannot be descent targets
    assert "poodle" not in concepts


def test_labelled_candidates_roles_do_not_overlap():
    from tc.hierarchy import labelled_candidates

    labels = labelled_candidates(_fragment(), "dog", n_desc=4, n_anc=2,
                                 n_wrong=2, n_sib=4, n_unrel=4, seed=0)
    assert labels["poodle"] == "desc" and labels["beagle"] == "desc"
    assert labels["animal"] == "anc"
    assert labels["cat"] == "sib_branch"
    assert labels["car"] == "unrelated"
    # exactly one label per word, and no descendant is also a distractor
    assert len(labels) == len(set(labels))


def test_fragment_entailment_pairs_true_and_reverse():
    from tc.hierarchy import fragment_entailment_pairs

    pairs = fragment_entailment_pairs(_fragment(), max_distance=2)
    by_pair = {(x, y): (t, d) for x, y, t, d in pairs}
    assert by_pair[("poodle", "dog")] == (1, 1)
    assert by_pair[("poodle", "animal")] == (1, 2)
    assert by_pair[("dog", "animal")] == (1, 1)
    assert by_pair[("animal", "dog")] == (0, 0)


def test_write_and_load_fragment_roundtrip(tmp_path):
    from tc.hierarchy import load_fragment, write_fragment

    frag = _fragment()
    path = tmp_path / "frag.tsv"
    write_fragment(frag, path)
    loaded = load_fragment(path)
    assert set(loaded["nodes"]) == set(frag["nodes"])
    assert loaded["nodes"]["poodle"]["depth"] == 2
    assert loaded["nodes"]["car"]["root"] == "vehicle"
