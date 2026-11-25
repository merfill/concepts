"""Offline tests for axis selection / validation helpers (experiment 008)."""

import numpy as np

from tc.axes import (
    axis_verdict,
    axis_verdict_anchored,
    adjacent,
    ladder,
    likert_expected,
    likert_prompt,
    names,
    ordering_accuracy,
    spearman_index,
    two_way,
)


def test_axes_well_formed():
    assert len(names()) >= 6
    for name in names():
        lad = ladder(name)
        assert 4 <= len(lad["words"]) <= 5
        assert len(lad["texts"]) == len(lad["words"])
        assert "{w}" not in lad["texts"][0]
        assert lad["texts"][0].endswith(".")


def test_adjacent_pairs():
    assert adjacent(["a", "b", "c"]) == [("a", "b"), ("b", "c")]


def test_two_way_is_a_distribution():
    for la, lb in [(0.0, 0.0), (2.0, -1.0), (-3.0, 1.0)]:
        pa, pb = two_way(la, lb)
        assert abs(pa + pb - 1.0) < 1e-12
        assert 0.0 <= pa <= 1.0 and 0.0 <= pb <= 1.0
    assert two_way(1.0, 0.0)[0] > 0.5


def test_ordering_accuracy_and_spearman():
    assert ordering_accuracy([(0.0, 2.0), (1.0, 0.5), (2.0, 3.0)]) == 2 / 3
    assert spearman_index([0, 1, 2, 3, 4]) > 0.99
    assert spearman_index([0, 1, 0, 1, 0]) < 0.5


def test_likert_helpers():
    prompt = likert_prompt("temperature", "cold")
    assert "cold" in prompt and "coldest" in prompt and "hottest" in prompt
    probs = np.array([0, 0.05, 0.1, 0.15, 0.3, 0.4])
    # digit ids 1..5 are positions 1..5 here
    val = likert_expected(probs, [1, 2, 3, 4, 5])
    assert 3.5 < val < 4.5


def test_axis_verdict_anchored_levels():
    cont = {"js_endpoints": 0.2, "js_adjacent": [0.02, 0.03, 0.02, 0.03]}
    assert axis_verdict_anchored(1.0, 0.9, cont)["verdict"] == "valid"
    assert axis_verdict_anchored(1.0, 0.9, {"js_endpoints": 0.0,
                                            "js_adjacent": [0.01, 0.1, 0.01, 0.01]}
                                 )["verdict"].startswith("valid-order")
    assert axis_verdict_anchored(0.5, 0.2, cont)["verdict"].startswith("weak")
    assert axis_verdict_anchored(0.9, 0.2, cont)["verdict"].startswith("partial")


def test_axis_verdict_levels():
    cont = {"js_endpoints": 0.2, "js_adjacent": [0.02, 0.03, 0.02, 0.03]}
    assert axis_verdict(1.0, 1.0, 0.9, cont)["verdict"] == "valid"

    weak_cont = {"js_endpoints": 0.01, "js_adjacent": [0.001, 0.05, 0.001, 0.001]}
    assert axis_verdict(1.0, 1.0, 0.9, weak_cont)["verdict"].startswith("valid-order")

    assert axis_verdict(0.5, 0.5, 0.2, cont)["verdict"].startswith("weak")
    assert axis_verdict(0.9, 0.9, 0.2, cont)["verdict"].startswith("partial")
