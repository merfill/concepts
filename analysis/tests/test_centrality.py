import math

import numpy as np
import pytest

from tc.centrality import (
    article,
    candidate_frames,
    compute_purity,
    is_function_token,
    member_match,
    partial_pearson,
)
from tc.lexicon import normalize


def test_normalize():
    assert normalize(" Polar Bear ") == "polar_bear"
    assert normalize("dog") == "dog"


def test_is_function_token():
    assert is_function_token("a")
    assert is_function_token("the")
    assert is_function_token("*")
    assert is_function_token("\u201c")
    assert not is_function_token("dog")
    assert not is_function_token("large")


def test_member_match_exact_and_plural():
    members = {"dog", "polar_bear"}
    assert member_match("dog", members) == "dog"
    assert member_match("dogs", members) == "dog"
    assert member_match("dog.", members) is None  # punctuation is not stripped here
    assert member_match("cat", members) is None


def test_compute_purity_partitions_mass():
    top = [("dog", 0.5), ("a", 0.2), ("large", 0.3)]
    out = compute_purity(top, {"dog"})
    assert out["member_mass"] == pytest.approx(0.5)
    assert out["function_mass"] == pytest.approx(0.2)
    assert out["other_mass"] == pytest.approx(0.3)
    assert out["window_mass"] == pytest.approx(1.0)
    assert out["purity"] == pytest.approx(0.5)
    assert [m["token"] for m in out["members"]] == ["dog"]
    assert [o["token"] for o in out["other"]] == ["large"]


def test_compute_purity_empty_window():
    out = compute_purity([], {"dog"})
    assert math.isnan(out["purity"])


def test_article_and_candidate_frames():
    assert article("bird") == "a"
    assert article("instrument") == "an"
    assert article("item", capitalize=True) == "An"
    frames = candidate_frames("bird")
    assert "A bird, such as a" in frames
    assert "Name one bird: a" in frames
    instr = candidate_frames("musical instrument")
    assert "An instrument used to make music, such as a" in instr
    assert "Name one instrument used to make music: a" in instr


def test_partial_pearson_identity_and_control():
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    assert partial_pearson(x, 2 * x + 1, np.ones(40)) == pytest.approx(1.0)

    confound = rng.normal(size=40)
    y = 3 * confound + 0.01 * rng.normal(size=40)
    raw = np.corrcoef(x, y)[0, 1]
    controlled = partial_pearson(x, y, confound)
    assert abs(controlled) < max(abs(raw), 0.5)
