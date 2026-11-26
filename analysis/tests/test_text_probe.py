import math

import pytest

from tc.text_probe import (
    normalize_phrase,
    parse_list,
    parse_ranking,
    parse_rating,
    ranking_spearman,
    rating_prompt,
)


def test_normalize_phrase():
    assert normalize_phrase("1. A pair of jeans.") == "jeans"
    assert normalize_phrase('"Hammer"') == "hammer"
    assert normalize_phrase("- an apple") == "apple"
    assert normalize_phrase("The  tape measure") == "tape measure"


def test_parse_list_matches_and_pair_of():
    members = {"hammer", "screwdriver", "jeans", "tape_measure"}
    out = parse_list("hammer, screwdriver, pair of jeans, computer, tape measure", members)
    assert out["matched"] == ["hammer", "screwdriver", "jeans", "tape_measure"]
    assert out["unmatched"] == ["computer"]


def test_parse_list_dedupes_and_numbering():
    members = {"dog", "cat"}
    out = parse_list("1. Dog\n2. cat\n3. dog", members)
    assert out["matched"] == ["dog", "cat"]


def test_parse_ranking_handles_multiword_overlap():
    words = ["bear", "polar bear", "dog"]
    out = parse_ranking("polar bear\ndog\nbear", words)
    assert out["ordered"] == ["polar bear", "dog", "bear"]


def test_parse_rating():
    words = ["screwdriver", "tape measure", "ratchet"]
    text = "1. screwdriver: 7\ntape measure - 5\nratchet 3"
    out = parse_rating(text, words)
    assert out == {"screwdriver": 7.0, "tape measure": 5.0, "ratchet": 3.0}


def test_rating_prompt_lists_words_and_scale():
    prompt = rating_prompt("tool", ["screwdriver", "hammer"], seed=0)
    assert "screwdriver" in prompt and "hammer" in prompt
    assert "1 (least typical)" in prompt and "7 (most typical)" in prompt


def test_ranking_spearman():
    human = {"a": 0.9, "b": 0.5, "c": 0.1}
    perfect = ranking_spearman(["a", "b", "c"], list(human), human)
    inverted = ranking_spearman(["c", "b", "a"], list(human), human)
    assert perfect == pytest.approx(1.0)
    assert inverted == pytest.approx(-1.0)
    assert math.isnan(ranking_spearman(["a"], list(human), human))
