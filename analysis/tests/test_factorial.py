"""Unit tests for the factorial corpus (no model): minimality and split."""

from tc.factorial import ATTRS, TEMPLATES, VALUES, build_corpus, build_pairs, render


def test_corpus_size():
    corpus = build_corpus()
    assert len(corpus) == len(TEMPLATES) * 2 ** len(ATTRS) * 3


def test_pairs_differ_only_in_target_attribute():
    for p in build_pairs():
        a, syn = p["attr"], p["syn"]
        # different attribute values exactly in `a`
        diff = {k for k in ATTRS if p["values_base"][k] != p["values_source"][k]}
        assert diff == {a}
        # textual minimality: replace the target phrase by a placeholder -> equal
        base = p["base"].replace(VALUES[a][p["values_base"][a]][syn], "<X>")
        source = p["source"].replace(VALUES[a][p["values_source"][a]][syn], "<X>")
        assert base == source
        assert p["base"] != p["source"]


def test_render_uses_all_attributes():
    values = {a: 0 for a in ATTRS}
    text = render(values, 0, 0)
    for a in ATTRS:
        assert VALUES[a][0][0] in text
