"""Unit tests for shared-support next-token distances (no model download).

These lock the protocol-level definitions of experiment 001: comparison is done
on the union of the two top-k supports, renormalized, so distances are defined
and bounded.
"""

import numpy as np

from tc.metrics import jaccard_words, js_topk, js_union, shared_topk, top_indices, tv_topk


def _rand_dist(rng, n, seed=0):
    p = rng.random(n)
    return p / p.sum()


def test_top_indices_sorted():
    p = np.array([0.1, 0.5, 0.3, 0.1])
    assert list(top_indices(p, 2)) == [1, 2]


def test_shared_topk_renormalized():
    rng = np.random.default_rng(0)
    p, q = _rand_dist(rng, 20), _rand_dist(rng, 20)
    pp, qq = shared_topk(p, q, 5)
    assert pp.shape == qq.shape
    assert abs(pp.sum() - 1.0) < 1e-9
    assert abs(qq.sum() - 1.0) < 1e-9


def test_js_topk_self_zero_and_symmetric():
    rng = np.random.default_rng(1)
    p, q = _rand_dist(rng, 30), _rand_dist(rng, 30)
    assert js_topk(p, p, 10) < 1e-12
    assert abs(js_topk(p, q, 10) - js_topk(q, p, 10)) < 1e-12


def test_js_topk_bounded():
    rng = np.random.default_rng(2)
    p, q = _rand_dist(rng, 30), _rand_dist(rng, 30)
    v = js_topk(p, q, 10)
    assert 0.0 <= v <= np.log(2) + 1e-9


def test_tv_topk_bounded():
    rng = np.random.default_rng(3)
    p, q = _rand_dist(rng, 30), _rand_dist(rng, 30)
    assert 0.0 <= tv_topk(p, q, 10) <= 1.0 + 1e-9


def test_js_union_self_zero_and_symmetric():
    a = {(1, 2): 0.5, (1, 3): 0.3, (4, 5): 0.2}
    b = {(1, 2): 0.6, (4, 5): 0.4}
    assert js_union(a, a) < 1e-12
    assert abs(js_union(a, b) - js_union(b, a)) < 1e-12
    assert js_union(a, b) > 0.0


def test_jaccard_basic():
    assert jaccard_words("a bee makes honey", "a bee produces honey") == 3 / 5
    assert jaccard_words("", "") == 1.0
    assert jaccard_words("bee", "wasp") == 0.0
