"""Offline tests for stage 1 (typicality / prototype) helpers."""

import numpy as np

from tc.typicality import (
    _even_indices,
    channel_summary,
    judgment_prompts,
    judgment_score,
    limit_per_category,
    load_norms,
    mean_partial_rho,
    mean_within_rho,
    multiple_baseline_rho,
    overall_verdict,
    per_category_rho,
    permutation_pvalue,
    prototype_cv,
    prototype_summary,
    ranking_probes,
    select_subset,
    signature_matrix,
    signature_prompts,
    write_norms,
)


def _synthetic_rows():
    rows = []
    for category, prefix in (("mammal", "m"), ("fruit", "f")):
        for k in range(10):
            rows.append({"category": category, "item": f"{prefix}{k}",
                         "typicality": k / 9.0})
    rows.append({"category": "mammal", "item": "shared", "typicality": 0.9})
    rows.append({"category": "fruit", "item": "shared", "typicality": 0.9})
    rows.append({"category": "mammal", "item": "mammal", "typicality": 1.0})
    rows.append({"category": "fruit", "item": "fruit", "typicality": 1.0})
    return rows


def test_probe_builders():
    prompts = judgment_prompts("jeans", "clothing")
    assert len(prompts) == 2
    for prompt in prompts:
        assert "jeans" in prompt and "clothing" in prompt and "{" not in prompt

    probes = ranking_probes("jeans", "clothing")
    assert len(probes) == 2
    for prompt, continuation in probes:
        assert "clothing" in prompt and "jeans" in continuation

    signatures = signature_prompts("jeans")
    assert len(signatures) == 2
    for prompt in signatures:
        assert "jeans" in prompt and "clothing" not in prompt and "{" not in prompt


def test_judgment_score():
    assert judgment_score(1.0, 0.0) == 1.0
    assert judgment_score(0.0, 1.0) == 0.0
    assert abs(judgment_score(1.0, 1.0) - 0.5) < 1e-12
    assert judgment_score(0.0, 0.0) == 0.0


def test_load_write_roundtrip(tmp_path):
    rows = [{"category": "mammal", "item": "dog", "typicality": 0.84},
            {"category": "fruit", "item": "apple", "typicality": 0.83}]
    path = tmp_path / "typicality.tsv"
    write_norms(rows, path)
    loaded = load_norms(path)
    assert len(loaded) == 2
    assert loaded[0]["item"] == "dog"
    assert abs(loaded[1]["typicality"] - 0.83) < 1e-6
    assert loaded[0]["source"]


def test_even_indices_endpoints_and_size():
    assert _even_indices(10, 4) == [0, 3, 6, 9]
    assert _even_indices(5, 5) == [0, 1, 2, 3, 4]
    assert _even_indices(3, 2) == [0, 2]
    assert _even_indices(0, 3) == []
    for n, k in [(10, 1), (10, 7), (2, 2)]:
        idx = _even_indices(n, k)
        assert len(idx) == min(n, k)
        assert len(set(idx)) == len(idx)


def test_select_subset_is_deterministic_and_cleaned():
    rows = _synthetic_rows()
    selected = select_subset(rows, categories=["mammal", "fruit"], per_category=4)
    assert len(selected) == 8
    items = {(r["category"], r["item"]) for r in selected}
    assert all("shared" not in item for _, item in items)
    assert all(category != item for category, item in items)
    assert selected == select_subset(rows, categories=["mammal", "fruit"], per_category=4)

    for category in ("mammal", "fruit"):
        group = [r for r in selected if r["category"] == category]
        assert len(group) == 4
        assert max(r["typicality"] for r in group) == max(
            r["typicality"] for r in rows
            if r["category"] == category and r["item"] not in ("shared", category))
        assert min(r["typicality"] for r in group) == min(
            r["typicality"] for r in rows
            if r["category"] == category and r["item"] not in ("shared", category))


def test_limit_per_category_keeps_gradient_endpoints():
    rows = _synthetic_rows()
    limited = limit_per_category(rows, 3)
    assert len(limited) == 6
    for category in ("mammal", "fruit"):
        group = [r for r in limited if r["category"] == category]
        assert len(group) == 3
        assert max(r["typicality"] for r in group) == 1.0


def test_mean_within_rho_and_per_category():
    cats = np.array(["a"] * 10 + ["b"] * 10)
    target = np.tile(np.arange(10, dtype=float), 2)
    score = 2.0 * target + 1.0
    assert mean_within_rho(score, target, cats) > 0.99
    per = per_category_rho(score, target, cats)
    assert per["a"] > 0.99 and per["b"] > 0.99
    assert np.isnan(mean_within_rho(np.ones(20), target, cats))


def test_permutation_pvalue_detects_and_rejects():
    cats = np.array(["a"] * 10 + ["b"] * 10)
    target = np.tile(np.arange(10, dtype=float), 2)
    score = target.copy()
    _, p, _ = permutation_pvalue(score, target, cats, n_perm=200, seed=0)
    assert p < 0.02

    rng = np.random.default_rng(1)
    noise = rng.normal(size=20)
    _, p_noise, _ = permutation_pvalue(noise, target, cats, n_perm=200, seed=0)
    assert p_noise > 0.05


def test_partial_rho_removes_confound():
    rng = np.random.default_rng(2)
    cats = np.array(["a"] * 20 + ["b"] * 20)
    confound = np.tile(np.linspace(0.0, 1.0, 20), 2)
    score = rng.normal(size=40)
    target = score + 4.0 * confound
    raw = mean_within_rho(score, target, cats)
    partial = mean_partial_rho(score, target, [confound], cats)
    assert partial > 0.9
    assert partial > raw


def test_multiple_baseline_recovers_linear_controls():
    rng = np.random.default_rng(3)
    cats = np.array(["a"] * 20 + ["b"] * 20)
    freq = rng.normal(size=40)
    length = rng.normal(size=40)
    target = 2.0 * freq - 1.5 * length
    rho = multiple_baseline_rho(target, [freq, length], cats)
    assert rho > 0.99


def test_signature_matrix_restricts_and_normalises():
    rng = np.random.default_rng(4)
    signatures = [rng.random(60) for _ in range(5)]
    matrix, support = signature_matrix(signatures, k=10)
    assert matrix.shape == (5, len(support))
    assert len(support) <= 50
    assert np.allclose(matrix.sum(axis=1), 1.0)


def test_prototype_cv_recovers_gradient():
    rng = np.random.default_rng(5)
    vocab = 100
    base = np.zeros(vocab)
    base[:10] = 1.0
    other1 = np.zeros(vocab)
    other1[10:20] = 1.0
    other2 = np.zeros(vocab)
    other2[20:30] = 1.0
    signatures, cats, target = [], [], []
    for category in ("a", "b"):
        for k in range(20):
            z = (k - 9.5) / 9.5
            if z >= 0:
                dist = (1.0 - z) * base + z * other1
            else:
                dist = (1.0 + z) * base + (-z) * other2
            dist = dist + 0.01 * rng.random(vocab)
            signatures.append(dist / dist.sum())
            cats.append(category)
            target.append(1.0 - abs(z))
    cats = np.array(cats)
    target = np.array(target)
    scores, nearest = prototype_cv(signatures, cats, [f"i{k}" for k in range(40)],
                                   k=10, folds=4, seed=0)
    assert len(scores) == 40
    assert not np.isnan(scores).any()
    assert mean_within_rho(scores, target, cats) > 0.9
    assert set(nearest) == {"a", "b"}


def test_channel_summary_and_verdict_levels():
    cats = np.array([c for c in ("a", "b", "c", "d", "e") for _ in range(10)])
    target = np.tile(np.arange(10, dtype=float), 5)
    rng = np.random.default_rng(6)
    freq = rng.normal(size=50)
    length = rng.normal(size=50)
    static = rng.normal(size=50)

    good = channel_summary("good", target.copy(), target, cats, freq, length,
                           static_cos=static, n_perm=200)
    assert good["P1a"] and good["P1b"]
    assert good["verdict"] == "P1a+P1b"

    bad = channel_summary("bad", rng.normal(size=50), target, cats, freq, length,
                          static_cos=static, n_perm=200)
    assert bad["verdict"] == "absent"


def test_prototype_summary_and_overall_verdict():
    cats = np.array([c for c in ("a", "b", "c", "d", "e") for _ in range(10)])
    target = np.tile(np.arange(10, dtype=float), 5)
    rng = np.random.default_rng(7)
    freq = rng.normal(size=50)
    length = rng.normal(size=50)
    scores = target + 0.01 * rng.normal(size=50)
    proto = prototype_summary("sig", scores, target, cats, freq, length, n_perm=200)
    assert proto["P1c"]

    channels = {
        "one": {"P1a": True, "P1b": True},
        "two": {"P1a": True, "P1b": True},
        "three": {"P1a": False, "P1b": False},
    }
    prototypes = {"sig0": {"P1c": True}, "sig1": {"P1c": False}}
    verdict = overall_verdict(channels, prototypes)
    assert verdict["typicality"] == "readout-robust"
    assert verdict["prototype"] == "readout-local"
    assert set(verdict["robust_channels"]) == {"one", "two"}
