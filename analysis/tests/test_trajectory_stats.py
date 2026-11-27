"""Offline tests for the trajectory-ensemble statistics (experiment 016b)."""

import numpy as np

from tc.trajectory_stats import (
    bootstrap_ci,
    class_report,
    logistic_rank_effect,
    outcome_classes,
    position_curve,
    reconvergence_by_rank,
    takeover_summary,
    wilson_interval,
)


def make_records(n, effect, seed=0, near=0.01):
    rng = np.random.default_rng(seed)
    ranks = rng.integers(2, 11, size=n)
    tstar = rng.integers(2, 39, size=n)
    ent = rng.random(n)
    if effect is None:
        y = rng.random(n) < 0.4
    else:
        logit = 1.0 - effect * ranks
        p = 1.0 / (1.0 + np.exp(-logit))
        y = rng.random(n) < p
    js = np.where(y, 0.0, 0.69)
    recs = []
    for i in range(n):
        recs.append({
            "prompt": "p", "rank": int(ranks[i]), "tstar": int(tstar[i]),
            "js_end": float(js[i]), "base_entropy": float(ent[i]),
            "dom_share": float(0.3 + 0.4 * y[i]),
            "tail_period": 2 if y[i] and i % 3 == 0 else 0,
            "entropy_after": float(0.7 - 0.4 * y[i]),
        })
    return recs


def test_bootstrap_and_wilson():
    lo, hi = bootstrap_ci(np.full(50, 3.0), n_boot=200)
    assert abs(lo - 3.0) < 1e-12 and abs(hi - 3.0) < 1e-12
    lo, hi = wilson_interval(5, 10)
    assert lo < 0.5 < hi
    assert 0.0 <= lo <= hi <= 1.0
    assert np.isnan(wilson_interval(0, 0)[0])


def test_outcome_classes_and_report():
    js = np.array([0.0] * 60 + [0.69] * 35 + [0.3] * 5)
    oc = outcome_classes(js)
    assert oc["n"] == 100
    assert abs(oc["frac_near0"] - 0.60) < 1e-12
    assert abs(oc["frac_high"] - 0.35) < 1e-12
    assert abs(oc["frac_middle"] - 0.05) < 1e-12
    rep = class_report(js, n_boot=300)
    assert set(rep["ci"]) == {"frac_near0", "frac_middle", "frac_high",
                             "frac_near_ln2"}
    for lo, hi in rep["ci"].values():
        assert 0.0 <= lo <= hi <= 1.0


def test_reconvergence_by_rank_and_position():
    recs = [{"rank": 2, "tstar": 6, "js_end": 0.0},
            {"rank": 2, "tstar": 6, "js_end": 0.69},
            {"rank": 3, "tstar": 20, "js_end": 0.0}]
    rr = reconvergence_by_rank(recs)
    assert rr["2"]["n"] == 2 and abs(rr["2"]["rate"] - 0.5) < 1e-12
    assert rr["3"]["rate"] == 1.0
    pc = position_curve(recs, [6, 20])
    assert pc["6"]["n"] == 2 and pc["20"]["rate"] == 1.0


def test_logistic_rank_effect_null_and_effect():
    null = make_records(1200, effect=None, seed=1)
    r0 = logistic_rank_effect(null, n_boot=200, seed=2)
    assert np.isfinite(r0["or_rank"])
    assert r0["excludes_one"] is False

    strong = make_records(1500, effect=0.7, seed=3)
    r1 = logistic_rank_effect(strong, n_boot=200, seed=4)
    assert r1["or_rank"] < 1.0
    assert r1["excludes_one"] is True


def test_takeover_summary_split():
    recs = make_records(300, effect=None, seed=5)
    ts = takeover_summary(recs, n_boot=300)
    assert ts["n"] == 300
    assert set(ts) >= {"all", "reconverged", "diverged"}
    if ts["reconverged"]["n"] > 0 and ts["diverged"]["n"] > 0:
        assert ts["reconverged"]["dom_share_mean"] > ts["diverged"]["dom_share_mean"]
