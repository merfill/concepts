"""Unit tests for the interpolation-validity instrument (experiment 006).

No model: pure geometry (straight / curved / jump), behavioral curve comparison
and path-independence are validated on synthetic paths, so the model run only
confirms what is already locked here.
"""

import numpy as np

from tc.factorial import ATTRS
from tc.interpolate import (
    angular_metrics,
    background_stats,
    behavioral_curve,
    behavioral_profile,
    factorial_ladder,
    flip_orders,
    graded_texts,
    interpolation_verdict,
    order_step_signature,
    pairwise_consistency,
    path_metrics,
    remove_pcs_rows,
    standardize_rows,
    unit_rows,
)


def _orthogonal(rng, v, d):
    u = rng.standard_normal(d)
    u = u - (u @ v) / (v @ v) * v
    return u / np.linalg.norm(u)


def test_graded_texts_progress_and_length():
    g = graded_texts("temperature")
    assert len(g["texts"]) == 5
    assert g["progress"][0] == 0.0 and g["progress"][-1] == 1.0
    assert g["texts"][0] == "The water is freezing."


def test_factorial_ladder_flips_one_at_a_time():
    order = ("diet", "sociality", "body", "nest")
    lad = factorial_ladder(order)
    assert len(lad["texts"]) == 5
    assert all(v == 0 for v in lad["scenes"][0].values())
    assert all(v == 1 for v in lad["scenes"][-1].values())
    for a, b in zip(lad["scenes"][:-1], lad["scenes"][1:]):
        diff = [k for k in ATTRS if a[k] != b[k]]
        assert len(diff) == 1
    assert len(flip_orders()) == 24


def test_straight_path_metrics():
    rng = np.random.default_rng(0)
    d, K = 6, 4
    h0, delta = rng.standard_normal(d), rng.standard_normal(d)
    ts = np.linspace(0, 1, K + 1)
    H = h0 + np.outer(ts, delta)
    m = path_metrics(H)
    assert abs(m["monotonicity"] - 1.0) < 1e-9
    assert m["residual_mean"] < 1e-9
    assert abs(m["length_ratio"] - 1.0) < 1e-9
    assert abs(m["step_cos_mean"] - 1.0) < 1e-9
    assert abs(m["single_step_share"] - 1.0 / K) < 1e-9


def test_curved_path_has_residual_and_extra_length():
    rng = np.random.default_rng(1)
    d, K = 6, 4
    h0, delta = rng.standard_normal(d), rng.standard_normal(d)
    nd = np.linalg.norm(delta)
    u = _orthogonal(rng, delta, d)
    ts = np.linspace(0, 1, K + 1)
    H = h0 + np.outer(ts, delta) + np.outer(np.sin(np.pi * ts), 0.6 * nd * u)
    m = path_metrics(H)
    assert m["residual_mean"] > 0.15
    assert m["length_ratio"] > 1.15
    assert m["monotonicity"] > 0.99          # still monotone along the chord
    assert interpolation_verdict(m, None).startswith("I2")


def test_jump_path_flags_i3():
    rng = np.random.default_rng(2)
    d = 6
    h0, delta = rng.standard_normal(d), rng.standard_normal(d)
    ts = np.array([0.0, 0.9, 0.9, 0.9, 1.0])
    H = h0 + np.outer(ts, delta)
    m = path_metrics(H)
    assert m["single_step_share"] > 0.6
    assert interpolation_verdict(m, None).startswith("I3")


def test_zigzag_but_monotone_is_i2_not_i3():
    rng = np.random.default_rng(5)
    d, K = 6, 4
    h0, delta = rng.standard_normal(d), rng.standard_normal(d)
    nd = np.linalg.norm(delta)
    u = _orthogonal(rng, delta, d)
    ts = np.linspace(0, 1, K + 1)
    zig = np.array([0, 1, -1, 1, 0], dtype=float) * 0.3 * nd
    H = h0 + np.outer(ts, delta) + np.outer(zig, u)
    m = path_metrics(H)
    assert m["monotonicity"] > 0.99
    assert m["step_cos_mean"] < 0.0
    assert m["single_step_share"] < 0.6
    assert interpolation_verdict(m, None).startswith("I2")


def test_behavioral_curve_identity_match():
    ts = np.linspace(0, 1, 5)

    def prob(t):
        p = np.zeros(4)
        p[0], p[1] = 1.0 - t, t
        return p

    nat = [prob(t) for t in ts]
    steered = [prob(t) for t in ts]
    c = behavioral_curve(nat, steered, k=4)
    assert abs(c["match_rho"] - 1.0) < 1e-9
    assert c["natural_monotonicity"] > 0.99


def test_path_independence_commutative_vs_order_dependent():
    rng = np.random.default_rng(3)
    d = 5
    base = rng.standard_normal(d)
    deltas = {a: rng.standard_normal(d) for a in ATTRS}
    orders = flip_orders()

    def build(fn):
        hs = []
        for order in orders:
            x = base.copy()
            H = [x.copy()]
            for pos, a in enumerate(order):
                x = x + fn(a, pos)
                H.append(x.copy())
            hs.append(np.array(H))
        return [order_step_signature(H, o) for H, o in zip(hs, orders)]

    commutative = build(lambda a, pos: deltas[a])
    assert pairwise_consistency(commutative) > 1 - 1e-9

    def order_dependent(a, pos):
        sign = 1.0 if pos >= 2 else -1.0
        return sign * deltas[a]

    assert pairwise_consistency(build(order_dependent)) < 0.5


def test_unit_rows_are_scale_invariant():
    rng = np.random.default_rng(6)
    H = rng.standard_normal((5, 7)) + 3.0
    a = path_metrics(unit_rows(H))
    b = path_metrics(unit_rows(4.0 * H))
    assert np.allclose(a["t"], b["t"], atol=1e-10)
    assert abs(a["residual_mean"] - b["residual_mean"]) < 1e-10


def test_background_standardize_and_pc_removal():
    rng = np.random.default_rng(7)
    bg = rng.standard_normal((60, 6))
    mu, sigma, comps = background_stats(bg, n_pcs=2)
    Xw = standardize_rows(bg, mu, sigma)
    assert abs(float(Xw.mean())) < 1e-9
    assert np.allclose(Xw.std(axis=0), 1.0, atol=1e-8)
    Xr = remove_pcs_rows(bg, mu, sigma, comps)
    assert np.allclose(Xr @ comps.T, 0.0, atol=1e-9)


def test_angular_metrics_on_great_circle():
    d = 4
    e1, e2 = np.zeros(d), np.zeros(d)
    e1[0], e2[1] = 1.0, 1.0
    ang = np.linspace(0, np.pi / 2, 5)
    H = np.outer(np.cos(ang), e1) + np.outer(np.sin(ang), e2)
    m = angular_metrics(H)
    assert abs(m["geodesic_length_ratio"] - 1.0) < 1e-6
    assert abs(m["step_angle_mean"] - (np.pi / 2) / 4) < 1e-6


def test_behavioral_profile_is_graded():
    ts = np.linspace(0, 1, 5)

    def prob(t):
        p = np.zeros(4)
        p[0], p[1] = 1.0 - t, t
        return p

    prof = behavioral_profile([prob(t) for t in ts], k=4)
    assert prof["js_endpoints"] > 0
    assert all(a > 0 for a in prof["js_adjacent"])
    assert prof["js_from_start_monotonicity"] > 0.99


def test_interpolation_verdict_i1_with_behavioral_match():
    rng = np.random.default_rng(4)
    d, K = 6, 4
    h0, delta = rng.standard_normal(d), rng.standard_normal(d)
    H = h0 + np.outer(np.linspace(0, 1, K + 1), delta)
    m = path_metrics(H)
    assert interpolation_verdict(m, 0.9).startswith("I1")
