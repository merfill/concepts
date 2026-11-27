"""Offline tests for trajectory-geometry helpers (experiment 016)."""

import numpy as np

from tc.inverse import softmax
from tc.trajectory import (
    branch_top_index,
    distance_to_arc,
    euclid_path_length,
    fisher_norm,
    fisher_path_length,
    fisher_rao,
    geodesic_deviation,
    nucleus_index,
    null_index,
    participation_ratio,
    simplex_embed,
    simplex_geodesic,
    token_survival,
)


def rand_dist(v=50, seed=0):
    rng = np.random.default_rng(seed)
    return softmax(rng.standard_normal(v))


def x_to_p(x):
    x = np.asarray(x, dtype=float)
    return x * x / 4.0


def test_fisher_norm_gauge_and_variance():
    p = rand_dist()
    rng = np.random.default_rng(1)
    u = rng.standard_normal(p.size)
    mean = np.dot(p, u)
    var = np.dot(p, u * u) - mean * mean
    assert abs(fisher_norm(p, u) - np.sqrt(max(var, 0.0))) < 1e-12
    assert fisher_norm(p, np.full(p.size, 3.7)) < 1e-12
    assert fisher_norm(p, np.zeros(p.size)) == 0.0


def test_fisher_norm_matches_local_fisher_rao():
    rng = np.random.default_rng(2)
    z0 = rng.standard_normal(40)
    p0 = softmax(z0)
    dz = 1e-4 * rng.standard_normal(40)
    p1 = softmax(z0 + dz)
    local = fisher_norm(p0, dz)
    geodesic = fisher_rao(p0, p1)
    assert abs(local - geodesic) / max(geodesic, 1e-12) < 1e-3


def test_fisher_rao_properties():
    p = rand_dist(seed=3)
    q = rand_dist(seed=4)
    assert fisher_rao(p, p) < 1e-9
    assert abs(fisher_rao(p, q) - fisher_rao(q, p)) < 1e-12
    assert fisher_rao(p, q) > 0
    bc = np.sum(np.sqrt(p * q))
    assert abs(fisher_rao(p, q) - 2 * np.arccos(np.clip(bc, 0, 1))) < 1e-12


def test_fisher_path_length_positive_and_gauge_invariant():
    rng = np.random.default_rng(5)
    logits = [rng.standard_normal(30) for _ in range(6)]
    ps = [softmax(l) for l in logits]
    L = fisher_path_length(ps, logits)
    assert L > 0
    shifted = [l + 10.0 for l in logits]
    ps_shift = [softmax(l) for l in shifted]
    assert np.allclose(ps[0], ps_shift[0])
    assert abs(fisher_path_length(ps_shift, shifted) - L) < 1e-9


def test_euclid_path_length_scales():
    rng = np.random.default_rng(55)
    zs = [rng.standard_normal(8) for _ in range(5)]
    assert abs(euclid_path_length([2 * z for z in zs])
               - 2 * euclid_path_length(zs)) < 1e-9


def test_geodesic_points_have_zero_deviation():
    p_a = rand_dist(seed=6)
    p_b = rand_dist(seed=7)
    x_a, x_b = simplex_embed(p_a), simplex_embed(p_b)
    assert abs(np.linalg.norm(x_a) - 2.0) < 1e-12
    xs = [simplex_geodesic(x_a, x_b, s) for s in np.linspace(0, 1, 11)]
    assert abs(np.linalg.norm(xs[0]) - 2.0) < 1e-9
    assert abs(np.linalg.norm(xs[-1]) - 2.0) < 1e-9
    ps = [x_to_p(x) for x in xs]
    dev = geodesic_deviation(ps)
    assert dev["max"] < 1e-8
    assert dev["mean"] < 1e-8


def test_off_geodesic_point_has_positive_deviation():
    p_a = rand_dist(seed=8)
    p_b = rand_dist(seed=9)
    x_a, x_b = simplex_embed(p_a), simplex_embed(p_b)
    mid = simplex_geodesic(x_a, x_b, 0.5)
    a = x_a / np.linalg.norm(x_a)
    b = x_b / np.linalg.norm(x_b)
    b_perp = b - np.dot(b, a) * a
    b_perp /= np.linalg.norm(b_perp)
    e = np.zeros_like(a)
    e[0] = 1.0
    normal = e - np.dot(e, a) * a
    normal -= np.dot(normal, b_perp) * b_perp
    normal /= np.linalg.norm(normal)
    angle = 0.2
    off = 2.0 * (np.cos(angle) * (mid / np.linalg.norm(mid))
                 + np.sin(angle) * normal)
    off *= 2.0 / np.linalg.norm(off)
    assert abs(np.linalg.norm(off) - 2.0) < 1e-9
    dev = geodesic_deviation([x_to_p(x_a), x_to_p(off), x_to_p(x_b)])
    assert dev["max"] > 0.1
    assert abs(dev["max"] - 2.0 * angle) < 1e-6


def test_distance_to_arc_clamps_to_endpoints():
    p_a = rand_dist(seed=10)
    p_b = rand_dist(seed=11)
    x_a, x_b = simplex_embed(p_a), simplex_embed(p_b)
    assert distance_to_arc(x_a, x_a, x_b) < 1e-9
    assert distance_to_arc(x_b, x_a, x_b) < 1e-9
    outside = x_a - 0.5 * (x_b - x_a)
    outside = 2.0 * outside / np.linalg.norm(outside)
    d = distance_to_arc(outside, x_a, x_b)
    assert d > 0


def test_participation_ratio_line_and_subspace():
    rng = np.random.default_rng(12)
    n, D = 80, 40
    t = rng.standard_normal((n, 1))
    v = rng.standard_normal((1, D))
    line = t * v
    assert abs(participation_ratio(line)["pr"] - 1.0) < 0.05

    A = rng.standard_normal((3, D))
    Z = rng.standard_normal((n, 3))
    cloud = Z @ A
    assert abs(participation_ratio(cloud)["pr"] - 3.0) < 0.2


def test_token_survival():
    a = [1, 2, 3, 4]
    b = [1, 2, 9, 4]
    assert abs(token_survival(a, b, start=0) - 0.75) < 1e-12
    assert abs(token_survival(a, b, start=2) - 0.5) < 1e-12
    assert token_survival(a, b, start=10) == 1.0


def test_branch_indices():
    p = np.array([0.1, 0.7, 0.2])
    assert branch_top_index(p, 1) == 1
    assert branch_top_index(p, 2) == 2
    assert branch_top_index(p, 3) == 0
    assert branch_top_index(p, 2, exclude={2}) == 0
    assert nucleus_index(p, 0.5) == 1
    assert nucleus_index(p, 0.51) == 1
    assert nucleus_index(p, 0.75) == 2
    assert nucleus_index(p, 0.8) == 2
    assert nucleus_index(p, 0.95) == 0
    assert null_index(p) == 0
    assert null_index(p, exclude={0}) == 2


def test_token_dominance_and_tail_period():
    from tc.trajectory import tail_period, token_dominance
    d = token_dominance([1, 1, 2, 1])
    assert abs(d["dom_share"] - 0.75) < 1e-12
    assert d["dom_token"] == 1
    assert token_dominance([])["n"] == 0
    assert tail_period([9, 9, 1, 2, 1, 2, 1, 2]) == 2
    assert tail_period([1, 1, 1, 1]) == 1
    assert tail_period([1, 2, 3, 4]) == 0
