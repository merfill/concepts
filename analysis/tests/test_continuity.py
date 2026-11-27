"""Offline tests for response-function continuity helpers (experiment 013)."""

import numpy as np

from tc.continuity import (
    FRAMES,
    axis_projection,
    build_prompt,
    cliff_ratio,
    continuity_verdict,
    expected_valence,
    frame_robustness,
    frequency_residual,
    js_steps,
    ladder_words,
    load_ladder,
    monotonicity,
    path_independence,
    single_step_share,
)


def moving_probs(k=7, v=12):
    p0 = np.zeros(v)
    p0[0], p0[1] = 0.6, 0.4
    pK = np.zeros(v)
    pK[v - 2], pK[v - 1] = 0.6, 0.4
    out = []
    for t in np.linspace(0.0, 1.0, k):
        p = (1 - t) * p0 + t * pK
        out.append(p / p.sum())
    return out


def test_load_ladder(tmp_path):
    path = tmp_path / "lad.tsv"
    path.write_text("rung\tword\tword_alt\n2\tbad\tawful\n1\tterrible\tdreadful\n")
    lad = load_ladder(path, column="word")
    assert ladder_words(lad) == ["terrible", "bad"]
    alt = load_ladder(path, column="word_alt")
    assert ladder_words(alt) == ["dreadful", "awful"]


def test_build_prompt_and_filler():
    assert build_prompt(FRAMES["F1"], "good") == "The movie was good. Overall, it was ___"
    p = build_prompt(FRAMES["F1"], "good", "I watched it last night.")
    assert "good. I watched it last night. Overall" in p


def test_js_steps_identity_is_zero():
    p = np.ones(10) / 10
    assert js_steps([p, p])[0] == 0.0


def test_cliff_and_share():
    assert abs(cliff_ratio([0.01, 0.01, 0.04, 0.01]) - 4.0) < 1e-9
    assert cliff_ratio([]) != cliff_ratio([])  # nan
    assert abs(single_step_share([0.1, 0.1, 0.2]) - 0.5) < 1e-9


def test_monotonicity():
    assert monotonicity([0, 1, 2, 3, 4]) > 0.99
    assert monotonicity([0, 1, 0, 1, 0]) < 0.5


def test_axis_projection_monotone_and_endpoints():
    proj = axis_projection(moving_probs(), k=12)
    phi = np.array(proj["phi"])
    assert abs(phi[0]) < 1e-9 and abs(phi[-1] - 1.0) < 1e-9
    assert monotonicity(phi) > 0.99
    assert proj["js_endpoints"] > 0


def test_expected_valence():
    p = np.zeros(5)
    p[1], p[2] = 0.5, 0.5
    assert abs(expected_valence(p, {1: 1.0, 2: 7.0}) - 4.0) < 1e-9
    assert np.isnan(expected_valence(p, {}))


def test_path_independence():
    a = [0.0, 0.2, 0.4, 0.7, 1.0]
    same = path_independence(a, a)
    assert same["rho"] > 0.99 and same["max_abs_diff"] < 1e-12
    diff = path_independence(a, [1.0, 0.7, 0.4, 0.2, 0.0])
    assert diff["rho"] < -0.5


def test_frame_robustness_identical():
    phi = [0.0, 0.2, 0.4, 0.7, 1.0]
    steps = [0.02, 0.03, 0.02, 0.03]
    fr = frame_robustness({"F1": phi, "F2": phi}, {"F1": steps, "F2": steps})
    assert fr["mean_rho"] > 0.99
    assert fr["cliff_agree"]


def test_frequency_residual_removes_linear_confound():
    zipf = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    phi = 2.0 * zipf
    r = frequency_residual(phi, zipf, np.ones(7))
    assert len(r["resid"]) == 7
    assert np.allclose(r["resid"], 0.0, atol=1e-9)


def test_continuity_verdict_levels():
    phi = [i / 6 for i in range(7)]
    ok = {"ok": True}
    smooth = [0.02] * 6
    assert continuity_verdict(phi, smooth, 1.0, 0.2, ok, 0.0, 0.95)["label"].startswith("continuous")

    jumpy = [0.01, 0.01, 0.2, 0.01, 0.01, 0.01]
    assert continuity_verdict(phi, jumpy, 1.0, 0.2, ok, 0.0, 0.95)["label"].startswith("discrete")

    assert continuity_verdict(phi, smooth, 1.0, 0.001, ok, 0.0, 0.95)["label"].startswith("absent")
    assert continuity_verdict(phi, smooth, 0.1, 0.2, ok, 0.0, 0.95)["label"].startswith("absent")

    partial = continuity_verdict(phi, smooth, 1.0, 0.2, {"ok": False}, 0.0, 0.95)
    assert "partial" in partial["label"]
