"""Unit tests for the invariance / null-operator instrument (experiment 007).

No model is downloaded: pure vector ops, reference norms and a synthetic
pre-norm residual model are validated offline.  The claims from the protocol
(scale no-op at the final norm, LayerNorm shift-invariance vs RMSNorm not,
mid-layer scaling not a no-op) are checked here so the model run only confirms
them on a real transformer.
"""

import numpy as np

from tc.invariance import (
    apply_at_position,
    coord_shift,
    delta_cos,
    gaussian_noise,
    layernorm,
    remove_projection,
    rmsnorm,
    rotate_plane,
    scale,
    shift,
    softmax,
    synthetic_model,
    top1_flip,
    unit,
)


def test_scale_shift_coord():
    h = np.array([1.0, -2.0, 3.0])
    assert np.allclose(scale(h, 2.0), [2, -4, 6])
    assert np.allclose(shift(h, 1.5), [2.5, -0.5, 4.5])
    assert np.allclose(coord_shift(h, 2.0, 1), [1.0, 0.0, 3.0])


def test_remove_projection_is_orthogonal_to_direction():
    rng = np.random.default_rng(0)
    h = rng.standard_normal(7)
    u = rng.standard_normal(7)
    r = remove_projection(h, u)
    assert abs(float(unit(u) @ r)) < 1e-12


def test_rotate_plane_identity_and_norm():
    rng = np.random.default_rng(1)
    h = rng.standard_normal(6)
    u = unit(rng.standard_normal(6))
    v = rng.standard_normal(6)
    assert np.allclose(rotate_plane(h, u, v, 0.0), h, atol=1e-12)
    for theta in (0.3, np.pi / 2):
        r = rotate_plane(h, u, v, theta)
        assert abs(np.linalg.norm(r) - np.linalg.norm(h)) < 1e-10


def test_gaussian_noise_scale():
    rng = np.random.default_rng(2)
    h = np.ones(5)
    n = gaussian_noise(h, 0.0, rng)
    assert np.allclose(n, h)


def test_delta_cos_and_top1_flip():
    h = np.array([1.0, 2.0, -1.0])
    assert delta_cos(h, h) < 1e-12
    assert delta_cos(h, 3.0 * h) < 1e-12      # positive scaling keeps direction
    assert abs(delta_cos(h, -h) - 2.0) < 1e-12
    assert top1_flip(np.array([0.9, 0.1]), np.array([0.1, 0.9]))
    assert not top1_flip(np.array([0.9, 0.1]), np.array([0.8, 0.2]))


def test_softmax_shift_invariant_not_scale():
    z = np.array([0.5, -1.0, 2.0, 0.0])
    assert np.allclose(softmax(z + 3.0), softmax(z), atol=1e-12)
    assert not np.allclose(softmax(2.0 * z), softmax(z))


def test_layernorm_scale_and_shift_invariant():
    rng = np.random.default_rng(3)
    h = rng.standard_normal(8)
    assert np.allclose(layernorm(2.0 * h), layernorm(h), atol=1e-6)
    assert np.allclose(layernorm(h + 5.0), layernorm(h), atol=1e-6)


def test_rmsnorm_scale_invariant_but_not_shift():
    rng = np.random.default_rng(4)
    h = rng.standard_normal(8)
    assert np.allclose(rmsnorm(2.0 * h), rmsnorm(h), atol=1e-6)
    assert np.linalg.norm(rmsnorm(h + 5.0) - rmsnorm(h)) > 1e-3


def test_synthetic_shift_noop_on_layernorm_not_on_rmsnorm():
    rng = np.random.default_rng(5)
    h = rng.standard_normal(12)
    ln_a = synthetic_model(h, layernorm, 4)
    ln_b = synthetic_model(h + 3.5, layernorm, 4)
    assert np.allclose(ln_a, ln_b, atol=1e-5)

    rms_a = synthetic_model(h, rmsnorm, 4)
    rms_b = synthetic_model(h + 3.5, rmsnorm, 4)
    assert np.linalg.norm(rms_a - rms_b) > 1e-3


def test_synthetic_midlayer_scale_is_not_a_noop():
    rng = np.random.default_rng(6)
    h = rng.standard_normal(12)
    base = synthetic_model(h, layernorm, 4)
    scaled = synthetic_model(2.0 * h, layernorm, 4)
    assert np.linalg.norm(base - scaled) > 1e-2


def test_apply_at_position_last_vs_all():
    rng = np.random.default_rng(7)
    H = rng.standard_normal((5, 4))
    fn = lambda h: 2.0 * h
    last = apply_at_position(H, fn, position=-1)
    assert np.allclose(last[:-1], H[:-1])
    assert np.allclose(last[-1], 2.0 * H[-1])

    everything = apply_at_position(H, fn, all_positions=True)
    assert np.allclose(everything, 2.0 * H)
