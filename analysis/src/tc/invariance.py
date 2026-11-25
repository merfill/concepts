"""Invariance / null-operator battery (experiment 007, program stage 0).

The point is *calibration*: the transformer architectures give analytically
known no-ops and non-no-ops, which we use to validate the intervention plumbing
before interpreting steering / localization.

Facts encoded here (see ``../experiment-007-2026-11-25-invariance-battery.md``):

* ``softmax`` is shift-invariant but not scale-invariant (scale = temperature;
  this is a *logit* operation, not a hidden-state one).
* LayerNorm / RMSNorm are scale-invariant for ``c>0``: ``norm(c*h) = norm(h)``.
* LayerNorm is shift-invariant ``LN(h + c*1) = LN(h)``; RMSNorm is not.

Pure numpy ops (scale/shift/rotate/...) are separated from model hooks so they
can be unit-tested without downloading a model.
"""

from __future__ import annotations

import numpy as np
import torch

from .interchange import get_blocks


# --------------------------------------------------------------------------- #
# Pure vector operations
# --------------------------------------------------------------------------- #

def _as_vec(h) -> np.ndarray:
    return np.asarray(h, dtype=np.float64)


def unit(v) -> np.ndarray:
    v = _as_vec(v)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def identity(h) -> np.ndarray:
    return _as_vec(h)


def scale(h, c: float) -> np.ndarray:
    return c * _as_vec(h)


def shift(h, c: float) -> np.ndarray:
    """Add ``c`` to every coordinate: ``h + c*1``."""
    return _as_vec(h) + c


def coord_shift(h, c: float, k: int) -> np.ndarray:
    h = _as_vec(h).copy()
    h[k] += c
    return h


def rotate_plane(h, u, v, theta: float) -> np.ndarray:
    """Rotate ``h`` by ``theta`` inside the (orthonormalized) plane span(u, v)."""
    h = _as_vec(h)
    u = unit(u)
    v = _as_vec(v)
    v = unit(v - u * float(u @ v))
    a, b = float(u @ h), float(v @ h)
    ct, st = np.cos(theta), np.sin(theta)
    a2, b2 = ct * a - st * b, st * a + ct * b
    return h + (a2 - a) * u + (b2 - b) * v


def remove_projection(h, u) -> np.ndarray:
    h = _as_vec(h)
    u = unit(u)
    return h - float(u @ h) * u


def gaussian_noise(h, sigma: float, rng: np.random.Generator) -> np.ndarray:
    h = _as_vec(h)
    return h + rng.normal(scale=sigma, size=h.shape)


def top_norm_coord(h) -> int:
    return int(np.argmax(np.abs(_as_vec(h))))


# --------------------------------------------------------------------------- #
# Readout-space helpers
# --------------------------------------------------------------------------- #

def delta_cos(h1, h2) -> float:
    """``1 - cos(h1, h2)``: 0 for same direction (incl. positive scaling)."""
    a, b = _as_vec(h1), _as_vec(h2)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def top1(p) -> int:
    return int(np.argmax(np.asarray(p, dtype=np.float64)))


def top1_flip(p1, p2) -> bool:
    return top1(p1) != top1(p2)


def softmax(z) -> np.ndarray:
    z = _as_vec(z)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# --------------------------------------------------------------------------- #
# Reference normalizations and a synthetic residual model (model-free tests)
# --------------------------------------------------------------------------- #

def layernorm(h, eps: float = 1e-5) -> np.ndarray:
    h = _as_vec(h)
    mu = h.mean()
    var = ((h - mu) ** 2).mean()
    return (h - mu) / np.sqrt(var + eps)


def rmsnorm(h, eps: float = 1e-6) -> np.ndarray:
    h = _as_vec(h)
    return h / np.sqrt((h * h).mean() + eps)


def synthetic_block(h, norm) -> np.ndarray:
    """Pre-norm residual block: ``h + tanh(norm(h))``."""
    h = _as_vec(h)
    return h + np.tanh(norm(h))


def synthetic_model(h, norm, n_layers: int) -> np.ndarray:
    """Stack ``n_layers`` pre-norm blocks, then a final norm (like a LM head)."""
    x = _as_vec(h)
    for _ in range(n_layers):
        x = synthetic_block(x, norm)
    return norm(x)


# --------------------------------------------------------------------------- #
# Applying a transform at one position of an (n, d) activation matrix
# --------------------------------------------------------------------------- #

def apply_at_position(H, fn, position: int = -1, all_positions: bool = False) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64).copy()
    if all_positions:
        for i in range(H.shape[0]):
            H[i] = fn(H[i])
    elif -H.shape[0] <= position < H.shape[0]:
        H[position] = fn(H[position])
    else:
        raise IndexError(f"position {position} out of range for {H.shape[0]} tokens")
    return H


# --------------------------------------------------------------------------- #
# Model hooks
# --------------------------------------------------------------------------- #

def norm_type(model) -> str:
    """Best-effort architecture check: 'layernorm', 'rmsnorm' or 'unknown'."""
    names = {type(m).__name__ for m in model.modules()}
    if any("RMSNorm" in n for n in names):
        return "rmsnorm"
    if any("LayerNorm" in n for n in names):
        return "layernorm"
    return "unknown"


def _pos_view(H, position: int, all_positions: bool):
    return H if all_positions else H[position]


def transformed_probs(model, tok, text: str, site, fn,
                      position: int = -1, all_positions: bool = False):
    """Next-token probs and hidden states when ``fn`` is applied at ``site``.

    ``site`` is one of ``("block", layer, "in")``, ``("block", layer, "out")``
    or ``("final",)`` (= output of the last block, i.e. input to the final norm).
    Returns a dict with ``probs``, ``h_orig``, ``h_pert`` (at ``position`` unless
    ``all_positions``).
    """
    enc = tok(text, return_tensors="pt")
    cap: dict[str, np.ndarray] = {}

    def hook_out(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        H = h[0].detach().double().cpu().numpy()
        cap["orig"] = H.copy()
        H2 = apply_at_position(H, fn, position, all_positions)
        cap["pert"] = H2.copy()
        h2 = h.clone()
        h2[0] = torch.as_tensor(H2, dtype=h.dtype, device=h.device)
        return (h2,) + tuple(output[1:]) if is_tuple else h2

    def hook_in(module, inputs):
        h = inputs[0]
        H = h[0].detach().double().cpu().numpy()
        cap["orig"] = H.copy()
        H2 = apply_at_position(H, fn, position, all_positions)
        cap["pert"] = H2.copy()
        h2 = h.clone()
        h2[0] = torch.as_tensor(H2, dtype=h.dtype, device=h.device)
        return (h2,) + tuple(inputs[1:])

    blocks = get_blocks(model)
    if site[0] == "block":
        layer, kind = int(site[1]), site[2]
        block = blocks[layer]
        handle = (block.register_forward_hook(hook_out) if kind == "out"
                  else block.register_forward_pre_hook(hook_in))
    elif site == ("final",):
        handle = blocks[-1].register_forward_hook(hook_out)
    else:
        raise ValueError(f"unknown site {site!r}")

    try:
        with torch.no_grad():
            out = model(**enc)
    finally:
        handle.remove()

    probs = torch.softmax(out.logits[0, -1], dim=-1).double().numpy()
    return {
        "probs": probs,
        "h_orig": _pos_view(cap["orig"], position, all_positions),
        "h_pert": _pos_view(cap["pert"], position, all_positions),
    }


def transformed_logits(model, tok, text: str, logit_fn):
    """Probs after applying ``logit_fn`` to the next-token logits (no hook)."""
    enc = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    z = out.logits[0, -1].double().numpy()
    z2 = np.asarray(logit_fn(z), dtype=np.float64)
    return {"probs": softmax(z2), "logits_orig": z, "logits_pert": z2}


def hidden_at(model, tok, text: str, site, position: int = -1):
    """Hidden state at ``site`` without any transformation."""
    return transformed_probs(model, tok, text, site, identity, position=position)
