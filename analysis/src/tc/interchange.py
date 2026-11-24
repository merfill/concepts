"""Difference-in-means directions and interchange interventions (experiment 002).

Phase 0 (no gradient search): for a binary attribute take the direction
``mean(attr=1) - mean(attr=0)`` of the last-position residual stream at a layer,
then *interchange* the projection onto that direction from a source run into a
base run. If the attribute is carried by that direction, the base's next-token
distribution moves toward the source's.

Interventions are applied at the readout position (last token of ``text+probe``),
the only position aligned across texts of different length. Hooking a block and
swapping one direction keeps the rest of the state real (on-manifold-friendly).
"""

from __future__ import annotations

import numpy as np
import torch


def get_blocks(model):
    """Return the list of transformer blocks for common causal-LM families."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # GPT-2
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # Llama / Qwen
    raise ValueError(f"cannot locate transformer blocks in {type(model).__name__}")


def extract(model, tok, text: str):
    """Return (hidden_stack, probs) at the last position of `text`.

    hidden_stack: (L+1, d) hidden states; block layer `l` output is index l+1.
    probs: full next-token distribution.
    """
    enc = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    hs = np.stack([h[0, -1, :].double().numpy() for h in out.hidden_states])
    probs = torch.softmax(out.logits[0, -1], dim=-1).double().numpy()
    return hs, probs


def direction(hidden_states, labels):
    """Mean(1) - mean(0) over last-position hidden states (unit-normalized)."""
    hs = np.stack(hidden_states).astype(np.float64)
    labels = np.asarray(labels)
    v = hs[labels == 1].mean(0) - hs[labels == 0].mean(0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _swap_hook(v, src_vec):
    vn, sn = np.asarray(v), np.asarray(src_vec)

    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        h2 = h.clone()
        last = h2.shape[1] - 1
        hh = h2[0, last, :]
        vt = torch.as_tensor(vn, dtype=hh.dtype, device=hh.device)
        st = torch.as_tensor(sn, dtype=hh.dtype, device=hh.device)
        h2[0, last, :] = hh - torch.dot(hh, vt) * vt + torch.dot(st, vt) * vt
        return (h2,) + tuple(output[1:]) if is_tuple else h2

    return hook


def patched_probs(model, tok, text: str, layer: int, v, src_vec):
    """Next-token distribution of `text` with the v-component swapped from src_vec.

    `layer` is a block index; the swap happens on that block's output at the
    last position, before the remaining blocks and the unembedding.
    """
    enc = tok(text, return_tensors="pt")
    handle = get_blocks(model)[layer].register_forward_hook(_swap_hook(v, src_vec))
    try:
        with torch.no_grad():
            out = model(**enc)
    finally:
        handle.remove()
    return torch.softmax(out.logits[0, -1], dim=-1).double().numpy()


def _subspace_hook(U, src_vec):
    """Replace the projection onto subspace U (D x d) at the last position."""
    Un, sn = np.asarray(U), np.asarray(src_vec)

    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        h2 = h.clone()
        last = h2.shape[1] - 1
        hh = h2[0, last, :]
        Ut = torch.as_tensor(Un, dtype=hh.dtype, device=hh.device)
        src = torch.as_tensor(sn, dtype=hh.dtype, device=hh.device)
        h2[0, last, :] = hh + Ut @ (Ut.T @ (src - hh))
        return (h2,) + tuple(output[1:]) if is_tuple else h2

    return hook


def patched_probs_subspace(model, tok, text: str, layer: int, U, src_vec):
    """Next-token distribution with the U-subspace (D x d) swapped from src_vec."""
    enc = tok(text, return_tensors="pt")
    handle = get_blocks(model)[layer].register_forward_hook(_subspace_hook(U, src_vec))
    try:
        with torch.no_grad():
            out = model(**enc)
    finally:
        handle.remove()
    return torch.softmax(out.logits[0, -1], dim=-1).double().numpy()


def _ablate_hook(U):
    """Remove (project out) the U-subspace (D x d) at the last position."""
    Un = np.asarray(U)

    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        h2 = h.clone()
        last = h2.shape[1] - 1
        hh = h2[0, last, :]
        Ut = torch.as_tensor(Un, dtype=hh.dtype, device=hh.device)
        h2[0, last, :] = hh - Ut @ (Ut.T @ hh)
        return (h2,) + tuple(output[1:]) if is_tuple else h2

    return hook


def ablated_probs(model, tok, text: str, layer: int, U):
    """Next-token distribution with the U-subspace removed at the last position."""
    enc = tok(text, return_tensors="pt")
    handle = get_blocks(model)[layer].register_forward_hook(_ablate_hook(U))
    try:
        with torch.no_grad():
            out = model(**enc)
    finally:
        handle.remove()
    return torch.softmax(out.logits[0, -1], dim=-1).double().numpy()


def _add_hook(delta):
    """Add a fixed vector `delta` (D,) to the last-position hidden state."""
    dn = np.asarray(delta)

    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        h2 = h.clone()
        last = h2.shape[1] - 1
        h2[0, last, :] = h2[0, last, :] + torch.as_tensor(dn, dtype=h2.dtype, device=h2.device)
        return (h2,) + tuple(output[1:]) if is_tuple else h2

    return hook


def steered_probs(model, tok, text: str, layer: int, delta):
    """Next-token distribution after adding `delta` at the last position."""
    enc = tok(text, return_tensors="pt")
    handle = get_blocks(model)[layer].register_forward_hook(_add_hook(delta))
    try:
        with torch.no_grad():
            out = model(**enc)
    finally:
        handle.remove()
    return torch.softmax(out.logits[0, -1], dim=-1).double().numpy()


def _add_hook_at(delta, pos: int):
    """Add `delta` at a fixed absolute position `pos` in the sequence."""
    dn = np.asarray(delta)

    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        h2 = h.clone()
        h2[0, pos, :] = h2[0, pos, :] + torch.as_tensor(dn, dtype=h2.dtype, device=h2.device)
        return (h2,) + tuple(output[1:]) if is_tuple else h2

    return hook


def seq_logprob_steered(model, tok, prompt: str, concept: str, layer: int, delta):
    """Length-normalized log-prob of `concept` after `prompt`, with `delta`
    injected at the answer position (last prompt token), propagating to the
    concept tokens through attention. Works for multi-token concepts."""
    ep = tok(prompt, return_tensors="pt")
    ec = tok(" " + concept, add_special_tokens=False, return_tensors="pt")
    if ec.input_ids.shape[1] == 0:
        ec = tok(concept, add_special_tokens=False, return_tensors="pt")
    ids = torch.cat([ep.input_ids, ec.input_ids], dim=1)
    pos = ep.input_ids.shape[1] - 1
    handle = get_blocks(model)[layer].register_forward_hook(_add_hook_at(delta, pos))
    try:
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    finally:
        handle.remove()
    lps = torch.log_softmax(out.logits[0], dim=-1).double()
    start = ep.input_ids.shape[1]
    lp = sum(float(lps[start - 1 + i, tid]) for i, tid in enumerate(ec.input_ids[0]))
    return lp / ec.input_ids.shape[1]
