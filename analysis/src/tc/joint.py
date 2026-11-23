"""Joint-context encodings: the model reading two texts together.

Used to operationalize "native comparison": instead of encoding each text
separately, put both in one input and read how they interact (cross-attention
and the contextualized pooled representations of each segment).
"""

from __future__ import annotations

import numpy as np
import torch


def cosine_sim(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def _content_mask(enc, tok) -> np.ndarray:
    """True for real words (exclude [CLS]/[SEP]/[PAD])."""
    am = enc["attention_mask"][0].numpy().astype(bool)
    special = enc.get("special_tokens_mask")
    if special is not None:
        return am & ~special[0].numpy().astype(bool)
    return am


def encode_alone(text, model, tok, layer: int = -1):
    """Return (V, pooled) for a text encoded on its own at `layer` (-1 = last)."""
    enc = tok(text, return_tensors="pt", return_special_tokens_mask=True)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, output_attentions=True)
    V = out.hidden_states[layer][0].double().numpy()
    mask = _content_mask(enc, tok)
    pooled = V[mask].mean(0) if mask.any() else V.mean(0)
    return V, pooled


def encode_pair(a, b, model, tok):
    """Encode '[CLS] a [SEP] b [SEP]' and return segment-level comparison signals.

    Returns (pooled_a, pooled_b, cross_score, V, seg) where cross_score is the
    mean attention mass flowing between the two segments (last layer, heads avg).
    """
    enc = tok(a, b, return_tensors="pt", return_special_tokens_mask=True)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, output_attentions=True)
    V = out.hidden_states[-1][0].double().numpy()
    content = _content_mask(enc, tok)

    if "token_type_ids" in enc:
        seg = enc["token_type_ids"][0].numpy()
    else:  # fallback: split at first [SEP]
        ids = enc["input_ids"][0].numpy()
        sep = tok.sep_token_id
        first = int(np.where(ids == sep)[0][0]) if np.any(ids == sep) else len(ids) // 2
        seg = (np.arange(len(ids)) > first).astype(int)

    ma = content & (seg == 0)
    mb = content & (seg == 1)
    pa = V[ma].mean(0) if ma.any() else V.mean(0)
    pb = V[mb].mean(0) if mb.any() else V.mean(0)

    cross_score = 0.0
    att = getattr(out, "attentions", None)
    if att and att[-1] is not None:
        A = att[-1][0].mean(0).double().numpy()
        m_a = seg == 0
        m_b = seg == 1
        block = np.concatenate([A[np.ix_(m_a, m_b)].ravel(), A[np.ix_(m_b, m_a)].ravel()])
        if block.size:
            cross_score = float(block.mean())
    return pa, pb, cross_score, V, seg
