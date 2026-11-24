"""Transition-operator instrument (experiment 005).

Experiment 004 showed that the *state* is not an invariant: neither a vector, nor
an axis, nor an RDM over states transfers between readout types.  The next
candidate carrier is the operator that *produces* the state: the local Jacobian
of a block at the readout position

    O_l(x) = d h_{l+1}(t*) / d h_l(t*)   in R^{d x d}.

The full Jacobian is noisy and expensive, so we estimate its action on a fixed
probe basis P (d x m):

    Phi_l(x) = O_l(x) P   in R^{d x m}.

Operators are compared by CKA of their probe responses (or cosine of the
flattened response).  Because P is shared within one model/site, this is a clean
comparison; across sites/models only the induced RDM is compared.

Estimation is a batched central finite difference: all m probe directions are
stacked in the batch dimension, so one forward gives every column of Phi.

Pure numpy/torch; ``tests/test_operator.py`` validates the estimator and CKA on
synthetic maps without downloading a model.
"""

from __future__ import annotations

import numpy as np
import torch

from tc.interchange import get_blocks


# --------------------------------------------------------------------------- #
# Probe basis and finite differences
# --------------------------------------------------------------------------- #

def random_probe_basis(d: int, m: int, seed: int = 0) -> np.ndarray:
    """Orthonormal probe basis ``P`` (d x m) with random orientation."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d, m)))
    return q[:, :m]


def finite_diff_operator(fn, h0: np.ndarray, P: np.ndarray, eps: float = 1e-2) -> np.ndarray:
    """Estimate ``J(h0) @ P`` for a vector-valued ``fn`` by central differences.

    ``fn`` maps ``R^d -> R^d``.  Used for the synthetic validation and as the
    single-vector fallback; the model path uses batched ``estimate_phi``.
    """
    h0 = np.asarray(h0, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    out0 = np.asarray(fn(h0), dtype=np.float64)
    d = out0.size
    Phi = np.zeros((d, P.shape[1]), dtype=np.float64)
    for k in range(P.shape[1]):
        p = P[:, k]
        gp = np.asarray(fn(h0 + eps * p), dtype=np.float64)
        gm = np.asarray(fn(h0 - eps * p), dtype=np.float64)
        Phi[:, k] = (gp - gm) / (2.0 * eps)
    return Phi


def estimate_phi(model, tok, text: str, layer: int, P: np.ndarray, eps: float = 1e-2) -> np.ndarray:
    """Batched finite-difference estimate of ``Phi_l(x) = O_l(x) P``.

    All ``m`` probe directions are stacked in the batch dimension; the block
    ``layer`` input at the last position is perturbed by ``+eps*p_k`` / ``-eps*p_k``
    and its output is read at the same position.
    """
    P = np.asarray(P, dtype=np.float64)
    m = P.shape[1]
    enc = tok([text] * m, return_tensors="pt")
    block = get_blocks(model)[layer]
    pt = torch.as_tensor(P.T, dtype=torch.float32)  # (m, d)

    def run(sign: float) -> np.ndarray:
        store = {}

        def pre(module, args):
            hs = args[0].clone()
            delta = sign * eps * pt.to(hs.dtype)
            hs[:, -1, :] = hs[:, -1, :] + delta
            return (hs,) + tuple(args[1:])

        def post(module, inputs, output):
            o = output[0] if isinstance(output, tuple) else output
            store["h"] = o[:, -1, :].detach().clone()

        h1 = block.register_forward_pre_hook(pre)
        h2 = block.register_forward_hook(post)
        try:
            with torch.no_grad():
                model(**enc)
        finally:
            h1.remove()
            h2.remove()
        return store["h"].double().numpy()  # (m, d)

    hp, hm = run(+1.0), run(-1.0)
    return ((hp - hm) / (2.0 * eps)).T  # (d, m)


# --------------------------------------------------------------------------- #
# Comparison of operators
# --------------------------------------------------------------------------- #

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two representations (rows = samples, cols = features)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    num = float(np.sum((X.T @ Y) ** 2))
    dxx = float(np.sum((X.T @ X) ** 2))
    dyy = float(np.sum((Y.T @ Y) ** 2))
    denom = np.sqrt(dxx * dyy)
    return num / denom if denom > 0.0 else 0.0


def operator_distance(phi_a: np.ndarray, phi_b: np.ndarray, metric: str = "cka") -> float:
    """Distance between two probe-operator responses ``Phi`` (d x m)."""
    if metric == "cka":
        return float(1.0 - linear_cka(phi_a.T, phi_b.T))
    if metric == "cosine":
        a = np.asarray(phi_a, dtype=np.float64).ravel()
        b = np.asarray(phi_b, dtype=np.float64).ravel()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0.0 or nb == 0.0:
            return 1.0
        return float(1.0 - np.dot(a, b) / (na * nb))
    raise ValueError(f"unknown metric {metric!r}")


def operator_rdm(phis, metric: str = "cka") -> np.ndarray:
    """Pairwise operator RDM over a list of ``Phi`` responses."""
    n = len(phis)
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = operator_distance(phis[i], phis[j], metric)
    return M


def operator_effective_dimension(phis) -> float:
    """Participation ratio of centred, flattened probe responses (secondary)."""
    from tc.rdm import effective_rank

    X = np.stack([np.asarray(p, dtype=np.float64).ravel() for p in phis])
    X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    return effective_rank(s)
