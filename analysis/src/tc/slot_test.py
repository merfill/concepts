"""Slot test: concept/relation vs slot form under a fixed frame (experiment 019).

Ten triples (concept, related, opposite), each with three fillers of different
length.  Every filler goes into a fixed frame; we read the next-token
distribution.  We work on the Hellinger Gram matrix and ask whether the geometry
follows the semantic gradient (concept <-> related closer than concept <->
opposite) or the slot form (length/frequency).

Protocol: ``../experiment-019-2026-11-28-slot-form-vs-concept.md``.
"""

from __future__ import annotations

import numpy as np

from .concept_centroids import (
    centered_gram,
    cross_frame_transfer,
    cv_fisher,
    cv_fisher_covariates,
    frequency_length_covariates,
    hellinger_gram,
    train_pca,
)
from .controls import partial_spearman, spearman, upper_tri
from .separation import cosine_rdm, label_rdm

FRAME = "When we say {x}, we mean"
FRAME_B = "The expression {x} refers to"
ROLES = ["target", "related", "opposite"]

TRIPLES = [
    {"name": "hot",
     "target": ["hot", "very warm", "high in temperature"],
     "related": ["warm", "mildly heated", "slightly above room temperature"],
     "opposite": ["cold", "very chilly", "low in temperature"]},
    {"name": "big",
     "target": ["big", "very large", "great in size"],
     "related": ["large", "quite big", "above average in size"],
     "opposite": ["small", "quite little", "below average in size"]},
    {"name": "fast",
     "target": ["fast", "very quick", "moving at high speed"],
     "related": ["quick", "quite fast", "moving with speed"],
     "opposite": ["slow", "rather sluggish", "moving at low speed"]},
    {"name": "happy",
     "target": ["happy", "very glad", "feeling great joy"],
     "related": ["joyful", "full of cheer", "feeling deep delight"],
     "opposite": ["sad", "rather unhappy", "feeling deep sorrow"]},
    {"name": "bright",
     "target": ["bright", "very luminous", "giving off much light"],
     "related": ["luminous", "full of light", "shining with strong light"],
     "opposite": ["dark", "rather dim", "giving off little light"]},
    {"name": "young",
     "target": ["young", "very youthful", "early in life"],
     "related": ["youthful", "quite young", "having the freshness of youth"],
     "opposite": ["old", "rather aged", "late in life"]},
    {"name": "strong",
     "target": ["strong", "very powerful", "having great force"],
     "related": ["powerful", "full of strength", "having much force"],
     "opposite": ["weak", "rather feeble", "having little force"]},
    {"name": "clean",
     "target": ["clean", "very tidy", "free from dirt"],
     "related": ["tidy", "quite neat", "free from mess"],
     "opposite": ["dirty", "rather filthy", "covered with dirt"]},
    {"name": "empty",
     "target": ["empty", "completely vacant", "containing nothing inside"],
     "related": ["vacant", "quite empty", "left without contents"],
     "opposite": ["full", "completely filled", "containing as much as possible"]},
    {"name": "near",
     "target": ["near", "very close", "at a short distance"],
     "related": ["close", "quite near", "at a small distance"],
     "opposite": ["far", "rather distant", "at a long distance"]},
]


def build_items(frame: str = FRAME) -> list[dict]:
    items = []
    for ti, tr in enumerate(TRIPLES):
        for ri, role in enumerate(ROLES):
            for fi, phr in enumerate(tr[role]):
                items.append({
                    "triple": ti,
                    "triple_name": tr["name"],
                    "role": ri,
                    "role_name": role,
                    "filler": fi,
                    "paraphrase": phr,
                    "text": frame.format(x=phr),
                })
    return items


def labels(items):
    y_triple = np.asarray([it["triple"] for it in items], dtype=int)
    y_role = np.asarray([it["role"] for it in items], dtype=int)
    slots = np.asarray([it["filler"] for it in items], dtype=int)
    return y_triple, y_role, slots


def cluster_labels(items) -> np.ndarray:
    y_triple, y_role, _ = labels(items)
    return y_triple * len(ROLES) + y_role


def cluster_centroid_gram(G, labels):
    """Centroid inner products ``<m_a, m_b>`` from the item Gram ``G``."""
    labels = np.asarray(labels)
    G = np.asarray(G, dtype=np.float64)
    uniq = np.unique(labels)
    idx = [np.where(labels == c)[0] for c in uniq]
    M = np.zeros((len(uniq), len(uniq)), dtype=np.float64)
    for a in range(len(uniq)):
        for b in range(len(uniq)):
            M[a, b] = G[np.ix_(idx[a], idx[b])].mean()
    return uniq, M


def centroid_rdm(M) -> np.ndarray:
    """Hellinger distance between normalized centroids from their Gram ``M``."""
    M = np.asarray(M, dtype=np.float64)
    norms = np.sqrt(np.clip(np.diag(M), 1e-12, None))
    S = M / np.outer(norms, norms)
    np.clip(S, -1.0, 1.0, out=S)
    R = np.sqrt(np.clip(1.0 - S, 0.0, None))
    np.fill_diagonal(R, 0.0)
    return R


def _gradient_from_rdm(R, uniq, y_triple):
    pos = {int(c): i for i, c in enumerate(uniq)}
    d_tr, d_to = [], []
    for t in np.unique(y_triple):
        it, ir, io = (pos[int(t * 3 + 0)], pos[int(t * 3 + 1)], pos[int(t * 3 + 2)])
        d_tr.append(R[it, ir])
        d_to.append(R[it, io])
    d_tr = np.asarray(d_tr)
    d_to = np.asarray(d_to)
    return {
        "d_target_related": float(d_tr.mean()),
        "d_target_opposite": float(d_to.mean()),
        "gap": float(np.mean(d_to - d_tr)),
        "fraction_related_closer": float(np.mean(d_tr < d_to)),
        "d_tr": d_tr.tolist(),
        "d_to": d_to.tolist(),
    }


def semantic_gradient(G, y_triple, y_role) -> dict:
    """Concept-related closer than concept-opposite, from the Hellinger centroids."""
    cl = np.asarray(y_triple) * 3 + np.asarray(y_role)
    uniq, M = cluster_centroid_gram(G, cl)
    R = centroid_rdm(M)
    return _gradient_from_rdm(R, uniq, np.asarray(y_triple))


def _global_coords(G) -> np.ndarray:
    idx = np.arange(G.shape[0])
    Gc = centered_gram(G, idx)
    Z, _ = train_pca(Gc, idx, idx)
    return Z


def _residualize(Z, covariates) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float64)
    D = np.column_stack([np.ones(Z.shape[0]), np.asarray(covariates, dtype=np.float64)])
    beta, *_ = np.linalg.lstsq(D, Z, rcond=None)
    return Z - D @ beta


def gradient_from_coords(Z, y_triple, y_role) -> dict:
    cl = np.asarray(y_triple) * 3 + np.asarray(y_role)
    uniq = np.unique(cl)
    C = np.stack([Z[cl == c].mean(axis=0) for c in uniq])
    R = cosine_rdm(C)
    return _gradient_from_rdm(R, uniq, np.asarray(y_triple))


def residualized_gradient(G, y_triple, y_role, covariates) -> dict:
    Z = _residualize(_global_coords(G), covariates)
    return gradient_from_coords(Z, y_triple, y_role)


def rdm_controls(G, lengths, y_triple, y_role) -> dict:
    G = np.asarray(G, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    model = np.sqrt(np.clip(1.0 - G, 0.0, None))
    np.fill_diagonal(model, 0.0)
    length_rdm = np.abs(lengths[:, None] - lengths[None, :])
    triple_rdm = label_rdm(np.asarray(y_triple))
    role_rdm = label_rdm(np.asarray(y_role))
    return {
        "rho_model_length": spearman(upper_tri(model), upper_tri(length_rdm)),
        "rho_model_triple": spearman(upper_tri(model), upper_tri(triple_rdm)),
        "rho_model_role": spearman(upper_tri(model), upper_tri(role_rdm)),
        "partial_triple_given_length": partial_spearman(model, triple_rdm, length_rdm),
        "partial_role_given_length": partial_spearman(model, role_rdm, length_rdm),
    }


def slot_metrics(items, P, lengths, n_covariates) -> dict:
    """All step-1 metrics for one frame."""
    G = hellinger_gram(P)
    y_triple, y_role, slots = labels(items)
    cov = n_covariates
    grad = semantic_gradient(G, y_triple, y_role)
    grad_resid = residualized_gradient(G, y_triple, y_role, cov)
    return {
        "gradient": grad,
        "gradient_residualized": grad_resid,
        "role_accuracy": cv_fisher(G, y_role, slots, 3)["accuracy"],
        "triple_accuracy": cv_fisher(G, y_triple, slots, 10)["accuracy"],
        "role_accuracy_resid": cv_fisher_covariates(G, y_role, slots, cov, 3),
        "triple_accuracy_resid": cv_fisher_covariates(G, y_triple, slots, cov, 10),
        "rdm_controls": rdm_controls(G, lengths, y_triple, y_role),
        "chance_role": 1.0 / 3.0,
        "chance_triple": 0.1,
        "G": G,
    }
