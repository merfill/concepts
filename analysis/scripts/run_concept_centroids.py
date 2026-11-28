#!/usr/bin/env python
"""Experiment 018, step 1: do concept centroids distinguish concepts?

Protocol: ``../experiment-018-2026-11-28-concept-centroids.md``.

Each paraphrase is mapped to the full next-token distribution under the frame
``You can describe {X} as``.  We work on the Hellinger Gram matrix
``G_ij = <sqrt(p_i), sqrt(p_j)>``, do a train-only PCA and the Fisher (meaning)
projection, and classify held-out paraphrases by the nearest concept centroid.
Baselines: raw PCA coordinates and TF-IDF.  Null: shuffled concept labels.

Run:
    uv run python scripts/run_concept_centroids.py --model distilgpt2
    uv run python scripts/run_concept_centroids.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.concept_centroids import (
    FRAME,
    FRAME_PASSIVE,
    anchor_diagnostic,
    build_items,
    cross_frame_transfer,
    cv_fisher,
    cv_fisher_covariates,
    cv_fisher_residualized,
    cv_raw,
    frequency_length_covariates,
    hellinger_centroid_accuracy,
    hellinger_gram,
    lexical_overlap_stats,
    permutation_null,
    permutation_null_residualized,
    rdm_partial,
    tfidf_centroid_accuracy,
    tfidf_fisher_cv,
)

K_SWEEP = [5, 10, 15, 19]
PRIMARY_K = 10


def model_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def extract_probs(model, tok, texts):
    rows = []
    for text in texts:
        enc = tok(text, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
        p = torch.softmax(out.logits[0, -1].float(), dim=-1).double().cpu().numpy()
        rows.append(p)
    return np.stack(rows, axis=0)


def run(model, tok, args):
    items = build_items(FRAME)
    items_b = build_items(FRAME_PASSIVE)
    texts = [it["text"] for it in items]
    texts_b = [it["text"] for it in items_b]
    y = np.asarray([it["concept"] for it in items], dtype=int)
    slots = np.asarray([it["slot"] for it in items], dtype=int)

    print(f"[items] {len(items)}  concepts={len(np.unique(y))}  slots={len(np.unique(slots))}")
    P = extract_probs(model, tok, texts)
    P_b = extract_probs(model, tok, texts_b)
    G = hellinger_gram(P)

    k_acc = {str(k): cv_fisher(G, y, slots, k)["accuracy"] for k in K_SWEEP}
    raw_acc = cv_raw(G, y, slots, PRIMARY_K)["accuracy"]
    hell_acc = hellinger_centroid_accuracy(P, y, slots)
    tfidf_acc = tfidf_centroid_accuracy(texts, y, slots)
    null = permutation_null(G, y, slots, PRIMARY_K, args.n_perm)

    anchor = anchor_diagnostic(items)
    overlap = lexical_overlap_stats(items)
    lex_fisher = {str(k): tfidf_fisher_cv(texts, y, slots, k) for k in (10, 15)}
    resid_fisher = {str(k): cv_fisher_residualized(G, texts, y, slots, k) for k in (10, 15)}
    null_resid = permutation_null_residualized(G, texts, y, slots, PRIMARY_K, args.n_perm)
    rdm = rdm_partial(P, texts, y)

    lengths = [int(tok(t, return_tensors="pt")["input_ids"].shape[1]) for t in texts]
    cov = frequency_length_covariates(texts, lengths)
    freq_len_acc = cv_fisher_covariates(G, y, slots, cov, PRIMARY_K)

    G_pool = hellinger_gram(np.vstack([P, P_b]))
    y_pool = np.concatenate([y, y])
    frames = np.asarray([0] * len(y) + [1] * len(y))
    cross = cross_frame_transfer(G_pool, y_pool, frames, PRIMARY_K)

    result = {
        "model": args.model,
        "frame": FRAME,
        "frame_b": FRAME_PASSIVE,
        "n_items": int(len(items)),
        "n_concepts": int(np.unique(y).size),
        "n_slots": int(np.unique(slots).size),
        "V": int(P.shape[1]),
        "concepts": [it["concept_name"] for it in items[::5]],
        "chance": float(1.0 / np.unique(y).size),
        "accuracy_by_k": k_acc,
        "primary_k": PRIMARY_K,
        "primary_accuracy": k_acc[str(PRIMARY_K)],
        "raw_pca_baseline": raw_acc,
        "hellinger_centroid_baseline": hell_acc,
        "tfidf_baseline": tfidf_acc,
        "null": null,
        "anchor_diagnostic": anchor,
        "lexical_overlap": overlap,
        "tfidf_fisher": lex_fisher,
        "residualized_fisher": resid_fisher,
        "null_residualized": null_resid,
        "frequency_length_residual": freq_len_acc,
        "cross_frame": cross,
        "rdm": rdm,
        "folds": cv_fisher(G, y, slots, PRIMARY_K)["folds"],
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    print(f"[model] {args.model}  dtype={dtype}")

    result = run(model, tok, args)
    result["dtype"] = dtype

    out = Path(args.out) if args.out else \
        Path(__file__).resolve().parents[1] / "results" / f"concept_centroids_{model_slug(args.model)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"  chance={result['chance']:.3f}")
    for k in K_SWEEP:
        print(f"  fisher k={k:2d}: accuracy={result['accuracy_by_k'][str(k)]:.3f}")
    print(f"  raw PCA-{PRIMARY_K}: {result['raw_pca_baseline']:.3f}")
    print(f"  hellinger centroid: {result['hellinger_centroid_baseline']:.3f}")
    print(f"  tfidf centroid: {result['tfidf_baseline']:.3f}")
    n = result["null"]
    print(f"  null: mean={n['mean']:.3f}  p={n['p']:.4f}  (n_perm={n['n_perm']})")
    anchors = [a for a in result["anchor_diagnostic"] if a["anchors"]]
    print(f"  anchors: {len(anchors)} concepts with a unique shared word")
    print(f"  overlap: within={result['lexical_overlap']['within_mean']:.4f}  "
          f"between={result['lexical_overlap']['between_mean']:.4f}")
    print(f"  tfidf fisher k=10: {result['tfidf_fisher']['10']:.3f}")
    print(f"  residualized fisher k=10: {result['residualized_fisher']['10']:.3f}")
    nr = result["null_residualized"]
    print(f"  residualized null: mean={nr['mean']:.3f}  p={nr['p']:.4f}")
    print(f"  freq/length residual: {result['frequency_length_residual']:.3f}")
    c = result["cross_frame"]
    print(f"  cross-frame: A->B {c['accuracy_A_to_B']:.3f}  B->A {c['accuracy_B_to_A']:.3f}  "
          f"rdm_corr {c['rdm_corr_A_to_B']:+.3f}/{c['rdm_corr_B_to_A']:+.3f}")
    r = result["rdm"]
    print(f"  rdm: rho(model,concept)={r['rho_model_concept']:+.3f}  "
          f"rho(model,lexical)={r['rho_model_lexical']:+.3f}  "
          f"partial={r['partial_model_concept_given_lexical']:+.3f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
