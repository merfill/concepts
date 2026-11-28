#!/usr/bin/env python
"""Experiment 019: strong fixed frame + slot, concept vs slot form.

Protocol: ``../experiment-019-2026-11-28-slot-form-vs-concept.md``.

Ten triples (concept, related, opposite), three fillers of different length
each.  We read the next-token distribution under a fixed frame and ask whether
the geometry follows the semantic gradient (related closer than opposite) or the
slot form (length/frequency).

Run:
    uv run python scripts/run_slot_test.py --model distilgpt2
    uv run python scripts/run_slot_test.py --model Qwen/Qwen2.5-1.5B-Instruct
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
    cross_frame_transfer,
    frequency_length_covariates,
    hellinger_gram,
)
from tc.slot_test import FRAME, FRAME_B, build_items, labels, slot_metrics


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
    items_b = build_items(FRAME_B)
    texts = [it["text"] for it in items]
    texts_b = [it["text"] for it in items_b]
    y_triple, y_role, slots = labels(items)
    lengths = [int(tok(t, return_tensors="pt")["input_ids"].shape[1]) for t in texts]
    cov = frequency_length_covariates(texts, lengths)

    print(f"[items] {len(items)}  triples={np.unique(y_triple).size}  "
          f"roles={np.unique(y_role).size}  fillers={np.unique(slots).size}")
    P = extract_probs(model, tok, texts)
    P_b = extract_probs(model, tok, texts_b)

    metrics = slot_metrics(items, P, lengths, cov)
    G = metrics.pop("G")

    G_pool = hellinger_gram(np.vstack([P, P_b]))
    y_triple_pool = np.concatenate([y_triple, y_triple])
    y_role_pool = np.concatenate([y_role, y_role])
    frames = np.asarray([0] * len(y_triple) + [1] * len(y_triple))
    cross_role = cross_frame_transfer(G_pool, y_role_pool, frames, 3)
    cross_triple = cross_frame_transfer(G_pool, y_triple_pool, frames, 10)

    result = {
        "model": args.model,
        "frame": FRAME,
        "frame_b": FRAME_B,
        "n_items": int(len(items)),
        "n_triples": int(np.unique(y_triple).size),
        "V": int(P.shape[1]),
        "length_range": [int(min(lengths)), int(max(lengths))],
        "metrics": metrics,
        "cross_frame_role": cross_role,
        "cross_frame_triple": cross_triple,
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
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
        Path(__file__).resolve().parents[1] / "results" / f"slot_test_{model_slug(args.model)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    m = result["metrics"]
    g = m["gradient"]
    gr = m["gradient_residualized"]
    print(f"  gradient raw: related={g['d_target_related']:.3f}  "
          f"opposite={g['d_target_opposite']:.3f}  "
          f"closer={g['fraction_related_closer']:.2f}")
    print(f"  gradient resid: related={gr['d_target_related']:.3f}  "
          f"opposite={gr['d_target_opposite']:.3f}  "
          f"closer={gr['fraction_related_closer']:.2f}")
    print(f"  role acc: {m['role_accuracy']:.3f} (chance {m['chance_role']:.2f}); "
          f"resid {m['role_accuracy_resid']:.3f}")
    print(f"  triple acc: {m['triple_accuracy']:.3f} (chance {m['chance_triple']:.2f}); "
          f"resid {m['triple_accuracy_resid']:.3f}")
    r = m["rdm_controls"]
    print(f"  rdm: length={r['rho_model_length']:+.3f}  triple={r['rho_model_triple']:+.3f}  "
          f"partial_triple|len={r['partial_triple_given_length']:+.3f}")
    cr = result["cross_frame_role"]
    ct = result["cross_frame_triple"]
    print(f"  cross role: A->B {cr['accuracy_A_to_B']:.3f}  B->A {cr['accuracy_B_to_A']:.3f}")
    print(f"  cross triple: A->B {ct['accuracy_A_to_B']:.3f}  B->A {ct['accuracy_B_to_A']:.3f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
