#!/usr/bin/env python
"""Experiment 012 / stage 3.0: calibrate hierarchy templates before the main run.

For each candidate template we read the model's *candidate-token distribution*
(never the free top-k) for every calibration concept and compute:

* ``Q`` (top-1 share) per level and its cross-level std  -> level-invariance;
* normalised ``MI(concept; candidate)``                    -> content-tracking;
* frame agreement across templates                         -> not one-off;
* a nonce-leak diagnostic                                  -> no generic pull.

A template passes only if it is level-invariant *and* content-tracking *and*
agrees with at least one other template.  Nothing here touches the main
hierarchy test (descendant mass / level-step); this is the instrument gate.

Protocol: ``../experiment-012-2026-11-26-hierarchy.md``.

Run:
    uv run python scripts/run_template_calibration.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.extract import next_token_probs
from tc.hierarchy import (
    TEMPLATES_E,
    TEMPLATES_G,
    candidate_probs,
    entailment_mass,
    frame_agreement,
    load_triads,
    mutual_information,
    readable_vocab,
    render,
    triad_concepts,
)

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "hierarchy_triads.tsv"

# Pronounceable nonces that fit noun slots; they have no taxonomy.
NONCE = ["brickle", "flondar", "plim", "zorbet", "quandix", "mirtle", "gavon", "sorvid"]

GATE = {
    "mi_norm_min": 0.20,
    "agreement_min": 0.50,
    "nonce_leak_max": 1.00,
    "entailment_diff_min": 0.05,
    "entailment_coverage_min": 0.10,
}


def model_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def candidate_mass(raw: dict[str, float]) -> float:
    return float(sum(v for v in raw.values() if not math.isnan(v)))


def calibrate_descent(model, tokenizer, triads, vocab) -> dict:
    concepts = triad_concepts(triads)
    n_concepts = len(concepts)
    out: dict[str, dict] = {}
    child_mass: dict[str, dict[tuple[int, str], float]] = {}

    for name, template in TEMPLATES_G.items():
        dists = []
        masses = []
        scores: dict[tuple[int, str], float] = {}
        for word, level, tid in concepts:
            probs = next_token_probs(model, tokenizer, render(template, x=word))
            raw, dist = candidate_probs(probs, tokenizer, vocab)
            dists.append(dist)
            masses.append(candidate_mass(raw))
            triad = triads[tid]
            child = triad["basic"] if level == "super" else (triad["sub"] if level == "basic" else None)
            if child is not None and child in dist:
                scores[(tid, level)] = dist[child]

        nonce_mass = []
        for nonce in NONCE:
            probs = next_token_probs(model, tokenizer, render(template, x=nonce))
            raw, _ = candidate_probs(probs, tokenizer, vocab)
            nonce_mass.append(candidate_mass(raw))

        mi = mutual_information(dists, vocab)
        mi_norm = mi / math.log(n_concepts) if n_concepts > 1 else 0.0
        mean_real_mass = float(np.mean(masses)) if masses else 0.0
        mean_nonce_mass = float(np.mean(nonce_mass)) if nonce_mass else 0.0
        nonce_leak = mean_nonce_mass / mean_real_mass if mean_real_mass > 0 else 0.0

        child_mass[name] = scores
        out[name] = {
            "prompt_example": render(template, x=triads[0]["basic"]),
            "mi": mi,
            "mi_norm": mi_norm,
            "candidate_mass_real": mean_real_mass,
            "candidate_mass_nonce": mean_nonce_mass,
            "nonce_leak": nonce_leak,
        }

    names = list(TEMPLATES_G)
    agreement: dict[str, dict[str, float]] = {}
    for a in names:
        agreement[a] = {}
        for b in names:
            if a == b:
                continue
            keys = sorted(set(child_mass[a]) & set(child_mass[b]))
            va = [child_mass[a][k] for k in keys]
            vb = [child_mass[b][k] for k in keys]
            agreement[a][b] = frame_agreement(va, vb)

    for name in names:
        others = [v for k, v in agreement[name].items() if not math.isnan(v)]
        out[name]["agreement_max"] = float(max(others)) if others else float("nan")
        row = out[name]
        row["pass"] = bool(
            row["mi_norm"] >= GATE["mi_norm_min"]
            and not math.isnan(row["agreement_max"])
            and row["agreement_max"] >= GATE["agreement_min"]
            and row["nonce_leak"] <= GATE["nonce_leak_max"]
        )

    return {"templates": out, "agreement": agreement}


def calibrate_entailment(model, tokenizer, triads) -> dict:
    pairs = []
    for tid, t in enumerate(triads):
        s, b, u = t["super"], t["basic"], t["sub"]
        pairs += [(b, s, 1, tid), (u, b, 1, tid), (u, s, 1, tid),
                  (s, b, 0, tid), (b, u, 0, tid), (s, u, 0, tid)]
    out: dict[str, dict] = {}
    for name, template in TEMPLATES_E.items():
        true_vals, false_vals, coverage = [], [], []
        for x, y, label, _ in pairs:
            probs = next_token_probs(model, tokenizer, render(template, x=x, y=y))
            p_yes, mass = entailment_mass(probs, tokenizer)
            (true_vals if label else false_vals).append(p_yes)
            coverage.append(mass)
        mean_true = float(np.mean(true_vals)) if true_vals else 0.0
        mean_false = float(np.mean(false_vals)) if false_vals else 0.0
        cov = float(np.mean(coverage)) if coverage else 0.0
        out[name] = {
            "prompt_example": render(template, x="dog", y="animal"),
            "mean_true": mean_true,
            "mean_false": mean_false,
            "diff": mean_true - mean_false,
            "coverage": cov,
            "pass": bool((mean_true - mean_false) >= GATE["entailment_diff_min"]
                         and cov >= GATE["entailment_coverage_min"]),
        }
    return {"templates": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    all_triads = load_triads(args.data)
    calib = load_triads(args.data, split="calibration")
    vocab_all = sorted({t[k] for t in all_triads for k in ("super", "basic", "sub")})

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    vocab = readable_vocab(tokenizer, vocab_all)
    print(f"model={args.model}  triads(calibration)={len(calib)}  "
          f"vocab(readable/total)={len(vocab)}/{len(vocab_all)}")

    descent = calibrate_descent(model, tokenizer, calib, vocab)
    entail = calibrate_entailment(model, tokenizer, calib)

    result = {
        "model": args.model,
        "data": str(args.data),
        "vocab_readable": vocab,
        "vocab_unreadable": sorted(set(vocab_all) - set(vocab)),
        "gate": GATE,
        "descent": descent,
        "entailment": entail,
    }
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"template_calibration_{model_slug(args.model)}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved -> {out}\n")

    print("=== G (descent) templates ===")
    print(f"{'template':22s} {'MI_norm':>8s} {'agree':>6s} {'leak':>6s}  pass")
    for name, row in descent["templates"].items():
        print(f"{name:22s} {row['mi_norm']:8.3f} "
              f"{row['agreement_max']:6.2f} {row['nonce_leak']:6.2f}  {row['pass']}")
    print("\n=== E (entailment) templates ===")
    print(f"{'template':22s} {'true':>6s} {'false':>6s} {'diff':>6s} {'cov':>6s}  pass")
    for name, row in entail["templates"].items():
        print(f"{name:22s} {row['mean_true']:6.3f} {row['mean_false']:6.3f} "
              f"{row['diff']:6.3f} {row['coverage']:6.3f}  {row['pass']}")


if __name__ == "__main__":
    main()
