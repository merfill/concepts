#!/usr/bin/env python
"""Experiment 006 / program stage 4.0: is linear interpolation along the chord valid?

Protocol: ``../experiment-006-2026-11-25-interpolation-validity.md``.

For graded ladders (temperature, size) we compare the *natural* path of hidden
states across the rungs with the straight chord, and check whether steering along
the chord reproduces the natural behavioral curve.  The binary factorial ladder
is used only for path-independence (24 flip orders).

The residual stream is NOT on the unit sphere, so raw Euclidean is only a
baseline.  The path verdict is computed on several representations:
    raw | unit (direction) | centered | whitened | pc_removed
plus angular (geodesic) diagnostics and per-rung behavioral JS profiling.

Run:
    uv run python scripts/run_interp_validation.py --smoke
    uv run python scripts/run_interp_validation.py --model distilgpt2 --layers 0,3,5
    uv run python scripts/run_interp_validation.py --model Qwen/Qwen2.5-1.5B-Instruct --layers 12,20 --smoke
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
from tc.corpus import HARD_GROUPS
from tc.interchange import get_blocks, steered_probs
from tc.interpolate import (
    angular_metrics,
    background_stats,
    behavioral_curve,
    behavioral_profile,
    center_rows,
    chord_delta,
    factorial_ladder,
    flip_orders,
    graded_texts,
    interpolation_verdict,
    order_step_signature,
    pairwise_consistency,
    path_metrics,
    remove_pcs_rows,
    standardize_rows,
    unit_rows,
)
from tc.invariance import identity, norm_type, transformed_probs

REPS = ("raw", "unit", "centered", "whitened", "pc_removed")


def model_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def hidden_and_probs(model, tok, text, site):
    r = transformed_probs(model, tok, text, site, identity)
    return r["h_orig"], r["probs"]


def background_texts(n: int):
    flat = [t for items in HARD_GROUPS.values() for t in items]
    return flat[:n]


def tag(verdict: str) -> str:
    return verdict.split(":")[0]


def consensus(tags) -> str:
    return tags[0] if all(t == tags[0] for t in tags) else "metric-dependent"


def run_graded(model, tok, layer, kind, stats, args):
    lad = graded_texts(kind)
    site = ("block", int(layer), "out")
    hs, probs = [], []
    for t in lad["texts"]:
        h, p = hidden_and_probs(model, tok, t, site)
        hs.append(h)
        probs.append(p)
    H = np.stack(hs)
    mu, sigma, comps = stats

    norms = np.linalg.norm(H, axis=1)
    norm_cv = float(norms.std() / norms.mean()) if norms.mean() > 0 else float("nan")

    reps = {
        "raw": H,
        "unit": unit_rows(H),
        "centered": center_rows(H, mu),
        "whitened": standardize_rows(H, mu, sigma),
        "pc_removed": remove_pcs_rows(H, mu, sigma, comps),
    }
    geoms = {name: path_metrics(X) for name, X in reps.items()}

    # behavior is independent of the geometric representation
    bp = behavioral_profile(probs, k=args.k)
    delta = chord_delta(H)
    steered = [steered_probs(model, tok, lad["texts"][0], int(layer), a * delta)
               for a in geoms["raw"]["t"]]
    beh = behavioral_curve(probs, steered, k=args.k)

    metrics = {}
    for name in REPS:
        metrics[name] = {
            "geometry": geoms[name],
            "verdict": interpolation_verdict(geoms[name], beh["match_rho"]),
        }
    ang = angular_metrics(H)
    tags = [tag(metrics[name]["verdict"]) for name in REPS]
    verdict = consensus(tags)

    print(f"  [{kind:11s} L{layer}] cv={norm_cv:.2f}  "
          + " ".join(f"{name}:{tag(metrics[name]['verdict'])}" for name in REPS)
          + f"  angL={ang['geodesic_length_ratio']:.2f}  beh={beh['match_rho']:+.2f}"
          + f"  js_end={bp['js_endpoints']:.3f}  js_mono={bp['js_from_start_monotonicity']:+.2f}"
          + f"  => {verdict}")

    return {
        "ladder": kind, "layer": int(layer),
        "norm_cv": norm_cv, "norms": norms.tolist(),
        "metrics": metrics, "angular": ang,
        "behavioral_curve": beh, "behavioral_profile": bp,
        "verdict": verdict, "metric_tags": dict(zip(REPS, tags)),
        "delta_norm": float(np.linalg.norm(delta)),
    }


def run_path_independence(model, tok, layer, args):
    orders = flip_orders()[: args.orders]
    site = ("block", int(layer), "out")
    signatures = []
    for order in orders:
        lad = factorial_ladder(order)
        H = np.stack([hidden_and_probs(model, tok, t, site)[0] for t in lad["texts"]])
        signatures.append(order_step_signature(H, order))
    score = pairwise_consistency(signatures)
    verdict = ("order-independent (commutative)" if score >= 0.7
               else "order-dependent (link to 003)")
    print(f"  [path-independence L{layer}, {len(orders)} orders] "
          f"consistency={score:+.3f}  => {verdict}")
    return {"layer": int(layer), "n_orders": len(orders), "consistency": score,
            "threshold": 0.7, "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--layers", default="auto")
    ap.add_argument("--ladders", default="temperature,size")
    ap.add_argument("--orders", type=int, default=24)
    ap.add_argument("--bg-texts", type=int, default=20, help="background set for whitening/PC")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    n_layers = len(get_blocks(model))
    ntype = norm_type(model)
    dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    if args.layers == "auto":
        layers = [n_layers // 2] if args.smoke else sorted({0, n_layers // 2, n_layers - 1})
    else:
        layers = [int(x) for x in args.layers.split(",")]
    ladders = [s for s in args.ladders.split(",") if s]

    bg = background_texts(args.bg_texts)
    print(f"[model] {args.model}  blocks={n_layers}  norm={ntype}  dtype={dtype}")
    print(f"[layers] {layers}  ladders={ladders}  orders={args.orders}  bg={len(bg)}  k={args.k}\n")

    graded = []
    for layer in layers:
        site = ("block", int(layer), "out")
        bgH = np.stack([hidden_and_probs(model, tok, t, site)[0] for t in bg])
        stats = background_stats(bgH, n_pcs=3)
        print(f"[L{layer}] background {len(bg)} texts; top-PC removed = 3")
        for kind in ladders:
            graded.append(run_graded(model, tok, layer, kind, stats, args))

    pi_layer = layers[len(layers) // 2]
    pi = run_path_independence(model, tok, pi_layer, args)

    out = Path(args.out) if args.out else Path(f"results/interp_validation_{model_slug(args.model)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "norm_type": ntype, "dtype": dtype, "n_blocks": n_layers,
        "layers": layers, "ladders": ladders, "orders": args.orders,
        "bg_texts": len(bg), "k": args.k, "reps": list(REPS),
        "graded": graded, "path_independence": pi,
    }, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
