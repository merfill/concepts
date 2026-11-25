#!/usr/bin/env python
"""Experiment 007 / program stage 0: invariance and null-operator battery.

Protocol: ``../experiment-007-2026-11-25-invariance-battery.md``.

We calibrate the intervention plumbing on analytically known no-ops:

* Z1: ``h -> c*h`` (c>0) at the *output of the last block* is a no-op (norm eats
  the scale); at a block *input* it is not.
* Z2: ``h -> h + c*1`` is a no-op on LayerNorm models (distilgpt2) at every site,
  but changes output on RMSNorm models (Qwen2.5).
* Z3: there are behaviourally-null (JS~0) yet geometrically large (dcos>=0.1)
  directions.

No model is needed for the core physics; this script confirms it on a real
transformer and prints actual ``(site, layer, position)`` so hooks can be audited.

Run:
    uv run python scripts/run_invariance.py --smoke
    uv run python scripts/run_invariance.py --model distilgpt2 --layers 0,3,5
    uv run python scripts/run_invariance.py --model Qwen/Qwen2.5-1.5B-Instruct --layers 12,20,27
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.corpus import HARD_GROUPS
from tc.interchange import get_blocks
from tc.invariance import (
    coord_shift,
    delta_cos,
    gaussian_noise,
    identity,
    norm_type,
    remove_projection,
    rotate_plane,
    scale,
    shift,
    top1_flip,
    top_norm_coord,
    transformed_logits,
    transformed_probs,
    unit,
)
from tc.metrics import js_topk

TAU_NOOP = 1e-6       # float32
TAU_NOOP_LOW = 1e-3   # bf16/fp16 (protocol §9)
TAU_EFF = 1e-2
K = 50


def model_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def pick_texts(n: int):
    flat = [t for items in HARD_GROUPS.values() for t in items]
    return flat[:n]


def sites_for(layers, n_layers):
    """Input/output of chosen blocks + final hidden state (no duplicate final)."""
    sites = []
    for l in layers:
        sites.append(("block", int(l), "in"))
        if int(l) != n_layers - 1:
            sites.append(("block", int(l), "out"))
    sites.append(("final",))
    return sites


def site_label(site) -> str:
    if site[0] == "final":
        return "final"
    return f"L{site[1]}{site[2]}"


def ops_for_site(mean_h, rng, names=None):
    d = mean_h.shape[0]
    ops = []
    for c in (0.5, 0.9, 1.1, 2.0, -1.0):
        ops.append({"op": "scale", "param": c, "fn": (lambda h, c=c: scale(h, c))})
    for c in (1.0, -1.0, 10.0, -10.0):
        ops.append({"op": "shift", "param": c, "fn": (lambda h, c=c: shift(h, c))})
    ops.append({"op": "coord_top", "param": 1.0,
                "fn": (lambda h: coord_shift(h, 1.0, top_norm_coord(h)))})
    u = unit(rng.standard_normal(d))
    v = rng.standard_normal(d)
    ops.append({"op": "rotate", "param": float(np.pi / 2),
                "fn": (lambda h, u=u, v=v: rotate_plane(h, u, v, np.pi / 2))})
    noise_rng = np.random.default_rng(12345)
    ops.append({"op": "noise", "param": 0.1,
                "fn": (lambda h, r=noise_rng: gaussian_noise(h, 0.1 * float(np.linalg.norm(h)), r))})
    ops.append({"op": "proj_mean", "param": 0.0,
                "fn": (lambda h, m=mean_h: remove_projection(h, m))})
    ops.append({"op": "proj_ones", "param": 0.0,
                "fn": (lambda h: remove_projection(h, np.ones_like(h)))})
    ur = unit(rng.standard_normal(d))
    ops.append({"op": "proj_random", "param": 0.0,
                "fn": (lambda h, u=ur: remove_projection(h, u))})
    if names is not None:
        ops = [o for o in ops if o["op"] in names]
    return ops


def run(args):
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    n_layers = len(get_blocks(model))
    ntype = norm_type(model)
    dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    tau_noop = TAU_NOOP if dtype in ("float32", "float64") else TAU_NOOP_LOW
    if args.layers == "auto":
        layers = sorted({0, n_layers // 2, n_layers - 1})
        if args.smoke:
            layers = [n_layers // 2]
    else:
        layers = [int(x) for x in args.layers.split(",")]

    # Guard: a full battery on a large model on CPU is hours. In smoke mode,
    # keep distilgpt2 intact but cut the Qwen run to a couple of cheap checks.
    op_names = None
    if args.smoke and n_layers > 12:
        op_names = {"scale", "shift", "coord_top", "proj_ones", "proj_mean"}
    texts = pick_texts(min(args.n_texts, 2) if (args.smoke and n_layers > 12) else args.n_texts)
    sites = sites_for(layers, n_layers)
    if args.smoke and n_layers > 12:
        sites = [("block", layers[0], "in"), ("final",)]
    print(f"[model] {args.model}  blocks={n_layers}  norm={ntype}  dtype={dtype}  "
          f"tau_noop={tau_noop:.0e}")
    print(f"[sites] {[site_label(s) for s in sites]}  texts={len(texts)}  "
          f"ops={'all' if op_names is None else sorted(op_names)}  k={K}\n")

    records = []
    for site_idx, site in enumerate(sites):
        label = site_label(site)
        base_probs, hs = {}, []
        for t in texts:
            r = transformed_probs(model, tok, t, site, identity)
            base_probs[t] = r["probs"]
            hs.append(r["h_orig"])
        mean_h = np.mean(hs, axis=0)
        ops = ops_for_site(mean_h, np.random.default_rng(site_idx), op_names)
        for op in ops:
            for t in texts:
                r = transformed_probs(model, tok, t, site, op["fn"])
                records.append({
                    "site": label, "kind": "block" if site[0] == "block" else "final",
                    "layer": (site[1] if site[0] == "block" else n_layers - 1),
                    "op": op["op"], "param": float(op["param"]), "text": t,
                    "js": float(js_topk(base_probs[t], r["probs"], K)),
                    "dcos": float(delta_cos(r["h_orig"], r["h_pert"])),
                    "flip": bool(top1_flip(base_probs[t], r["probs"])),
                })
        print(f"  done {label:7s} ({len(ops)} ops x {len(texts)} texts)")

    # ---- logit-space ops (temperature / uniform shift) --------------------
    logit_records = []
    base_logits = {}
    for t in texts:
        base_logits[t] = transformed_logits(model, tok, t, identity)["probs"]
    logit_ops = [
        ("logit_scale", 0.5, lambda z: 0.5 * z),
        ("logit_scale", 2.0, lambda z: 2.0 * z),
        ("logit_shift", 1.0, lambda z: z + 1.0),
        ("logit_shift", -1.0, lambda z: z - 1.0),
    ]
    for name, param, fn in logit_ops:
        for t in texts:
            r = transformed_logits(model, tok, t, fn)
            logit_records.append({
                "op": name, "param": float(param), "text": t,
                "js": float(js_topk(base_logits[t], r["probs"], K)),
                "flip": bool(top1_flip(base_logits[t], r["probs"])),
            })

    # ---- aggregate --------------------------------------------------------
    def aggregate(recs, keys):
        agg = defaultdict(list)
        for r in recs:
            agg[tuple(r[k] for k in keys)].append(r)
        out = []
        for key, rs in agg.items():
            js = [x["js"] for x in rs]
            has_dc = "dcos" in rs[0]
            dc = [x["dcos"] for x in rs] if has_dc else []
            row = {k: v for k, v in zip(keys, key)}
            row.update({
                "n": len(rs), "js_mean": float(np.mean(js)), "js_max": float(np.max(js)),
                "dcos_mean": float(np.mean(dc)) if has_dc else None,
                "dcos_max": float(np.max(dc)) if has_dc else None,
                "flips": int(sum(x["flip"] for x in rs)),
            })
            out.append(row)
        return out

    summary = aggregate(records, ["site", "kind", "layer", "op", "param"])
    logit_summary = aggregate(logit_records, ["op", "param"])

    # ---- verdicts --------------------------------------------------------
    final_scale_pos = [s for s in summary
                       if s["kind"] == "final" and s["op"] == "scale" and s["param"] > 0]
    mid_scale_pos = [s for s in summary
                     if s["kind"] == "block" and s["op"] == "scale" and s["param"] > 0]
    shifts = [s for s in summary if s["op"] == "shift"]

    z1_final = (all(s["js_max"] <= tau_noop for s in final_scale_pos)
                if final_scale_pos else None)
    z1_mid = (any(s["js_mean"] >= TAU_EFF for s in mid_scale_pos)
              if mid_scale_pos else None)
    if ntype == "layernorm":
        z2 = all(s["js_max"] <= tau_noop for s in shifts) if shifts else None
    elif ntype == "rmsnorm":
        z2 = any(s["js_mean"] >= TAU_EFF for s in shifts) if shifts else None
    else:
        z2 = None

    z3 = [s for s in summary if s["js_max"] <= tau_noop and s["dcos_mean"] >= 0.1]

    verdicts = {
        "norm_type": ntype, "dtype": dtype,
        "Z1_scale_final_noop": z1_final,
        "Z1_scale_mid_effect": z1_mid,
        "Z2_shift_expected": z2,
        "Z3_null_but_geometric": [{"site": s["site"], "op": s["op"], "param": s["param"],
                                    "js_max": s["js_max"], "dcos_mean": s["dcos_mean"]}
                                   for s in z3],
        "tau_noop": tau_noop, "tau_eff": TAU_EFF,
    }

    # ---- print -----------------------------------------------------------
    print("\n== key rows (JS mean/max, dcos mean) ==")
    for s in summary:
        if s["op"] in ("scale", "shift") or s["kind"] == "final":
            print(f"  {s['site']:7s} {s['op']:9s} {s['param']:+.2f}  "
                  f"JS {s['js_mean']:.2e}/{s['js_max']:.2e}  dcos {s['dcos_mean']:.3f}")
    print("\n== logit-space ==")
    for s in logit_summary:
        print(f"  {s['op']:12s} {s['param']:+.1f}  JS {s['js_mean']:.2e}/{s['js_max']:.2e}  "
              f"flips {s['flips']}")
    print("\n== verdicts ==")
    print(f"  norm = {ntype}")
    print(f"  Z1 scale on final no-op   : {z1_final}")
    print(f"  Z1 scale mid  has effect  : {z1_mid}")
    print(f"  Z2 shift as predicted     : {z2}")
    print(f"  Z3 null-but-geometric     : {len(z3)} rows")
    for s in z3[:8]:
        print(f"      {s['site']:7s} {s['op']:9s} {s['param']:+.2f} "
              f"JS={s['js_max']:.1e} dcos={s['dcos_mean']:.3f}")
    if z1_final is False or z2 is False:
        print("  !! predicted no-op failed -> audit site/position/dtype before interpreting")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "norm_type": ntype, "dtype": dtype, "n_blocks": n_layers,
        "layers": layers, "sites": [site_label(s) for s in sites],
        "texts": texts, "k": K, "tau_noop": tau_noop, "tau_eff": TAU_EFF,
        "records": records, "logit_records": logit_records,
        "summary": summary, "logit_summary": logit_summary, "verdicts": verdicts,
    }, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--layers", default="auto",
                    help="comma list, or 'auto' (0, mid, last)")
    ap.add_argument("--n-texts", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.smoke:
        args.n_texts = min(args.n_texts, 4)

    if not args.out:
        args.out = f"results/invariance_{model_slug(args.model)}.json"
    run(args)


if __name__ == "__main__":
    main()
