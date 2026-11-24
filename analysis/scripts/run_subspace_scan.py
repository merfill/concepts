#!/usr/bin/env python
"""Experiment 002, phase 1: linear (d=1) vs distributed (d>1) attribute subspace.

DAS-style, training-free: for attribute a build the subspace spanned by the top-d
principal directions of the interchange deltas (source - base) at a site, swap
that subspace, and measure how much of the base->source readout gap closes on
held-out pairs. Compared against a *random* d-subspace at the same d.

Answer: if real(d=1) already saturates (≈ real(d=4)), the attribute is a compact
1-D feature; if real(d) grows with d while random(d) does not, it is distributed.

Run:  uv run python scripts/run_subspace_scan.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.factorial import ATTRIBUTES_ALL, build_corpus, build_pairs, prompt
from tc.interchange import extract, get_blocks, patched_probs_subspace
from tc.metrics import js_topk

PROBES = {"diet": "It feeds on", "sociality": "It lives", "body": "It has a",
          "nest": "It builds", "setting": "It is found"}
DS = [1, 2, 4]
N_TRAIN = 32
N_HELD = 10


def rs(rng, D, d):
    Q, _ = np.linalg.qr(rng.standard_normal((D, d)))
    return Q[:, :d]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--layers", default="3,4")
    ap.add_argument("--out", default="results/subspace_scan.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    layers = [int(x) for x in args.layers.split(",")]
    print(f"[model] {args.model} layers={layers} ds={DS}\n")

    pairs = build_pairs(ATTRIBUTES_ALL)
    rng = np.random.default_rng(0)
    result = {}

    for a in ATTRIBUTES_ALL:
        probe = PROBES[a]
        tr = [p for p in pairs if p["attr"] == a and p["template"] in {0, 1}][:N_TRAIN]
        he = [p for p in pairs if p["attr"] == a and p["template"] == 2][:N_HELD]
        result[a] = {}
        for layer in layers:
            deltas = []
            for p in tr:
                hb, _ = extract(model, tok, prompt(p["base"], probe))
                hs, _ = extract(model, tok, prompt(p["source"], probe))
                deltas.append(hs[layer + 1] - hb[layer + 1])
            deltas = np.stack(deltas)
            _, _, Vt = np.linalg.svd(deltas, full_matrices=False)

            held = []
            for p in he:
                hb, pb = extract(model, tok, prompt(p["base"], probe))
                hs, ps = extract(model, tok, prompt(p["source"], probe))
                js0 = js_topk(pb, ps, 50)
                if js0 > 1e-6:
                    held.append({"prompt": prompt(p["base"], probe), "src": hs[layer + 1],
                                 "pb": pb, "ps": ps, "js0": js0})

            row = {"js_sep": float(np.mean([h["js0"] for h in held])) if held else float("nan")}
            D = Vt.shape[1]
            for d in DS:
                U = Vt[:d].T
                Ur = rs(rng, D, d)
                def closing(Ux):
                    vals = []
                    for h in held:
                        patched = patched_probs_subspace(model, tok, h["prompt"], layer, Ux, h["src"])
                        vals.append((h["js0"] - js_topk(patched, h["ps"], 50)) / h["js0"])
                    return float(np.mean(vals)) if vals else float("nan")
                row[f"d{d}_real"] = closing(U)
                row[f"d{d}_rand"] = closing(Ur)
            result[a][layer] = row
            r = row
            print(f"  {a:9s} L{layer}: " + "  ".join(
                f"d{d} {r[f'd{d}_real']:+.2f}/rand {r[f'd{d}_rand']:+.2f}" for d in DS)
                + f"   js_sep={r['js_sep']:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "layers": layers, "ds": DS, "result": result},
                              indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
