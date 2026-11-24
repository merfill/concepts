#!/usr/bin/env python
"""Experiment 002b: necessity (ablation) + subspace stability (identifiability).

Builds on the isolated attributes (nest, sociality, body). For each:
* necessity: ablate the attribute subspace from the *source* run; if the attribute
  is carried there, source moves toward base and the base->source gap shrinks.
  Control: ablate a random subspace of the same dimension.
* stability: recover the subspace on two data shards and at two layers; compare by
  principal angles (cos -> 1 means the same subspace, i.e. identifiable).

Run:  uv run python scripts/run_validation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.factorial import ATTRS, build_pairs, prompt
from tc.interchange import ablated_probs, extract
from tc.metrics import js_topk

PROBES = {"diet": "It feeds on", "sociality": "It lives", "body": "It has a", "nest": "It builds"}
D_MAP = {"nest": 1, "sociality": 2, "body": 2}
ATTRS_USE = ["nest", "sociality", "body"]


def subspace(deltas, d):
    _, _, Vt = np.linalg.svd(np.stack(deltas), full_matrices=False)
    return Vt[:d].T


def principal_cos(UA, UB):
    s = np.linalg.svd(UA.T @ UB, compute_uv=False)
    return float(np.mean(s))


def rs(rng, D, d):
    Q, _ = np.linalg.qr(rng.standard_normal((D, d)))
    return Q[:, :d]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--layers", default="3,4")
    ap.add_argument("--max-pairs", type=int, default=12)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    layers = [int(x) for x in args.layers.split(",")]
    print(f"[model] {args.model} layers={layers}\n")

    pairs = build_pairs(ATTRS)
    rng = np.random.default_rng(0)

    for a in ATTRS_USE:
        d = D_MAP[a]
        tr = [p for p in pairs if p["attr"] == a and p["template"] in {0, 1}]
        he = [p for p in pairs if p["attr"] == a and p["template"] == 2][: args.max_pairs]
        print(f"=== {a}  (d={d}) ===")
        for layer in layers:
            # hidden for train pairs
            cache = {}
            def hb(p):
                if p["base"] not in cache:
                    cache[p["base"]] = extract(model, tok, prompt(p["base"], PROBES[a]))[0][layer + 1]
                return cache[p["base"]]
            def hs(p):
                if p["source"] not in cache:
                    cache[p["source"]] = extract(model, tok, prompt(p["source"], PROBES[a]))[0][layer + 1]
                return cache[p["source"]]

            dA = [hs(p) - hb(p) for p in tr if p["syn"] == 0]
            dB = [hs(p) - hb(p) for p in tr if p["syn"] != 0]
            Ufull = subspace([hs(p) - hb(p) for p in tr], d)
            cos_shard = principal_cos(subspace(dA, d), subspace(dB, d))

            # necessity + random control on held-out
            D = Ufull.shape[0]
            Ur = rs(rng, D, d)
            nec, necr = [], []
            for p in he:
                hb_p, pb = extract(model, tok, prompt(p["base"], PROBES[a]))
                _, psrc = extract(model, tok, prompt(p["source"], PROBES[a]))
                js0 = js_topk(pb, psrc, 50)
                if js0 < 1e-6:
                    continue
                for U, acc in ((Ufull, nec), (Ur, necr)):
                    ab = ablated_probs(model, tok, prompt(p["source"], PROBES[a]), layer, U)
                    acc.append((js0 - js_topk(pb, ab, 50)) / js0)
            print(f"  L{layer}: necessity real={np.mean(nec):+.3f} rand={np.mean(necr):+.3f}  "
                  f"shard-cos={cos_shard:.3f}")

    # site stability (L3 vs L4) computed per attribute using full-train subspaces
    print("\nsite stability (principal cos between L3 and L4 subspaces):")
    for a in ATTRS_USE:
        d = D_MAP[a]
        tr = [p for p in pairs if p["attr"] == a and p["template"] in {0, 1}]
        Us = {}
        for layer in layers:
            deltas = []
            for p in tr:
                hb = extract(model, tok, prompt(p["base"], PROBES[a]))[0][layer + 1]
                hs = extract(model, tok, prompt(p["source"], PROBES[a]))[0][layer + 1]
                deltas.append(hs - hb)
            Us[layer] = subspace(deltas, d)
        if len(layers) >= 2:
            print(f"  {a:9s} cos(L{layers[0]}, L{layers[1]}) = {principal_cos(Us[layers[0]], Us[layers[1]]):.3f}")


if __name__ == "__main__":
    main()
