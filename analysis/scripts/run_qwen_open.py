#!/usr/bin/env python
"""Experiment 002e: attribute isolation on Qwen with an OPEN readout.

002c/002d showed a binary Yes/No judgment collapses content into the verdict
axis. 003C showed the open continuation ("...most likely a") keeps attributes
distinct. Here we localize on that open readout: direction v_a = mean(a=1) -
mean(a=0) at the last position of the open prompt; sufficiency/necessity by
interchange/ablation measured as JS closing of the next-token distribution;
specificity matrix and shard identifiability.

Run:  uv run python scripts/run_qwen_open.py --layers 20,24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.extract import next_token_probs
from tc.factorial import ATTRIBUTES_ALL, build_corpus, build_pairs
from tc.interchange import ablated_probs, extract, patched_probs
from tc.metrics import js_topk

TAIL = "This insect is most likely a"
N_TR = 16
N_HE = 4


def opt(text):
    return f"{text}\n\n{TAIL}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--layers", default="20,24")
    ap.add_argument("--k", type=int, default=50)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    layers = [int(x) for x in args.layers.split(",")]
    rng = np.random.default_rng(0)
    print(f"[model] {args.model} layers={layers} k={args.k}\n")

    corpus = [c for c in build_corpus(ATTRIBUTES_ALL) if c["template"] in {0, 1}]
    pairs = build_pairs(ATTRIBUTES_ALL)
    he = {b: [p for p in pairs if p["attr"] == b and p["template"] == 2][:N_HE] for b in ATTRIBUTES_ALL}

    held = {}
    for b in ATTRIBUTES_ALL:
        held[b] = []
        for p in he[b]:
            hs, _ = extract(model, tok, opt(p["source"]))
            pb = next_token_probs(model, tok, opt(p["base"]))
            ps = next_token_probs(model, tok, opt(p["source"]))
            js0 = js_topk(pb, ps, args.k)
            if js0 > 1e-6:
                held[b].append({"pb": opt(p["base"]), "sb": opt(p["source"]),
                                "pbase": pb, "psrc": ps, "hs": hs, "js0": js0})
    print("readout js_sep by attribute:",
          {b: round(float(np.mean([h["js0"] for h in held[b]])), 3) for b in ATTRIBUTES_ALL})

    def attr_dir(a, layer):
        train = [c for c in corpus if c["values"][a] in (0, 1)]
        pick = [train[i] for i in rng.permutation(len(train))[:N_TR]]
        cache = [(extract(model, tok, opt(c["text"]))[0][layer + 1], c["values"][a], c["syn"]) for c in pick]
        H1 = [h for h, val, s in cache if val == 1]; H0 = [h for h, val, s in cache if val == 0]
        v = np.mean(H1, 0) - np.mean(H0, 0); v /= np.linalg.norm(v)
        H1a = [h for h, val, s in cache if val == 1 and s == 0]; H0a = [h for h, val, s in cache if val == 0 and s == 0]
        H1b = [h for h, val, s in cache if val == 1 and s != 0]; H0b = [h for h, val, s in cache if val == 0 and s != 0]
        v0 = np.mean(H1a, 0) - np.mean(H0a, 0); v1 = np.mean(H1b, 0) - np.mean(H0b, 0)
        shard = abs(float(np.dot(v0 / np.linalg.norm(v0), v1 / np.linalg.norm(v1))))
        return v, shard

    def closing(target, layer, v):
        vals = []
        for h in held[target]:
            patched = patched_probs(model, tok, h["pb"], layer, v, h["hs"][layer + 1])
            vals.append((h["js0"] - js_topk(patched, h["psrc"], args.k)) / h["js0"])
        return float(np.mean(vals)) if vals else float("nan")

    def necessity(target, layer, v):
        vals = []
        for h in held[target]:
            ab = ablated_probs(model, tok, h["sb"], layer, np.asarray(v).reshape(-1, 1))
            vals.append((h["js0"] - js_topk(h["pbase"], ab, args.k)) / h["js0"])
        return float(np.mean(vals)) if vals else float("nan")

    for layer in layers:
        V, sh = {}, {}
        for a in ATTRIBUTES_ALL:
            V[a], sh[a] = attr_dir(a, layer)
        print(f"\n=== layer {layer} ===")
        for a in ATTRIBUTES_ALL:
            row = {b: closing(b, layer, V[a]) for b in ATTRIBUTES_ALL}
            off = np.mean([row[b] for b in ATTRIBUTES_ALL if b != a])
            nec = necessity(a, layer, V[a])
            rv = rng.standard_normal(V[a].shape[0]); rv /= np.linalg.norm(rv)
            print(f"  {a:9s} diag={row[a]:+.2f} off={off:+.2f} nec={nec:+.2f} "
                  f"shard={sh[a]:.2f}  [" + " ".join(f"{b}={row[b]:+.2f}" for b in ATTRIBUTES_ALL if b != a) + "]")


if __name__ == "__main__":
    main()
