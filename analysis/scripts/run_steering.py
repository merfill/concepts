#!/usr/bin/env python
"""Experiment 003C: causal composition by steering the forced-choice answer.

Take the "bee" scene (all attributes 0) and inject each attribute's delta
direction at the answer position; measure how the forced-choice concept
distribution shifts toward "wasp". If features compose causally, the shift should
track their behavioral weight (nest largest) and a random direction should do
little. Works with multi-token concepts via sequence log-prob.

Run:  uv run python scripts/run_steering.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.factorial import ATTRS, build_corpus, prompt
from tc.interchange import extract, seq_logprob_steered

CONCEPTS = ["bee", "wasp", "ant", "fly", "moth", "beetle", "hornet"]
TAIL = "This insect is most likely a"


def weights(model, tok, prm, layer, delta):
    lps = np.array([seq_logprob_steered(model, tok, prm, c, layer, delta) for c in CONCEPTS])
    w = np.exp(lps - lps.max()); w = w / w.sum()
    return dict(zip(CONCEPTS, w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--layer", type=int, default=20)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    L = args.layer
    print(f"[model] {args.model} layer={L}\n")

    corpus = [c for c in build_corpus(ATTRS) if c["template"] == 0 and c["syn"] == 0]
    scenes = {tuple(c["values"][a] for a in ATTRS): c["text"] for c in corpus}
    prm = {k: prompt(t, TAIL) for k, t in scenes.items()}
    hidden = {k: extract(model, tok, p)[0][L + 1] for k, p in prm.items()}
    keys = list(scenes)
    hnorm = float(np.mean([np.linalg.norm(hidden[k]) for k in keys]))

    # delta_a = mean(a=1) - mean(a=0) at the answer position
    delta = {}
    for i, a in enumerate(ATTRS):
        d = np.mean([hidden[k] for k in keys if k[i] == 1], 0) - \
            np.mean([hidden[k] for k in keys if k[i] == 0], 0)
        delta[a] = d / np.linalg.norm(d)
    rng = np.random.default_rng(0)
    rvec = rng.standard_normal(hidden[keys[0]].shape[0]); rvec /= np.linalg.norm(rvec)
    delta["random"] = rvec

    bee_key = tuple(0 for _ in ATTRS)
    wasp_key = tuple(1 for _ in ATTRS)
    zero = np.zeros_like(delta["random"])
    base_bee = weights(model, tok, prm[bee_key], L, zero)
    base_wasp = weights(model, tok, prm[wasp_key], L, zero)
    print(f"bee scene  -> top={max(base_bee, key=base_bee.get)}  P(bee)={base_bee['bee']:.3f} P(wasp)={base_bee['wasp']:.3f}")
    print(f"wasp scene -> top={max(base_wasp, key=base_wasp.get)} P(wasp)={base_wasp['wasp']:.3f} P(bee)={base_wasp['bee']:.3f}\n")

    print("steer BEE scene by +dir (shift toward wasp): dP(wasp) / dP(bee)")
    result = {}
    for name in ATTRS + ["random"]:
        result[name] = {}
        row = []
        for c in (0.5, 1.0):
            dlt = c * hnorm * delta[name]
            w = weights(model, tok, prm[bee_key], L, dlt)
            dw, db = w["wasp"] - base_bee["wasp"], w["bee"] - base_bee["bee"]
            result[name][f"c{c}"] = {"dP_wasp": float(dw), "dP_bee": float(db)}
            row.append(f"c={c}: dW={dw:+.3f} dB={db:+.3f}")
        print(f"  {name:9s} " + "   ".join(row))

    out = Path("results/steering_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "layer": L,
                               "base_bee": base_bee, "base_wasp": base_wasp,
                               "steering": result}, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
