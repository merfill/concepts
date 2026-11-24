#!/usr/bin/env python
"""Experiment 003: compositionality of isolated features (plan in
``../experiment-003-2026-11-24-compositionality.md``).

A. Representational additivity: fit h(scene) ~ b0 + sum_a delta_a * x_a over the
   2^4 factorial and report R^2 (main effects only); a label-shuffle fit gives the
   null. Residual of the real fit = interactions.
B. Behavioral forced-choice: give a conjunction, score each candidate concept by
   its length-normalized sequence log-probability after "...most likely a", turn
   into weights over candidates, and read the top/P(bee)/P(wasp)/entropy.

Run:  uv run python scripts/run_composition.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.factorial import ATTRS, build_corpus
from tc.interchange import extract

CONCEPTS = ["bee", "wasp", "ant", "fly", "moth", "beetle", "hornet", "bumblebee", "cricket"]


def _scenes():
    corpus = [c for c in build_corpus(ATTRS) if c["template"] == 0]
    by_scene = {}
    for c in corpus:
        key = tuple(c["values"][a] for a in ATTRS)
        by_scene.setdefault(key, []).append(c["text"])
    return dict(sorted(by_scene.items()))


def additivity(model, tok, layers, rng):
    by_scene = _scenes()
    keys = list(by_scene)
    H = {l: [] for l in layers}
    for key in keys:
        hs = [extract(model, tok, t)[0] for t in by_scene[key]]
        for l in layers:
            H[l].append(np.mean([h[l + 1] for h in hs], axis=0))
    X = np.array([[1.0] + list(key) for key in keys])

    def r2(Hm):
        B, *_ = np.linalg.lstsq(X, Hm, rcond=None)
        res = float(np.sum((Hm - X @ B) ** 2))
        tot = float(np.sum((Hm - Hm.mean(0)) ** 2))
        return 1.0 - res / tot if tot else float("nan")

    out = {}
    for l in layers:
        Hm = np.stack(H[l])
        real = r2(Hm)
        null = float(np.mean([r2(Hm[rng.permutation(len(keys))]) for _ in range(30)]))
        B, *_ = np.linalg.lstsq(X, Hm, rcond=None)
        out[l] = {"r2": real, "r2_shuffle_null": null,
                  "main_effect_norm": {a: float(np.linalg.norm(B[i + 1])) for i, a in enumerate(ATTRS)}}
    return out


def seq_logprob(model, tok, prompt, concept):
    ep = tok(prompt, return_tensors="pt")
    ec = tok(" " + concept, add_special_tokens=False, return_tensors="pt")
    ids = torch.cat([ep.input_ids, ec.input_ids], dim=1)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    lps = torch.log_softmax(out.logits[0], dim=-1).double()
    start = ep.input_ids.shape[1]
    lp = sum(float(lps[start - 1 + i, tid]) for i, tid in enumerate(ec.input_ids[0]))
    return lp / ec.input_ids.shape[1]


def behavior(model, tok, tail="This insect is most likely a"):
    by_scene = _scenes()
    rows = []
    for key, texts in by_scene.items():
        text = texts[0]
        prompt = f"{text}\n\n{tail}"
        lps = np.array([seq_logprob(model, tok, prompt, n) for n in CONCEPTS])
        w = np.exp(lps - lps.max()); w = w / w.sum()
        dist = dict(zip(CONCEPTS, w))
        ent = float(-(w * np.log(w + 1e-12)).sum())
        rows.append({"values": dict(zip(ATTRS, key)), "n_wasp_traits": int(sum(key)),
                     "p_bee": float(dist["bee"]), "p_wasp": float(dist["wasp"]),
                     "top": CONCEPTS[int(np.argmax(w))], "entropy": ent,
                     "dist": {k: round(float(v), 4) for k, v in dist.items()}})
    nw = [r["n_wasp_traits"] for r in rows]
    rho_w = float(stats.spearmanr(nw, [r["p_wasp"] for r in rows]).correlation)
    rho_b = float(stats.spearmanr(nw, [r["p_bee"] for r in rows]).correlation)
    bee = next(r for r in rows if r["n_wasp_traits"] == 0)
    single = {}
    for a in ATTRS:
        combo = {k: 0 for k in ATTRS}; combo[a] = 1
        r = next(x for x in rows if x["values"] == combo)
        single[a] = float(r["p_wasp"] - bee["p_wasp"])
    return rows, {"rho_traits_vs_pwasp": rho_w, "rho_traits_vs_pbee": rho_b,
                  "single_flip_dP_wasp": single}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--layers", default="8,16,20,24")
    ap.add_argument("--out", default="results/composition_summary.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    layers = [int(x) for x in args.layers.split(",")]
    print(f"[model] {args.model} layers={layers}\n")

    add = additivity(model, tok, layers, np.random.default_rng(0))
    print("A) additivity R^2 (main effects) vs shuffle-null:")
    for l in layers:
        print(f"  layer {l}: R2={add[l]['r2']:.3f}  null={add[l]['r2_shuffle_null']:.3f}  |delta_a| "
              + " ".join(f"{a}={add[l]['main_effect_norm'][a]:.2f}" for a in ATTRS))

    rows, summ = behavior(model, tok)
    print(f"\nB) rho(#wasp-traits, P(wasp))={summ['rho_traits_vs_pwasp']:+.3f}  "
          f"rho(..., P(bee))={summ['rho_traits_vs_pbee']:+.3f}")
    print("  single-attribute flip dP(wasp): " +
          " ".join(f"{a}={v:+.3f}" for a, v in summ["single_flip_dP_wasp"].items()))
    print("  scenes (vals -> top, bee/wasp, H):")
    for r in rows:
        print(f"    {'/'.join(str(r['values'][a]) for a in ATTRS)} ({r['n_wasp_traits']}) -> "
              f"{r['top']:9s} bee={r['p_bee']:.3f} wasp={r['p_wasp']:.3f} H={r['entropy']:.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "layers": layers,
                               "additivity": {str(l): add[l] for l in layers},
                               "behavior": rows, "behavior_summary": summ},
                              indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
