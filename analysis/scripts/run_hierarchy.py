#!/usr/bin/env python
"""Experiment 012 / stage 3.1: main hierarchy test on curated triads (arm a).

For each triad's *super* and *basic* concept we read the model's choice among a
small labelled candidate set and ask: does the answer descend one level in the
right branch?  The candidate labels give the factor control:

* ``desc``               the true child (exactly one level down)   -> target
* ``anc``                the parent                                -> trap "up"
* ``desc_wrong_level``   in the subtree but two levels down         -> step control
* ``sib_branch``         same level, other branch                   -> "not just frequent"
* ``unrelated``          other branch, other level                  -> background

Controls: nonce substitution (no concepts), label shuffle, and a frequency-only
baseline (Zipf of each candidate).  Second readout: entailment ``X is a Y`` AUC.

Protocol: ``../experiment-012-2026-11-26-hierarchy.md``.

Run:
    uv run python scripts/run_hierarchy.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from wordfreq import zipf_frequency

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.extract import next_token_probs
from tc.hierarchy import (
    TEMPLATES_E,
    TEMPLATES_G,
    auc,
    build_candidates,
    descendant_metrics,
    entailment_mass,
    load_triads,
    read_candidates,
    render,
)

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "hierarchy_triads.tsv"
NONCE = ["brickle", "flondar", "plim", "zorbet", "quandix", "mirtle", "gavon", "sorvid"]

GATE = {
    "g_auc_min": 0.80,
    "g_margin_min": 0.05,
    "e_auc_min": 0.80,
    "templates_min": 2,
}


def model_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def shuffled_labels(labels: dict[str, str], rng) -> dict[str, str]:
    words = list(labels)
    vals = [labels[w] for w in words]
    rng.shuffle(vals)
    return dict(zip(words, vals))


def freq_auc(labels: dict[str, str]) -> float:
    words = list(labels)
    scores = [zipf_frequency(w, "en") for w in words]
    positive = [labels[w] == "desc" for w in words]
    return auc(scores, positive)


def run_descent(model, tokenizer, triads) -> dict:
    out: dict[str, dict] = {}
    for name, template in TEMPLATES_G.items():
        aucs, masses, tops, freqs, nonces, shuffles = [], [], [], [], [], []
        for tid in range(len(triads)):
            cands = build_candidates(triads, tid, n_sib=4, n_unrel=4, seed=0)
            for idx, ((concept, level), labels) in enumerate(cands.items()):
                words = list(labels)
                prompt = render(template, x=concept)
                _, dist = read_candidates(model, tokenizer, prompt, words)
                m = descendant_metrics(dist, labels)
                aucs.append(m["auc"])
                masses.append(m["desc_mass"])
                tops.append(float(m["top_is_desc"]))
                freqs.append(freq_auc(labels))

                nonce_word = NONCE[(tid + idx) % len(NONCE)]
                _, dist_n = read_candidates(model, tokenizer,
                                            render(template, x=nonce_word), words)
                nonces.append(descendant_metrics(dist_n, labels)["auc"])

                rng = np.random.default_rng(tid * 10 + idx)
                shuffles.append(descendant_metrics(dist, shuffled_labels(labels, rng))["auc"])

        mean_auc = float(np.nanmean(aucs))
        mean_freq = float(np.nanmean(freqs))
        margin = mean_auc - max(mean_freq, 1.0 - mean_freq)
        out[name] = {
            "prompt_example": render(template, x=triads[0]["basic"]),
            "auc": mean_auc,
            "desc_mass": float(np.nanmean(masses)),
            "top_is_desc": float(np.nanmean(tops)),
            "freq_auc": mean_freq,
            "margin": margin,
            "nonce_auc": float(np.nanmean(nonces)),
            "shuffle_auc": float(np.nanmean(shuffles)),
            "pass": bool(mean_auc >= GATE["g_auc_min"] and margin >= GATE["g_margin_min"]),
        }
    return {"templates": out}


def run_entailment(model, tokenizer, triads) -> dict:
    pairs = []
    for tid, t in enumerate(triads):
        s, b, u = t["super"], t["basic"], t["sub"]
        pairs += [(b, s, 1), (u, b, 1), (u, s, 1),
                  (s, b, 0), (b, u, 0), (s, u, 0)]
    out: dict[str, dict] = {}
    for name, template in TEMPLATES_E.items():
        scores, labels = [], []
        for x, y, label in pairs:
            probs = next_token_probs(model, tokenizer, render(template, x=x, y=y))
            p_yes, _ = entailment_mass(probs, tokenizer)
            scores.append(p_yes)
            labels.append(bool(label))
        a = auc(scores, labels)
        out[name] = {
            "prompt_example": render(template, x="dog", y="animal"),
            "auc": a,
            "pass": bool(a >= GATE["e_auc_min"]),
        }
    return {"templates": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--split", default="main")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    triads = load_triads(args.data, split=args.split)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    print(f"model={args.model}  triads={len(triads)} (split={args.split})")

    descent = run_descent(model, tokenizer, triads)
    entail = run_entailment(model, tokenizer, triads)

    g_pass = sum(r["pass"] for r in descent["templates"].values())
    e_pass = sum(r["pass"] for r in entail["templates"].values())
    verdict = "hierarchy-confirmed" if (g_pass >= GATE["templates_min"] and e_pass >= 1) else "absent"

    result = {
        "model": args.model,
        "data": str(args.data),
        "split": args.split,
        "gate": GATE,
        "descent": descent,
        "entailment": entail,
        "summary": {"g_pass": g_pass, "e_pass": e_pass, "verdict": verdict},
    }
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"hierarchy_{args.split}_{model_slug(args.model)}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved -> {out}\n")

    print("=== G (descent) ===")
    print(f"{'template':22s} {'AUC':>6s} {'mass':>6s} {'top%d':>6s} {'freq':>6s} {'margin':>7s} {'nonce':>6s} {'shuf':>6s} pass")
    for name, r in descent["templates"].items():
        print(f"{name:22s} {r['auc']:6.3f} {r['desc_mass']:6.3f} {r['top_is_desc']*100:6.0f} "
              f"{r['freq_auc']:6.3f} {r['margin']:7.3f} {r['nonce_auc']:6.3f} {r['shuffle_auc']:6.3f} {r['pass']}")
    print("\n=== E (entailment) ===")
    print(f"{'template':22s} {'AUC':>6s}  pass")
    for name, r in entail["templates"].items():
        print(f"{name:22s} {r['auc']:6.3f}  {r['pass']}")
    print(f"\nverdict: {verdict}  (G pass {g_pass}/{len(TEMPLATES_G)}, E pass {e_pass}/{len(TEMPLATES_E)})")


if __name__ == "__main__":
    main()
