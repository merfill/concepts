#!/usr/bin/env python
"""Experiment 013 / program stage 4 (reactivated): continuity of the concept
space read as a *response function* along an on-manifold graded axis.

Protocol: ``../experiment-013-2026-11-27-continuity.md``.

Primary readout: ``continuation`` next-token distributions per rung; the concept
coordinate is the projection onto the ``q_last - q_first`` axis (output side).
Secondary readout: generative ordering (worst -> best).  Controls: nonce,
empty baseline, shuffle, frequency residual, cross-frame robustness, a second
synonymous ladder (parameterisation independence), and an optional external
valence lexicon.

Run:
    uv run python scripts/run_continuity.py --smoke
    uv run python scripts/run_continuity.py --model Qwen/Qwen2.5-1.5B-Instruct
    uv run python scripts/run_continuity.py --filler "I watched it last night."
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import wordfreq
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.axes import parse_order
from tc.continuity import (
    CARRIER_NEG,
    CARRIER_POS,
    DEFAULT_K,
    DEFAULT_NONCE,
    FRAME_TAU,
    FRAMES,
    NEUTRAL_FRAME,
    axis_projection,
    build_prompt,
    cliff_ratio,
    continuity_verdict,
    expected_valence,
    frame_robustness,
    frequency_residual,
    js_steps,
    load_ladder,
    monotonicity,
    path_independence,
    single_step_share,
)
from tc.extract import next_token_probs
from tc.metrics import js_topk

DEFAULT_LADDER = Path(__file__).resolve().parents[1] / "data" / "valence_ladder.tsv"
BASELINE = "The movie. Overall, it was ___"


def model_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def is_single_token(tok, word: str) -> bool:
    return len(tok(word, add_special_tokens=False).input_ids) == 1


def probs_for(model, tok, template, words, filler=""):
    return [next_token_probs(model, tok, build_prompt(template, w, filler)) for w in words]


def frame_metrics(model, tok, template, words, filler, k):
    probs = probs_for(model, tok, template, words, filler)
    proj = axis_projection(probs, k)
    steps = js_steps(probs, k)
    phi = proj["phi"]
    return probs, {
        "phi": phi,
        "steps": steps,
        "js_endpoints": proj["js_endpoints"],
        "rho_rank": monotonicity(phi),
        "cliff": cliff_ratio(steps),
        "single_share": single_step_share(steps),
    }


def order_readout(model, tok, words):
    shuffled = [words[i] for i in np.random.RandomState(0).permutation(len(words))]
    prompt = (f"Order these words from worst to best: {', '.join(shuffled)}.\nAnswer:")
    enc = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=32, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(gen[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    got = parse_order(text, words)
    tau = float("nan")
    if len(got) >= 3:
        common = [w for w in words if w in got]
        if len(common) >= 3:
            exp = {w: i for i, w in enumerate(words)}
            tau = float(stats.kendalltau([got.index(w) for w in common],
                                         [exp[w] for w in common]).correlation)
    return {"tau": tau, "raw": text.strip()}


def nonce_control(model, tok, template, nonce, filler, k):
    words = list(nonce)
    proj = axis_projection(probs_for(model, tok, template, words, filler), k)
    return {"words": words, "phi": proj["phi"], "rho": monotonicity(proj["phi"])}


def shuffle_control(model, tok, template, words, filler, k):
    perm = np.random.RandomState(1).permutation(len(words))
    perm_words = [words[i] for i in perm]
    proj = axis_projection(probs_for(model, tok, template, perm_words, filler), k)
    true_ranks = [int(perm[j]) for j in range(len(perm))]
    return {"rho": float(stats.spearmanr(proj["phi"], true_ranks).correlation),
            "perm": [int(x) for x in perm], "phi": proj["phi"]}


def load_rows(path):
    """Read both ladders keyed by rung so primary/alt stay aligned."""
    primary = load_ladder(path, column="word")
    alt = {r["rung"]: r["word"] for r in load_ladder(path, column="word_alt")}
    return [{"rung": r["rung"], "word": r["word"], "alt": alt.get(r["rung"], "")}
            for r in primary]


def load_valence_ids(tok, path):
    id_valence = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            word = (row.get("word") or "").strip()
            if not word:
                continue
            val = float(row["valence"])
            for form in (f" {word}", word):
                ids = tok(form, add_special_tokens=False).input_ids
                if len(ids) == 1:
                    id_valence[int(ids[0])] = val
    return id_valence


def run(model, tok, args):
    frame_names = [s.strip() for s in args.frames.split(",") if s.strip()]
    frames = {name: FRAMES[name] for name in frame_names if name in FRAMES}
    if args.neutral_probe:
        frames["neutral"] = NEUTRAL_FRAME
    if args.counterbalance:
        frames["pos_carrier"] = CARRIER_POS
        frames["neg_carrier"] = CARRIER_NEG

    ladder = load_rows(args.ladder)
    if args.single_token_only:
        kept_idx = [i for i, r in enumerate(ladder) if is_single_token(tok, r["word"])]
    else:
        kept_idx = list(range(len(ladder)))
    words = [ladder[i]["word"] for i in kept_idx]
    alt_words = [ladder[i]["alt"] for i in kept_idx]
    counts = [len(tok(w, add_special_tokens=False).input_ids) for w in words]
    result = {"model": args.model, "ladder": str(args.ladder),
              "words": words, "token_counts": counts, "frames": list(frames)}

    if len(words) < args.min_words:
        result["verdict"] = {"label": "insufficient single-token rungs",
                             "kept": len(words)}
        return result

    probs_by_frame, per_frame = {}, {}
    for name, template in frames.items():
        probs, m = frame_metrics(model, tok, template, words, args.filler, args.k)
        probs_by_frame[name] = probs
        per_frame[name] = m
    result["per_frame"] = per_frame

    core = [f for f in frame_names if f in per_frame]
    primary = max(core, key=lambda f: per_frame[f]["js_endpoints"])
    result["primary_frame"] = primary
    phi = per_frame[primary]["phi"]
    steps = per_frame[primary]["steps"]

    frame_rob = frame_robustness(
        {f: per_frame[f]["phi"] for f in frames},
        {f: per_frame[f]["steps"] for f in frames},
    )
    frame_rob["ok"] = bool(len(core) >= 2 and not np.isnan(frame_rob["mean_rho"])
                           and frame_rob["mean_rho"] >= FRAME_TAU
                           and frame_rob["cliff_agree"])
    result["frame_robustness"] = frame_rob

    result["nonce"] = nonce_control(model, tok, frames[primary], DEFAULT_NONCE,
                                    args.filler, args.k)

    base = next_token_probs(model, tok, BASELINE)
    result["empty"] = {"js_from_baseline":
                       [float(js_topk(base, p, args.k)) for p in probs_by_frame[primary]]}

    result["shuffle"] = shuffle_control(model, tok, frames[primary], words,
                                        args.filler, args.k)

    zipf = [float(wordfreq.zipf_frequency(w, "en")) for w in words]
    lengths = [len(w) for w in words]
    freq = frequency_residual(phi, zipf, lengths)
    freq.update({"zipf": zipf, "lengths": lengths})
    result["frequency"] = freq

    result["ordering"] = ({} if args.no_ordering
                          else order_readout(model, tok, words))

    if args.alt_ladder:
        result["alt_words"] = alt_words
        a_probs = probs_for(model, tok, frames[primary], alt_words, args.filler)
        alt_phi = axis_projection(a_probs, args.k)["phi"]
        result["alt_ladder"] = {
            "phi": alt_phi, "rho_rank": monotonicity(alt_phi),
            "independence": path_independence(phi, alt_phi),
        }

    if args.valence_file:
        id_valence = load_valence_ids(tok, args.valence_file)
        phi_val = [expected_valence(p, id_valence) for p in probs_by_frame[primary]]
        result["expected_valence"] = {"phi": phi_val,
                                      "rho_rank": monotonicity(phi_val)}

    nonce_rho = result["nonce"].get("rho", float("nan"))
    verdict = continuity_verdict(
        phi, steps, per_frame[primary]["rho_rank"], per_frame[primary]["js_endpoints"],
        frame_rob, nonce_rho, freq["rho_resid"],
    )
    if result.get("ordering") and not np.isnan(result["ordering"].get("tau", np.nan)):
        verdict["ordering_tau"] = result["ordering"]["tau"]
        if verdict["label"].startswith("continuous") and result["ordering"]["tau"] < 0.8:
            verdict["label"] = "readout-local (continuation continuous, ordering weak)"
    result["verdict"] = verdict
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--ladder", default=str(DEFAULT_LADDER))
    ap.add_argument("--frames", default="F1,F2,F3")
    ap.add_argument("--filler", default="")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--min-words", type=int, default=5)
    ap.add_argument("--single-token-only", action="store_true")
    ap.add_argument("--neutral-probe", action="store_true")
    ap.add_argument("--counterbalance", action="store_true")
    ap.add_argument("--no-ordering", action="store_true")
    ap.add_argument("--no-alt-ladder", dest="alt_ladder", action="store_false")
    ap.add_argument("--valence-file", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.smoke:
        args.frames = "F1"
        args.no_ordering = True

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    print(f"[model] {args.model}  dtype={dtype}  frames={args.frames}  k={args.k}")

    result = run(model, tok, args)
    result["dtype"] = dtype
    result["k"] = args.k
    result["filler"] = args.filler

    out = Path(args.out) if args.out else \
        Path(__file__).resolve().parents[1] / "results" / f"continuity_{model_slug(args.model)}_valence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    v = result.get("verdict", {})
    print(f"[words] {result.get('words')}")
    print(f"[verdict] {v.get('label')}  primary={result.get('primary_frame')}")
    for name, m in result.get("per_frame", {}).items():
        print(f"  {name:12s} rho={m['rho_rank']:+.2f} cliff={m['cliff']:.2f} "
              f"js_end={m['js_endpoints']:.3f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
