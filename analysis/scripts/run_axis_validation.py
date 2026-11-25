#!/usr/bin/env python
"""Experiment 008 / program step 4.0a: which graded axes does the model confirm?

Protocol: ``../experiment-008-2026-11-25-axis-validation.md``.

Single-token rungs are a HARD filter here (user requirement).  The scale is
anchored explicitly (a 1..5 rating with named extremes) instead of relying on
bare next-token after ".".

Channels:
  * Likert: anchored 1..5 rating -> expected value must rise with the rung;
  * bidirectional forced-choice with letter options (forward "hotter" + reverse
    "colder"); correct only if both directions agree;
  * continuation next-token JS profile (secondary).

Run:
    uv run python scripts/run_axis_validation.py --smoke
    uv run python scripts/run_axis_validation.py --axes size,quality,loudness,speed
    uv run python scripts/run_axis_validation.py --model distilgpt2 --skip-fc --skip-likert
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tc.axes import (
    axis_verdict_anchored,
    adjacent,
    ladder,
    likert_prompt,
    likert_prompt_rev,
    names,
    order_prompt,
    parse_order,
    spearman_index,
)
from tc.extract import next_token_probs
from tc.interpolate import behavioral_profile


def model_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def seq_logprob(model, tok, prompt: str, continuation: str) -> float:
    ep = tok(prompt, return_tensors="pt")
    ec = tok(continuation, add_special_tokens=False, return_tensors="pt")
    if ec.input_ids.shape[1] == 0:
        return float("nan")
    ids = torch.cat([ep.input_ids, ec.input_ids], dim=1)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    lps = torch.log_softmax(out.logits[0], dim=-1).double()
    start = ep.input_ids.shape[1]
    lp = sum(float(lps[start - 1 + i, tid]) for i, tid in enumerate(ec.input_ids[0]))
    return lp / ec.input_ids.shape[1]


def sentence_logprob(model, tok, text: str) -> float:
    """Length-normalised log-prob of a whole sentence (semantic-order probe)."""
    enc = tok(text, return_tensors="pt")
    ids = enc.input_ids[0]
    if ids.shape[0] < 2:
        return float("nan")
    with torch.no_grad():
        out = model(**enc)
    lps = torch.log_softmax(out.logits[0], dim=-1).double()
    lp = sum(float(lps[i - 1, ids[i]]) for i in range(1, ids.shape[0]))
    return lp / (ids.shape[0] - 1)


def digit_ids(tok):
    """Map rating value 1..5 -> list of candidate next-token ids."""
    out = {}
    for d in range(1, 6):
        ids = []
        for s in (f" {d}", f"{d}"):
            e = tok(s, add_special_tokens=False).input_ids
            if len(e) == 1:
                ids.append(int(e[0]))
        out[d] = sorted(set(ids))
    return out


def rating(probs, dids) -> float:
    m = np.array([float(np.sum([probs[i] for i in dids[d]])) for d in range(1, 6)])
    s = m.sum()
    if s <= 0:
        return float("nan")
    return float((m / s) @ np.arange(1, 6))


def run_axis(model, tok, name, args, dids):
    lad = ladder(name)
    words, dim = lad["words"], lad["dim"]
    counts = [int(len(tok(w, add_special_tokens=False).input_ids)) for w in words]
    kept = [w for w, c in zip(words, counts) if c == 1]
    out = {"axis": name, "words": words, "kept_words": kept,
           "token_counts": counts, "single_token": all(c == 1 for c in counts),
           "dim": dim, "opposite": lad["opposite"] if "opposite" in lad else None}

    if len(kept) < args.min_words:
        out["verdict"] = "insufficient single-token rungs"
        print(f"  {name:11s} tokens={counts}  -> insufficient single-token rungs (<{args.min_words})")
        return out

    cont = None
    if not args.skip_cont:
        probs = [next_token_probs(model, tok, lad["frame"].format(w=w)) for w in kept]
        cont = behavioral_profile(probs, k=args.k)
        out["continuation"] = cont

    likert_ev, likert_rho, likert_rho_rev = [], float("nan"), float("nan")
    if not args.skip_likert:
        for w in kept:
            p = next_token_probs(model, tok, likert_prompt(name, w))
            likert_ev.append(rating(p, dids))
        likert_rho = spearman_index(likert_ev)
        out["likert_expected"] = likert_ev
        rev_ev = [rating(next_token_probs(model, tok, likert_prompt_rev(name, w)), dids)
                  for w in kept]
        out["likert_expected_rev"] = rev_ev
        likert_rho_rev = spearman_index(rev_ev)
    else:
        likert_rho_rev = float("nan")

    fc_bidir = float("nan")
    if not args.skip_fc:
        trials, correct = [], 0
        for low, high in adjacent(kept):
            good = f"{high} is {dim} than {low}."
            bad = f"{low} is {dim} than {high}."
            llg, llb = sentence_logprob(model, tok, good), sentence_logprob(model, tok, bad)
            ok = bool(llg > llb)
            correct += int(ok)
            trials.append({"pair": [low, high], "good": good, "ll": [llg, llb], "ok": ok})
        fc_bidir = correct / len(trials) if trials else float("nan")
        out["plausibility_trials"] = trials

    order_tau = {}
    if not args.skip_order:
        for rev in (False, True):
            prompt = order_prompt(name, kept, reverse=rev)
            enc = tok(prompt, return_tensors="pt")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=16, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            text = tok.decode(gen[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
            got = parse_order(text, kept)
            expected = list(reversed(kept)) if rev else list(kept)
            tau = float("nan")
            if len(got) >= 3:
                common = [w for w in expected if w in got]
                if len(common) >= 3:
                    tau = float(stats.kendalltau(
                        [got.index(w) for w in common],
                        [expected.index(w) for w in common]).correlation)
            order_tau["rev" if rev else "fwd"] = tau
            out.setdefault("order_raw", {})["rev" if rev else "fwd"] = text.strip()
        out["order_tau"] = order_tau
    order_ok = bool(not np.isnan(order_tau.get("fwd", np.nan)) and
                    not np.isnan(order_tau.get("rev", np.nan)) and
                    order_tau["fwd"] >= 0.8 and order_tau["rev"] >= 0.8)

    anchor_ok = bool(likert_rho >= 0.8 and
                     (np.isnan(likert_rho_rev) or likert_rho_rev <= -0.8))
    channels = []
    if anchor_ok:
        channels.append("likert(anchor)")
    if not np.isnan(fc_bidir) and fc_bidir >= 0.8:
        channels.append("plausibility")
    if order_ok:
        channels.insert(0, "ordering(fwd+rev)")
    if len(channels) >= 2:
        verdict = "valid"
    elif order_ok:
        verdict = "valid (ordering fwd+rev)"
    elif len(channels) == 1:
        verdict = "readout-local (Likert-confirmed)"
    else:
        verdict = "weak/unsupported"
    out.update({"fc_bidir": fc_bidir, "likert_rho": likert_rho,
                "likert_rho_rev": likert_rho_rev, "anchor_ok": anchor_ok,
                "verdict": verdict, "channels": channels, "order_ok": order_ok})

    print(f"  {name:11s} tokens={counts}  likert={likert_rho:+.2f}/rev={likert_rho_rev:+.2f} "
          f"ev={[round(x,2) for x in likert_ev]}  fc2={fc_bidir:.2f}  "
          + (f"js_end={cont['js_endpoints']:.3f}  " if cont else "")
          + f"order={order_tau}  => {verdict} {channels}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--axes", default=",".join(names()))
    ap.add_argument("--skip-cont", action="store_true")
    ap.add_argument("--skip-fc", action="store_true")
    ap.add_argument("--skip-likert", action="store_true")
    ap.add_argument("--skip-order", action="store_true")
    ap.add_argument("--min-words", type=int, default=4)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    axes = [s for s in args.axes.split(",") if s]
    if args.smoke:
        axes = axes[:2]

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    dids = digit_ids(tok)

    print(f"[model] {args.model}  dtype={dtype}  axes={axes}  k={args.k}")
    print(f"[single-token filter] min words = {args.min_words}\n")
    results = [run_axis(model, tok, name, args, dids) for name in axes]

    out = Path(args.out) if args.out else Path(f"results/axis_validation_{model_slug(args.model)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "dtype": dtype, "k": args.k, "min_words": args.min_words,
        "channels": {"continuation": not args.skip_cont,
                     "forced_choice_bidir": not args.skip_fc,
                     "likert": not args.skip_likert},
        "axes": results,
    }, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
