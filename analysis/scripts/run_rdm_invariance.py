#!/usr/bin/env python
"""Experiment 004: is the *structure of relations* (RDM) stable across readouts?

Protocol / frozen parameters: ``../experiment-004-2026-11-24-rdm-invariance.md``
(§11).  For a fixed object set we build the representational dissimilarity
matrix (RDM) under several *different* readout families, then test whether RSA
between readouts is high even when the recovered subspace orientation (principal
angles) is not.

Readout families (must differ; agreement inside one family is not an invariant):
    open        -- text + "… is most likely a"      -> next-token JS
    open_alt    -- same frame swapped                -> next-token JS (within-family)
    completion  -- text + "It builds" / "It has"     -> next-token JS
    concepts    -- forced-choice over a concept list -> JS over the list
    judgment    -- P(Yes) profile over fixed questions (vector readout)

Run:
    uv run python scripts/run_rdm_invariance.py --smoke
    uv run python scripts/run_rdm_invariance.py --model Qwen/Qwen2.5-1.5B-Instruct
    uv run python scripts/run_rdm_invariance.py --split train --no-orientation
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
from tc import rdm as R
from tc.controls import separation_auc
from tc.corpus import HARD_GROUPS
from tc.extract import next_token_probs, token_count
from tc.factorial import ATTRS, build_corpus
from tc.interchange import extract as extract_hidden

# Frozen §11.2 judgment questions for the factorial set (two per attribute value).
J_QUESTIONS = [
    "Does it feed on nectar and pollen?",
    "Does it hunt other insects?",
    "Does it live in a large colony?",
    "Does it live mostly alone?",
    "Does it have a hairy body?",
    "Does it have a smooth body?",
    "Does it build cells of wax?",
    "Does it build nests of paper?",
    "Is it found near a lake?",
    "Is it found near a mountain?",
]


# --------------------------------------------------------------------------- #
# Objects (§11.1)
# --------------------------------------------------------------------------- #

def factorial_objects(templates):
    objs = []
    for c in build_corpus(ATTRS):
        if c["template"] not in templates:
            continue
        objs.append({
            "text": c["text"],
            "group": "".join(str(c["values"][a]) for a in ATTRS),
            "shard": 0 if c["syn"] == 0 else 1,
            "template": c["template"], "syn": c["syn"],
        })
    return objs


def natural_objects():
    objs = []
    for name, items in HARD_GROUPS.items():
        for i, t in enumerate(items):
            objs.append({"text": t, "group": name,
                         "shard": 0 if i == 0 else 1, "concept": name})
    return objs


# --------------------------------------------------------------------------- #
# Model utilities
# --------------------------------------------------------------------------- #

def class_token_ids(tok, target: str):
    t = target.strip().lower()
    return [i for i in range(len(tok)) if tok.decode([i]).strip().lower() == t]


def continuation_logprob(model, tok, prompt: str, word: str) -> float:
    ep = tok(prompt, return_tensors="pt")
    ec = tok(" " + word, add_special_tokens=False, return_tensors="pt")
    if ec.input_ids.shape[1] == 0:
        ec = tok(word, add_special_tokens=False, return_tensors="pt")
    ids = torch.cat([ep.input_ids, ec.input_ids], dim=1)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    lps = torch.log_softmax(out.logits[0], dim=-1).double().numpy()
    start = ep.input_ids.shape[1]
    lp = sum(float(lps[start - 1 + i, int(tid)]) for i, tid in enumerate(ec.input_ids[0]))
    return lp / ec.input_ids.shape[1]


# --------------------------------------------------------------------------- #
# Readout specs (§11.2)
# --------------------------------------------------------------------------- #

def _dist_spec(name, rtype, tail, model, tok, k, role="main"):
    def make_fn(tail):
        return lambda t: next_token_probs(model, tok, f"{t}\n\n{tail}")
    return {"name": name, "type": rtype, "role": role, "kind": "dist",
            "prompt": (lambda t, tail=tail: f"{t}\n\n{tail}"),
            "fn": make_fn(tail), "k": k}


def make_readouts(set_name, model, tok, yes_ids, no_ids, k):
    specs = []
    if set_name == "factorial":
        specs.append(_dist_spec("open", "continuation", "This insect is most likely a", model, tok, k))
        specs.append(_dist_spec("open_alt", "continuation",
                                "In short, this insect is most likely a", model, tok, k, role="robustness"))
        specs.append(_dist_spec("completion", "continuation", "It builds", model, tok, k))

        jp = "{t}\n\nQuestion: {q}\nAnswer Yes or No.\nAnswer:"

        def judg(t):
            vec = []
            for q in J_QUESTIONS:
                p = next_token_probs(model, tok, jp.format(t=t, q=q))
                py, pn = float(p[yes_ids].sum()), float(p[no_ids].sum())
                vec.append(py / (py + pn + 1e-12))
            return np.asarray(vec)
        specs.append({"name": "judgment", "type": "judgment", "role": "main", "kind": "vec",
                      "prompt": None, "fn": judg, "k": None})
    else:
        specs.append(_dist_spec("open", "continuation", "This is most likely a", model, tok, k))
        specs.append(_dist_spec("open_alt", "continuation",
                                "In short, this is most likely a", model, tok, k, role="robustness"))
        specs.append(_dist_spec("completion", "continuation", "It has", model, tok, k))

        concepts = list(HARD_GROUPS.keys())
        frame = "This is most likely a"

        def concepts_fn(t):
            prompt = f"{t}\n\n{frame}"
            s = np.array([continuation_logprob(model, tok, prompt, c) for c in concepts])
            s -= s.max()
            e = np.exp(s)
            return e / e.sum()
        specs.append({"name": "concepts", "type": "continuation", "role": "main", "kind": "dist",
                      "prompt": (lambda t: f"{t}\n\n{frame}"), "fn": concepts_fn,
                      "k": len(concepts)})

        jp = "{t}\n\nDoes this describe a {c}? Answer Yes or No.\nAnswer:"

        def judg_nat(t):
            vec = []
            for c in concepts:
                p = next_token_probs(model, tok, jp.format(t=t, c=c))
                py, pn = float(p[yes_ids].sum()), float(p[no_ids].sum())
                vec.append(py / (py + pn + 1e-12))
            return np.asarray(vec)
        specs.append({"name": "judgment", "type": "judgment", "role": "main", "kind": "vec",
                      "prompt": None, "fn": judg_nat, "k": None})
    return specs


# --------------------------------------------------------------------------- #
# One object set
# --------------------------------------------------------------------------- #

def run_set(set_name, objs, specs, model, tok, args):
    texts = [o["text"] for o in objs]
    groups = [o["group"] for o in objs]
    shards = [o["shard"] for o in objs]
    lengths = [token_count(t, tok) for t in texts]
    lexical = R.rdm_lexical(texts)
    length = R.rdm_length(lengths)

    rdms, types, roles, sep_auc = {}, {}, {}, {}
    for spec in specs:
        reps = [spec["fn"](t) for t in texts]
        if spec["kind"] == "dist":
            kk = spec["k"] or args.k
            rdms[spec["name"]] = R.rdm_js(reps, kk)
        else:
            mat = R.zscore_columns(np.vstack(reps))
            rdms[spec["name"]] = R.rdm_vectors(mat, "euclidean")
        types[spec["name"]] = spec["type"]
        roles[spec["name"]] = spec["role"]
        sep_auc[spec["name"]] = float(separation_auc(rdms[spec["name"]], groups))
        print(f"  RDM {spec['name']:11s} type={spec['type']:12s} role={spec['role']:10s} "
              f"separation_auc={sep_auc[spec['name']]:.3f}")

    names = list(rdms)
    cross, within, robust = [], [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            rec = {"a": a, "b": b,
                   "type_a": types[a], "type_b": types[b],
                   "rsa": R.rsa(rdms[a], rdms[b]),
                   "rsa_partial": R.rsa_partial(rdms[a], rdms[b], [lexical, length])}
            if "robustness" in (roles[a], roles[b]):
                robust.append(rec)
                continue
            base = max(R.baseline_rsa(rdms[a], lexical, length),
                       R.baseline_rsa(rdms[b], lexical, length))
            _, p, _ = R.rsa_permutation(rdms[a], rdms[b], n_perm=args.n_perm, seed=0)
            rec.update({"baseline": base, "margin": rec["rsa"] - base, "p": p})
            if types[a] == types[b]:
                within.append(rec)
            else:
                cross.append(rec)

    mean_cross = float(np.mean([r["rsa"] for r in cross])) if cross else float("nan")
    mean_margin = float(np.mean([r["margin"] for r in cross])) if cross else float("nan")
    mean_within = float(np.mean([r["rsa"] for r in within])) if within else float("nan")
    n_above = int(sum(r["margin"] > 0.15 for r in cross))

    orientation = {}
    if args.orientation:
        for spec in specs:
            if spec["kind"] != "dist" or spec["prompt"] is None:
                continue
            for layer in args.layers_list:
                H = []
                for t in texts:
                    hs, _ = extract_hidden(model, tok, spec["prompt"](t))
                    H.append(hs[layer + 1] if layer + 1 < len(hs) else hs[-1])
                cos, mean = R.orientation_stability(
                    np.vstack(H), groups, shards, k=args.orient_k)
                orientation[f"{spec['name']}@L{layer}"] = {
                    "mean_cos": mean, "cosines": [float(c) for c in cos]}
    oc = [v["mean_cos"] for v in orientation.values() if np.isfinite(v["mean_cos"])]
    mean_orientation = float(np.mean(oc)) if oc else float("nan")

    call = R.verdict(mean_cross, mean_margin, mean_orientation,
                     mean_within, n_above, len(cross))

    print(f"\n  [{set_name}] cross-TYPE RSA (primary) = {mean_cross:.3f} "
          f"(margin over baseline {mean_margin:+.3f}); within-type = {mean_within:.3f}; "
          f"orientation cos = {mean_orientation:.3f}")
    for r in cross:
        print(f"    {r['a']:9s} ~ {r['b']:11s} rsa={r['rsa']:+.3f} "
              f"(partial {r['rsa_partial']:+.3f}) base={r['baseline']:+.3f} "
              f"margin={r['margin']:+.3f} p={r['p']:.4f}")
    if within:
        print("  within-type (H4c reference):")
        for r in within:
            print(f"    {r['a']:9s} ~ {r['b']:11s} rsa={r['rsa']:+.3f} "
                  f"(partial {r['rsa_partial']:+.3f})")
    if robust:
        print("  excluded pairs involving a robustness readout (not in aggregates):")
        for r in robust:
            print(f"    {r['a']:9s} ~ {r['b']:11s} rsa={r['rsa']:+.3f} "
                  f"(partial {r['rsa_partial']:+.3f})")
    print(f"  => {call}\n")

    return {
        "objects": set_name, "n_objects": len(objs),
        "split": args.split, "model": args.model,
        "separation_auc": sep_auc,
        "rdm": {n: rdms[n].round(6).tolist() for n in names},
        "cross_type": cross, "within_type": within, "frame_robustness": robust,
        "mean_cross_rsa": mean_cross, "mean_margin": mean_margin,
        "mean_within_rsa": mean_within, "n_above_threshold": n_above,
        "orientation": orientation, "mean_orientation_cos": mean_orientation,
        "verdict": call,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--object-set", default="both", choices=["factorial", "natural", "both"])
    ap.add_argument("--split", default="eval", choices=["eval", "train"])
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--layers", default="4,8")
    ap.add_argument("--orient-k", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="first N objects (smoke)")
    ap.add_argument("--orientation", dest="orientation", action="store_true", default=True)
    ap.add_argument("--no-orientation", dest="orientation", action="store_false")
    ap.add_argument("--smoke", action="store_true", help="tiny run: limit=6, no orientation")
    ap.add_argument("--out", default="results/rdm_invariance_summary.json")
    args = ap.parse_args()

    if args.smoke:
        args.limit = args.limit or 8
        args.orientation = False
        args.n_perm = min(args.n_perm, 200)

    args.layers_list = [int(x) for x in args.layers.split(",")]

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    yes_ids, no_ids = class_token_ids(tok, "yes"), class_token_ids(tok, "no")
    if not yes_ids or not no_ids:
        raise SystemExit(f"could not locate Yes/No tokens (yes={yes_ids}, no={no_ids})")
    print(f"[model] {args.model}  split={args.split}  layers={args.layers_list}  "
          f"k={args.k}  n_perm={args.n_perm}\n")

    reports = []
    if args.object_set in ("factorial", "both"):
        templates = {2} if args.split == "eval" else {0, 1}
        objs = factorial_objects(templates)
        if args.limit:
            objs = objs[: args.limit]
        print(f"=== factorial ({args.split}): {len(objs)} objects ===")
        specs = make_readouts("factorial", model, tok, yes_ids, no_ids, args.k)
        reports.append(run_set("factorial", objs, specs, model, tok, args))

    if args.object_set in ("natural", "both"):
        objs = natural_objects()
        if args.limit:
            objs = objs[: args.limit]
        print(f"=== natural ({args.split}): {len(objs)} objects ===")
        specs = make_readouts("natural", model, tok, yes_ids, no_ids, args.k)
        reports.append(run_set("natural", objs, specs, model, tok, args))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "split": args.split, "k": args.k,
        "layers": args.layers_list, "n_perm": args.n_perm,
        "sets": reports,
    }, indent=2, ensure_ascii=False))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
