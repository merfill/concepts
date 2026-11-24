#!/usr/bin/env python
"""Experiment 005: is the *transition operator* (not the state) invariant across readouts?

Protocol: ``../experiment-005-2026-11-24-operator-invariance.md``.

For each object and readout *context* we estimate the block Jacobian action
``Phi_l(x) = O_l(x) P`` at the readout position (P = fixed probe basis), build the
operator-RDM, and test whether it transfers across readout types — while the
paired state/readout RDM (the 004 negative control) does not.

Contexts: open / completion (continuation type) and judgment (judgment type;
a single representative question defines the operator input context).
The "transient" variant re-estimates the same operator along the first K
generated tokens (experiment 001's transient window).

Run:
    uv run python scripts/run_operator_invariance.py --smoke
    uv run python scripts/run_operator_invariance.py --model Qwen/Qwen2.5-1.5B-Instruct --layers 12,20
    uv run python scripts/run_operator_invariance.py --transient 2 --limit 12
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
from tc.corpus import HARD_GROUPS
from tc.extract import next_token_probs
from tc.factorial import ATTRS, build_corpus
from tc.operator import (
    estimate_phi,
    operator_effective_dimension,
    operator_rdm,
    random_probe_basis,
)


# --------------------------------------------------------------------------- #
# Objects (same as 004)
# --------------------------------------------------------------------------- #

def factorial_objects(templates):
    objs = []
    for c in build_corpus(ATTRS):
        if c["template"] not in templates:
            continue
        objs.append({
            "text": c["text"],
            "group": "".join(str(c["values"][a]) for a in ATTRS),
            "template": c["template"], "syn": c["syn"],
        })
    return objs


def natural_objects():
    objs = []
    for name, items in HARD_GROUPS.items():
        for t in items:
            objs.append({"text": t, "group": name, "concept": name})
    return objs


# --------------------------------------------------------------------------- #
# Operator input contexts (readout framing)
# --------------------------------------------------------------------------- #

def make_contexts(set_name):
    if set_name == "factorial":
        return [
            {"name": "open", "type": "continuation",
             "prompt": lambda t: f"{t}\n\nThis insect is most likely a"},
            {"name": "completion", "type": "continuation",
             "prompt": lambda t: f"{t}\n\nIt builds"},
            {"name": "judge", "type": "judgment",
             "prompt": lambda t: (f"{t}\n\nQuestion: Does it feed on nectar and pollen?\n"
                                  f"Answer Yes or No.\nAnswer:")},
        ]
    return [
        {"name": "open", "type": "continuation", "prompt": lambda t: f"{t}\n\nThis is most likely a"},
        {"name": "completion", "type": "continuation", "prompt": lambda t: f"{t}\n\nIt has"},
        {"name": "judge", "type": "judgment",
         "prompt": lambda t: (f"{t}\n\nDoes this describe a bee? Answer Yes or No.\nAnswer:")},
    ]


OPEN_FRAME = {"factorial": "This insect is most likely a", "natural": "This is most likely a"}


def hidden_size(model):
    cfg = model.config
    return int(getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd"))


def gen_prefix_texts(model, tok, text, K):
    """Return object texts after 0..K greedily generated tokens."""
    enc = tok(text, return_tensors="pt")
    out = model.generate(**enc, max_new_tokens=K, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    new = out[0, enc.input_ids.shape[1]:]
    texts = [text]
    for k in range(1, K + 1):
        cont = tok.decode(new[:k], skip_special_tokens=True).strip()
        texts.append(f"{text} {cont}".strip())
    return texts


# --------------------------------------------------------------------------- #
# Pipeline for one object set
# --------------------------------------------------------------------------- #

def run_set(set_name, objs, model, tok, args, P, P2):
    texts = [o["text"] for o in objs]
    groups = [o["group"] for o in objs]
    contexts = make_contexts(set_name)
    lexical, length = R.rdm_lexical(texts), R.rdm_length(
        [len(tok(t).input_ids) for t in texts])

    state_rdm, op_rdm, phis = {}, {}, {}
    for ctx in contexts:
        states = [next_token_probs(model, tok, ctx["prompt"](t)) for t in texts]
        state_rdm[ctx["name"]] = R.rdm_js(states, args.k)
        op_rdm[ctx["name"]] = {}
        phis[ctx["name"]] = {}
        for layer in args.layers_list:
            phis[ctx["name"]][layer] = [
                estimate_phi(model, tok, ctx["prompt"](t), layer, P, args.eps) for t in texts]
            op_rdm[ctx["name"]][layer] = operator_rdm(phis[ctx["name"]][layer])
        print(f"  [{set_name}] context {ctx['name']:11s} ({ctx['type']}) "
              f"state {len(texts)} objects; operator dims computed")

    pairs = []
    for layer in args.layers_list:
        for i in range(len(contexts)):
            for j in range(i + 1, len(contexts)):
                a, b = contexts[i], contexts[j]
                op_a, op_b = op_rdm[a["name"]][layer], op_rdm[b["name"]][layer]
                st_a, st_b = state_rdm[a["name"]], state_rdm[b["name"]]
                base = max(R.baseline_rsa(op_a, lexical, length),
                           R.baseline_rsa(op_b, lexical, length))
                _, p, _ = R.rsa_permutation(op_a, op_b, n_perm=args.n_perm, seed=0)
                rec = {
                    "layer": layer, "a": a["name"], "b": b["name"],
                    "type_a": a["type"], "type_b": b["type"],
                    "op_rsa": R.rsa(op_a, op_b),
                    "op_rsa_partial": R.rsa_partial(op_a, op_b, [lexical, length]),
                    "state_rsa": R.rsa(st_a, st_b),
                    "baseline": base, "p": p,
                }
                rec["op_margin"] = rec["op_rsa"] - base
                rec["delta_op_minus_state"] = rec["op_rsa"] - rec["state_rsa"]
                pairs.append(rec)

    cross = [r for r in pairs if r["type_a"] != r["type_b"]]
    within = [r for r in pairs if r["type_a"] == r["type_b"]]
    mean_op = float(np.mean([r["op_rsa"] for r in cross])) if cross else float("nan")
    mean_state = float(np.mean([r["state_rsa"] for r in cross])) if cross else float("nan")
    mean_margin = float(np.mean([r["op_margin"] for r in cross])) if cross else float("nan")
    mean_delta = float(np.mean([r["delta_op_minus_state"] for r in cross])) if cross else float("nan")
    within_op = float(np.mean([r["op_rsa"] for r in within])) if within else float("nan")
    n_above = int(sum(r["op_margin"] > 0.15 for r in cross))

    call = verdict_005(mean_op, mean_margin, mean_delta, within_op, n_above, len(cross))

    # probe-basis stability (open context, first layer)
    layer0 = args.layers_list[0]
    phis2 = [estimate_phi(model, tok, contexts[0]["prompt"](t), layer0, P2, args.eps) for t in texts]
    rdm2 = operator_rdm(phis2)
    probe_stability = R.rsa(op_rdm[contexts[0]["name"]][layer0], rdm2)
    eff_dim = {c["name"]: {l: operator_effective_dimension(phis[c["name"]][l])
                           for l in args.layers_list} for c in contexts}

    transient = {}
    if args.transient > 0:
        frame = OPEN_FRAME[set_name]
        layer = layer0
        ptexts0 = [f"{t}\n\n{frame}" for t in texts]
        rdm0 = operator_rdm([estimate_phi(model, tok, p, layer, P, args.eps) for p in ptexts0])
        for k in range(1, args.transient + 1):
            rdmk = []
            for t in texts:
                pt = gen_prefix_texts(model, tok, t, k)[k]
                rdmk.append(estimate_phi(model, tok, f"{pt}\n\n{frame}", layer, P, args.eps))
            transient[k] = R.rsa(rdm0, operator_rdm(rdmk))
        print(f"  transient stability (open@L{layer}): "
              + " ".join(f"k={k}:{v:+.3f}" for k, v in transient.items()))

    print(f"\n  [{set_name}] cross-type operator RSA = {mean_op:+.3f} "
          f"(margin {mean_margin:+.3f}); state RSA = {mean_state:+.3f}; "
          f"Δ(op−state) = {mean_delta:+.3f}; within op = {within_op:+.3f}; "
          f"probe-P stability = {probe_stability:+.3f}")
    for r in cross:
        print(f"    L{r['layer']} {r['a']:9s}~{r['b']:9s} op={r['op_rsa']:+.3f} "
              f"(partial {r['op_rsa_partial']:+.3f}) state={r['state_rsa']:+.3f} "
              f"base={r['baseline']:+.3f} margin={r['op_margin']:+.3f} p={r['p']:.4f}")
    print(f"  => {call}\n")

    return {
        "objects": set_name, "n_objects": len(texts), "model": args.model,
        "layers": args.layers_list, "m_probes": args.m, "eps": args.eps,
        "cross_type": cross, "within_type": within,
        "mean_op_rsa": mean_op, "mean_state_rsa": mean_state,
        "mean_margin": mean_margin, "mean_delta": mean_delta,
        "within_op_rsa": within_op, "n_above_threshold": n_above,
        "probe_basis_stability": probe_stability,
        "effective_dimension": eff_dim, "transient": transient,
        "verdict": call,
    }


def verdict_005(mean_op, margin, delta, within, n_above, n_cross):
    if (mean_op >= 0.5 and margin > 0.15 and delta >= 0.2
            and n_above == n_cross):
        return "H5a: transition-operator invariant found"
    if within >= 0.6 and mean_op < within - 0.3:
        return "H5c: operator stable only within readout type"
    if mean_op < 0.3 or margin <= 0.15:
        return "H5b: no operator invariant"
    return "partial/ambiguous operator invariant"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--object-set", default="both", choices=["factorial", "natural", "both"])
    ap.add_argument("--split", default="eval", choices=["eval", "train"])
    ap.add_argument("--layers", default="3,4")
    ap.add_argument("--m", type=int, default=16, help="probe directions (>=16 for basis stability)")
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--transient", type=int, default=0, help="K generated tokens (0=off)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results/operator_invariance_summary.json")
    args = ap.parse_args()

    if args.smoke:
        args.limit = args.limit or 10
        args.m = min(args.m, 4)
        args.n_perm = min(args.n_perm, 150)

    args.layers_list = [int(x) for x in args.layers.split(",")]
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    d = hidden_size(model)
    P = random_probe_basis(d, args.m, seed=0)
    P2 = random_probe_basis(d, args.m, seed=1)
    print(f"[model] {args.model} d={d} layers={args.layers_list} m={args.m} "
          f"eps={args.eps} transient={args.transient}\n")

    reports = []
    if args.object_set in ("factorial", "both"):
        objs = factorial_objects({2} if args.split == "eval" else {0, 1})
        if args.limit:
            objs = objs[: args.limit]
        print(f"=== factorial ({args.split}): {len(objs)} objects ===")
        reports.append(run_set("factorial", objs, model, tok, args, P, P2))
    if args.object_set in ("natural", "both"):
        objs = natural_objects()
        if args.limit:
            objs = objs[: args.limit]
        print(f"=== natural ({args.split}): {len(objs)} objects ===")
        reports.append(run_set("natural", objs, model, tok, args, P, P2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "split": args.split, "layers": args.layers_list,
        "m": args.m, "eps": args.eps, "n_perm": args.n_perm,
        "transient": args.transient, "sets": reports,
    }, indent=2, ensure_ascii=False))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
