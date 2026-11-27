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
from tc.controls import spearman
from tc.extract import next_token_probs
from tc.hierarchy import (
    TEMPLATES_E,
    TEMPLATES_G,
    auc,
    build_candidates,
    descendant_metrics,
    entailment_mass,
    fragment_concepts,
    fragment_entailment_pairs,
    labelled_candidates,
    load_fragment,
    load_triads,
    normalize_probs,
    read_candidates,
    render,
    synset_is_abstract,
)

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "hierarchy_triads.tsv"
DEFAULT_FRAGMENT = Path(__file__).resolve().parents[1] / "data" / "wordnet_fragment.tsv"
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


def _word_depths(fragment) -> dict[str, int]:
    """Surface word -> depth (first occurrence wins; senses were fixed upstream)."""
    depths: dict[str, int] = {}
    for node in fragment["nodes"].values():
        depths.setdefault(node["word"], node["depth"])
    return depths


def run_descent_wordnet(model, tokenizer, fragment, concepts) -> dict:
    """Descent on a WordNet fragment, with a per-(template, concept) gate.

    A pair is valid only if the real concept brings more candidate mass than a
    nonce does for the same template; invalid pairs are dropped and counted so a
    bad phrasing cannot masquerade as a result.
    """
    nodes = fragment["nodes"]
    word_depth = _word_depths(fragment)
    abstract_flags = {c: synset_is_abstract(c) for c in concepts}
    out: dict[str, dict] = {}
    records: list[dict] = []
    for name, template in TEMPLATES_G.items():
        aucs, masses, tops, freqs, nonces, shuffles = [], [], [], [], [], []
        c_depths, a_depths, steps = [], [], []
        total_pairs = valid_pairs = 0
        for ci, concept in enumerate(concepts):
            labels = labelled_candidates(fragment, concept, n_desc=6, n_anc=2,
                                         n_wrong=2, n_sib=4, n_unrel=4, seed=ci)
            base_record = {
                "template": name,
                "concept": nodes[concept]["word"],
                "synset": concept,
                "depth": nodes[concept]["depth"],
                "abstract": abstract_flags[concept],
            }
            if "desc" not in labels.values():
                records.append({**base_record, "valid": False, "reason": "no_desc"})
                continue
            words = list(labels)
            prompt = render(template, x=nodes[concept]["word"])
            raw, dist = read_candidates(model, tokenizer, prompt, words)

            nonce = NONCE[ci % len(NONCE)]
            raw_nonce, _ = read_candidates(model, tokenizer,
                                           render(template, x=nonce), words)
            total_pairs += 1
            if sum(raw.values()) <= sum(raw_nonce.values()):
                records.append({**base_record, "valid": False, "reason": "nonce_gate"})
                continue
            valid_pairs += 1

            metrics = descendant_metrics(dist, labels)
            aucs.append(metrics["auc"])
            masses.append(metrics["desc_mass"])
            tops.append(float(metrics["top_is_desc"]))
            freq_value = freq_auc(labels)
            freqs.append(freq_value)
            nonces.append(descendant_metrics(
                normalize_probs(raw_nonce, words), labels)["auc"])
            rng = np.random.default_rng(ci * 10 + 1)
            shuffles.append(descendant_metrics(
                dist, shuffled_labels(labels, rng))["auc"])

            top = metrics["top"]
            step = None
            if top is not None and top in word_depth:
                answer_depth = word_depth[top]
                c_depths.append(nodes[concept]["depth"])
                a_depths.append(answer_depth)
                step = answer_depth - nodes[concept]["depth"]
                steps.append(step)
            records.append({
                **base_record,
                "valid": True,
                "auc": metrics["auc"],
                "top": top,
                "top_is_desc": metrics["top_is_desc"],
                "desc_mass": metrics["desc_mass"],
                "freq_auc": freq_value,
                "step": step,
            })

        mean_auc = float(np.nanmean(aucs)) if aucs else float("nan")
        mean_freq = float(np.nanmean(freqs)) if freqs else float("nan")
        margin = mean_auc - max(mean_freq, 1.0 - mean_freq)
        monotonic = (spearman(c_depths, a_depths)
                     if len(set(c_depths)) > 1 and len(set(a_depths)) > 1
                     else float("nan"))
        step_median = float(np.median(steps)) if steps else float("nan")
        out[name] = {
            "prompt_example": render(
                template, x=nodes[concepts[0]]["word"] if concepts else "X"),
            "valid_pairs": valid_pairs,
            "total_pairs": total_pairs,
            "auc": mean_auc,
            "desc_mass": float(np.nanmean(masses)) if masses else float("nan"),
            "top_is_desc": float(np.nanmean(tops)) if tops else float("nan"),
            "freq_auc": mean_freq,
            "margin": margin,
            "nonce_auc": float(np.nanmean(nonces)) if nonces else float("nan"),
            "shuffle_auc": float(np.nanmean(shuffles)) if shuffles else float("nan"),
            "step_median": step_median,
            "monotonic": monotonic,
            "pass": bool(valid_pairs > 0 and mean_auc >= GATE["g_auc_min"]
                         and margin >= GATE["g_margin_min"]),
        }
    return {"templates": out, "records": records}


def run_entailment_wordnet(model, tokenizer, fragment, concepts) -> dict:
    """Entailment on WordNet pairs built around the selected concepts."""
    pairs = fragment_entailment_pairs(fragment, concepts=concepts)
    out: dict[str, dict] = {}
    for name, template in TEMPLATES_E.items():
        scores, labels, true_dist, true_yes = [], [], [], []
        coverage = []
        for x, y, is_true, distance in pairs:
            probs = next_token_probs(model, tokenizer, render(template, x=x, y=y))
            p_yes, mass = entailment_mass(probs, tokenizer)
            scores.append(p_yes)
            labels.append(bool(is_true))
            coverage.append(mass)
            if is_true:
                true_dist.append(distance)
                true_yes.append(p_yes)
        a = auc(scores, labels)
        gradient = (spearman(true_dist, true_yes)
                    if len(set(true_dist)) > 1 and len(set(true_yes)) > 1
                    else float("nan"))
        out[name] = {
            "prompt_example": render(template, x="dog", y="animal"),
            "pairs": len(pairs),
            "auc": a,
            "gradient": gradient,
            "coverage": float(np.mean(coverage)) if coverage else 0.0,
            "pass": bool(a >= GATE["e_auc_min"]),
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


def _print_descent(descent, detailed: bool = False) -> None:
    print("=== G (descent) ===")
    if detailed:
        print(f"{'template':22s} {'AUC':>6s} {'mass':>6s} {'top%d':>6s} {'freq':>6s} {'margin':>7s} "
              f"{'nonce':>6s} {'shuf':>6s} {'step':>5s} {'mono':>6s} {'valid':>7s} pass")
        for name, r in descent["templates"].items():
            print(f"{name:22s} {r['auc']:6.3f} {r['desc_mass']:6.3f} {r['top_is_desc']*100:6.0f} "
                  f"{r['freq_auc']:6.3f} {r['margin']:7.3f} {r['nonce_auc']:6.3f} "
                  f"{r['shuffle_auc']:6.3f} {r['step_median']:5.2f} {r['monotonic']:6.2f} "
                  f"{r['valid_pairs']:3d}/{r['total_pairs']:<3d} {r['pass']}")
        return
    print(f"{'template':22s} {'AUC':>6s} {'mass':>6s} {'top%d':>6s} {'freq':>6s} {'margin':>7s} {'nonce':>6s} {'shuf':>6s} pass")
    for name, r in descent["templates"].items():
        print(f"{name:22s} {r['auc']:6.3f} {r['desc_mass']:6.3f} {r['top_is_desc']*100:6.0f} "
              f"{r['freq_auc']:6.3f} {r['margin']:7.3f} {r['nonce_auc']:6.3f} {r['shuffle_auc']:6.3f} {r['pass']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--arm", choices=["triads", "wordnet"], default="triads")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--fragment", default=str(DEFAULT_FRAGMENT))
    ap.add_argument("--split", default="main")
    ap.add_argument("--max-concepts", type=int, default=0,
                    help="limit concepts (0 = all); for smoke runs")
    ap.add_argument("--min-children", type=int, default=3,
                    help="WordNet arm: keep concepts with >= this many direct children")
    ap.add_argument("--balance-abstract", action="store_true",
                    help="WordNet arm: with --max-concepts, take half abstract / half concrete")
    ap.add_argument("--g-only", action="store_true",
                    help="WordNet arm: run descent only (skip entailment)")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    if args.arm == "triads":
        triads = load_triads(args.data, split=args.split)
        print(f"model={args.model}  arm=triads  triads={len(triads)} (split={args.split})")
        descent = run_descent(model, tokenizer, triads)
        entail = run_entailment(model, tokenizer, triads)
        source = str(args.data)
        out_name = f"hierarchy_{args.split}_{model_slug(args.model)}.json"
    else:
        fragment = load_fragment(args.fragment)
        concepts = fragment_concepts(fragment, min_children=args.min_children)
        nodes = fragment["nodes"]
        if args.balance_abstract and args.max_concepts:
            abstract = sorted(c for c in concepts if synset_is_abstract(c))
            concrete = sorted(c for c in concepts if not synset_is_abstract(c))
            half = args.max_concepts // 2
            concepts = abstract[:half] + concrete[:args.max_concepts - half]
        # Interleave depths so a truncated run still spans the hierarchy.
        by_depth: dict[int, list[str]] = {}
        for c in sorted(concepts, key=lambda c: (nodes[c]["depth"], nodes[c]["root"], c)):
            by_depth.setdefault(nodes[c]["depth"], []).append(c)
        ordered: list[str] = []
        queues = [list(v) for v in by_depth.values()]
        while any(queues):
            for queue in queues:
                if queue:
                    ordered.append(queue.pop(0))
        concepts = ordered
        if args.max_concepts:
            concepts = concepts[:args.max_concepts]
        print(f"model={args.model}  arm=wordnet  fragment={args.fragment}  "
              f"concepts={len(concepts)} (min_children={args.min_children})")
        descent = run_descent_wordnet(model, tokenizer, fragment, concepts)
        entail = ({"templates": {}, "skipped": True} if args.g_only
                  else run_entailment_wordnet(model, tokenizer, fragment, concepts))
        source = str(args.fragment)
        suffix = ""
        if args.g_only:
            suffix += "_gonly"
        if args.balance_abstract:
            suffix += f"_bal{args.max_concepts}"
        out_name = f"hierarchy_wordnet{suffix}_{model_slug(args.model)}.json"

    g_pass = sum(r["pass"] for r in descent["templates"].values())
    e_pass = sum(r["pass"] for r in entail["templates"].values())
    verdict = "hierarchy-confirmed" if (g_pass >= GATE["templates_min"] and e_pass >= 1) else "absent"

    result = {
        "model": args.model,
        "arm": args.arm,
        "source": source,
        "gate": GATE,
        "descent": descent,
        "entailment": entail,
        "summary": {"g_pass": g_pass, "e_pass": e_pass, "verdict": verdict},
    }
    if args.arm == "triads":
        result["data"] = source
        result["split"] = args.split
    else:
        result["fragment"] = source
        result["concepts"] = concepts
        result["min_children"] = args.min_children

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / out_name
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved -> {out}\n")

    _print_descent(descent, detailed=(args.arm == "wordnet"))
    print("\n=== E (entailment) ===")
    if args.arm == "triads":
        print(f"{'template':22s} {'AUC':>6s}  pass")
        for name, r in entail["templates"].items():
            print(f"{name:22s} {r['auc']:6.3f}  {r['pass']}")
    else:
        print(f"{'template':22s} {'AUC':>6s} {'grad':>6s} {'cov':>6s} {'pairs':>5s}  pass")
        for name, r in entail["templates"].items():
            print(f"{name:22s} {r['auc']:6.3f} {r['gradient']:6.2f} {r['coverage']:6.3f} "
                  f"{r['pairs']:5d}  {r['pass']}")
    print(f"\nverdict: {verdict}  (G pass {g_pass}/{len(TEMPLATES_G)}, E pass {e_pass}/{len(TEMPLATES_E)})")


if __name__ == "__main__":
    main()
