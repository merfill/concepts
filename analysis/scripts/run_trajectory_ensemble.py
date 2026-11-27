#!/usr/bin/env python
"""Experiment 016b: ensemble trajectory branching for statistical power.

Protocol: ``../experiment-016-2026-11-27-trajectory-geometry.md`` (section 016b).

Instead of one hand-picked token per branch point, we branch over **all** tokens
in the top-``m`` (excluding the argmax) at a **dense** grid of positions ``t*``.
This removes the "one special token took over" confound.

Per branch we record ``JS`` to the greedy base at a fixed lag and at the end,
token survival, base entropy at ``t*`` (a covariate for "already locked in"),
and takeover descriptors (token dominance / tail period / predictive entropy).

Stats (bootstrap CIs, rank logistic regression, position curve) are computed
and stored alongside the raw records.

Run:
    uv run python scripts/run_trajectory_ensemble.py --model distilgpt2 \
        --limit-prompts 1 --length 12 --topm 4 --tstar-step 4
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
from tc.inverse import js
from tc.trajectory import (
    participation_ratio,
    tail_period,
    token_dominance,
    token_survival,
)
from tc.trajectory_stats import (
    class_report,
    logistic_rank_effect,
    position_curve,
    reconvergence_by_rank,
    takeover_summary,
)

DEFAULT_PROMPTS = Path(__file__).resolve().parents[1] / "data" / "trajectory_prompts.txt"


def model_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def load_prompts(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def entropy(p) -> float:
    p = np.asarray(p, dtype=np.float64)
    return float(-(p * np.log(p + 1e-12)).sum())


def get_state(model, W_U, bias, input_ids):
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    z = out.hidden_states[-1][0, -1].float()
    logits = z @ W_U.T
    if bias is not None:
        logits = logits + bias
    p = torch.softmax(logits, dim=-1)
    return (p.cpu().numpy().astype(np.float64),
            logits.cpu().numpy().astype(np.float64))


def run_greedy(model, W_U, bias, ids, length: int, forced: dict | None = None):
    forced = forced or {}
    ps, toks = [], []
    cur = ids
    for step in range(length):
        p, _ = get_state(model, W_U, bias, cur)
        ps.append(p)
        t = int(forced[step]) if step in forced else int(np.argmax(p))
        toks.append(t)
        cur = torch.cat([cur, torch.tensor([[t]], dtype=cur.dtype)], dim=1)
    return {"ps": ps, "toks": toks}


def decode_token(tok, tid: int) -> str:
    return tok.decode([int(tid)])


def feed_reservoir(reservoir, seen, vec, cap, rng):
    seen += 1
    if len(reservoir) < cap:
        reservoir.append(vec)
    else:
        j = int(rng.integers(0, seen))
        if j < cap:
            reservoir[j] = vec
    return seen


def cloud_pr(reservoir, cap):
    if not reservoir:
        return {"pr_p": 0.0, "n": 0}
    cloud = np.asarray(reservoir, dtype=np.float64)
    pr = participation_ratio(cloud, max_points=cap)["pr"]
    return {"pr_p": float(pr), "n": int(cloud.shape[0])}


def run_prompt(model, tok, W_U, bias, prompt, L, topm, tstars, lags, cap,
               pr_rng, seed):
    ids = tok(prompt, return_tensors="pt").input_ids
    base = run_greedy(model, W_U, bias, ids, L)
    base_ent = [entropy(p) for p in base["ps"]]
    base_maxp = [float(np.max(p)) for p in base["ps"]]

    reservoir, seen = [], 0
    records = []
    for tstar in tstars:
        p_at = base["ps"][tstar]
        order = np.argsort(-p_at, kind="stable")
        rank_of = {int(t): i + 1 for i, t in enumerate(order)}
        candidates = [int(t) for t in order[:topm] if int(t) != int(order[0])]
        if not candidates:
            continue
        prefix = torch.cat(
            [ids, torch.tensor([base["toks"][:tstar]], dtype=ids.dtype)], dim=1
        )
        remaining = L - tstar
        for tid in candidates:
            br = run_greedy(model, W_U, bias, prefix, remaining, forced={0: tid})
            nb = len(br["ps"])
            js_curve = [max(0.0, float(js(br["ps"][j], base["ps"][tstar + j])))
                        for j in range(nb)]
            lag_vals = {int(k): (float(js_curve[k]) if k < nb else None)
                        for k in lags}
            tail = br["toks"][1:]
            dom = token_dominance(tail)
            ent_after = (float(np.mean([entropy(p) for p in br["ps"][1:]]))
                         if nb > 1 else None)
            records.append({
                "prompt": prompt,
                "tstar": int(tstar),
                "rank": int(rank_of[tid]),
                "token_id": int(tid),
                "token": decode_token(tok, tid),
                "n_steps": int(nb),
                "js_end": float(js_curve[-1]),
                "js_lag": {str(k): v for k, v in lag_vals.items()},
                "survival_after": token_survival(br["toks"], base["toks"][tstar:],
                                                 start=1),
                "base_entropy": float(base_ent[tstar]),
                "base_maxp": float(base_maxp[tstar]),
                "entropy_after": ent_after,
                "dom_share": float(dom["dom_share"]),
                "tail_period": int(tail_period(tail)),
            })
            for p in br["ps"]:
                seen = feed_reservoir(reservoir, seen, p, cap, pr_rng)

    return {
        "prompt": prompt,
        "base_tokens": [int(t) for t in base["toks"]],
        "base_entropy_mean": float(np.mean(base_ent)),
        "cloud": cloud_pr(reservoir, cap),
        "records": records,
    }


def build_stats(records, positions, lags):
    js_end = [r["js_end"] for r in records]
    stats = {
        "class_end": class_report(js_end),
        "by_rank": reconvergence_by_rank(records),
        "by_position": position_curve(records, positions),
        "rank_logistic": logistic_rank_effect(records),
        "takeover": takeover_summary(records),
    }
    for k in lags:
        lag_recs = [{**r, "js_end": r["js_lag"][str(k)]} for r in records
                    if r["js_lag"].get(str(k)) is not None]
        stats[f"class_lag{k}"] = class_report([r["js_end"] for r in lag_recs])
        stats[f"by_rank_lag{k}"] = reconvergence_by_rank(lag_recs)
        stats[f"by_position_lag{k}"] = position_curve(lag_recs, positions)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--prompts-file", default=str(DEFAULT_PROMPTS))
    ap.add_argument("--limit-prompts", type=int, default=0)
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--topm", type=int, default=10)
    ap.add_argument("--tstar-step", type=int, default=4)
    ap.add_argument("--tstar-min", type=int, default=2)
    ap.add_argument("--lags", default="5,10")
    ap.add_argument("--pr-max-points", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    lags = [int(x) for x in args.lags.split(",") if x.strip()]
    positions = list(range(args.tstar_min, max(args.tstar_min + 1, args.length - 1),
                           max(1, args.tstar_step)))
    prompts = load_prompts(Path(args.prompts_file))
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pr_rng = np.random.default_rng(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    W_U = model.lm_head.weight.detach().float()
    head_bias = getattr(model.lm_head, "bias", None)
    bias = head_bias.detach().float() if head_bias is not None else None
    V, d = W_U.shape
    print(f"[model] {args.model} d={d} V={V} L={args.length} topm={args.topm} "
          f"t*={positions} lags={lags}")

    per_prompt = []
    for prompt in prompts:
        r = run_prompt(model, tok, W_U, bias, prompt, args.length, args.topm,
                       positions, lags, args.pr_max_points, pr_rng, args.seed)
        per_prompt.append(r)
        print(f"  {prompt[:26]!r:28s} branches={len(r['records']):3d} "
              f"PR_p={r['cloud']['pr_p']:.2f}")

    records = [rec for r in per_prompt for rec in r["records"]]
    stats = build_stats(records, positions, lags)

    out = {
        "model": args.model, "d": int(d), "V": int(V),
        "length": args.length, "topm": args.topm,
        "tstar_positions": positions, "lags": lags,
        "n_prompts": len(prompts), "n_branches": len(records),
        "per_prompt": per_prompt, "stats": stats,
    }
    out_path = Path(args.out) if args.out else \
        Path(__file__).resolve().parents[1] / "results" / \
        f"trajectory_ensemble_{model_slug(args.model)}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    ce = stats["class_end"]
    print(f"\n[branches] n={ce['n']}  near0={ce['frac_near0']:.3f} "
          f"middle={ce['frac_middle']:.3f} high={ce['frac_high']:.3f} "
          f"near_ln2={ce['frac_near_ln2']:.3f}")
    print(f"  middle CI={[round(x,3) for x in ce['ci']['frac_middle']]}")
    rl = stats["rank_logistic"]
    if rl.get("or_rank") is None:
        print(f"  rank OR=n/a ({rl.get('note', 'no estimate')})")
    else:
        print(f"  rank OR={rl['or_rank']:.3f} "
              f"CI={[round(x, 3) for x in rl.get('or_ci', [])]} "
              f"excludes1={rl.get('excludes_one')}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
