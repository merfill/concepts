#!/usr/bin/env python
"""Experiment 016: geometry of generation trajectories in distribution space.

Protocol: ``../experiment-016-2026-11-27-trajectory-geometry.md``.

Greedy generation is the base trajectory.  At a swept position ``t*`` we force
one alternative *plausible* token (rank ``r`` in top-k, or the nucleus boundary)
and continue greedily.  For every branch we report: per-step JS to the base,
token survival, Fisher vs Euclidean path length, Fisher-Rao distance to the
base, geodesic deviation on the simplex sphere, and the participation ratio of
the cloud of visited distributions (all branches).

All distances use the full vocabulary: ``g = diag(p) − p pᵀ`` via the O(V)
quadratic form.  Full ``p_t``/``z_t`` are not persisted (size); only metrics.

Run:
    uv run python scripts/run_trajectory_geometry.py --model distilgpt2 \
        --limit-prompts 2 --length 20
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
    branch_top_index,
    euclid_path_length,
    fisher_path_length,
    fisher_rao,
    geodesic_deviation,
    null_index,
    nucleus_index,
    participation_ratio,
    token_survival,
)

DEFAULT_PROMPTS = Path(__file__).resolve().parents[1] / "data" / "trajectory_prompts.txt"


def model_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def load_prompts(path: Path) -> list[str]:
    out = [ln.strip() for ln in path.read_text().splitlines()]
    return [ln for ln in out if ln]


def get_state(model, W_U, bias, input_ids):
    """Return post-norm ``z``, ``p = softmax(ℓ)``, and logits ``ℓ = W_U z`` (all V)."""
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    z = out.hidden_states[-1][0, -1].float()
    logits = z @ W_U.T
    if bias is not None:
        logits = logits + bias
    p = torch.softmax(logits, dim=-1)
    return (z.cpu().numpy().astype(np.float64),
            p.cpu().numpy().astype(np.float64),
            logits.cpu().numpy().astype(np.float64))


def run_greedy(model, W_U, bias, ids, length: int, forced: dict | None = None):
    """Greedy rollout with optional forced token(s) at local step index."""
    forced = forced or {}
    zs, ps, logits, toks = [], [], [], []
    cur = ids
    for step in range(length):
        z, p, lg = get_state(model, W_U, bias, cur)
        zs.append(z)
        ps.append(p)
        logits.append(lg)
        t = int(forced[step]) if step in forced else int(np.argmax(p))
        toks.append(t)
        nxt = torch.tensor([[t]], dtype=cur.dtype)
        cur = torch.cat([cur, nxt], dim=1)
    return {"zs": zs, "ps": ps, "logits": logits, "toks": toks}


def decode_token(tok, tid: int) -> str:
    return tok.decode([int(tid)])


def feed_reservoir(reservoir, seen, vec, cap, rng):
    """Reservoir-sample rows into ``reservoir`` so memory stays O(cap·V)."""
    seen += 1
    if len(reservoir) < cap:
        reservoir.append(vec)
    else:
        j = int(rng.integers(0, seen))
        if j < cap:
            reservoir[j] = vec
    return seen


def _embed_rows(cloud):
    return 2.0 * np.sqrt(np.clip(cloud, 0.0, None))


def cloud_metrics(reservoir, cap, seed):
    if not reservoir:
        return {"pr_p": 0.0, "pr_x": 0.0, "n": 0}
    cloud = np.asarray(reservoir, dtype=np.float64)
    pr_p = participation_ratio(cloud, max_points=cap, seed=seed)["pr"]
    pr_x = participation_ratio(_embed_rows(cloud), max_points=cap, seed=seed)["pr"]
    return {"pr_p": float(pr_p), "pr_x": float(pr_x), "n": int(cloud.shape[0])}


def random_null_cloud(V: int, cap: int, rng, seed: int):
    u = rng.standard_exponential((cap, V))
    null_p = u / u.sum(axis=1, keepdims=True)
    pr_p = participation_ratio(null_p, max_points=cap, seed=seed)["pr"]
    pr_x = participation_ratio(_embed_rows(null_p), max_points=cap, seed=seed)["pr"]
    return {"pr_p": float(pr_p), "pr_x": float(pr_x)}


def branch_metrics(base, branch, tstar):
    """Align a local branch (from ``tstar``) against the global base trajectory."""
    offset = tstar
    nb = len(branch["ps"])
    js_curve = [max(0.0, float(js(branch["ps"][j], base["ps"][offset + j])))
                for j in range(nb)]
    dist_curve = [float(fisher_rao(branch["ps"][j], base["ps"][offset + j]))
                  for j in range(nb)]
    surv = token_survival(branch["toks"], base["toks"][offset:], start=1)

    bL_F = fisher_path_length(branch["ps"], branch["logits"])
    bL_2 = euclid_path_length(branch["zs"])
    return {
        "tstar": int(tstar),
        "n_steps": int(nb),
        "js_curve": [round(v, 8) for v in js_curve],
        "js_end": js_curve[-1],
        "survival_after": surv,
        "L_F": bL_F,
        "L_2": bL_2,
        "fisher_over_euclid": float(bL_F / max(bL_2, 1e-12)),
        "dist_to_base_end": dist_curve[-1],
        "dist_to_base_mean": float(np.mean(dist_curve)),
        "geodesic": geodesic_deviation(branch["ps"]),
    }


def run_prompt(model, tok, W_U, bias, prompt, L, ranks, nucleus_mass,
               tstars, cap, seed, pr_rng, include_null=True):
    ids = tok(prompt, return_tensors="pt").input_ids
    base = run_greedy(model, W_U, bias, ids, L)
    base_L_F = fisher_path_length(base["ps"], base["logits"])
    base_L_2 = euclid_path_length(base["zs"])

    exclude = set(int(i) for i in getattr(tok, "all_special_ids", []))

    reservoir: list = []
    seen = 0
    min_t = min(tstars) if tstars else 0
    for t in range(min_t, L):
        seen = feed_reservoir(reservoir, seen, base["ps"][t], cap, pr_rng)

    branches = []
    n_toks = len(ids[0])
    for tstar in tstars:
        prefix = torch.cat(
            [ids, torch.tensor([base["toks"][:tstar]], dtype=ids.dtype)], dim=1
        )
        remaining = L - tstar
        p_at = base["ps"][tstar]

        base_argmax = int(np.argmax(p_at))
        controls = []
        for r in ranks:
            tid = branch_top_index(p_at, r, exclude=exclude)
            controls.append(("rank", r, tid))
        nuc = nucleus_index(p_at, nucleus_mass)
        if nuc == base_argmax:
            nuc = branch_top_index(p_at, 2, exclude=exclude)
        controls.append(("nucleus", None, nuc))
        if include_null:
            controls.append(("null", None, null_index(p_at, exclude=exclude)))

        for kind, rank, tid in controls:
            br = run_greedy(model, W_U, bias, prefix, remaining, forced={0: tid})
            m = branch_metrics(base, br, tstar)
            m["kind"] = kind
            m["rank"] = rank
            m["token_id"] = int(tid)
            m["token"] = decode_token(tok, tid)
            branches.append(m)
            for p in br["ps"]:
                seen = feed_reservoir(reservoir, seen, p, cap, pr_rng)

    cloud = cloud_metrics(reservoir, cap, seed)
    return {
        "prompt": prompt,
        "prompt_tokens": int(n_toks),
        "base": {
            "tokens": [int(t) for t in base["toks"]],
            "L_F": base_L_F,
            "L_2": base_L_2,
            "fisher_over_euclid": float(base_L_F / max(base_L_2, 1e-12)),
            "geodesic": geodesic_deviation(base["ps"]),
        },
        "cloud": cloud,
        "branches": branches,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--prompts-file", default=str(DEFAULT_PROMPTS))
    ap.add_argument("--limit-prompts", type=int, default=0)
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--ranks", default="2,3,5,10")
    ap.add_argument("--tstar-fracs", default="0.15,0.5,0.85")
    ap.add_argument("--nucleus-mass", type=float, default=0.5)
    ap.add_argument("--no-null", action="store_true")
    ap.add_argument("--pr-max-points", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
    fracs = [float(x) for x in args.tstar_fracs.split(",") if x.strip()]
    tstars = sorted({max(0, min(args.length - 2, int(round(f * args.length))))
                     for f in fracs if args.length >= 2})

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
    print(f"[model] {args.model}  d={d} V={V}  L={args.length}  "
          f"t*={tstars}  ranks={ranks}  nucleus={args.nucleus_mass}")

    null_ref = random_null_cloud(V, args.pr_max_points, pr_rng, args.seed)
    out = {
        "model": args.model,
        "d": int(d),
        "V": int(V),
        "length": args.length,
        "ranks": ranks,
        "nucleus_mass": args.nucleus_mass,
        "tstar_fracs": fracs,
        "tstar_positions": tstars,
        "pr_max_points": args.pr_max_points,
        "pr_null": null_ref,
        "prompts_file": str(args.prompts_file),
        "n_prompts": len(prompts),
        "per_prompt": [],
    }

    for prompt in prompts:
        r = run_prompt(model, tok, W_U, bias, prompt, args.length, ranks,
                       args.nucleus_mass, tstars, args.pr_max_points, args.seed,
                       pr_rng, include_null=not args.no_null)
        out["per_prompt"].append(r)
        ranks_js = {b["rank"]: round(b["js_end"], 4)
                    for b in r["branches"] if b["kind"] == "rank"
                    and b["tstar"] == tstars[0]}
        print(f"  {prompt[:26]!r:28s} base L_F={r['base']['L_F']:.2f} "
              f"L2={r['base']['L_2']:.2f}  PR_p={r['cloud']['pr_p']:.2f}  "
              f"JS_end@t*={ranks_js}")

    out_path = Path(args.out) if args.out else \
        Path(__file__).resolve().parents[1] / "results" / \
        f"trajectory_geometry_{model_slug(args.model)}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
