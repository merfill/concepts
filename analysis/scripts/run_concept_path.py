#!/usr/bin/env python
"""Experiment 015: lift a path in token-distribution space to state space.

Protocol: ``../experiment-015-2026-11-27-concept-path-lift.md``.

The straight simplex path ``p(t)=(1-t)p_A+t p_B`` is projected onto the model's
reachable manifold via the unique convex minimiser ``z*(t)``; the KL residual is
the off-manifold distance.  The on-manifold reference is the lifted ``z``-line.

Run:
    uv run python scripts/run_concept_path.py --model distilgpt2
    uv run python scripts/run_concept_path.py --model Qwen/Qwen2.5-1.5B-Instruct --points 9
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
from tc.inverse import cosine, js, kl, simplex_interp

PAIRS = [
    ("The capital of France is", "The capital of Germany is"),
    ("The opposite of hot is", "The opposite of cold is"),
    ("A typical example of a bird is a", "A typical example of a fish is a"),
    ("In the morning the sky is", "At night the sky is"),
]


def model_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def state(model, tok, prompt, W_U):
    enc = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    z = out.hidden_states[-1][0, -1].float()
    p = torch.softmax(z @ W_U.T, dim=-1)
    return z, p


def inverse_state(W_U, p_target, z0, iters, lr):
    """Convex KL projection: argmin_z KL(p_target || softmax(W_U z))."""
    z = z0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        logp = torch.log_softmax(z @ W_U.T, dim=-1)
        loss = -(p_target * logp).sum()
        if not torch.isfinite(loss):
            break
        loss.backward()
        opt.step()
    with torch.no_grad():
        p_model = torch.softmax(z @ W_U.T, dim=-1)
    return z.detach(), p_model


def run_pair(model, tok, W_U, prompt_a, prompt_b, points, iters, lr):
    zA, pA = state(model, tok, prompt_a, W_U)
    zB, pB = state(model, tok, prompt_b, W_U)
    pA_np, pB_np = pA.cpu().numpy(), pB.cpu().numpy()
    ts = np.linspace(0.0, 1.0, points)

    rows, solved = [], {}
    for t in ts:
        p_t = torch.tensor(simplex_interp(pA_np, pB_np, t), dtype=torch.float32)
        z_lin = (1.0 - t) * zA + t * zB
        q_lin = torch.softmax(z_lin @ W_U.T, dim=-1)
        z_star, q_star = inverse_state(W_U, p_t, z_lin, iters, lr)
        solved[round(float(t), 6)] = z_star

        p_np = p_t.cpu().numpy()
        rows.append({
            "t": float(t),
            "R_offmanifold": kl(p_np, q_star.cpu().numpy()),
            "js_p_qlin": js(p_np, q_lin.cpu().numpy()),
            "js_p_qstar": js(p_np, q_star.cpu().numpy()),
            "cos_zstar_zlin": cosine(z_star.cpu().numpy(), z_lin.cpu().numpy()),
            "rel_dist": float((z_star - z_lin).norm() / z_lin.norm().clamp_min(1e-12)),
        })

    t04 = float(min(ts, key=lambda x: abs(x - 0.4)))
    t06 = float(min(ts, key=lambda x: abs(x - 0.6)))
    delta_p = (t06 - t04) * (pB_np - pA_np)
    delta_z = (solved[round(t06, 6)] - solved[round(t04, 6)]).cpu().numpy()
    local = {
        "amplification": float(np.linalg.norm(delta_z) / max(np.linalg.norm(delta_p), 1e-12)),
        "cos_dz_chord": cosine(delta_z, (zB - zA).cpu().numpy()),
    }
    return {"prompt_a": prompt_a, "prompt_b": prompt_b, "rows": rows, "local": local}


def run(model, tok, args):
    W_U = model.lm_head.weight.detach().float()
    V, d = W_U.shape
    out = {"model": args.model, "d": int(d), "V": int(V), "points": args.points,
           "iters": args.iters, "pairs": []}
    for pa, pb in args.pairs:
        r = run_pair(model, tok, W_U, pa, pb, args.points, args.iters, args.lr)
        out["pairs"].append(r)
        R = [x["R_offmanifold"] for x in r["rows"]]
        print(f"  {pa[:26]!r:28s}->{pb[:26]!r:28s} "
              f"R0={R[0]:.1e} R1={R[-1]:.1e} maxR_int={max(R[1:-1]):.4f} "
              f"amp={r['local']['amplification']:.2f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--points", type=int, default=11)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    args.pairs = PAIRS[: args.max_pairs] if args.max_pairs else PAIRS

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    print(f"[model] {args.model}")
    result = run(model, tok, args)
    result["dtype"] = str(next(model.parameters()).dtype).replace("torch.", "")

    out = Path(args.out) if args.out else \
        Path(__file__).resolve().parents[1] / "results" / f"concept_path_{model_slug(args.model)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
