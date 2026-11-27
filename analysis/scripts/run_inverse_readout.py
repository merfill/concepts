#!/usr/bin/env python
"""Experiment 014: is the final post-norm state identifiable from the output?

Protocol: ``../experiment-014-2026-11-27-inverse-readout.md``.

For fixed prompts we read ``z_true`` (final norm output) and ``p_true``; then:
  * I3: verify the norm gauge (scale for RMSNorm; scale+shift for LayerNorm);
  * I2: from top-m probabilities, rank/nullity of the probability Jacobian;
  * I2 constructive: perturb z along a null direction -> top-m nearly fixed while the
    full distribution changes; contrast with a random direction.

No optimisation, no invented axis: ``W_U`` is the model's real decoder.

Run:
    uv run python scripts/run_inverse_readout.py --model distilgpt2
    uv run python scripts/run_inverse_readout.py --model Qwen/Qwen2.5-1.5B-Instruct
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
from tc.inverse import (
    gauge_dim,
    kl,
    null_basis,
    prob_jacobian,
    rank_nullity,
    softmax,
    topk_indices,
)

PROMPTS = [
    "The capital of France is",
    "A typical example of a bird is a",
    "The chemical symbol for water is",
    "When the sun rises in the morning, the sky becomes",
    "She opened the book and began to",
    "The opposite of hot is",
]
M_SWEEP = [5, 10, 20, 50, 100, 200, 500, 1000]
PERTURB_M = 50


def model_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def final_norm(model):
    for path in ("model.norm", "transformer.ln_f", "norm"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if obj is not None:
                return obj
        except AttributeError:
            continue
    raise AttributeError("final norm not found")


def norm_kind(norm) -> str:
    name = type(norm).__name__.lower()
    if "rms" in name:
        return "rmsnorm"
    return "layernorm"


def make_prompt_state(model, tok, prompt):
    enc = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    z = out.hidden_states[-1][0, -1].float()
    head = model.lm_head
    W_U = head.weight.detach().float()
    bias = head.bias.detach().float() if getattr(head, "bias", None) is not None else None
    with torch.no_grad():
        logits = z @ W_U.T
        if bias is not None:
            logits = logits + bias
    p = torch.softmax(logits, dim=-1)
    return z, p, out.logits[0, -1].float()


def gauge_check(norm, kind, W_U, x):
    """Gauge invariance of the final norm on a vector ``x`` (scale / scale+shift)."""
    with torch.no_grad():
        a = norm(x).float()
        alt = 2.0 * x if kind == "rmsnorm" else 2.0 * x + torch.ones_like(x)
        b = norm(alt).float()
        pa = torch.softmax(a @ W_U.T, dim=-1)
        pb = torch.softmax(b @ W_U.T, dim=-1)
    dz = float((a - b).abs().max())
    js = float(0.5 * (pa * torch.log(pa / pb.clamp_min(1e-12))).sum()
               + 0.5 * (pb * torch.log(pb / pa.clamp_min(1e-12))).sum())
    return {"max_abs_dz": dz, "js_p_alt": js, "kind": kind,
            "gauge_dim": gauge_dim(kind)}


def perturb(model, W_U, z, p, S, direction, eps, z_norm):
    z_np = z.detach().cpu().numpy()
    dir_np = direction.detach().cpu().numpy()
    z2 = torch.tensor(z_np + eps * z_norm * dir_np, dtype=torch.float32)
    with torch.no_grad():
        logits = z2 @ W_U.T
    p2 = torch.softmax(logits, dim=-1)
    dp = (p2[S] - p[S]).abs().max().item()
    full_kl = kl(p.cpu().numpy(), p2.cpu().numpy())
    return {"eps": eps, "d_top": float(dp), "kl_full": float(full_kl)}


def run(model, tok, args):
    norm = final_norm(model)
    kind = norm_kind(norm)
    W_U = model.lm_head.weight.detach().float()
    V, d = W_U.shape
    dt = str(next(model.parameters()).dtype)
    recon_tol = 0.2 if "bfloat16" in dt else 1e-2
    result = {"model": args.model, "d": int(d), "V": int(V), "norm": kind,
              "gauge_dim": gauge_dim(kind), "recon_tol": recon_tol, "prompts": PROMPTS,
              "m_sweep": [m for m in M_SWEEP if m < V], "perturb_m": PERTURB_M,
              "per_prompt": []}

    for prompt in PROMPTS:
        z, p, logits_model = make_prompt_state(model, tok, prompt)
        check = float((p.log() - torch.log_softmax(logits_model, dim=-1)).abs().max())
        gauge = gauge_check(norm, kind, W_U, z)
        S = torch.tensor(topk_indices(p.cpu().numpy(), PERTURB_M))

        rank_curve = {}
        for m in result["m_sweep"]:
            idx = topk_indices(p.cpu().numpy(), m)
            J = prob_jacobian(W_U.cpu().numpy(), p.cpu().numpy(), idx)
            rn = rank_nullity(J)
            rank_curve[str(m)] = rn

        idx = topk_indices(p.cpu().numpy(), PERTURB_M)
        J = prob_jacobian(W_U.cpu().numpy(), p.cpu().numpy(), idx)
        basis = null_basis(J)
        z_norm = float(z.norm())
        null_dir = torch.tensor(basis[0], dtype=torch.float32)
        null_dir = null_dir / null_dir.norm()
        rand = torch.randn_like(null_dir)
        rand = rand / rand.norm()

        null_scan = [perturb(model, W_U, z, p, S, null_dir, e, z_norm)
                     for e in (0.001, 0.005, 0.01, 0.05, 0.1)]
        rand_scan = [perturb(model, W_U, z, p, S, rand, e, z_norm)
                     for e in (0.001, 0.005, 0.01, 0.05, 0.1)]

        result["per_prompt"].append({
            "prompt": prompt,
            "recon_max_logp_err": check,
            "gauge": gauge,
            "rank_curve": rank_curve,
            "null_dim_at_50": rank_nullity(J)["null_dim"],
            "null_scan": null_scan,
            "rand_scan": rand_scan,
        })
        print(f"  {prompt!r:55s} null_dim@50={rank_nullity(J)['null_dim']:4d} "
              f"gauge_dz={gauge['max_abs_dz']:.2e}")

    # aggregate
    null_dims = [pp["rank_curve"]["50"]["null_dim"] for pp in result["per_prompt"]]
    result["summary"] = {
        "mean_null_dim_at_50": float(np.mean(null_dims)),
        "d_minus_50": int(d - PERTURB_M),
        "recon_ok": bool(max(pp["recon_max_logp_err"] for pp in result["per_prompt"]) < recon_tol),
        "gauge_ok": bool(max(pp["gauge"]["max_abs_dz"] for pp in result["per_prompt"]) < 1e-4),
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    print(f"[model] {args.model}  dtype={dtype}")

    result = run(model, tok, args)
    result["dtype"] = dtype

    out = Path(args.out) if args.out else \
        Path(__file__).resolve().parents[1] / "results" / f"inverse_{model_slug(args.model)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    s = result["summary"]
    print(f"\n[d]={result['d']}  [V]={result['V']}  norm={result['norm']}  "
          f"gauge_dim={result['gauge_dim']}")
    print(f"[null_dim@50] mean={s['mean_null_dim_at_50']:.1f}  (d-50={s['d_minus_50']})")
    print(f"[recon_ok]={s['recon_ok']}  [gauge_ok]={s['gauge_ok']}")
    pp = result["per_prompt"][0]
    print("example null scan:", [(round(x["eps"], 2), round(x["d_top"], 4), round(x["kl_full"], 4))
                                 for x in pp["null_scan"]])
    print("example rand scan:", [(round(x["eps"], 2), round(x["d_top"], 4), round(x["kl_full"], 4))
                                 for x in pp["rand_scan"]])
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
