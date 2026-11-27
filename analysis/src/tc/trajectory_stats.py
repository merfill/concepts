"""Statistics for the trajectory-ensemble experiment (016b).

Branch records are plain dicts with at least: ``rank``, ``tstar``, ``js_end``,
``base_entropy``, ``prompt`` and (optionally) ``dom_share``, ``tail_period``,
``entropy_after``.  Everything here is offline numpy / sklearn.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

__all__ = [
    "bootstrap_ci",
    "wilson_interval",
    "outcome_classes",
    "class_report",
    "reconvergence_by_rank",
    "position_curve",
    "logistic_rank_effect",
    "takeover_summary",
]


def bootstrap_ci(values, stat=np.mean, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0):
    """Percentile bootstrap CI for ``stat`` over iid ``values``."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return float("nan"), float("nan")
    if v.size == 1:
        s = float(stat(v))
        return s, s
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boot = np.asarray(stat(v[idx], axis=1), dtype=np.float64)
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def wilson_interval(k: int, n: int, z: float = 1.959963984540054):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def outcome_classes(js, near: float = 0.01, high: float = 0.5,
                    near_ln2: float = 0.68) -> dict:
    """Counts/fractions of near-zero, middle, high, near-ln2 outcomes."""
    js = np.asarray(js, dtype=np.float64).reshape(-1)
    n = js.size
    if n == 0:
        return {"n": 0}
    return {
        "n": int(n),
        "frac_near0": float(np.mean(js < near)),
        "frac_middle": float(np.mean((js >= near) & (js <= high))),
        "frac_high": float(np.mean(js > high)),
        "frac_near_ln2": float(np.mean(js > near_ln2)),
        "mean": float(np.mean(js)),
        "median": float(np.median(js)),
    }


def class_report(js, near: float = 0.01, high: float = 0.5,
                 near_ln2: float = 0.68, n_boot: int = 2000,
                 seed: int = 0) -> dict:
    """``outcome_classes`` with bootstrap CIs for each fraction."""
    js = np.asarray(js, dtype=np.float64).reshape(-1)
    out = outcome_classes(js, near, high, near_ln2)
    if out["n"] == 0:
        return out
    ind = {
        "frac_near0": (js < near).astype(np.float64),
        "frac_middle": ((js >= near) & (js <= high)).astype(np.float64),
        "frac_high": (js > high).astype(np.float64),
        "frac_near_ln2": (js > near_ln2).astype(np.float64),
    }
    out["ci"] = {}
    for i, (key, arr) in enumerate(ind.items()):
        lo, hi = bootstrap_ci(arr, n_boot=n_boot, seed=seed + i)
        out["ci"][key] = [lo, hi]
    return out


def reconvergence_by_rank(records, near: float = 0.01) -> dict:
    """Reconvergence rate per branch rank with Wilson intervals."""
    out = {}
    ranks = sorted({int(r["rank"]) for r in records if r.get("rank") is not None})
    for rank in ranks:
        vals = [r["js_end"] < near for r in records if r.get("rank") == rank]
        n = len(vals)
        k = int(sum(vals))
        lo, hi = wilson_interval(k, n)
        out[str(rank)] = {"n": n, "rate": (k / n if n else float("nan")),
                          "ci": [lo, hi]}
    return out


def position_curve(records, positions, near: float = 0.01) -> dict:
    """Reconvergence rate per branch position ``t*`` with Wilson intervals."""
    out = {}
    for t in positions:
        vals = [r["js_end"] < near for r in records if int(r.get("tstar", -1)) == int(t)]
        n = len(vals)
        k = int(sum(vals))
        lo, hi = wilson_interval(k, n)
        out[str(int(t))] = {"n": n, "rate": (k / n if n else float("nan")),
                            "ci": [lo, hi]}
    return out


def _logit_design(rows, prompts):
    X, y = [], []
    for r in rows:
        feats = [float(r["rank"]), float(r["tstar"]), float(r["base_entropy"])]
        feats += [1.0 if r["prompt"] == p else 0.0 for p in prompts[1:]]
        X.append(feats)
        y.append(1 if r["js_end"] < 0.01 else 0)
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


def _fit_rank_coef(X, y, scale_mean, scale_std):
    Xs = X.copy()
    Xs[:, 1:3] = (Xs[:, 1:3] - scale_mean) / scale_std
    if y.min() == y.max():
        return np.nan
    clf = LogisticRegression(C=1e4, solver="lbfgs", max_iter=5000)
    try:
        clf.fit(Xs, y)
    except Exception:
        return np.nan
    return float(clf.coef_[0][0])


def logistic_rank_effect(records, near: float = 0.01, n_boot: int = 500,
                         seed: int = 0) -> dict:
    """Log-odds of reconvergence per unit of branch rank, controlling for
    ``t*``, base entropy and prompt (fixed effects).  Bootstrap CI on the OR.
    """
    rows = [r for r in records
            if r.get("rank") is not None and r.get("js_end") is not None]
    if len(rows) < 20:
        return {"n": len(rows), "note": "too few rows"}
    prompts = sorted({r["prompt"] for r in rows})
    X, y = _logit_design(rows, prompts)
    scale_mean = X[:, 1:3].mean(axis=0)
    scale_std = X[:, 1:3].std(axis=0) + 1e-9
    coef = _fit_rank_coef(X, y, scale_mean, scale_std)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(rows), size=len(rows))
        c = _fit_rank_coef(X[idx], y[idx], scale_mean, scale_std)
        if np.isfinite(c):
            boots.append(c)
    boots = np.asarray(boots, dtype=np.float64)
    result = {"n": len(rows), "n_boot": int(boots.size),
              "or_rank": float(np.exp(coef)) if np.isfinite(coef) else float("nan")}
    if boots.size >= 20:
        lo, hi = np.quantile(np.exp(boots), [0.025, 0.975])
        result["or_ci"] = [float(lo), float(hi)]
        result["excludes_one"] = bool(lo > 1.0 or hi < 1.0)
    else:
        result["or_ci"] = [float("nan"), float("nan")]
        result["excludes_one"] = None
    return result


def takeover_summary(records, near: float = 0.01, n_boot: int = 2000,
                     seed: int = 0) -> dict:
    """Token-takeover descriptors overall and split by reconvergence."""
    rows = [r for r in records if r.get("dom_share") is not None]
    if not rows:
        return {"n": 0}

    def block(rs):
        if not rs:
            return {"n": 0}
        dom = [r["dom_share"] for r in rs]
        per = [1.0 if int(r.get("tail_period") or 0) > 0 else 0.0 for r in rs]
        ent = [r["entropy_after"] for r in rs if r.get("entropy_after") is not None]
        lo_d, hi_d = bootstrap_ci(dom, n_boot=n_boot, seed=seed)
        lo_p, hi_p = bootstrap_ci(per, n_boot=n_boot, seed=seed + 1)
        return {
            "n": len(rs),
            "dom_share_mean": float(np.mean(dom)),
            "dom_share_ci": [lo_d, hi_d],
            "frac_periodic": float(np.mean(per)),
            "frac_periodic_ci": [lo_p, hi_p],
            "entropy_after_mean": float(np.mean(ent)) if ent else float("nan"),
        }

    rec = [r for r in rows if r["js_end"] < near]
    div = [r for r in rows if r["js_end"] >= near]
    return {"n": len(rows), "all": block(rows),
            "reconverged": block(rec), "diverged": block(div)}
