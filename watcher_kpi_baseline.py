#!/usr/bin/env python3
"""
watcher_kpi_baseline.py
=======================
Track B aggregator — turn a walk-forward backtest into WATCHER KPI baselines.

Consumes the backtest_results.json emitted by backtest_pools.py (the real
per-draw hit/survivor stream over CA Daily 3 history) and produces, for each
KPI that a sieve-only walk-forward can baseline:

  * the REAL healthy baseline (hit rate + Wilson CI, survivor distribution),
  * the EMPIRICAL firing rate of the trigger on the real stream (replayed
    through the same fire-functions the resolvability probe uses, so real
    autocorrelation is captured — not i.i.d. simulation),
  * a RECOMMENDED threshold value that meets a target false-fire rate.

COVERAGE (honest)
-----------------
Baselineable here (sieve-only backtest provides the inputs):
    hit_rate_collapse, max_consecutive_misses, minimum_hit_rate, survivor_churn
Needs the heavier full-pipeline-per-draw run (NOT covered):
    confidence_drift, window_decay, llm_confidence
The report states this explicitly so nobody calibrates a KPI on absent data.

WHY REPLAY INSTEAD OF SIMULATE
------------------------------
The resolvability probe assumed independent draws. Real hit sequences cluster
(streaks around regime edges). Replaying the recorded 0/1 hit stream through the
fire-functions gives the firing rate your system would ACTUALLY have shown,
autocorrelation included. That is the baseline that matters.

USAGE
-----
    python3 watcher_kpi_baseline.py \
        --backtest-results backtest_results.json \
        --hit-column hit_100 \
        --policies watcher_policies.json \
        --out watcher_kpi_baseline_findings.json

Read-only: recommends values; changes no policy. Route via THRESHOLD_GOVERNANCE.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

Z_95 = 1.959963984540054


# ── binomial helpers ─────────────────────────────────────────────────────────
def binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i)
               for i in range(0, min(k, n) + 1))


def wilson_ci(p_hat: float, n: int, z: float = Z_95):
    if n <= 0:
        return (0.0, 1.0)
    denom = 1 + z*z/n
    center = (p_hat + z*z/(2*n)) / denom
    half = (z/denom) * math.sqrt(p_hat*(1-p_hat)/n + z*z/(4*n*n))
    return (max(0.0, center - half), min(1.0, center + half))


def expected_trials_to_run(run_len: int, q: float) -> float:
    if q <= 0:
        return math.inf
    if q >= 1:
        return float(run_len)
    qr = q ** run_len
    return (1 - qr) / (qr * (1 - q))


# ── replay a fire-function across the REAL stream ────────────────────────────
def empirical_firing(hits: np.ndarray, fire_fn) -> Dict:
    fires = 0
    gaps: List[int] = []
    last = None
    for t in range(len(hits)):
        if fire_fn(hits[: t + 1]):
            fires += 1
            if last is not None:
                gaps.append(t - last)
            last = t
    n = len(hits)
    return {
        "fires": fires,
        "cycles": n,
        "empirical_fire_rate": round(fires / n, 5) if n else None,
        "median_gap_between_fires": (int(np.median(gaps)) if gaps else None),
    }


# ── loaders ──────────────────────────────────────────────────────────────────
def load_backtest(path: str, hit_column: str):
    with open(path) as f:
        data = json.load(f)
    rows = data.get("results", [])
    if not rows:
        print("[FATAL] backtest_results.json has no 'results' rows.", file=sys.stderr)
        sys.exit(2)
    order = {"midday": 0, "evening": 1}
    rows.sort(key=lambda r: (r.get("date", ""), order.get(r.get("session", ""), 9)))
    if hit_column not in rows[0]:
        print(f"[FATAL] hit column '{hit_column}' not in rows "
              f"(have: {[k for k in rows[0] if k.startswith('hit')]}).", file=sys.stderr)
        sys.exit(2)
    hits = np.array([1 if r.get(hit_column) in (True, "True", 1) else 0 for r in rows],
                    dtype=np.int8)
    survivors = np.array([int(r.get("survivors", 0)) for r in rows], dtype=np.int64)
    return data, rows, hits, survivors


def load_policy_vals(path: Optional[str]) -> Dict:
    defaults = {"hit_rate_collapse_threshold": 0.01, "hit_rate_collapse_window": 20,
                "max_consecutive_misses": 5, "minimum_hit_rate": 0.05,
                "survivor_churn_threshold": 0.4}
    if path and Path(path).exists():
        with open(path) as f:
            pol = json.load(f)
        rt = pol.get("retrain_triggers", {})
        rs = pol.get("regime_shift_triggers", {})
        cv = pol.get("convergence_targets", {})
        return {
            "hit_rate_collapse_threshold": rt.get("hit_rate_collapse_threshold", defaults["hit_rate_collapse_threshold"]),
            "hit_rate_collapse_window": rt.get("hit_rate_collapse_window", defaults["hit_rate_collapse_window"]),
            "max_consecutive_misses": rt.get("max_consecutive_misses", defaults["max_consecutive_misses"]),
            "minimum_hit_rate": cv.get("minimum_hit_rate", defaults["minimum_hit_rate"]),
            "survivor_churn_threshold": rs.get("survivor_churn_threshold", defaults["survivor_churn_threshold"]),
        }
    print("[warn] no policies file — using documented defaults for configured values.",
          file=sys.stderr)
    return defaults


# ── recommendation math ──────────────────────────────────────────────────────
def recommend_collapse(p: float, window: int, alpha: float) -> Dict:
    """Largest hits-threshold h keeping healthy per-window false-fire <= alpha;
    if even h=0 exceeds alpha, recommend a longer window."""
    h = -1
    for cand in range(0, window + 1):
        if binom_cdf(cand, window, p) <= alpha:
            h = cand
        else:
            break
    if h < 0:
        need_w = math.ceil(math.log(alpha) / math.log(1 - p)) if 0 < p < 1 else None
        return {"feasible_at_window": False,
                "recommend_min_window_for_zero_hit_rule": need_w,
                "note": f"At p={p:.3f}, window={window} is too short: even a "
                        f"zero-hit rule false-fires above {alpha}. Need window "
                        f">= {need_w} for a 0-hit collapse rule to be safe."}
    return {"feasible_at_window": True,
            "recommend_threshold_hits_per_window": h,
            "recommend_threshold_rate": round(h / window, 4),
            "healthy_false_fire_per_window": round(binom_cdf(h, window, p), 5)}


def recommend_consecutive(p: float, horizon: int) -> Dict:
    """Smallest run length M whose expected healthy draws-to-false-fire >= horizon."""
    q = 1 - p
    for m in range(1, 100):
        if expected_trials_to_run(m, q) >= horizon:
            return {"recommend_run_len": m,
                    "expected_draws_to_false_fire": round(expected_trials_to_run(m, q), 1)}
    return {"recommend_run_len": None, "note": "no run length <100 reaches horizon"}


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Baseline WATCHER KPIs from a walk-forward backtest")
    ap.add_argument("--backtest-results", required=True)
    ap.add_argument("--hit-column", default="hit_100", choices=["hit_20", "hit_100", "hit_300"])
    ap.add_argument("--policies", default="watcher_policies.json")
    ap.add_argument("--target-false-fire", type=float, default=0.05,
                    help="Acceptable healthy false-fire rate for recommendations")
    ap.add_argument("--false-fire-horizon", type=int, default=200,
                    help="Draws a healthy consecutive-miss rule should survive")
    ap.add_argument("--churn-percentile", type=float, default=95.0,
                    help="Percentile of healthy survivor-churn to set the threshold at")
    ap.add_argument("--out", default="watcher_kpi_baseline_findings.json")
    args = ap.parse_args()

    _, rows, hits, survivors = load_backtest(args.backtest_results, args.hit_column)
    cfg = load_policy_vals(args.policies)
    n = len(hits)

    baseline_p = float(hits.mean())
    lo, hi = wilson_ci(baseline_p, n)

    print("=" * 74)
    print("WATCHER KPI BASELINE  (from walk-forward backtest)")
    print("=" * 74)
    print(f"draws: {n}   hit column: {args.hit_column}")
    print(f"REAL baseline hit rate: {baseline_p:.4f}  (95% Wilson [{lo:.4f}, {hi:.4f}])")
    print(f"survivors: mean={survivors.mean():.1f} std={survivors.std():.1f} "
          f"min={survivors.min()} max={survivors.max()}")
    if n < 100:
        print(f"!! n={n} is small — treat baselines as provisional; extend the "
              f"walk-forward for tighter CIs before governance.")
    print("-" * 74)

    findings: List[Dict] = []

    # 1) hit_rate_collapse
    W = int(cfg["hit_rate_collapse_window"])
    T = float(cfg["hit_rate_collapse_threshold"])
    thr_hits = math.floor(T * W)

    def collapse_fire(h):
        return len(h) >= W and h[-W:].sum() <= thr_hits
    emp = empirical_firing(hits, collapse_fire)
    rec = recommend_collapse(baseline_p, W, args.target_false_fire)
    findings.append({"kpi": "hit_rate_collapse", "configured": {"threshold": T, "window": W},
                     "empirical_on_real_stream": emp, "recommendation": rec})

    # 2) max_consecutive_misses
    M = int(cfg["max_consecutive_misses"])

    def miss_fire(h):
        return len(h) >= M and h[-M:].sum() == 0
    emp2 = empirical_firing(hits, miss_fire)
    rec2 = recommend_consecutive(baseline_p, args.false_fire_horizon)
    findings.append({"kpi": "max_consecutive_misses", "configured": {"run_len": M},
                     "empirical_on_real_stream": emp2, "recommendation": rec2})

    # 3) minimum_hit_rate
    floor = float(cfg["minimum_hit_rate"])
    findings.append({"kpi": "minimum_hit_rate", "configured": {"minimum_hit_rate": floor},
                     "real_baseline_hit_rate": round(baseline_p, 4),
                     "baseline_clears_floor": bool(lo > floor),
                     "note": f"Real hit rate {baseline_p:.4f} vs floor {floor:.4f}; "
                             f"floor is {'below' if floor < baseline_p else 'above'} "
                             f"the measured healthy rate."})

    # 4) survivor_churn (relative cycle-to-cycle change in survivor count)
    if len(survivors) > 1:
        prev = survivors[:-1].astype(float)
        delta = np.abs(np.diff(survivors.astype(float)))
        rel = np.divide(delta, np.maximum(prev, 1.0))
        churn_thr = float(cfg["survivor_churn_threshold"])
        pctl = float(np.percentile(rel, args.churn_percentile))
        fire_rate = float((rel > churn_thr).mean())
        findings.append({"kpi": "survivor_churn", "configured": {"threshold": churn_thr},
                         "healthy_relative_churn": {
                             "mean": round(float(rel.mean()), 4),
                             "p50": round(float(np.percentile(rel, 50)), 4),
                             f"p{int(args.churn_percentile)}": round(pctl, 4)},
                         "configured_threshold_fire_rate": round(fire_rate, 4),
                         "recommendation": {
                             f"set_threshold_at_p{int(args.churn_percentile)}": round(pctl, 4),
                             "note": "A threshold at the healthy upper percentile fires "
                                     "only on genuinely anomalous churn."}})

    # report
    for r in findings:
        print(f"\n{r['kpi']}   configured={r.get('configured')}")
        if "empirical_on_real_stream" in r:
            e = r["empirical_on_real_stream"]
            print(f"    empirical fire rate on real stream: {e['empirical_fire_rate']} "
                  f"({e['fires']}/{e['cycles']}), median gap {e['median_gap_between_fires']}")
        if "recommendation" in r:
            print(f"    recommend: {r['recommendation']}")
        if "baseline_clears_floor" in r:
            print(f"    baseline clears floor: {r['baseline_clears_floor']} — {r['note']}")
        if "healthy_relative_churn" in r:
            print(f"    healthy churn: {r['healthy_relative_churn']} | "
                  f"configured fires {r['configured_threshold_fire_rate']*100:.1f}% of cycles")

    out = {
        "aggregator": "watcher_kpi_baseline_v1",
        "source": args.backtest_results,
        "hit_column": args.hit_column,
        "n_draws": n,
        "real_baseline_hit_rate": round(baseline_p, 5),
        "hit_rate_wilson_95": [round(lo, 5), round(hi, 5)],
        "survivor_stats": {"mean": float(survivors.mean()), "std": float(survivors.std()),
                           "min": int(survivors.min()), "max": int(survivors.max())},
        "coverage": {
            "baselined_here": ["hit_rate_collapse", "max_consecutive_misses",
                               "minimum_hit_rate", "survivor_churn"],
            "needs_full_pipeline_run": ["confidence_drift", "window_decay", "llm_confidence"],
        },
        "findings": findings,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + "-" * 74)
    print(f"[done] baseline findings written to {args.out}")
    print("[note] recommends values only — route through THRESHOLD_GOVERNANCE + Team Beta.")


if __name__ == "__main__":
    main()
