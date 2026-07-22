#!/usr/bin/env python3
"""
watcher_kpi_resolvability_probe.py
==================================
Type-1 (analytic) resolvability probe for WATCHER governance KPIs.

PURPOSE
-------
Answer one question per numeric KPI in watcher_policies.json:

    "At its configured value and window, can this trigger distinguish a
     real event from normal statistical variance of a HEALTHY system?"

A threshold that fires on healthy data is DECORATIVE (false alarms dominate).
A threshold that cannot fire even on a genuine collapse is INSENSITIVE.
Only a threshold with acceptable false-positive AND detection behaviour is
RESOLVABLE. This probe classifies each KPI and recommends a direction.

This is *analytic* calibration only (Type 1). It needs no cluster, no GPU,
and no recorded runs. Thresholds that require labelled event series or LLM
runs (regime-shift decay, survivor churn, confidence drift, LLM confidence)
are reported as NEEDS-DATA with the series they would require — they are NOT
faked here.

CRITICAL CAVEAT — read before trusting any verdict
--------------------------------------------------
Every result depends on the OPERATIONAL DEFINITION of "hit rate": is it a
per-draw Bernoulli probability, and what is the healthy baseline value?
The convergence_targets.minimum_hit_rate (0.05) is used as the default
baseline, and a pool-size model (P(hit) = pool_size / draw_space) is offered
as an alternative. If the live code computes hit rate differently, pass the
correct baseline via --baseline-hit-rate. Step zero of any real calibration
is to extract that definition from the source (grep the hit-rate computation
in the Step-5 / WATCHER path) and feed it here. This probe does not read that
definition for you; it makes the assumption explicit and loud.

USAGE
-----
    python3 watcher_kpi_resolvability_probe.py \
        --policies watcher_policies.json \
        --pool-size 50 \
        --out watcher_kpi_findings.json

    # or specify the baseline hit rate directly if you know it:
    python3 watcher_kpi_resolvability_probe.py \
        --policies watcher_policies.json \
        --baseline-hit-rate 0.08

Author: Team Alpha (delivered for Claude Code execution on VM101)
Read-only: computes and reports; changes no policy value, commits nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ── Documented defaults (fallback only; real values come from --policies) ─────
DOCUMENTED_DEFAULTS = {
    "retrain_triggers": {
        "retrain_after_n_draws": 10,
        "confidence_drift_threshold": 0.2,
        "max_consecutive_misses": 5,
        "hit_rate_collapse_threshold": 0.01,
        "hit_rate_collapse_window": 20,
    },
    "regime_shift_triggers": {
        "window_decay_threshold": 0.5,
        "survivor_churn_threshold": 0.4,
        "llm_confidence_threshold": 0.8,
    },
    "convergence_targets": {
        "minimum_hit_rate": 0.05,
    },
}

Z_95 = 1.959963984540054  # two-sided 95%


# ── Closed-form helpers (stdlib only) ────────────────────────────────────────
def binom_pmf(k: int, n: int, p: float) -> float:
    if k < 0 or k > n:
        return 0.0
    return math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    return sum(binom_pmf(i, n, p) for i in range(0, min(k, n) + 1))


def expected_trials_to_run(run_len: int, event_prob: float) -> float:
    """
    Expected number of Bernoulli trials until the first run of `run_len`
    consecutive events, each event occurring with probability `event_prob`.
    Closed form: E = (1 - q^r) / (q^r * (1 - q)) with q = event_prob.
    """
    q = event_prob
    if q <= 0.0:
        return math.inf
    if q >= 1.0:
        return float(run_len)
    qr = q ** run_len
    return (1.0 - qr) / (qr * (1.0 - q))


def wilson_halfwidth(p_hat: float, n: int, z: float = Z_95) -> float:
    """Half-width of the Wilson score interval — closed form, no scipy."""
    if n <= 0:
        return math.inf
    denom = 1.0 + z * z / n
    margin = (z / denom) * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n))
    return margin


# ── Monte-Carlo confirmation on a rolling healthy stream ──────────────────────
def mc_draws_to_first_false_fire(
    fire_fn,
    baseline_p: float,
    stream_len: int,
    trials: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Simulate healthy draw streams (each draw a hit with prob baseline_p) and
    record how many draws elapse before `fire_fn(history)` first returns True.
    Returns median/mean draws-to-false-fire and the fraction of streams that
    NEVER falsely fired within stream_len.
    """
    times: List[float] = []
    never = 0
    for _ in range(trials):
        hits = (rng.random(stream_len) < baseline_p).astype(np.int8)
        fired_at = None
        for t in range(stream_len):
            if fire_fn(hits[: t + 1]):
                fired_at = t + 1
                break
        if fired_at is None:
            never += 1
            times.append(float(stream_len))  # censored at horizon
        else:
            times.append(float(fired_at))
    arr = np.array(times)
    return {
        "mc_median_draws_to_false_fire": float(np.median(arr)),
        "mc_mean_draws_to_false_fire": float(np.mean(arr)),
        "mc_frac_clean_within_horizon": never / trials,
        "mc_horizon": stream_len,
    }


# ── Per-KPI analysers ─────────────────────────────────────────────────────────
def analyse_hit_rate_collapse(
    threshold: float, window: int, baseline_p: float, collapsed_p: float,
    cfg: "Config", rng: np.random.Generator,
) -> Dict:
    """Fires when hits/window <= threshold over the last `window` draws."""
    max_fire_hits = math.floor(threshold * window)  # integer hits that trip it
    fp_per_window = binom_cdf(max_fire_hits, window, baseline_p)
    detect_per_window = binom_cdf(max_fire_hits, window, collapsed_p)

    def fire_fn(hist: np.ndarray) -> bool:
        if len(hist) < window:
            return False
        return hist[-window:].sum() <= max_fire_hits

    mc = mc_draws_to_first_false_fire(fire_fn, baseline_p, cfg.mc_stream_len,
                                      cfg.mc_trials, rng)
    verdict, note = _verdict(fp_per_window, detect_per_window, mc, cfg)
    return {
        "kpi": "retrain_triggers.hit_rate_collapse_threshold",
        "configured": {"threshold": threshold, "window": window},
        "integer_hits_that_trip": max_fire_hits,
        "fp_rate_per_window_healthy": round(fp_per_window, 5),
        "detection_per_window_at_collapsed_rate": round(detect_per_window, 5),
        "collapsed_rate_tested": collapsed_p,
        **mc,
        "verdict": verdict,
        "note": note,
    }


def analyse_consecutive_misses(
    run_len: int, baseline_p: float, collapsed_p: float,
    cfg: "Config", rng: np.random.Generator,
) -> Dict:
    """Fires on `run_len` consecutive misses (a miss = no hit)."""
    q_healthy = 1.0 - baseline_p
    q_collapsed = 1.0 - collapsed_p
    e_healthy = expected_trials_to_run(run_len, q_healthy)
    e_collapsed = expected_trials_to_run(run_len, q_collapsed)

    def fire_fn(hist: np.ndarray) -> bool:
        if len(hist) < run_len:
            return False
        return hist[-run_len:].sum() == 0

    mc = mc_draws_to_first_false_fire(fire_fn, baseline_p, cfg.mc_stream_len,
                                      cfg.mc_trials, rng)
    # False-positive framing: how often does a healthy stream trip it?
    fp_proxy = 1.0 - mc["mc_frac_clean_within_horizon"]
    verdict, note = _verdict(fp_proxy, 1.0, mc, cfg,
                             detect_is_latency=True,
                             e_healthy=e_healthy)
    return {
        "kpi": "retrain_triggers.max_consecutive_misses",
        "configured": {"run_len": run_len},
        "expected_draws_to_false_fire_healthy_analytic": round(e_healthy, 2),
        "expected_draws_to_fire_at_collapsed_rate": round(e_collapsed, 2),
        "collapsed_rate_tested": collapsed_p,
        **mc,
        "verdict": verdict,
        "note": note,
    }


def analyse_retrain_after_n(n_draws: int, baseline_p: float, cfg: "Config") -> Dict:
    """Is n_draws enough to resolve a rate change from baseline?"""
    hw = wilson_halfwidth(baseline_p, n_draws)
    # Minimum detectable effect ~ interval half-width; express relative to base.
    rel = hw / baseline_p if baseline_p > 0 else math.inf
    resolvable = hw < cfg.max_ci_halfwidth
    return {
        "kpi": "retrain_triggers.retrain_after_n_draws",
        "configured": {"n_draws": n_draws},
        "baseline_hit_rate": baseline_p,
        "wilson_95_halfwidth_at_baseline": round(hw, 5),
        "relative_uncertainty": round(rel, 3),
        "verdict": "RESOLVABLE" if resolvable else "INSUFFICIENT-N",
        "note": (
            f"At n={n_draws}, a hit rate of {baseline_p:g} is only pinned to "
            f"+/-{hw:.3f} (95%). Rate changes smaller than that half-width are "
            f"statistically indistinguishable from noise at this sample size."
        ),
    }


def analyse_minimum_hit_rate(
    min_hit_rate: float, pool_size: Optional[int], draw_space: int,
    eval_draws: int, cfg: "Config",
) -> Dict:
    """Is the floor above pure-chance for the given pool, with margin?"""
    rec: Dict = {
        "kpi": "convergence_targets.minimum_hit_rate",
        "configured": {"minimum_hit_rate": min_hit_rate},
    }
    if pool_size is None:
        rec["verdict"] = "NEEDS-DATA"
        rec["note"] = ("Pass --pool-size to compare against the chance baseline "
                       "P(hit)=pool_size/draw_space. Without it, cannot judge "
                       "whether 0.05 clears random.")
        return rec

    chance = pool_size / draw_space
    margin = min_hit_rate - chance
    # One-sided significance that true rate > chance, given the floor is met
    # exactly, over eval_draws draws (normal approx to binomial).
    if chance <= 0 or chance >= 1:
        z = math.inf
    else:
        se = math.sqrt(chance * (1 - chance) / eval_draws)
        z = margin / se if se > 0 else math.inf
    if margin <= 0:
        verdict = "DECORATIVE (at/below chance)"
    elif z < 1.64:  # < ~95% one-sided
        verdict = "WEAK (margin not significant at eval size)"
    else:
        verdict = "RESOLVABLE"
    rec.update({
        "pool_size": pool_size,
        "draw_space": draw_space,
        "chance_baseline_hit_rate": round(chance, 5),
        "margin_over_chance": round(margin, 5),
        "z_vs_chance_at_eval_draws": round(z, 3) if math.isfinite(z) else None,
        "eval_draws_assumed": eval_draws,
        "verdict": verdict,
        "note": (
            f"A random pool of {pool_size} over {draw_space} outcomes hits at "
            f"{chance:.3f}. The floor {min_hit_rate:g} sits {margin:+.3f} from "
            f"chance."
        ),
    })
    return rec


def report_needs_data() -> List[Dict]:
    """KPIs that cannot be resolved analytically — state what they'd need."""
    return [
        {"kpi": "regime_shift_triggers.window_decay_threshold",
         "verdict": "NEEDS-DATA",
         "note": "Requires recorded window-performance series + injected decay "
                 "events (Type-2 ROC calibration)."},
        {"kpi": "regime_shift_triggers.survivor_churn_threshold",
         "verdict": "NEEDS-DATA",
         "note": "Requires recorded survivor-population deltas across reruns + "
                 "labelled true regime boundaries (Type-2)."},
        {"kpi": "retrain_triggers.confidence_drift_threshold",
         "verdict": "NEEDS-DATA",
         "note": "Requires recorded predicted-vs-actual correlation series to "
                 "measure drift-detection FP/FN (Type-2)."},
        {"kpi": "regime_shift_triggers.llm_confidence_threshold",
         "verdict": "NEEDS-DATA",
         "note": "Requires advisor runs over labelled shift/no-shift windows to "
                 "build a confidence-vs-precision calibration curve (Type-3, "
                 "needs LLM services)."},
    ]


# ── Verdict logic (configurable bounds, no hardcoded pass/fail) ───────────────
def _verdict(fp_rate, detection, mc, cfg, detect_is_latency=False, e_healthy=None):
    clean = mc["mc_frac_clean_within_horizon"]
    if clean < (1.0 - cfg.max_fp_rate):
        extra = ""
        if e_healthy is not None:
            extra = f" (analytic E[draws to false fire]={e_healthy:.1f})"
        return ("DECORATIVE (fires on healthy data)",
                f"Only {clean*100:.1f}% of healthy {mc['mc_horizon']}-draw "
                f"streams stayed clean; FP-per-window={fp_rate:.3f}.{extra}")
    if not detect_is_latency and detection < cfg.min_detection:
        return ("INSENSITIVE (misses real events)",
                f"Detection at the tested collapsed rate is only "
                f"{detection:.3f} per window (< {cfg.min_detection}).")
    return ("RESOLVABLE",
            f"Healthy streams stayed clean {clean*100:.1f}% of the time; "
            f"detection={detection:.3f}.")


class Config:
    def __init__(self, args):
        self.mc_trials = args.mc_trials
        self.mc_stream_len = args.mc_stream_len
        self.max_fp_rate = args.max_fp_rate
        self.min_detection = args.min_detection
        self.max_ci_halfwidth = args.max_ci_halfwidth


# ── Policy loading ────────────────────────────────────────────────────────────
def load_policies(path: Optional[str]) -> Dict:
    if path and Path(path).exists():
        with open(path) as f:
            pol = json.load(f)
        print(f"[probe] Loaded configured KPI values from {path}")
        return pol
    print("[probe] WARNING: watcher_policies.json not found — using DOCUMENTED "
          "DEFAULTS. Pass --policies to analyse the real configured values.",
          file=sys.stderr)
    return DOCUMENTED_DEFAULTS


def get(pol: Dict, section: str, key: str, default):
    return pol.get(section, {}).get(key, default)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="WATCHER KPI analytic resolvability probe")
    ap.add_argument("--policies", default="watcher_policies.json",
                    help="Path to watcher_policies.json (uses defaults if absent)")
    ap.add_argument("--baseline-hit-rate", type=float, default=None,
                    help="Healthy per-draw hit probability. If omitted, taken "
                         "from convergence_targets.minimum_hit_rate.")
    ap.add_argument("--pool-size", type=int, default=None,
                    help="Prediction pool size (for chance-baseline comparison "
                         "and pool-model hit rate).")
    ap.add_argument("--draw-space", type=int, default=1000,
                    help="Number of distinct outcomes (CA Daily 3 = 1000).")
    ap.add_argument("--collapse-factor", type=float, default=0.2,
                    help="Genuine-collapse alternative = baseline * factor "
                         "(default 0.2 = an 80%% drop).")
    ap.add_argument("--eval-draws", type=int, default=200,
                    help="Draws assumed available to evaluate minimum_hit_rate.")
    ap.add_argument("--mc-trials", type=int, default=5000)
    ap.add_argument("--mc-stream-len", type=int, default=500,
                    help="Healthy-stream horizon for false-fire simulation.")
    ap.add_argument("--max-fp-rate", type=float, default=0.05,
                    help="Acceptable chance a healthy stream falsely fires "
                         "within the horizon.")
    ap.add_argument("--min-detection", type=float, default=0.80,
                    help="Required per-window detection at the collapsed rate.")
    ap.add_argument("--max-ci-halfwidth", type=float, default=0.03,
                    help="Max acceptable Wilson half-width for retrain_after_n.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="watcher_kpi_findings.json")
    args = ap.parse_args()

    pol = load_policies(args.policies)
    cfg = Config(args)
    rng = np.random.default_rng(args.seed)

    # Resolve the baseline hit rate — the load-bearing assumption.
    if args.baseline_hit_rate is not None:
        baseline_p = args.baseline_hit_rate
        p_src = "explicit --baseline-hit-rate"
    elif args.pool_size is not None:
        baseline_p = args.pool_size / args.draw_space
        p_src = f"pool model ({args.pool_size}/{args.draw_space})"
    else:
        baseline_p = get(pol, "convergence_targets", "minimum_hit_rate", 0.05)
        p_src = "convergence_targets.minimum_hit_rate"
    collapsed_p = max(baseline_p * args.collapse_factor, 1e-6)

    print("=" * 74)
    print("WATCHER KPI ANALYTIC RESOLVABILITY PROBE")
    print("=" * 74)
    print(f"Baseline hit rate p = {baseline_p:g}   (source: {p_src})")
    print(f"Collapsed alt rate  = {collapsed_p:g}   (baseline x {args.collapse_factor})")
    print(f"MC: {args.mc_trials} streams x {args.mc_stream_len} draws | "
          f"accept FP<{args.max_fp_rate}, detection>{args.min_detection}")
    print("!! Verdicts are only as valid as the hit-rate definition above. "
          "Confirm it against source.")
    print("-" * 74)

    findings: List[Dict] = []
    findings.append(analyse_hit_rate_collapse(
        get(pol, "retrain_triggers", "hit_rate_collapse_threshold", 0.01),
        get(pol, "retrain_triggers", "hit_rate_collapse_window", 20),
        baseline_p, collapsed_p, cfg, rng))
    findings.append(analyse_consecutive_misses(
        get(pol, "retrain_triggers", "max_consecutive_misses", 5),
        baseline_p, collapsed_p, cfg, rng))
    findings.append(analyse_retrain_after_n(
        get(pol, "retrain_triggers", "retrain_after_n_draws", 10),
        baseline_p, cfg))
    findings.append(analyse_minimum_hit_rate(
        get(pol, "convergence_targets", "minimum_hit_rate", 0.05),
        args.pool_size, args.draw_space, args.eval_draws, cfg))
    findings.extend(report_needs_data())

    # ── Report ────────────────────────────────────────────────────────────
    for r in findings:
        print(f"\n[{r['verdict']}]  {r['kpi']}")
        if "configured" in r:
            print(f"    configured: {r['configured']}")
        for k in ("fp_rate_per_window_healthy",
                  "mc_frac_clean_within_horizon",
                  "expected_draws_to_false_fire_healthy_analytic",
                  "mc_median_draws_to_false_fire",
                  "detection_per_window_at_collapsed_rate",
                  "wilson_95_halfwidth_at_baseline",
                  "chance_baseline_hit_rate", "margin_over_chance"):
            if k in r:
                print(f"    {k}: {r[k]}")
        print(f"    -> {r['note']}")

    out = {
        "probe": "watcher_kpi_resolvability_v1",
        "assumptions": {
            "baseline_hit_rate": baseline_p,
            "baseline_source": p_src,
            "collapsed_rate": collapsed_p,
            "pool_size": args.pool_size,
            "draw_space": args.draw_space,
            "acceptance": {"max_fp_rate": args.max_fp_rate,
                           "min_detection": args.min_detection,
                           "max_ci_halfwidth": args.max_ci_halfwidth},
        },
        "findings": findings,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + "-" * 74)
    print(f"[probe] Findings written to {args.out}")
    print("[probe] Read-only: no policy value was changed. Route recommended "
          "changes through THRESHOLD_GOVERNANCE + Team Beta.")


if __name__ == "__main__":
    main()
