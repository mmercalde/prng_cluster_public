#!/usr/bin/env python3
"""
Metric-C deterministic KPI analysis (S176 — WATCHER KPI validation, Task 2b).

Metric C is the quantity the LIVE WATCHER retrain triggers actually threshold:

    current_hit_rate = exact_hits / pool_size      (chapter_13_diagnostics.py:531)

where exact_hits = count of stage-6 pool predictions exactly equal to the single
actual draw (compute_prediction_validation, chapter_13_diagnostics.py:242-249),
so per-draw exact_hits in {0,1} for a distinct-valued pool. With the production
stage-6 pool (prediction_generator_config.json: pool_size=20, k=10) this makes
current_hit_rate a DETERMINISTIC two-valued quantity per draw:

    hit  draw -> 1 / pool_size
    miss draw -> 0

Because it is deterministic given pool_size, NO Monte Carlo is needed. This tool
derives, in closed form, exactly what each of the three retrain KPIs resolves to
against metric C, and writes a findings JSON.

Read-only: writes only its findings file. Recommends only; changes nothing.
"""
import argparse
import json
from datetime import datetime, timezone


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def analyze(pool_size: int, k: int, draw_space: int,
            collapse_threshold: float, max_misses: int) -> dict:
    # Per-draw metric C takes exactly two values.
    hit_value = 1.0 / max(pool_size, 1)      # exact_hits=1
    miss_value = 0.0                          # exact_hits=0
    # Chance probability the single actual draw lands in a size-`pool_size`
    # pool over `draw_space` outcomes = pool_size / draw_space (one actual number).
    p_hit_chance = pool_size / draw_space

    # ---- KPI 1: hit_rate_collapse (fires if current_hit_rate < threshold) ----
    fires_on_hit = hit_value < collapse_threshold
    fires_on_miss = miss_value < collapse_threshold   # 0 < threshold -> always True for threshold>0
    # The window param (hit_rate_collapse_window) is NOT read by the trigger:
    # chapter_13_triggers.py:269 compares a single instantaneous snapshot.
    if fires_on_hit and fires_on_miss:
        collapse_verdict = "DECORATIVE-ALWAYS"   # fires every draw regardless of outcome
        collapse_note = (
            f"pool_size={pool_size} > {int(1/collapse_threshold)} => hit-draw value "
            f"{hit_value:.4f} is itself < {collapse_threshold}; trigger fires on EVERY "
            f"draw including hits. Carries zero information.")
    elif fires_on_miss and not fires_on_hit:
        collapse_verdict = "DECORATIVE-FIRES-ON-EVERY-MISS"
        collapse_note = (
            f"hit-draw value {hit_value:.4f} >= {collapse_threshold} (no fire); "
            f"miss-draw value 0.0 < {collapse_threshold} (fires). The trigger is an "
            f"exact re-encoding of the raw per-draw miss boolean — it detects 'did this "
            f"one draw miss?', not a rate collapse. Under healthy chance operation "
            f"(P(hit)={p_hit_chance:.3f}) it fires on ~{(1-p_hit_chance)*100:.1f}% of draws.")
    else:
        collapse_verdict = "RESOLVABLE"
        collapse_note = "hit and miss draws straddle the threshold in an informative way."

    # ---- KPI 2: max_consecutive_misses (fires if run of misses >= max_misses) ----
    # Deterministic framing: a "miss" is the DOMINANT per-draw outcome under healthy
    # chance operation. Mean gap between hits = 1/p_hit draws; a run of `max_misses`
    # misses is expected unless a hit interrupts it.
    mean_gap_between_hits = 1.0 / p_hit_chance if p_hit_chance > 0 else float("inf")
    # P(a given hit-to-hit gap contains a run of >= max_misses) ~ 1 when mean gap >> max_misses.
    # Closed-form expected draws until first run of M misses, each miss w.p. q=1-p:
    q = 1.0 - p_hit_chance
    p = p_hit_chance
    expected_draws_to_run = ((1.0 - q ** max_misses) / (p * q ** max_misses)) if (p > 0 and q > 0) else float("inf")
    misses_verdict = ("DECORATIVE-FIRES-UNDER-HEALTHY"
                      if mean_gap_between_hits > max_misses else "RESOLVABLE")
    misses_note = (
        f"Under healthy chance operation P(hit)={p_hit_chance:.3f}, mean gap between "
        f"hits = {mean_gap_between_hits:.1f} draws >> max_consecutive_misses={max_misses}. "
        f"A run of {max_misses} misses is the NORM, not an anomaly. Closed-form expected "
        f"draws to first {max_misses}-miss run = {expected_draws_to_run:.1f}. The trigger "
        f"fires routinely on a healthy stream, so it cannot separate healthy from degraded.")

    # ---- KPI 3: minimum_hit_rate — UNWIRED ----
    minhr_verdict = "UNWIRED-NO-RUNTIME-CONSUMER"
    minhr_note = (
        "grep is definitive: minimum_hit_rate has ZERO runtime consumers "
        "(no trigger, no flag, no diagnostic reads it). It appears only in "
        "watcher_policies.json convergence_targets and in these KPI tools. "
        f"Numerically it (0.05) equals metric C's hit-draw value at pool_size=20 "
        f"(1/20={1/20:.3f}), but nothing compares against it.")

    return {
        "probe": {
            "name": "watcher_kpi_metricC_deterministic",
            "run_id": iso_now(),
            "method": "closed-form / deterministic (no Monte Carlo)",
        },
        "metric_C": {
            "definition": "current_hit_rate = exact_hits / pool_size",
            "source": "chapter_13_diagnostics.py:531 (built from :242-249)",
            "pool_source": "prediction_generator_config.json (pool_size, k)",
            "pool_size": pool_size,
            "k_predictions": k,
            "draw_space": draw_space,
            "per_draw_values": {"hit": round(hit_value, 4), "miss": miss_value},
            "chance_hit_probability": round(p_hit_chance, 4),
            "note": ("Metric C is two-valued and deterministic given pool_size; the "
                     "hit EVENT (exact_hits>=1) is the same event that Top-20 recall "
                     "(metric A) measures as a rate, but C is evaluated per-draw and "
                     "instantaneously by the triggers."),
        },
        "findings": [
            {
                "kpi": "retrain_triggers.hit_rate_collapse_threshold",
                "configured": {"threshold": collapse_threshold, "window_UNUSED": 20},
                "consumer": "chapter_13_triggers.py:269-276",
                "fires_on_hit_draw": fires_on_hit,
                "fires_on_miss_draw": fires_on_miss,
                "verdict": collapse_verdict,
                "note": collapse_note,
            },
            {
                "kpi": "retrain_triggers.max_consecutive_misses",
                "configured": {"run_len": max_misses},
                "consumer": "chapter_13_triggers.py:247-254 (miss = exact_hits==0, :544-547)",
                "mean_gap_between_hits_draws": round(mean_gap_between_hits, 2),
                "expected_draws_to_first_run": round(expected_draws_to_run, 2),
                "verdict": misses_verdict,
                "note": misses_note,
            },
            {
                "kpi": "convergence_targets.minimum_hit_rate",
                "configured": {"minimum_hit_rate": 0.05},
                "consumer": None,
                "verdict": minhr_verdict,
                "note": minhr_note,
            },
        ],
        "recommend_only": ("Read-only. No policy value changed. Route any change "
                           "through THRESHOLD_GOVERNANCE + Team Beta."),
    }


def main():
    ap = argparse.ArgumentParser(description="Metric-C deterministic KPI analysis (no MC).")
    ap.add_argument("--pool-size", type=int, default=20,
                    help="Stage-6 prediction pool size (prediction_generator_config.json).")
    ap.add_argument("--k", type=int, default=10, help="k predictions generated.")
    ap.add_argument("--draw-space", type=int, default=1000, help="CA Daily 3 = 1000.")
    ap.add_argument("--collapse-threshold", type=float, default=0.01)
    ap.add_argument("--max-misses", type=int, default=5)
    ap.add_argument("--out", default="watcher_kpi_metricC_deterministic_findings.json")
    args = ap.parse_args()

    result = analyze(args.pool_size, args.k, args.draw_space,
                     args.collapse_threshold, args.max_misses)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[metricC] pool_size={args.pool_size} k={args.k} draw_space={args.draw_space}")
    for fnd in result["findings"]:
        print(f"  [{fnd['verdict']}] {fnd['kpi']}")
    print(f"[metricC] Findings written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
