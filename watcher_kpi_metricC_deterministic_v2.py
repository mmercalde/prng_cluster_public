#!/usr/bin/env python3
"""
Metric-C deterministic KPI analysis — v2 (S176 follow-up, per TB ruling §8).

v2 applies all twelve Team Beta §8 corrections to v1
(`watcher_kpi_metricC_deterministic.py`, kept untouched for the audit trail):

  1  validate pool_size > 0
  2  validate draw_space > 0
  3  validate max_misses >= 1
  4  no hardcoded window value (the live trigger does not read the window at all)
  5  no hardcoded minimum_hit_rate
  6  policy values come from --policies OR explicit flags; FAIL LOUDLY otherwise
     (no silent policy defaults)
  7  `chance_hit_probability` renamed -> `uniform_null_hit_probability`
  8  `assumed_healthy_hit_rate` kept strictly separate from the uniform null
  9  the unique-pool-size assumption is asserted and reported
 10  the verdict criterion is a defined false-alarm-horizon / expected-waiting-time
     test (`--false-alarm-horizon`), NOT `mean_gap > max_misses`
 11  output states that only TWO live triggers consume metric C, and
     `minimum_hit_rate` is a configured target, not a live trigger
 12  Metric-A and Metric-C are described as complementary views of the SAME
     Bernoulli event, not independent evidence

WORDING DISCIPLINE (TB §3): every quantitative finding is stated **at the uniform
random null** (Hit@K = K/draw_space). This tool makes NO "healthy TFM" claim —
TFM has no measured empirical baseline yet (TB §3, §4.3).

METRIC C (what the live retrain triggers actually threshold):
    current_hit_rate = exact_hits / pool_size          (chapter_13_diagnostics.py:531)
Two live consumers only (chapter_13_triggers.py):
    - hit_rate_collapse_threshold   (instantaneous, per-draw)
    - max_consecutive_misses        (run length of exact_hits==0 draws)
`minimum_hit_rate` is a convergence TARGET with zero runtime consumers
(confirmed S176-followup 1b on tree 0c3166a) — reported, never treated as a trigger.

Read-only: writes only its findings file. Recommends only; changes nothing.
"""
import argparse
import json
import math
from datetime import datetime, timezone


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------------
# Policy loading — no silent defaults for policy values (TB §8.4, §8.5, §8.6)
# ----------------------------------------------------------------------------
def resolve_policy_values(args) -> dict:
    """Resolve the two live-trigger policy values from --policies or explicit
    flags. Fail loudly if a required value is available from neither source."""
    pol = {}
    if args.policies:
        with open(args.policies, "r", encoding="utf-8") as f:
            raw = json.load(f)
        rt = raw.get("retrain_triggers", {})
        pol["collapse_threshold"] = rt.get("hit_rate_collapse_threshold")
        pol["max_misses"] = rt.get("max_consecutive_misses")
        # minimum_hit_rate is a TARGET, not a trigger (TB §8.11) — read for
        # reporting only, never used as a firing threshold.
        pol["minimum_hit_rate_target"] = raw.get("convergence_targets", {}).get("minimum_hit_rate")
        pol["_policy_source"] = args.policies
    # explicit flags override / supply
    if args.collapse_threshold is not None:
        pol["collapse_threshold"] = args.collapse_threshold
    if args.max_misses is not None:
        pol["max_misses"] = args.max_misses

    missing = [k for k in ("collapse_threshold", "max_misses") if pol.get(k) is None]
    if missing:
        raise SystemExit(
            f"[FATAL] No value for {missing}. Supply --policies <watcher_policies.json> "
            f"or explicit --collapse-threshold / --max-misses. This tool refuses to "
            f"invent policy defaults (TB §8.6).")
    return pol


def expected_draws_to_miss_run(p_miss: float, m: int) -> float:
    """Closed-form expected number of draws until the first run of `m` misses,
    where each draw is a miss with probability p_miss (Bernoulli). Standard
    run-length waiting time: E[T_m] = (1 - q^m) / (p * q^m), q=p_miss, p=1-q."""
    q = p_miss
    p = 1.0 - q
    if p <= 0:
        return float("inf")   # never a hit => run reached at draw m, but degenerate
    if q <= 0:
        return float("inf")   # never a miss => run never occurs
    return (1.0 - q ** m) / (p * q ** m)


def analyze(pool_size, k, draw_space, pol, false_alarm_horizon,
            assumed_healthy_hit_rate) -> dict:
    # ---- structural validation (TB §8.1-8.3) ----
    if pool_size <= 0:
        raise SystemExit("[FATAL] pool_size must be > 0 (TB §8.1).")
    if draw_space <= 0:
        raise SystemExit("[FATAL] draw_space must be > 0 (TB §8.2).")
    max_misses = int(pol["max_misses"])
    if max_misses < 1:
        raise SystemExit("[FATAL] max_consecutive_misses must be >= 1 (TB §8.3).")
    collapse_threshold = float(pol["collapse_threshold"])
    if false_alarm_horizon <= 0:
        raise SystemExit("[FATAL] --false-alarm-horizon must be > 0.")

    # ---- metric C values (unique-pool-size assumption, TB §8.9) ----
    # exact_hits in {0,1} PER DRAW *iff* the pool contains distinct numbers; then
    # current_hit_rate is two-valued. If the pool held duplicate predictions equal
    # to the actual draw, exact_hits could exceed 1. This assumption is asserted
    # and reported, not silently assumed.
    unique_pool_assumption = (
        "current_hit_rate is two-valued {0, 1/pool_size} ONLY under the "
        "distinct-prediction assumption (exact_hits in {0,1} per draw). Duplicate "
        "pool entries equal to the actual draw would make exact_hits > 1. Verify "
        "pool distinctness in compute_prediction_validation output before relying "
        "on the two-valued form.")
    hit_value = 1.0 / pool_size          # exact_hits = 1
    miss_value = 0.0                      # exact_hits = 0

    # ---- the uniform random null (TB §3, §8.7) ----
    # NOT a healthy-TFM rate. Hit@pool_size under the uniform null = pool_size/draw_space.
    uniform_null_hit_probability = pool_size / draw_space
    p_miss_null = 1.0 - uniform_null_hit_probability

    # ---- KPI 1: hit_rate_collapse — instantaneous, per-draw ----
    fires_on_hit = hit_value < collapse_threshold
    fires_on_miss = miss_value < collapse_threshold
    if fires_on_hit and fires_on_miss:
        per_draw_fire_prob_at_null = 1.0                 # fires every draw
        collapse_shape = "fires on EVERY draw (hit-draw value already below threshold)"
    elif fires_on_miss and not fires_on_hit:
        per_draw_fire_prob_at_null = p_miss_null         # fires exactly when the draw misses
        collapse_shape = "exact re-encoding of the per-draw miss boolean"
    else:
        per_draw_fire_prob_at_null = 0.0
        collapse_shape = "does not fire at the tested values"
    # false-alarm-horizon test (TB §8.10): expected draws to first false fire AT NULL
    exp_draws_to_collapse_fire = (1.0 / per_draw_fire_prob_at_null
                                  if per_draw_fire_prob_at_null > 0 else float("inf"))
    collapse_fires_within_horizon = exp_draws_to_collapse_fire <= false_alarm_horizon

    # ---- KPI 2: max_consecutive_misses — run-length ----
    exp_draws_to_miss_run = expected_draws_to_miss_run(p_miss_null, max_misses)
    misses_fire_within_horizon = exp_draws_to_miss_run <= false_alarm_horizon

    # ---- optional: same arithmetic at an ASSUMED healthy rate (kept SEPARATE, §8.8) ----
    assumed_block = None
    if assumed_healthy_hit_rate is not None:
        a = float(assumed_healthy_hit_rate)
        if not (0.0 <= a <= 1.0):
            raise SystemExit("[FATAL] --assumed-healthy-hit-rate must be in [0,1].")
        p_miss_a = 1.0 - a
        assumed_block = {
            "_caveat": ("ASSUMPTION, not a measured baseline and not the null. TFM has "
                        "no empirical baseline yet (TB §3/§4.3). Provided only to show "
                        "sensitivity; do NOT read as 'healthy TFM'."),
            "assumed_healthy_hit_rate": a,
            "hit_rate_collapse_exp_draws_to_fire": (round(1.0 / p_miss_a, 3)
                                                    if p_miss_a > 0 else None),
            "max_consecutive_misses_exp_draws_to_run": round(
                expected_draws_to_miss_run(p_miss_a, max_misses), 3),
        }

    return {
        "probe": {
            "name": "watcher_kpi_metricC_deterministic_v2",
            "run_id": iso_now(),
            "method": "closed-form / deterministic (no Monte Carlo)",
            "wording_discipline": ("All findings stated at the UNIFORM RANDOM NULL "
                                   "(Hit@K = K/draw_space). No healthy-TFM claim is made; "
                                   "TFM has no measured empirical baseline (TB §3)."),
            "live_trigger_scope": ("Metric C is consumed by exactly TWO live triggers "
                                   "(hit_rate_collapse, max_consecutive_misses; "
                                   "chapter_13_triggers.py). minimum_hit_rate is a "
                                   "convergence TARGET, not a live trigger, with zero "
                                   "runtime consumers (S176-followup 1b, tree 0c3166a)."),
            "metric_a_c_relationship": ("Metric A (Top-K recall rate over draws) and "
                                        "Metric C (per-draw exact_hits/pool_size) are "
                                        "COMPLEMENTARY VIEWS OF THE SAME Bernoulli event "
                                        "(actual draw in pool, p=pool_size/draw_space); "
                                        "they are not independent evidence."),
        },
        "inputs": {
            "pool_size": pool_size,
            "k_predictions": k,
            "draw_space": draw_space,
            "false_alarm_horizon_draws": false_alarm_horizon,
            "policy_source": pol.get("_policy_source", "explicit-flags"),
            "collapse_threshold": collapse_threshold,
            "max_consecutive_misses": max_misses,
            "minimum_hit_rate_target_reported_only": pol.get("minimum_hit_rate_target"),
        },
        "metric_C": {
            "definition": "current_hit_rate = exact_hits / pool_size",
            "source": "chapter_13_diagnostics.py:531 (from :242-249)",
            "per_draw_values": {"hit": round(hit_value, 6), "miss": miss_value},
            "uniform_null_hit_probability": round(uniform_null_hit_probability, 6),
            "unique_pool_size_assumption": unique_pool_assumption,
        },
        "findings": [
            {
                "kpi": "retrain_triggers.hit_rate_collapse_threshold",
                "consumer": "chapter_13_triggers.py:269-276 (reads one instantaneous snapshot)",
                "configured_threshold": collapse_threshold,
                "window_read_by_trigger": False,   # §8.4: the live trigger does not read the window
                "shape_at_null": collapse_shape,
                "per_draw_fire_probability_at_uniform_null": round(per_draw_fire_prob_at_null, 4),
                "expected_draws_to_first_false_fire_at_null": (
                    round(exp_draws_to_collapse_fire, 4)
                    if exp_draws_to_collapse_fire != float("inf") else None),
                "false_alarm_horizon_draws": false_alarm_horizon,
                "fires_within_horizon_at_null": collapse_fires_within_horizon,
                "verdict": ("FIRES-WITHIN-HORIZON-AT-NULL" if collapse_fires_within_horizon
                            else "WITHIN-HORIZON-SAFE-AT-NULL"),
            },
            {
                "kpi": "retrain_triggers.max_consecutive_misses",
                "consumer": "chapter_13_triggers.py:247-254 (miss = exact_hits==0, :544-547)",
                "configured_run_len": max_misses,
                "expected_draws_to_first_run_at_null": round(exp_draws_to_miss_run, 4),
                "false_alarm_horizon_draws": false_alarm_horizon,
                "fires_within_horizon_at_null": misses_fire_within_horizon,
                "verdict": ("FIRES-WITHIN-HORIZON-AT-NULL" if misses_fire_within_horizon
                            else "WITHIN-HORIZON-SAFE-AT-NULL"),
            },
            {
                "kpi": "convergence_targets.minimum_hit_rate",
                "consumer": None,
                "runtime_consumers_on_tree_0c3166a": 0,
                "verdict": "NOT-A-LIVE-TRIGGER (configured target; deprecation precondition met)",
                "note": ("Reported for completeness only. It is a convergence target, not "
                         "a firing threshold; the analyzer never compares against it."),
            },
        ],
        "assumed_healthy_sensitivity": assumed_block,
        "recommend_only": ("Read-only. No policy value changed. Route any change through "
                           "THRESHOLD_GOVERNANCE + Team Beta. Verdicts are stated at the "
                           "uniform null and are NOT threshold recommendations."),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Metric-C deterministic KPI analysis v2 (no MC; null-stated; TB §8).")
    # structural (validated; draw_space default documented as CA Daily 3)
    ap.add_argument("--pool-size", type=int, required=True,
                    help="Stage-6 prediction pool size (prediction_generator_config.json). Validated > 0.")
    ap.add_argument("--k", type=int, default=None, help="k predictions generated (reporting only).")
    ap.add_argument("--draw-space", type=int, default=1000,
                    help="Distinct outcomes. Default 1000 = CA Daily 3 (documented). Validated > 0.")
    # policy — no silent defaults (TB §8.6); supply via --policies or explicit flags
    ap.add_argument("--policies", default=None,
                    help="watcher_policies.json — source of hit_rate_collapse_threshold "
                         "and max_consecutive_misses (and minimum_hit_rate for reporting).")
    ap.add_argument("--collapse-threshold", type=float, default=None,
                    help="Explicit override of retrain_triggers.hit_rate_collapse_threshold.")
    ap.add_argument("--max-misses", type=int, default=None,
                    help="Explicit override of retrain_triggers.max_consecutive_misses.")
    # false-alarm-horizon test (TB §8.10) — explicit, documented default
    ap.add_argument("--false-alarm-horizon", type=int, default=1000,
                    help="Operator's acceptable draws-before-a-false-alarm horizon at the "
                         "uniform null. A trigger whose expected draws-to-first-false-fire "
                         "is <= this horizon is flagged FIRES-WITHIN-HORIZON-AT-NULL. "
                         "Default 1000 draws (~1.4 yr of twice-daily CA D3); documented, "
                         "not load-bearing — set to your governance horizon.")
    # assumed-healthy rate, kept strictly separate from the null (TB §8.8)
    ap.add_argument("--assumed-healthy-hit-rate", type=float, default=None,
                    help="OPTIONAL sensitivity assumption, kept separate from the null. "
                         "NOT a measured baseline; TFM has none yet.")
    ap.add_argument("--out", default="watcher_kpi_metricC_v2_findings.json")
    args = ap.parse_args()

    pol = resolve_policy_values(args)
    result = analyze(args.pool_size, args.k, args.draw_space, pol,
                     args.false_alarm_horizon, args.assumed_healthy_hit_rate)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[metricC-v2] pool_size={args.pool_size} draw_space={args.draw_space} "
          f"null_hit_p={result['metric_C']['uniform_null_hit_probability']} "
          f"horizon={args.false_alarm_horizon}")
    print(f"[metricC-v2] policy source: {result['inputs']['policy_source']}  "
          f"collapse_threshold={pol['collapse_threshold']}  max_misses={pol['max_misses']}")
    for fnd in result["findings"]:
        print(f"  [{fnd['verdict']}] {fnd['kpi']}")
    print(f"[metricC-v2] Findings written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
