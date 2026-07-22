#!/usr/bin/env python3
"""
Metric-C deterministic KPI analysis — v2.2 (S178 resubmission, per TB ruling §9-§11).

v2.2 applies the FIVE Team Beta §11-analyzer corrections on top of v2.1
(`watcher_kpi_metricC_deterministic_v2_1.py`, kept untouched for the audit trail).
v2.1 already carried the twelve S176 §8 corrections + the six §4 edge fixes; v2.2
adds ONLY the two reproducible defects TB flagged in §9-§10:

  1  THRESHOLD-SHAPE-AWARE assumed-rate sensitivity (TB §9). The optional
     assumed_healthy_sensitivity block previously computed the hit_rate_collapse
     waiting time as `1 / p_miss` unconditionally — i.e. as though the collapse
     trigger always fires on misses only. That contradicts the primary analysis,
     which recognises three shapes (fires on neither / on misses only / on every
     draw). v2.2 reuses the SAME (fires_on_hit, fires_on_miss) shape the primary
     KPI derives, so the assumed-rate block agrees with the configured trigger:
         fires on hit AND miss  -> p = 1.0            (fires every draw; wait 1)
         fires on miss only     -> p = 1 - assumed_hit_rate
         fires on neither       -> p = 0.0            (never fires; null)
     collapse_threshold = 0 now reports never-fires; collapse_threshold = 1 now
     reports wait = 1 draw at ANY assumed rate (TB §9 contradiction cases).

  2  EXPLICIT-ROOT PROVENANCE (TB §10). v2.1 recorded `git rev-parse HEAD` against
     the process working directory, so it could record `null` (outside a repo) or
     an UNRELATED repo's commit. v2.2 takes `--repo-root <path>` and resolves:
         git -C <repo-root> rev-parse HEAD
         git -C <repo-root> status --porcelain
     and records analyzed_repo_root, analyzed_source_commit, analyzed_tree_dirty,
     policy_file_path, policy_file_sha256, analyzer_file_sha256.

  3  FAIL-FATAL provenance (TB §10/§11.4): for an authoritative run, failure to
     resolve the repository commit is FATAL — never a silent null. An explicitly
     non-authoritative mode is available (`--no-provenance`) and marks the findings
     `authoritative = false` rather than fabricating a commit.

  4  REJECT BOOLEAN collapse_threshold (TB §10). `float(True)` silently yielded
     1.0; v2.2 rejects a bool collapse_threshold before coercion (bool is a subclass
     of int, so an explicit isinstance(x, bool) guard is required).

WORDING DISCIPLINE (TB §3): every quantitative finding is stated **at the uniform
random null** (Hit@K = K/draw_space). This tool makes NO "healthy TFM" claim —
TFM has no measured empirical baseline yet (TB §3, §4.3).

METRIC C (what the live retrain triggers actually threshold):
    current_hit_rate = exact_hits / pool_size          (chapter_13_diagnostics.py:531)
Two live consumers only (chapter_13_triggers.py):
    - hit_rate_collapse_threshold   (instantaneous, per-draw)  :269-276
    - max_consecutive_misses        (run length of exact_hits==0 draws)  :247-254
`minimum_hit_rate` is a convergence TARGET with zero runtime consumers
(confirmed S176-followup 1b on tree 0c3166a) — reported, never treated as a trigger.

Read-only: writes only its findings file. Recommends only; changes nothing.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------------
# Provenance (TB §10) — resolved from an EXPLICIT repo root, never the cwd.
# ----------------------------------------------------------------------------
def _sha256_file(path) -> str:
    """Streaming sha256 of a file; None when the path is missing/unreadable."""
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(repo_root, git_args):
    """Run `git -C <repo_root> <args>` read-only. Returns stdout (stripped) on
    success, or None on ANY failure (so callers can distinguish an empty-but-valid
    result "" from a hard failure None)."""
    try:
        out = subprocess.run(
            ["git", "-C", repo_root, *git_args],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def resolve_provenance(repo_root, policy_path, analyzer_path, authoritative) -> dict:
    """TB §10: bind the findings to a SPECIFIC source tree via --repo-root, not the
    working directory. For an authoritative run, an unresolvable commit is FATAL
    (no silent null); --no-provenance selects an explicitly non-authoritative run
    that records nulls and marks authoritative=false."""
    prov = {
        "analyzed_repo_root": os.path.abspath(repo_root) if repo_root else None,
        "analyzed_source_commit": None,
        "analyzed_tree_dirty": None,
        "policy_file_path": os.path.abspath(policy_path) if policy_path else None,
        "policy_file_sha256": _sha256_file(policy_path),
        "analyzer_file_sha256": _sha256_file(analyzer_path),
        "authoritative": bool(authoritative),
    }
    if not authoritative:
        # Explicitly non-authoritative: do not resolve, do not fatal (TB §11.4).
        return prov
    if not repo_root:
        raise SystemExit(
            "[FATAL] --repo-root is required for an authoritative run so provenance "
            "identifies WHICH TFM tree was analyzed (not the cwd). Pass --repo-root "
            "<path>, or --no-provenance to mark findings authoritative=false (TB §10).")
    if not os.path.isdir(repo_root):
        raise SystemExit(f"[FATAL] --repo-root {repo_root!r} is not a directory (TB §10).")
    commit = _git(repo_root, ["rev-parse", "HEAD"])
    if not commit:
        raise SystemExit(
            f"[FATAL] could not resolve HEAD at --repo-root {repo_root!r} — not a git "
            f"repository (refusing to write a silent null commit) (TB §10/§11.4).")
    porcelain = _git(repo_root, ["status", "--porcelain"])
    if porcelain is None:
        raise SystemExit(
            f"[FATAL] could not read `git status` at --repo-root {repo_root!r} (TB §10).")
    prov["analyzed_source_commit"] = commit
    prov["analyzed_tree_dirty"] = bool(porcelain.strip())
    return prov


def waiting_time_field(value):
    """Map a waiting-time float to (json_safe_value, status) for strict JSON
    output (TB §4.C). Infinite / NaN / None -> (None, <status>)."""
    if value is None:
        return None, "unavailable"
    if isinstance(value, float) and math.isnan(value):
        return None, "unavailable_nan"
    if isinstance(value, float) and math.isinf(value):
        return None, "infinite_never_fires"
    return round(value, 4), "finite"


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


def _validate_exact_integer(value, name: str) -> int:
    """TB §4.E: accept only an EXACT integer. Reject bool, NaN/inf, and any
    float with a fractional part (e.g. 5.5)."""
    if isinstance(value, bool):
        raise SystemExit(f"[FATAL] {name} must be an exact integer, got bool (TB §4.E).")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value) or not value.is_integer():
            raise SystemExit(
                f"[FATAL] {name} must be an exact integer, got {value!r} (TB §4.E).")
        return int(value)
    raise SystemExit(f"[FATAL] {name} must be an exact integer, got {type(value).__name__} (TB §4.E).")


def _validate_rate(value, name: str) -> float:
    """TB §10: reject a Boolean where a real rate is expected (bool is an int
    subclass, so float(True) == 1.0 would otherwise pass silently), then require a
    finite value within the supported rate range [0, 1]."""
    if isinstance(value, bool):
        raise SystemExit(
            f"[FATAL] {name} must be a real number in [0,1], got bool {value!r}; "
            f"float(True)=1.0 must NOT pass silently (TB §10).")
    coerced = float(value)
    if not math.isfinite(coerced):
        raise SystemExit(f"[FATAL] {name} must be finite (TB §4.E).")
    if not (0.0 <= coerced <= 1.0):
        raise SystemExit(
            f"[FATAL] {name} must be within the supported rate range [0,1] (TB §4.E).")
    return coerced


def expected_draws_to_miss_run(p_miss: float, m: int) -> float:
    """Closed-form expected number of draws until the first run of `m` misses,
    where each draw is a miss with probability p_miss (Bernoulli). Standard
    run-length waiting time: E[T_m] = (1 - q^m) / (p * q^m), q=p_miss, p=1-q.

    TB §4.B degenerate cases (exact):
        q == 1  (p_miss = 1, hit rate 0): every draw misses, so the first run of
                m misses is reached deterministically AT draw m -> return m.
        q == 0  (p_miss = 0, hit rate 1): a miss never occurs, so the run never
                happens -> return +inf.
    """
    q = p_miss
    if q >= 1.0:
        return float(m)          # never a hit => run reached exactly at draw m (TB §4.B)
    if q <= 0.0:
        return float("inf")      # never a miss => run never occurs
    p = 1.0 - q
    return (1.0 - q ** m) / (p * q ** m)


def analyze(pool_size, k, draw_space, pol, fire_horizon,
            assumed_healthy_hit_rate, provenance) -> dict:
    # ---- structural validation (TB §8.1-8.3 + §4.A) ----
    if pool_size <= 0:
        raise SystemExit("[FATAL] pool_size must be > 0 (TB §8.1).")
    if draw_space <= 0:
        raise SystemExit("[FATAL] draw_space must be > 0 (TB §8.2).")
    if pool_size > draw_space:
        raise SystemExit(
            f"[FATAL] pool_size ({pool_size}) cannot exceed draw_space ({draw_space}); "
            f"the null probability would exceed 1 (TB §4.A).")
    # TB §4.E: max_consecutive_misses must be an exact integer, then >= 1.
    max_misses = _validate_exact_integer(pol["max_misses"], "max_consecutive_misses")
    if max_misses < 1:
        raise SystemExit("[FATAL] max_consecutive_misses must be >= 1 (TB §8.3).")
    # TB §10: reject a Boolean collapse_threshold; then require finite, within [0,1].
    collapse_threshold = _validate_rate(pol["collapse_threshold"], "collapse_threshold")
    if fire_horizon <= 0:
        raise SystemExit("[FATAL] --fire-horizon must be > 0.")

    # ---- metric C values (unique-pool-size assumption, TB §8.9) ----
    unique_pool_assumption = (
        "current_hit_rate is two-valued {0, 1/pool_size} ONLY under the "
        "distinct-prediction assumption (exact_hits in {0,1} per draw). Duplicate "
        "pool entries equal to the actual draw would make exact_hits > 1. Verify "
        "pool distinctness in compute_prediction_validation output before relying "
        "on the two-valued form.")
    hit_value = 1.0 / pool_size          # exact_hits = 1
    miss_value = 0.0                      # exact_hits = 0

    # ---- the uniform random null (TB §3, §8.7) ----
    uniform_null_hit_probability = pool_size / draw_space
    p_miss_null = 1.0 - uniform_null_hit_probability

    # ---- KPI 1: hit_rate_collapse — instantaneous, per-draw ----
    # The trigger SHAPE (which per-draw outcomes push current_hit_rate below the
    # threshold) is a property of pool_size + collapse_threshold ALONE — it does not
    # depend on the hit RATE. Both the null verdict and the assumed-rate sensitivity
    # (TB §9) must therefore use these same two booleans.
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
    # fire-horizon test (TB §8.10 / §4.D): expected draws to first fire AT the uniform null.
    exp_draws_to_collapse_fire = (1.0 / per_draw_fire_prob_at_null
                                  if per_draw_fire_prob_at_null > 0 else float("inf"))
    collapse_fires_within_horizon = exp_draws_to_collapse_fire <= fire_horizon
    collapse_val, collapse_status = waiting_time_field(exp_draws_to_collapse_fire)

    # ---- KPI 2: max_consecutive_misses — run-length ----
    exp_draws_to_miss_run = expected_draws_to_miss_run(p_miss_null, max_misses)
    misses_fire_within_horizon = exp_draws_to_miss_run <= fire_horizon
    miss_val, miss_status = waiting_time_field(exp_draws_to_miss_run)

    # ---- optional: same arithmetic at an ASSUMED healthy rate (kept SEPARATE, §8.8) ----
    # TB §9 FIX: apply the SAME trigger shape as KPI 1 instead of assuming fire-on-miss.
    assumed_block = None
    if assumed_healthy_hit_rate is not None:
        a = _validate_rate(assumed_healthy_hit_rate, "--assumed-healthy-hit-rate")
        p_miss_a = 1.0 - a
        if fires_on_hit and fires_on_miss:
            collapse_fire_prob_a = 1.0            # fires every draw regardless of rate
        elif fires_on_miss and not fires_on_hit:
            collapse_fire_prob_a = p_miss_a       # fires only on a miss -> rate-dependent
        else:
            collapse_fire_prob_a = 0.0            # never fires (e.g. threshold = 0)
        collapse_a = (1.0 / collapse_fire_prob_a) if collapse_fire_prob_a > 0.0 else float("inf")
        collapse_a_val, collapse_a_status = waiting_time_field(collapse_a)
        miss_a = expected_draws_to_miss_run(p_miss_a, max_misses)
        miss_a_val, miss_a_status = waiting_time_field(miss_a)
        assumed_block = {
            "_caveat": ("ASSUMPTION, not a measured baseline and not the null. TFM has "
                        "no empirical baseline yet (TB §3/§4.3). Provided only to show "
                        "sensitivity; do NOT read as 'healthy TFM'."),
            "assumed_healthy_hit_rate": a,
            # TB §9: the collapse block now respects the configured threshold shape.
            "hit_rate_collapse_shape": collapse_shape,
            "hit_rate_collapse_fire_probability": round(collapse_fire_prob_a, 6),
            "hit_rate_collapse_exp_draws_to_fire": collapse_a_val,
            "hit_rate_collapse_status": collapse_a_status,
            "max_consecutive_misses_exp_draws_to_run": miss_a_val,
            "max_consecutive_misses_status": miss_a_status,
        }

    probe = {
        "name": "watcher_kpi_metricC_deterministic_v2_2",
        "run_id": iso_now(),
        "method": "closed-form / deterministic (no Monte Carlo)",
        "wording_discipline": ("All findings stated at the UNIFORM RANDOM NULL "
                               "(Hit@K = K/draw_space). No healthy-TFM claim is made; "
                               "TFM has no measured empirical baseline (TB §3). The null "
                               "is NOT a healthy baseline, so no event here is a 'false "
                               "alarm' (TB §4.D)."),
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
    }
    probe.update(provenance)   # TB §10: explicit-root provenance + authoritative flag

    return {
        "probe": probe,
        "inputs": {
            "pool_size": pool_size,
            "k_predictions": k,
            "draw_space": draw_space,
            "fire_horizon_draws": fire_horizon,
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
                "expected_draws_to_first_fire_at_uniform_null": collapse_val,   # TB §4.D rename
                "expected_draws_status": collapse_status,                       # TB §4.C
                "fire_horizon_draws": fire_horizon,
                "fires_within_horizon_at_null": collapse_fires_within_horizon,
                "verdict": ("FIRES-WITHIN-HORIZON-AT-NULL" if collapse_fires_within_horizon
                            else "WITHIN-HORIZON-SAFE-AT-NULL"),
            },
            {
                "kpi": "retrain_triggers.max_consecutive_misses",
                "consumer": "chapter_13_triggers.py:247-254 (miss = exact_hits==0)",
                "configured_run_len": max_misses,
                "expected_draws_to_first_run_at_uniform_null": miss_val,   # TB §4.D wording
                "expected_draws_status": miss_status,                      # TB §4.C
                "fire_horizon_draws": fire_horizon,
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
                           "the KPI-governance schema + Team Beta. Verdicts are stated at the "
                           "uniform null and are NOT threshold recommendations."),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Metric-C deterministic KPI analysis v2.2 (no MC; null-stated; TB §9-§11).")
    # structural (validated; draw_space default documented as CA Daily 3)
    ap.add_argument("--pool-size", type=int, required=True,
                    help="Stage-6 prediction pool size (prediction_generator_config.json). Validated 0 < pool_size <= draw_space.")
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
    # fire-horizon test (TB §8.10 / §4.D) — explicit, documented default.
    ap.add_argument("--fire-horizon", "--false-alarm-horizon", dest="fire_horizon",
                    type=int, default=1000,
                    help="Operator's acceptable draws-before-first-fire horizon at the "
                         "uniform null. A trigger whose expected draws-to-first-fire "
                         "is <= this horizon is flagged FIRES-WITHIN-HORIZON-AT-NULL. "
                         "Default 1000 draws (~1.4 yr of twice-daily CA D3); documented, "
                         "not load-bearing — set to your governance horizon. "
                         "(Deprecated alias: --false-alarm-horizon.)")
    # assumed-healthy rate, kept strictly separate from the null (TB §8.8)
    ap.add_argument("--assumed-healthy-hit-rate", type=float, default=None,
                    help="OPTIONAL sensitivity assumption, kept separate from the null. "
                         "NOT a measured baseline; TFM has none yet. Uses the same trigger "
                         "shape as the primary analysis (TB §9).")
    # provenance (TB §10) — explicit repo root, fail-fatal when authoritative.
    ap.add_argument("--repo-root", default=None,
                    help="Path to the TFM source tree whose commit this analysis is bound "
                         "to. Resolved via `git -C <root> rev-parse HEAD` + `status "
                         "--porcelain`. REQUIRED for an authoritative run (TB §10).")
    ap.add_argument("--no-provenance", action="store_true",
                    help="Explicitly non-authoritative run: skip repo resolution and mark "
                         "findings authoritative=false instead of failing (TB §11.4).")
    ap.add_argument("--out", default="watcher_kpi_metricC_v2_2_findings.json")
    args = ap.parse_args()

    pol = resolve_policy_values(args)
    provenance = resolve_provenance(
        repo_root=args.repo_root,
        policy_path=args.policies,
        analyzer_path=os.path.abspath(__file__),
        authoritative=not args.no_provenance,
    )
    result = analyze(args.pool_size, args.k, args.draw_space, pol,
                     args.fire_horizon, args.assumed_healthy_hit_rate, provenance)
    # TB §4.C: strict JSON — reject non-standard Infinity/NaN tokens.
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)

    print(f"[metricC-v2.2] pool_size={args.pool_size} draw_space={args.draw_space} "
          f"null_hit_p={result['metric_C']['uniform_null_hit_probability']} "
          f"fire_horizon={args.fire_horizon}")
    print(f"[metricC-v2.2] policy source: {result['inputs']['policy_source']}  "
          f"collapse_threshold={pol['collapse_threshold']}  max_misses={result['inputs']['max_consecutive_misses']}")
    print(f"[metricC-v2.2] authoritative={result['probe']['authoritative']}  "
          f"repo_root={result['probe']['analyzed_repo_root']}  "
          f"commit={result['probe']['analyzed_source_commit']}  "
          f"dirty={result['probe']['analyzed_tree_dirty']}")
    for fnd in result["findings"]:
        print(f"  [{fnd['verdict']}] {fnd['kpi']}")
    print(f"[metricC-v2.2] Findings written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
