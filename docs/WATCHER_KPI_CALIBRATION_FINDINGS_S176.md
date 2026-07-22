# WATCHER KPI Calibration Findings — S176

**Date:** 2026-07-20
**Session:** S176
**Author:** Team Alpha
**Status:** Recommend-only — for Team Beta ruling. Changes nothing in `watcher_policies.json`.
**Scope:** Analytic (Type-1) + deterministic validation of the WATCHER hit/survivor
governance KPIs. Empirical real-stream baselining (Track B) is deferred to the
post-walk-forward session and is **not** claimed here.

---

## 0. Headline finding — the coverage-governance gap

**TFM's stated 65–75% final-pool coverage quality target is watched by NO retrain
trigger.** The pipeline optimizes toward pool coverage (Balanced-100 "> 60% weight,
Hit@100 > 70%"; Wide-300 "> 85% weight, Hit@300 > 90%" — `docs/CHAPTER_7_PREDICTION_GENERATOR.md:393`),
but nothing in the WATCHER retrain-trigger layer thresholds that target. The only
runtime check on anything called "coverage" is:

```
if prediction_validation["pool_coverage"] < 0.01:  flags.append("LOW_POOL_COVERAGE")
        # chapter_13_diagnostics.py:609
```

That is a **different, looser measure** (`pool_coverage = unique_predictions / 1000`,
`chapter_13_diagnostics.py:266` — outcome-space breadth, not weight-mass coverage),
set at a **triviality floor of 1%**, and it is a **summary flag, not a trigger** — it
raises no retrain/rerun action. So the quality dimension the system is actually built
to deliver (concentrated, high-coverage pools) has **zero closed-loop governance**,
while the three KPIs that *do* have governance are thresholding a per-draw exact-match
quantity that (as shown below) fires on healthy data.

**This is decision #1 for Team Beta (§6.1).**

---

## 1. Purpose

Establish whether the WATCHER hit/survivor governance KPIs are calibrated to the
system's real behavior, or are leftover synthetic-era guesses. Two independent
methods were used and they converge:

- **Metric A (recall probe):** analytic + Monte-Carlo resolvability of the KPIs
  against per-draw Top-20 recall — the metric the *backtest / Track-B* path uses.
- **Metric C (deterministic):** closed-form analysis of `exact_hits / pool_size` —
  the metric the *live retrain triggers* actually threshold.

The two-method convergence (both flag the same two KPIs as decorative) is the
evidence base for the recommendations in §6.

---

## 2. Background — three different "hit/quality" quantities

A recurring source of confusion is that the repo contains **three distinct
quantities** all loosely called "hit rate" or "coverage," computed in different
subsystems and monitored (or not) by different governance:

| # | Quantity | Formula | Source | Watched by a retrain trigger? |
|---|----------|---------|--------|-------------------------------|
| **A** | Per-draw Top-20 **recall** (~2%) | `hits / total_draws`, `hit = actual ∈ top-K` | `evaluate_pools.py:34`; `backtest_pools.py:318-320` | **No** — backtest/Track-B only |
| **B** | 65–75% final-pool **coverage** | weight-mass `Σ(top-k weight)/Σweight` | `build_pools.py:249`; targets `CHAPTER_7:393` | **No** — see §0 |
| **C** | Per-draw exact-match **precision** | `exact_hits / pool_size` | `chapter_13_diagnostics.py:531` (from `:242-249`) | **Yes** — all live KPIs bind here |

A and C describe the *same underlying event* (the actual draw landing in the pool,
P ≈ pool/1000 = 2% for the 20-pool). A aggregates it as a rate over draws; C
evaluates it **per-draw and instantaneously**. The live triggers use the C form.

**Production pool (confirmed from live config, not memory):**
`prediction_generator_config.json` → `pool_size = 20`, `k = 10`; mirrored in
`agent_manifests/prediction.json` defaults. So metric C per draw ∈ {0, 1/20 = 0.05}.

---

## 3. Methodology

### 3.1 Metric A — recall probe (Task 2a)

`watcher_kpi_resolvability_probe.py --pool-size 20 --draw-space 1000`, swept over
four assumed *healthy* Top-20 recall rates {0.02, 0.05, 0.10, 0.15}. No measured
baseline exists before the walk-forward, so the assumed rate is a **stated
assumption**; the sweep's purpose is to show whether any verdict is an artifact of
that assumption. Chance baseline = 20/1000 = **0.020**.
Findings file: `watcher_kpi_resolvability_findings_metricA.json`.

### 3.2 Metric C — deterministic analysis (Task 2b)

`watcher_kpi_metricC_deterministic.py --pool-size 20 --k 10`. Metric C is
two-valued and deterministic given `pool_size`, so **no Monte Carlo is used**; each
KPI's behavior is derived in closed form. Findings file:
`watcher_kpi_metricC_deterministic_findings.json`.

---

## 4. Raw Results

### 4.1 Metric A — recall probe, verdict matrix (stable across the full sweep)

| KPI | Verdict (all four assumed rates) |
|-----|----------------------------------|
| `hit_rate_collapse_threshold` (0.01 / 20-draw) | **DECORATIVE** (fires on healthy data) |
| `max_consecutive_misses` (5) | **DECORATIVE** (fires on healthy data) |
| `retrain_after_n_draws` (10) | **INSUFFICIENT-N** (Wilson ½-width ±0.15→±0.21 at n=10) |
| `minimum_hit_rate` (0.05) | **RESOLVABLE** (analytic: +0.030 over the 0.020 chance floor) |
| `window_decay_threshold` | **NEEDS-DATA** (Type-2) |
| `survivor_churn_threshold` | **NEEDS-DATA** (Type-2) |
| `confidence_drift_threshold` | **NEEDS-DATA** (Type-2) |
| `llm_confidence_threshold` | **NEEDS-DATA** (Type-3, LLM) |

No verdict flips across the sweep — categories are robust to the unknown baseline.
Magnitudes do move: `hit_rate_collapse` healthy false-fire-per-window falls
0.67 → 0.36 → 0.12 → 0.039 as assumed skill rises, but only 0–3.9% of healthy
500-draw streams stay clean even at 0.15.

### 4.2 Metric C — deterministic, pool_size = 20

Per-draw metric C ∈ {**hit → 0.05**, **miss → 0.0**}; chance P(hit) = 0.020.

| KPI | Closed-form result | Verdict |
|-----|--------------------|---------|
| `hit_rate_collapse_threshold` (0.01) | hit-draw 0.05 ≥ 0.01 (no fire); miss-draw 0.0 < 0.01 (**fires**). Fires on ~98% of draws under healthy chance operation. `hit_rate_collapse_window: 20` is **dead config** — the trigger reads one instantaneous snapshot (`chapter_13_triggers.py:269`), never a 20-draw average. | **DECORATIVE — exact re-encoding of the raw per-draw miss boolean** |
| `max_consecutive_misses` (5) | Mean gap between hits = 1/0.020 = **50 draws** ≫ 5. A 5-miss run is the norm. Closed-form E[draws to first 5-run] = **5.31** (matches the metric-A MC value 5.31). | **DECORATIVE — fires under healthy operation** |
| `minimum_hit_rate` (0.05) | **Zero runtime consumers** (grep definitive: no trigger, no flag, no diagnostic). Numerically equals C's hit-draw value (1/20), but nothing compares against it. | **UNWIRED** |

**Convergence:** both methods independently flag `hit_rate_collapse` and
`max_consecutive_misses` as decorative. The metric-C view is the stronger statement
— it shows *why* (the arithmetic), not just *that* they false-fire.

> **Pool-size sensitivity note (for TB context):** `hit_rate_collapse` fires even on
> **hit** draws whenever `pool_size > 100` (since 1/pool_size < 0.01). At the
> production pool of 20 it fires only on misses, but that is still every ~98% of
> draws. Either way the trigger carries no more information than the raw miss bit.

---

## 5. Findings

- **F1 — Coverage gap (headline, §0).** The 65–75% pool-coverage target (metric B)
  has no retrain trigger; only the 1% `LOW_POOL_COVERAGE` flag, a different and
  looser measure, and not an action-producing trigger.
- **F2 — Two decorative triggers.** `hit_rate_collapse` and `max_consecutive_misses`
  both fire on healthy data, confirmed by two independent methods. `hit_rate_collapse`
  is arithmetically an instantaneous miss-detector with a dead `window` param.
- **F3 — `minimum_hit_rate` is unwired.** It is analytically resolvable but no code
  reads it; it governs nothing at runtime.
- **F4 — Two config-path defects surfaced during stage mapping** (logged separately,
  §6.4 — **not** part of the KPI calibration and not to be folded into it).

---

## 6. Decisions requested of Team Beta

> All items below are **recommend-only**. No `watcher_policies.json` value has been
> changed; no config-path bug has been fixed. Team Beta rules; Michael implements
> approved changes via `THRESHOLD_GOVERNANCE`.

### 6.1 Decision 1 — Coverage gap: add a retrain trigger for the 65–75% target?

**Recommendation: yes, add closed-loop governance on metric B.** The system's
headline quality target is currently ungoverned. Options for TB:

- **(a)** Add a `pool_coverage_floor` retrain/rerun trigger on the **weight-mass**
  coverage (`build_pools.py:249`) at, e.g., 0.60 (Balanced-100 band) — the measure
  that matches the CHAPTER_7 target.
- **(b)** Additionally raise/relabel the `LOW_POOL_COVERAGE` flag from its 1% floor
  so it stops reading as "coverage is fine" when it is merely non-degenerate.
- **Open input needed:** which pool (20/100/300) is the governance reference for the
  coverage target, and the exact floor. This mirrors the Top-20-vs-Top-100 hit-rate
  ambiguity already resolved for the hit KPIs (Top-20).

### 6.2 Decision 2 — The two decorative triggers: retune vs repoint?

- **`hit_rate_collapse`:** at pool=20 it is a per-draw miss detector with a dead
  window. **Recommendation: repoint, don't retune** — a threshold on `exact_hits/
  pool_size` cannot express "rate collapse" when the metric is two-valued per draw.
  Repoint at a **windowed recall** (metric A over N draws, e.g. `hit_20` rate over
  20 draws vs the 0.02 chance floor) so the `window` semantics become real.
- **`max_consecutive_misses`:** at P(hit)≈2%, a 5-run is normal. **Recommendation:**
  either raise the run length far above the healthy mean gap (≫50) so it flags only
  true droughts, or repoint at the same windowed-recall metric. A bare retune of the
  run length to a healthy-plausible value is possible but fragile to pool size.
- TB to rule: retune-in-place vs repoint-at-windowed-recall (Team Alpha recommends
  repoint for both).

### 6.3 Decision 3 — `minimum_hit_rate`: wire it or remove it?

It is resolvable but governs nothing. **Recommendation: wire it** as the floor of a
windowed-recall collapse trigger (pairing naturally with the Decision-2 repoint), or
else **remove it** from `convergence_targets` so the policy file stops implying a
control that does not exist. TB to choose wire vs remove.

### 6.4 Decision 4 — Two config-path defects (log as SEPARATE tickets)

These were found while mapping stage metrics. They are **not** KPI-calibration items
and should be tracked as their own defects, not folded into this work:

- **D1 — Stage 4 output/metric mismatch.** `adaptive_meta_optimizer.py` writes
  `reinforcement_engine_config.json` (`:156`), **not** the `optimal_ml_config.json`
  the WATCHER step-file map expects (`watcher_agent.py:1320`), and it computes **no
  R²** (`:10`, `:33-35`) despite the WATCHER step-4 gate `best_r2 > 0.5`
  (`CHAPTER_12:425`). WATCHER must be reading that R² from some other artifact —
  needs verification that the gate is actually fed.
- **D2 — Stage 3 output-name mismatch.** Stage 3 aggregates to
  `survivors_with_scores.json` (`run_step3_full_scoring.sh:306`), not the
  `full_scoring_results.json` the WATCHER step-file map expects
  (`watcher_agent.py:1319`).

TB to acknowledge these as separate defects and assign, not resolve here.

---

## 7. Caveats and Limitations

1. **No measured baseline yet.** Metric A used a *stated assumption* swept over four
   rates; real baselining (hit rate + Wilson CI, survivor distribution, empirical
   firing rates) requires Michael's walk-forward (`backtest_results.json`) and is the
   deferred Track-B session. Verdicts here are analytic/deterministic, not empirical.
2. **`confidence_drift`, `window_decay`, `llm_confidence`, `survivor_churn`** are
   NEEDS-DATA (Type-2/Type-3) — they cannot be judged without recorded
   performance/advisor series and are out of scope for this analytic pass.
3. **Pool size is config-driven.** All metric-C results assume `pool_size = 20` from
   `prediction_generator_config.json`; if a production run overrides it, re-run
   `watcher_kpi_metricC_deterministic.py --pool-size <N>` — the `>100` regime flips
   `hit_rate_collapse` to firing on hit draws too.

---

## 8. Deliverables

| File | Contents |
|------|----------|
| `watcher_kpi_resolvability_findings_metricA.json` | Recall probe, 4-rate sweep, verdict matrix (metric A / backtest path) |
| `watcher_kpi_metricC_deterministic_findings.json` | Closed-form metric-C analysis at pool=20 (live-trigger path) |
| `watcher_kpi_metricC_deterministic.py` | The deterministic analysis tool (CPU, read-only) |
| `docs/WATCHER_KPI_CALIBRATION_FINDINGS_S176.md` | This document |

Re-run any time:

```bash
# metric A (recall) sweep
for b in 0.02 0.05 0.10 0.15; do \
  python3 watcher_kpi_resolvability_probe.py --policies watcher_policies.json \
    --baseline-hit-rate $b --pool-size 20 --out scratch_kpi_sweep/findings_b${b}.json; done

# metric C (deterministic)
python3 watcher_kpi_metricC_deterministic.py --pool-size 20 --k 10
```

---

## 9. Change History

| Date | Change | Author |
|------|--------|--------|
| 2026-07-20 | Initial KPI calibration findings from S176 (Tasks 2a/2b) | Team Alpha |

---

## References

- `watcher_policies.json` — governance KPI values (unchanged)
- `chapter_13_triggers.py` — retrain-trigger consumers (metric C)
- `chapter_13_diagnostics.py` — `compute_prediction_validation` / `compute_pipeline_health` (metric C source)
- `evaluate_pools.py`, `backtest_pools.py` — metric A (recall) source
- `build_pools.py`, `docs/CHAPTER_7_PREDICTION_GENERATOR.md` — metric B (coverage) source/target
- `prediction_generator_config.json`, `agent_manifests/prediction.json` — production pool_size/k
- `docs/CHAPTER_12_WATCHER_AGENT.md:420-427` — per-step Layer-1 health gates
- `docs/THRESHOLD_CALIBRATION_FINDINGS_S148.md` — format precedent
