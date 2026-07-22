# PROPOSAL — WATCHER KPI Governance States (BOOTSTRAP → CALIBRATING → GOVERNED) v1.0

**Date:** 2026-07-20
**Session:** S176 follow-up
**Author:** Team Alpha
**Status:** DRAFT PROPOSAL — for Team Beta code-review before any implementation is scoped.
**Authority:** Recommend-only. This document changes nothing. No thresholds are selected;
no autonomous enforcement is enabled; `watcher_policies.json` is not edited.
**Basis:** `docs/TB_RULING_S176_WATCHER_KPI.md` (ruling on `0c3166a`) and
`docs/WATCHER_KPI_CALIBRATION_FINDINGS_S176.md`.

> Scope discipline: this proposal implements the **architecture** TB ruled for
> (three states, audit-only performance triggers, separated metric classes). It
> deliberately selects **no** numerical thresholds — those come only after the
> Phase C walk-forward produces empirical distributions (§7, non-scope).

---

## 1. The state model

TB §1/§5 require WATCHER to carry an explicit governance state because TFM has **no
empirical operating baseline** yet. Three states:

| State | Meaning | Performance triggers | Structural/safety gates |
|-------|---------|----------------------|-------------------------|
| **BOOTSTRAP** | Insufficient historical TFM observations exist | **audit-only** (record hypothetical decision; do NOT dispatch retraining) | **ACTIVE** |
| **CALIBRATING** | Enough walk-forward/prospective observations to estimate distributions | **shadow mode** (candidate thresholds, false-alarm/overlap counted, human review required) | ACTIVE |
| **GOVERNED** | Candidate policies demonstrated acceptable behavior per TB §5.3/Phase E | **autonomous** (per-trigger) | ACTIVE |

### Entry/exit criteria (mapped to TB Phase A–E, §7)

- **→ BOOTSTRAP (default):** set before the first real-data cycle (TB Phase A). Exit
  requires enough observations to estimate KPI distributions (TB §5.2).
- **BOOTSTRAP → CALIBRATING:** after Phase B (first complete real cycle) **and** Phase C
  (causally-correct historical walk-forward) yield a KPI history of sufficient count
  and stability (TB §4.3, §5.2). Count/stability bars are set by TB at calibration
  time — **not** in this proposal.
- **CALIBRATING → GOVERNED:** only after Phase D simulation shows, per trigger, the TB
  §5.3 evidence bundle (adequate observations, stable across folds, acceptable
  false-alarm frequency, tested cooldown+hysteresis, clear trigger→action mapping,
  auditable history, safe rollback).
- **Per-trigger transition (TB §5/Phase E, explicit):** triggers transition to GOVERNED
  **individually**; the state field is per-KPI (see schema §2), not a single global
  gate. Structural gates (§3) are ACTIVE in all three states from the first cycle.

---

## 2. The `kpi_governance` schema

Extends TB's Q3 example. **Two metric classes are kept separate** per TB Q1's explicit
requirement (a pool can concentrate model weight yet miss future draws, and vice-versa).
This block is a **proposed addition** to `watcher_policies.json`; it replaces the
generic `minimum_hit_rate` (see §5). All performance fields start `null` /
`audit_only` — no threshold is chosen here.

```json
{
  "kpi_governance": {
    "state": "BOOTSTRAP",
    "_state_note": "Global default; per-metric 'enforcement' governs actual dispatch.",

    "prospective_outcome_metrics": {
      "_class_note": "Do future actual draws land in the pools? (Metric-A family.)",
      "hit20":  { "null_rate": 0.02, "empirical_baseline": null, "minimum_samples": null, "collapse_threshold": null, "collapse_window": null, "enforcement": "audit_only" },
      "hit100": { "null_rate": 0.10, "empirical_baseline": null, "minimum_samples": null, "collapse_threshold": null, "collapse_window": null, "enforcement": "audit_only" },
      "hit300": { "null_rate": 0.30, "empirical_baseline": null, "minimum_samples": null, "collapse_threshold": null, "collapse_window": null, "enforcement": "audit_only" },
      "lift20":  { "null_rate": 1.0, "empirical_baseline": null, "enforcement": "audit_only" },
      "lift100": { "null_rate": 1.0, "empirical_baseline": null, "enforcement": "audit_only" },
      "lift300": { "null_rate": 1.0, "empirical_baseline": null, "enforcement": "audit_only" },
      "consecutive_misses": { "empirical_run_distribution": null, "false_alarm_horizon": null, "run_len_threshold": null, "enforcement": "audit_only" }
    },

    "pool_structure_metrics": {
      "_class_note": "How concentrated/structured are predictions? (SEPARATE from Hit@K, TB Q1.)",
      "top20_weight_share":  { "empirical_baseline": null, "floor": null, "enforcement": "audit_only" },
      "top100_weight_share": { "empirical_baseline": null, "floor": null, "enforcement": "audit_only" },
      "top300_weight_share": { "empirical_baseline": null, "floor": null, "enforcement": "audit_only" },
      "unique_prediction_count": { "empirical_baseline": null, "enforcement": "audit_only" },
      "outcome_space_breadth":   { "empirical_baseline": null, "enforcement": "audit_only" },
      "prediction_entropy":      { "empirical_baseline": null, "enforcement": "audit_only" },
      "effective_pool_size":     { "empirical_baseline": null, "enforcement": "audit_only" },
      "duplicate_count":         { "empirical_baseline": null, "max": null, "enforcement": "audit_only" },
      "pool_to_pool_stability":  { "empirical_baseline": null, "enforcement": "audit_only" }
    }
  }
}
```

`enforcement ∈ {audit_only, shadow, active}` maps to the state model: BOOTSTRAP ⇒
`audit_only`, CALIBRATING ⇒ `shadow`, GOVERNED ⇒ `active` (per metric).

---

## 3. Audit-only trigger mode — exactly which code paths change

**Principle (TB §5.1):** in BOOTSTRAP, uncalibrated **performance** triggers RECORD a
hypothetical decision (shadow log) but do **not** dispatch retraining; **structural/
catastrophic** gates stay ACTIVE and may block.

### 3.1 Performance triggers → audit-only

- **Where they fire today:** `chapter_13_triggers.py` evaluates `pipeline_health` /
  `prediction_validation` and returns a prioritized fired-trigger list
  (`hit_rate_collapse` `:269-276`; `consecutive_misses` `:247-254`; priority/return
  `:339-365`).
- **Change:** the **caller** (WATCHER, `watcher_agent.py`) must consult
  `kpi_governance` and, for any metric whose `enforcement != "active"`, route the
  fired trigger to a **shadow log** (§4) instead of dispatching a Steps 3→5→6 rerun.
  The trigger evaluation code itself need not change; the **dispatch decision** is
  gated by state. (Implementation detail for TB review: add a state check at the
  dispatch site, not inside the pure evaluator.)
- No performance threshold is selected; `hit_rate_collapse` and `max_consecutive_misses`
  are held `audit_only` in BOOTSTRAP regardless of their current policy values.

### 3.2 Structural / catastrophic gates → stay ACTIVE (TB §4.2 mapped to existing checks)

| TB §4.2 structural gate | Existing WATCHER check (cite) | Status |
|-------------------------|-------------------------------|--------|
| Required files exist / stage output newer than inputs | `check_output_freshness()` `watcher_agent.py:419-450` via `get_step_io_from_manifest()` `:386-416` (manifest `primary_output`, hard-fail if missing `:410-411`) | **present — keep active** |
| Manifest-driven input/output validation | `get_step_io_from_manifest()` `watcher_agent.py:406,414` | present — keep active |
| No malformed / empty prediction pool; pool non-degeneracy | `NO_EXACT_HITS` `chapter_13_diagnostics.py:606-607`; `LOW_POOL_COVERAGE` `:609` (breadth < 1%) | present (flags) — **elevate to gate?** (TB review) |
| Minimum survivor population | `min_survivor_count: 1000` `watcher_policies.json:98` | present — keep active |
| JSON/NPZ artifacts valid; feature-schema hashes match; model sidecar exists; finite prediction values; pool sizes match contract; duplicate counts within policy; weights normalized | partial — NPZ contract wall is S172 Phase 5 (`EXPECTED_NPZ_KEYS`, CLAUDE.md §6); schema/finite/normalized checks **NOT located in the diagnostics path this session** | **GAP — verify/implement (TB review)** |

Structural gates are ACTIVE in BOOTSTRAP, CALIBRATING, and GOVERNED. The audit-only
rule applies **only** to performance KPIs, never to these.

---

## 4. KPI recording plumbing

**What to persist per draw** (TB Phase B/C): `hit20/hit100/hit300`, `lift@K`, pool-structure
metrics (§2), `consecutive_misses`, survivor churn, confidence drift, window decay,
plus each performance trigger's **hypothetical** (shadow) decision and the state at
evaluation time.

**Where it hooks:**
- Producer: `generate_diagnostics()` `chapter_13_diagnostics.py:750` already assembles
  `prediction_validation` + `pipeline_health` and writes `post_draw_diagnostics.json`
  (`DEFAULT_OUTPUT` `:56`); an archival copy is written to the history dir
  (`chapter_13_diagnostics.py:920`, glob `:960`). **Proposed:** append the §2 metric
  set + null rates to this record (single append point; no new pipeline stage).
- Shadow log: extend the existing trigger-history writer
  (`_save_trigger_history()` `chapter_13_triggers.py` ~`:190`, `TRIGGER_HISTORY_FILE`)
  with a `dispatched: false` / `hypothetical: true` field and the governing state, so
  Phase D can replay real vs hypothetical decisions.

**File/format:** JSONL append per draw (one record/draw) under the diagnostics history
dir, keyed by draw id + session, so Phase C/D can load a clean time series. Exact
schema to be fixed at implementation; this proposal fixes the **hook points**, not the
serialization details.

---

## 5. `minimum_hit_rate` disposition (conditional on Deliverable 1b — now met)

S176-followup **1b verdict:** `minimum_hit_rate` has **zero runtime consumers** on tree
`0c3166a` (definitive `/bin/grep` across `.py/.json/.sh/.md/.gbnf`; sole non-doc/
non-artifact occurrence is the config definition `watcher_policies.json:74`). TB Q3's
deprecation precondition is therefore **met**.

**Path:**
1. Add the `kpi_governance` schema (§2) with per-pool `null_rate` + `empirical_baseline: null`.
2. **Deprecate** `convergence_targets.minimum_hit_rate` (0.05): it is a synthetic-data
   target (`_hitrate_note` `watcher_policies.json:75`), not a real-TFM floor (TB §2.6),
   and nothing reads it. Mark deprecated in-policy first (documentation), remove after
   the schema lands and a re-confirmation search shows no new consumer.
3. Its role — "is Hit@K above the floor?" — is subsumed by
   `prospective_outcome_metrics.hitK.empirical_baseline` once Phase C populates it.

No value is edited by this proposal; this is the recommended sequence for TB/Michael.

---

## 6. Metric-name uniqueness audit (TB Phase A item 6)

Every "hit-rate"/"coverage"-family name on the tree, its single definition, and owner.
**Two names carry more than one meaning — flagged.**

| Name | Definition | Owner (cite) | Class |
|------|-----------|--------------|-------|
| `current_hit_rate` | `exact_hits / pool_size` (per-draw, instantaneous) | `chapter_13_diagnostics.py:531` | Metric C |
| `hit_rate` (bare) | `exact_hits / max(pool_size,1)` — same as C | `chapter_13_llm_advisor.py:284`, `llm_proposal_schema.py:295` | Metric C |
| `hit_20_rate`/`hit_100_rate`/`hit_300_rate` | `hit_K_count / total_draws` (recall over draws) | `backtest_pools.py:318-320` | Metric A |
| `hit_at_20`/`hit_at_100`/`hit_at_300` | Hit@K **average** (advisor bands) | `advisor_bundle.py:201-203`; `parameter_advisor.py:145-147` | Metric A (recall) |
| `exact_hits` | count of pool predictions with `abs(pred−actual)==0` | `chapter_13_diagnostics.py:242-249` | count (feeds C) |
| `pool_coverage` | `unique_predictions / 1000` (outcome-space breadth) | `chapter_13_diagnostics.py:266` | Pool-structure (breadth) |
| `coverage` | `Σ(top-k weight)/Σ weight` (weight-mass) | `build_pools.py:249` | Pool-structure (weight share) |
| `coverage` (stage-2 objective term) | `unique(tn[mask])/unique(tn)` (temporal) | `scorer_trial_worker.py:430` | stage-2 optimizer term |
| `coverage_pct` | agent-framework coverage percentage | `agents/contexts/prediction_context.py:84` | agent gate |
| `minimum_hit_rate` | 0.05 synthetic target (unwired) | `watcher_policies.json:74` | target (deprecate, §5) |
| `score` | mean exact digit-match ×100 | `survivor_scorer.py:366-369` | stage-3 survivor score |

**⚠️ Names with two meanings (fix at implementation):**
- **"hit rate"** — used for both **Metric A** (recall over draws, `hit_K_rate`/`hit_at_K`)
  and **Metric C** (per-draw `exact_hits/pool_size`, `current_hit_rate`/bare `hit_rate`).
  The `kpi_governance` schema (§2) resolves this by naming `hitK` explicitly as
  prospective-outcome (recall) and retiring the ambiguous bare usage.
- **"coverage"** — at least three distinct measures: outcome-space breadth
  (`pool_coverage`), model **weight-mass** (`build_pools.coverage`), and stage-2
  temporal coverage (`scorer_trial_worker.coverage`). The schema separates these into
  named pool-structure fields (`topK_weight_share`, `outcome_space_breadth`).

---

## 7. Explicit non-scope

- **Phase C walk-forward is deferred** to **post-S172 Phase 7** on the RANGE-MINER path.
  Rationale (one sentence): the walk-forward is GPU/cluster-heavy and would create a
  GCVM launch-storm competing with the S172 Phase 5–7 acceptance runs already queued
  on the same rigs, so it must not run concurrently.
- **No thresholds selected.** Every performance field in §2 is `null` / `audit_only`.
- **No autonomous enforcement enabled.** GOVERNED activation is per-trigger and gated on
  the TB §5.3 evidence bundle after Phase D.
- **D1/D2 are not in scope here** — RESOLVED on `main` (S176-followup 1a, accepted); the
  stale `_find_results` `step_files` map (`watcher_agent.py:1317-1322`) is logged as a
  **separate P3 cleanup**, outside this KPI work.

---

## 8. Implementation checklist (for TB scoping — not executed here)

- [ ] Add `kpi_governance` block (§2) to `watcher_policies.json` (all `audit_only`).
- [ ] Add a governance-state read + per-metric `enforcement` gate at the WATCHER
      **dispatch** site (not inside `chapter_13_triggers` evaluation).
- [ ] Extend `generate_diagnostics()` to record the §2 metric set + null rates per draw.
- [ ] Extend the trigger-history writer with `hypothetical/dispatched` + state fields.
- [ ] Elevate/keep structural gates active (§3.2); close the schema/finite/normalized GAP.
- [ ] Deprecate `minimum_hit_rate` (§5) after schema lands + re-confirm no consumer.
- [ ] Resolve the two overloaded names ("hit rate", "coverage") per §6.
- [ ] Team Beta reviews before any of the above is implemented.

---

## 9. References

- `docs/TB_RULING_S176_WATCHER_KPI.md` — the ruling this proposal implements
- `docs/WATCHER_KPI_CALIBRATION_FINDINGS_S176.md` — S176 structural findings
- `chapter_13_triggers.py`, `chapter_13_diagnostics.py` — trigger + diagnostics paths
- `watcher_agent.py:386-450` — manifest-driven structural validation
- `watcher_policies.json` — current policy (unchanged)
- `watcher_kpi_metricC_deterministic_v2.py` — null-stated deterministic analyzer (TB §8)
