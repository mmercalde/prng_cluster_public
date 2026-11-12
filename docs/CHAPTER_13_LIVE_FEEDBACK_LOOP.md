# Chapter 13: Live Feedback Loop & Autonomous Learning

> **What TFM is.** TFM filters candidate generators against observed draws in two directions,
> keeps a population of survivors on purpose, ranks them, and lets the top candidates vote their
> next value into a prediction pool. A survivor is a scored candidate, not a verdict. The
> generator family is a substrate for mimicry, not a claim about the machine. The target is a
> pool hit rate. TFM does not recover a seed or reconstruct state, and an objection that assumes
> it does is out of scope.

## PRNG Analysis Pipeline — Complete Operating Guide

**Version:** 1.3.0 — corrected against code (2026-11-11)
**Status:** **CORRECTED at `a15b524`, 2026-11-11 — documentation pass, not a closure.** The
design text of v1.0–v1.2 is kept below; every place where the code at `a15b524` does something
different carries a visible *"Corrected 2026-11-11"* note or a **SUPERSEDED (2026-11-11)** marker.
The consolidated list of disagreements is **§22 Known gaps at a15b524**. Verification declaration
§25, closure statement and sentinel §26.
**Authority for this pass:** `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md` (REV1),
authorized by Michael 2026-11-11. Per its §0, Team Beta is a reviewer; a Beta ruling cited here
as "binding" is a recorded review position the owner may keep or lift.
**Base:** `a15b5247336255801de8cf67d52653701b1e8c80` (`a15b524`) on VM 101, working tree clean
for tracked files (`git diff --stat HEAD` empty).
**Audit this pass relies on:** `docs/CHAPTER_13_AUDIT.md` (§1–§9, made from the public mirror) and
its **"Addendum — live-host re-verification, 2026-11-11"** (made on VM 101 with host state,
gitignored files and git history).
**Numbering scheme:** **EXECUTABLE** step numbers throughout (`agents/watcher_agent.py:387-394`
`STEP_SCRIPTS`: Step 1 = `window_optimizer.py`, which runs the sieve; Step 2 = scorer
meta-optimizer; Step 3 = full scoring; Step 4 = ML meta-optimizer; Step 5 = anti-overfit training;
Step 6 = prediction generator). See `docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1. "Chapter 13" is a
chapter number, not a step: it is not a WATCHER step.
**Depends On:** Chapters 1-12
**Executes After:** Step 6 (Prediction Generator) — *as designed; see §6 for what the code checks*
**Purpose:** Close the autonomy loop — convert predictions into learning. *(Corrected 2026-11-11:
at `a15b524` the loop is a human-gated, defensive feedback path whose two inputs are not connected
to the pipeline's outputs; see §0.3 and §22.)*

---

## 0. How this chapter was produced (2026-11-11 pass)

### 0.1 What was done

1. The existing audit `docs/CHAPTER_13_AUDIT.md` (68 claims; made from the public mirror at
   `a15b524` with no host state, gitignored files or git history) was re-verified on VM 101.
   Every tracked file is byte-identical to `a15b524`; about 200 of its code anchors were opened by
   line and all matched. Results were appended to the audit as an addendum: 61 claims confirmed,
   3 supporting statements corrected (no label changed), 5 UNVERIFIED items resolved and 1 partly
   resolved.
2. The Chapter 13 host question of the authority file (§5) was settled read-only: **there is no
   host-side bridge** that writes `prediction_pool.json` or raises `new_draw.flag` (audit addendum
   A.7). Unavailable surfaces are named, not reported as clean.
3. This chapter was then corrected **in place, section by section**. Nothing was deleted. Design
   text that the code does not implement is kept and annotated.

### 0.2 What this chapter claims, and what it does not

It states what the chapter says against what the code at `a15b524` does. It does **not** judge
whether the design is right, does not fix code, and does not propose removing or simplifying any
component. Each disagreement carries exactly one label: **BROKEN**, **SWITCHED-OFF-BY-RULING** or
**DESCRIPTION-ONLY**, with the IDs used in the audit (B1–B18, S1–S9, D1–D9).

### 0.3 The short version of the as-built state

- **Null result first.** On the host, **233 of 233** archived Chapter 13 diagnostics graded an
  empty prediction pool (`pool_size 0`), and **233 of 233** were synthetic draws; **0** real draws
  have ever passed through the loop (audit addendum A.0).
- What runs end to end: flag-gated ingestion of test/manual draws, the daemon cycle, diagnostics
  and archive, trigger evaluation, the LLM advisory with heuristic fallback, the
  accept/reject/escalate engine, human approval through `agents/watcher_agent.py --approve`, the
  WATCHER request queue, and the selfplay inner episode (audit §3).
- What does not connect: Chapter 13 grades a file Step 6 does not write (B1) and appends draws to a
  history the relearning steps do not read (B2). Selfplay's outputs cannot change the active policy
  (B10, B11, B13, B14).

---

## Table of Contents

1. [Purpose](#1-purpose)
2. [Core Design Principle](#2-core-design-principle)
3. [The Circular Learning Loop](#3-the-circular-learning-loop)
4. [What Becomes Autonomous](#4-what-becomes-autonomous)
5. [What Remains Stable](#5-what-remains-stable)
6. [Execution Trigger](#6-execution-trigger)
   - 6.1 [Synthetic Draw Injection (Test Mode)](#61-synthetic-draw-injection-test-mode)
7. [Inputs](#7-inputs)
8. [Post-Draw Diagnostics](#8-post-draw-diagnostics)
9. [Label Refresh Mechanism](#9-label-refresh-mechanism)
10. [Retrain Trigger Policies](#10-retrain-trigger-policies)
11. [LLM Role: Strategist](#11-llm-role-strategist)
12. [WATCHER Role: Executor](#12-watcher-role-executor)
13. [Acceptance & Rejection Rules](#13-acceptance--rejection-rules)
14. [Outputs](#14-outputs)
15. [Integration With Steps 1-6](#15-integration-with-steps-1-6)
16. [Convergence Guarantees](#16-convergence-guarantees)
17. [Safety & Ethics](#17-safety--ethics)
18. [Configurable Parameter Reference](#18-configurable-parameter-reference)
19. [Implementation Checklist](#19-implementation-checklist)
20. [Deferred Extensions (Roadmap)](#20-deferred-extensions-roadmap)
21. [Phase 9A: Selfplay Integration](#21-phase-9a-selfplay-integration) *(added v1.2.0, sits after the original end marker)*
22. [Known gaps at a15b524](#22-known-gaps-at-a15b524) *(added 2026-11-11)*
23. [RANGE-MINER, the NPZ contract and the window-anchor change](#23-range-miner-the-npz-contract-and-the-window-anchor-change) *(added 2026-11-11)*
24. [Open items](#24-open-items) *(added 2026-11-11)*
25. [Verification declaration (VIR-1…6)](#25-verification-declaration-vir-16) *(added 2026-11-11)*
26. [Closure statement](#26-closure-statement) *(added 2026-11-11)*

(Section 0, *How this chapter was produced*, sits above this table.)

---

## 1. Purpose

Steps 1–6 generate predictions.  
**Chapter 13 makes those predictions learn from reality.**

This chapter defines the **autonomous feedback loop** that:

- Validates predictions against real-world outcomes
- Refreshes training labels from accumulated history
- Triggers selective retraining (Steps 3 → 5 → 6)
- Enables continuous improvement without human intervention

**Before Chapter 13:** The pipeline runs correctly but learning stops at execution.  
**After Chapter 13:** The system observes reality, measures error, adapts, and repeats indefinitely.

That is autonomy.

> **Corrected 2026-11-11 (D3, DESCRIPTION-ONLY; S1, SWITCHED-OFF-BY-RULING).** Previously this
> section said the loop runs "without human intervention". At `a15b524` every trigger evaluation
> carries `requires_approval=True` (`chapter_13_triggers.py:369`), and in production the WATCHER
> daemon notifies and waits; it auto-dispatches only when `test_mode` and
> `auto_approve_in_test_mode` are both true (`agents/watcher_agent.py:3317-3352`; tracked policy
> `watcher_policies.json:7` `test_mode: false`, `:102` `auto_approve_in_test_mode: false`). Human
> approval is the deliberate gate (§10.1, §20.3). "Validates predictions against real-world
> outcomes" is also not true today: Chapter 13 grades `prediction_pool.json`, which nothing writes
> (B1, §7), and the host has never processed a real draw (§0.3).

---

## 2. Core Design Principle

### Immutable Structure, Configurable Intelligence

The system is architected with a fundamental separation:

| Aspect | Mutability | Examples |
|--------|------------|----------|
| **Structure** | 🔒 Frozen | Step ordering, sieve math, feature schema, Pydantic models, validation rules |
| **Parameters** | 🔧 Configurable | Thresholds, weights, pool sizes, retraining cadence |

**The LLM cannot:**
- Rewrite mathematical logic
- Invent new features
- Bypass validation
- Mutate control flow
- Change step ordering

**The LLM can:**
- Interpret diagnostics
- Detect drift patterns
- Propose parameter adjustments
- Recommend retraining

This separation is enforced by:
- Pydantic schemas (contract enforcement)
- GBNF grammars (output constraint)
- Feature hash validation (drift prevention)
- Sidecar metadata (provenance tracking)
- Manifest-scoped parameters (LLM containment)

> **Corrected 2026-11-11 (D6, B17).** The proposal types are Python `dataclass`es, not Pydantic
> models (`llm_proposal_schema.py:70-71` `ParameterProposal`, `:104-105` `LLMProposal`). The GBNF
> grammar exists in two copies: `agent_grammars/chapter_13.gbnf` and the older
> `grammars/chapter_13.gbnf`, and the advisor loads the second (`chapter_13_llm_advisor.py:310-312`).
> The grammar lets the LLM name any parameter as a free string, and the frozen check is eight literal
> names (`chapter_13_acceptance.py:72-84` `FROZEN_PARAMETERS`), so "manifest-scoped parameters" is not
> enforced from the manifests (B17, BROKEN, latent while nothing is applied — S3).

---

## 3. The Circular Learning Loop

### 3.1 The Complete Loop

```
Step 6 → Predictions Emitted
           ↓
    Live Draw Occurs
           ↓
    History Updated (append-only)
           ↓
    Chapter 13 Diagnostics
           ↓
    WATCHER Policy Check
           ↓
    Step 3 Re-run (labels refresh)
           ↓
    Step 5 Re-run (model retrains)
           ↓
    Step 6 Re-run (better predictions)
           ↓
        Repeat
```

### 3.2 Static vs Dynamic Steps

| Category | Steps | Trigger |
|----------|-------|---------|
| **Static** | 1, 2, 4 | Run once; re-run only on regime shift |
| **Dynamic** | 3, 5, 6 | Re-run as part of learning loop |

**Key insight:** The system learns by weighting survivors, not by endlessly searching new ones.

### 3.3 Why This Works

- **Step 3** recomputes `holdout_hits` using expanded history
- **Step 5** retrains model weights on refreshed labels
- **Step 6** generates improved predictions from better model

No code changes to Steps 1-6. Only orchestration of when to re-invoke.

> **Corrected 2026-11-11 (D4, DESCRIPTION-ONLY; B2, BROKEN).** Step 5 now trains on
> `holdout_quality`, not `holdout_hits` (`meta_prediction_optimizer_anti_overfit.py:612`
> `target_field: str = "holdout_quality"`, `:1481`). And Step 3 does not see the "expanded history":
> it reads `train_history.json` / `holdout_history.json` (`agent_manifests/full_scoring.json:120-121`),
> which only Step 1 writes (`window_optimizer.py:1100-1102`), while Chapter 13 appends to
> `lottery_history.json` (§9, B2).

---

## 4. What Becomes Autonomous

Chapter 13 enables autonomous adjustment of:

### 4.1 Model Weights (Step 5)
- Retraining triggered by real outcomes
- `holdout_hits` becomes a rolling label
- Survivor weights increase/decrease based on live performance

### 4.2 Prediction Pool Quality (Step 6)
- Confidence calibration improves over time
- Pool composition adapts (tight vs wide effectiveness)
- Poor-performing survivors naturally decay

### 4.3 Retraining Cadence
- N draws accumulated
- Confidence drift detected
- Accuracy collapse identified
- Regime change flagged

### 4.4 Parameter Tuning
- Thresholds
- Feature importance shifts
- Model choice (XGBoost vs others)
- Ensemble weighting

### 4.5 Pipeline Re-execution
- WATCHER becomes event-driven, not manual
- Partial reruns (3→5→6) or full reruns (1→6) based on policy

> **Corrected 2026-11-11 — what is autonomous at `a15b524`.**
> - **§4.1 / §4.2 survivor reinforcement and decay (B8, BROKEN).** `decay_candidates` is initialised
>   empty and never appended (`chapter_13_diagnostics.py:405`, comment `:419`); hits are matched on
>   `predicted_value` / `value` (`:396`), which no Step 3 writer produces. Host: the last diagnostics
>   file has `reinforce_candidates []`, `decay_candidates []`.
> - **§4.3 cadence (B3, B4, B7).** The triggers exist but measure other things than named here (§10).
> - **§4.4 parameter tuning (S3, SWITCHED-OFF-BY-RULING).** Accepted proposals end in a string or an
>   approval request (`chapter_13_orchestrator.py:398-441`); there is no apply function and
>   `record_applied_changes` (`chapter_13_acceptance.py:635`) has no caller. Held by the Chain D
>   review position quoted in `docs/CLAUDE_CODE_INSTRUCTIONS_CHAIN_C_TRUTHFULNESS_HOTFIX.md:104-106`.
> - **§4.5 event-driven re-execution (S1).** Execution requires a human (§10.1).

---

## 5. What Remains Stable

These are **never** subject to autonomous modification:

| Component | Reason |
|-----------|--------|
| Step 1 (Window Optimizer) | Defines search space; expensive; structural |
| Step 2 (Scorer Meta) | Scoring logic stability is critical |
| Step 4 (ML Meta-Optimizer) | Architecture planning, not per-draw |
| Core sieve logic | Mathematical constraints must remain invariant |
| Feature schema | Changing silently would invalidate learning |
| Step ordering | Causal dependencies are fixed |
| Validation rules | Safety guarantees |

This separation prevents:
- Overfitting to recent noise
- Structural drift
- Chaotic parameter explosion
- Silent corruption

---

## 6. Execution Trigger

Chapter 13 executes **only when all conditions are met**:

1. ✅ Step 6 completed successfully
2. ✅ A new official draw has occurred
3. ✅ Draw fingerprint differs from previous run
4. ✅ No halt flag set

```python
def should_execute_chapter_13() -> bool:
    return (
        step_6_complete() and
        new_draw_available() and
        draw_fingerprint_changed() and
        not halt_flag_set()
    )
```

> **Corrected 2026-11-11 (B18, BROKEN).** The function above does not exist. The orchestrator's
> daemon loop checks only the halt files and `new_draw.flag` (`chapter_13_orchestrator.py:501-519`;
> `_check_safety` `:158`, `_check_new_draw_flag` `:215`). A fingerprint check exists, but in the
> ingestion daemon, not here (`draw_ingestion_daemon.py:101` `fingerprint_changed`). **No
> Step-6-completion check exists** in `chapter_13_orchestrator.py`. Who raises the flag: only
> `synthetic_draw_injector.py` (`create_new_draw_flag`, `:349`, `:408`) and `draw_ingestion_daemon.py`
> (`:265`). **Host (2026-11-11):** no unit, timer, cron job or process raises it (audit addendum A.1).

---

## 6.1 Synthetic Draw Injection (Test Mode)

For testing and validation, the system supports synthetic draw injection.

### Configuration

In `watcher_policies.json`:
```json
{
  "test_mode": true,
  "synthetic_injection": {
    "enabled": true,
    "interval_seconds": 60,
    "true_seed": 12345
  }
}
```

**Note:** PRNG type is **not** specified here. It is inherited from `optimal_window_config.json` to maintain consistency with Steps 1-6.

### `synthetic_draw_injector.py`

```python
"""
Synthetic Draw Injector — Test mode draw generation

CRITICAL: No hardcoded PRNG type.
- Reads prng_type from optimal_window_config.json
- Uses prng_registry.py (same as all pipeline steps)
- Ensures test behavior matches production behavior
"""

from prng_registry import get_prng_function
import json

def inject_synthetic_draw():
    # Load PRNG config from existing pipeline config
    with open('optimal_window_config.json') as f:
        config = json.load(f)
    
    prng_type = config.get('prng_type', 'java_lcg')
    prng_func = get_prng_function(prng_type)
    
    # Load injection settings
    with open('watcher_policies.json') as f:
        policies = json.load(f)
    
    true_seed = policies['synthetic_injection']['true_seed']
    
    # Generate next draw using same PRNG as pipeline
    # ... implementation
```

### Modes

| Mode | Command | Use Case |
|------|---------|----------|
| Manual | `python3 synthetic_draw_injector.py --inject-one` | Single test |
| Daemon | `python3 synthetic_draw_injector.py --daemon --interval 60` | Continuous testing |
| Flag-triggered | Automatic when `test_mode: true` after Step 6 | Integrated testing |

> **Corrected 2026-11-11.** The code block above is an illustration, not the file: the injector
> imports `get_cpu_reference, list_available_prngs, get_kernel_info` from `prng_registry`
> (`synthetic_draw_injector.py:30`) and `load_prng_config` (`:84`) refuses to run if
> `optimal_window_config.json` is missing or has no `prng_type` (`:93-108`). `--inject-one`,
> `--daemon`, `--interval`, `--no-flag` exist (`:535`, `:537`, `:545`, `:553`). **No "flag-triggered
> after Step 6" mode exists:** the only tracked launcher of the injector is the soak script
> (`run_soak_c.sh:390`, `--daemon --interval 60`); nothing in `run_pipeline` starts it
> (DESCRIPTION-ONLY). **Host state (2026-11-11):** `optimal_window_config.json` is absent from the
> project root, so the injector would refuse to run today (audit addendum A.4).

### Validation Purpose

With known `true_seed`:
- System generates **consistent, reproducible test sequences**
- Allows measurement of whether pattern learning improves over iterations
- Validates that the feedback loop actually learns surface patterns

> **Note:** The `true_seed` is used for reproducible test data generation, NOT as a discovery target. The system learns output patterns, not seed values.

### Convergence Expectation

> **With a correct PRNG hypothesis, learned patterns should produce measurable prediction lift over random baseline.**

| Metric | Target | Failure Indicates |
|--------|--------|-------------------|
| Hit Rate (Top-20) | > 5% (vs 0.1% random) | Pattern extraction not working |
| Confidence Calibration | Correlation > 0.3 | Confidence scores meaningless |
| Hit Rate Trend | Non-decreasing over N draws | Learning loop not improving |

This provides a quantitative pass/fail test for functional mimicry quality.

> **Status 2026-11-11 — UNVERIFIED (U5).** No host record measures Hit@20 against the > 5 % target.
> Through Chapter 13 it could not have been measured: every archived diagnostics file graded an
> empty pool (233 of 233, audit addendum A.0). The policy file still carries this target
> (`watcher_policies.json:74` `minimum_hit_rate: 0.05`); two of its sibling keys are marked
> DEPRECATED as seed-ranking metrics (`:68-71`), and `:13` still says of `true_seed` "System should
> 'find' this" — seed-finding wording the §1 framing excludes (recorded; config not edited here).

### Safety: Test Mode Gating

Synthetic injection **cannot run** unless explicitly enabled:

```json
{
  "test_mode": true,
  "synthetic_injection": {
    "enabled": true
  }
}
```

Both flags must be `true`. No ambiguity. No accidents.

> **Verified 2026-11-11.** `validate_injection_enabled` requires both
> (`synthetic_draw_injector.py:51-77`). Tracked policy: `test_mode: false` (`watcher_policies.json:7`),
> `synthetic_injection.enabled: true` (`:10`) — so injection is refused in the tracked configuration.

### Metadata Tagging

Synthetic draws are tagged for diagnostics (not logic):

```json
{
  "draw_id": "SYNTHETIC-2026-01-11-001",
  "draw_source": "synthetic",
  "true_seed": 12345,
  "generated_at": "2026-01-11T17:00:00Z"
}
```

This enables later analysis of convergence graphs without affecting pipeline behavior.

---

## 7. Inputs

All inputs are **read-only**. Chapter 13 never mutates source files.

| Artifact | Source | Purpose |
|----------|--------|---------|
| `prediction_pool.json` | Step 6 | Predictions to validate |
| `confidence_map.json` | Step 6 | Confidence calibration data |
| `lottery_history.json` | External updater | Ground truth (append-only) |
| `best_model.meta.json` | Step 5 | Model provenance |
| `run_metadata.json` | WATCHER | Lineage & parameters |
| `survivors_with_scores.json` | Step 3 | Current survivor state |

> **Corrected 2026-11-11 — what the code reads, and who writes it.** Defaults:
> `chapter_13_diagnostics.py:49-56`; the orchestrator calls `generate_diagnostics()` with all
> defaults (`chapter_13_orchestrator.py:282`).
>
> | Artifact (as read) | Real producer at `a15b524` | Label |
> |---|---|---|
> | `prediction_pool.json` | **None.** Step 6 writes `predictions/next_draw_prediction.json` (`prediction_generator.py:896-907` `_save_predictions`), whose `predictions` is a list of ints (`:849`), while the grader calls `pred.get("value", …)` on each element (`chapter_13_diagnostics.py:243`). A missing file silently becomes an empty pool (`:114`). | **B1, BROKEN** |
> | `confidence_map.json` | **None** (missing → `{}`, `:129`). | B1 |
> | `lottery_history.json` | Only the test/manual tools (`synthetic_draw_injector.py:311`, `draw_ingestion_daemon.py:231`). The production scraper writes `daily3.json` (`daily3_scraper.py:14`, `:99`). | **B2, BROKEN** |
> | `best_model.meta.json` (project root) | **None.** Step 5 writes `models/reinforcement/best_model.meta.json`; the reader also expects top-level `feature_hash` / `feature_importance` (`chapter_13_diagnostics.py:448-449`) that the sidecar nests under `feature_schema`. | **B19, BROKEN** (new this pass) |
> | `run_metadata.json` | Not read by `chapter_13_diagnostics.py` (no occurrence). | DESCRIPTION-ONLY |
> | `survivors_with_scores.json` | Step 3. | accurate |
>
> **History of B1 (git, newly visible):** `eb44bec` (2026-01-11) removed `prediction_pool.json` from
> the Step 6 manifest; the Chapter 13 code that reads it landed the next day (`a7c13f7`); `c6e3989`
> (2026-01-28) made `predictions/next_draw_prediction.json` Step 6's single canonical output "per Team
> Beta ruling" (commit message; no standalone ruling file found). `prediction_generator.py` never
> contained `prediction_pool.json` in any commit. The gitignored `agent_manifests/definitions.json:80-84`
> names a fourth set of Step 6 outputs (`next_draw_prediction.json`, `prediction_pools.json`,
> `ranked_survivors.json`); it has no tracked reader.
>
> **Host (2026-11-11):** `prediction_pool.json`, `confidence_map.json`, root `best_model.meta.json` and
> `run_metadata.json` do not exist; `predictions/next_draw_prediction.json` does (2026-03-07). No unit,
> cron job, untracked or ignored script, or commit writes `prediction_pool.json` (audit addendum A.7).
> **"All inputs are read-only"** is accurate for this module: `chapter_13_diagnostics.py` writes only
> its own output and archive (`save_diagnostics`, `:891-926`).

---

## 8. Post-Draw Diagnostics

### 8.1 Purpose

Generate structured diagnostics comparing predictions to reality.

Output file: `post_draw_diagnostics.json`

### 8.2 Schema

```json
{
  "schema_version": "1.0.0",
  "run_id": "chapter13_20260111_172000",
  "draw_id": "CA-D3-2026-01-11-EVE",
  "draw_timestamp": "2026-01-11T19:30:00Z",
  "data_fingerprint": "f7d1c918",
  
  "prediction_validation": {
    "pool_size": 20,
    "exact_hits": 1,
    "near_hits_within_5": 3,
    "best_rank": 3,
    "median_rank": 47,
    "pool_coverage": 0.15
  },
  
  "confidence_calibration": {
    "mean_confidence": 0.41,
    "max_confidence": 0.94,
    "confidence_spread": 0.66,
    "predicted_vs_actual_correlation": 0.23,
    "overconfidence_detected": false,
    "underconfidence_detected": true
  },
  
  "survivor_performance": {
    "hit_survivors": [12345, 67890],
    "top_10_hit_rate": 0.10,
    "decay_candidates": [11111, 22222, 33333],
    "reinforce_candidates": [12345]
  },
  
  "feature_diagnostics": {
    "dominant_feature_shift": false,
    "entropy_change": -0.18,
    "top_feature_turnover": 0.22,
    "schema_hash_match": true
  },
  
  "pipeline_health": {
    "window_decay": 0.31,
    "survivor_churn": 0.19,
    "model_stability": "stable",
    "consecutive_misses": 2
  },
  
  "summary_flags": [
    "WEAK_SIGNAL",
    "UNDERCONFIDENT_MODEL",
    "RETRAIN_RECOMMENDED"
  ],
  
  "recommended_actions": {
    "retrain_step_5": true,
    "rerun_step_3": true,
    "rerun_step_1": false,
    "parameter_adjustments": {
      "confidence_threshold": "-0.05",
      "pool_size": "+5"
    }
  }
}
```

### 8.3 Diagnostic Metrics

| Metric | Formula | Threshold |
|--------|---------|-----------|
| Exact Hit Rate | hits / pool_size | > 0.05 good |
| Confidence Correlation | corr(predicted, actual) | > 0.3 healthy |
| Feature Turnover | changed_top_10 / 10 | < 0.3 stable |
| Survivor Churn | new_survivors / total | < 0.2 stable |
| Consecutive Misses | sequential zero-hit draws | < 5 acceptable |

> **Corrected 2026-11-11.** The schema in §8.2 is an illustrative example. What the code emits:
> - Output and archive are as stated (`generate_diagnostics`, `chapter_13_diagnostics.py:750`;
>   `save_diagnostics` `:891-926`; archive dir `diagnostics_history`, `:57`). Accurate.
> - The near-hit key is `near_hits_within_threshold` with `near_threshold: 5`, plus `hit_values`,
>   `near_hit_values`; diagnostics also carry `selfplay_health` (`:871-881`) and
>   `recommended_actions.human_review_required` (host file, 2026-02-15). DESCRIPTION-ONLY.
> - **"Exact Hit Rate = hits / pool_size"** is computed for the **current draw only**
>   (`:529-531`), so it is 0 or 1/pool_size per draw (B3).
> - **"Survivor Churn = new_survivors / total"** is actually the relative change in survivor
>   **count** (`:555-559`), with the comment that identity tracking would be needed; **window decay**
>   is the relative fall in single-draw hit rate against the previous draw (`:549-553`), so 0 or 1.0
>   (B4, BROKEN).
> - `best_rank` is the minimum rank over exact hits **and** predictions within ±5 (`:246-258`) — so the
>   S140b downstream score (`chapter_13_orchestrator.py:292-319`) counts a near value as a hit (audit
>   note N1; with a 20-entry pool the three Hit@K tiers collapse to one event,
>   `docs/TB_RULING_S177_KPI_GOVERNANCE.md:303-325`).
> - Feature diagnostics are blind at `a15b524` (B19, §7): `current_hash` is always None and
>   `schema_hash_match` defaults to `True` (`:448-464`).
> - **Host:** the last file (draw `SYNTHETIC-2026-02-15-424`) has `pool_size 0`, `exact_hits 0`,
>   `consecutive_misses 233`; the 233 archived files all have `pool_size 0`.

---

## 9. Label Refresh Mechanism

### 9.1 How Labels Update

Live draws do **not** directly edit `holdout_hits`.

The mechanism:
1. New draw appended to `lottery_history.json`
2. Step 3 re-runs with expanded history
3. `holdout_hits` recomputed naturally for all survivors
4. Step 5 retrains on refreshed labels

### 9.2 Temporal Causality Preserved

```
Time T:   history = [draw_1, ..., draw_N]
          holdout = [draw_N+1, ..., draw_N+K]
          
Time T+1: history = [draw_1, ..., draw_N+1]  ← expanded
          holdout = [draw_N+2, ..., draw_N+K+1]  ← shifted
```

Labels evolve through data accumulation, not mutation.

### 9.3 Implementation

```python
def refresh_labels():
    """
    Re-run Step 3 to recompute holdout_hits with expanded history.
    
    Does NOT modify Step 3 code.
    Simply re-invokes with updated lottery_history.json.
    """
    subprocess.run([
        "python3", "survivor_scorer.py",
        "--lottery-data", "lottery_history.json",  # Now has +1 draw
        "--forward-survivors", "forward_survivors.json",
        "--reverse-survivors", "reverse_survivors.json",
        "--output", "survivors_with_scores.json"
    ])
```

> **SUPERSEDED (2026-11-11) — §9.1 step 1–2 and the §9.3 code block.** Replaced by this note and
> §22 B2. The function `refresh_labels()` does not exist, and Step 3 is not `survivor_scorer.py
> --lottery-data lottery_history.json`: WATCHER's Step 3 is `run_step3_full_scoring.sh`
> (`agents/watcher_agent.py:391`), whose manifest reads `bidirectional_survivors_binary.npz`,
> `train_history.json` and `holdout_history.json` (`agent_manifests/full_scoring.json:118-121`).
> Step 5 reads `train_history.json` (`agent_manifests/reinforcement.json:8`); Step 6 reads `daily3.json`
> (`agent_manifests/prediction.json:180`). The train/holdout split is written only by Step 1
> (`window_optimizer.py:1100-1102`), which is not in the 3→5→6 loop. **So a draw appended to
> `lottery_history.json` does not reach any step that relearns (B2, BROKEN).** Host: `lottery_history.json`
> holds 5,148 draws (5,000 `synthetic_bootstrap`, 148 `synthetic`, last 2026-02-15); `daily3.json` holds
> 18,068 (last 2026-02-26); no writer is shared. The governed single dataset authority is the
> pointer manifest of S180 Phase 6-P0 (commit `131787d`); `docs/TFM_PROJECT_FACTS_SKILL.md` §2.10
> ("Dataset authority") records the Beta position that adding a draw does not leave prior scores valid.
> §9.2 (temporal causality as a design statement) is kept as the intended behaviour.

---

## 10. Retrain Trigger Policies

### 10.1 What v1 Implements

Chapter 13 v1 implements trigger **definition** and **evaluation**:

| Aspect | v1 Status |
|--------|-----------|
| Trigger definition | ✅ Implemented |
| Trigger evaluation | ✅ Implemented |
| Trigger execution | ⚠️ **Requires human approval** |

**Important:** Automatic trigger execution without human approval is **deferred** to future versions. See [Section 20: Deferred Extensions](#20-deferred-extensions-roadmap).

### 10.2 Automatic Triggers (Require Approval)

WATCHER evaluates and **recommends** Steps 3→5→6 when:

| Condition | Threshold | Rationale |
|-----------|-----------|-----------|
| N draws accumulated | ≥ 10 | Statistical significance |
| Confidence drift | correlation < 0.2 | Model miscalibrated |
| Consecutive misses | ≥ 5 | Performance collapse |
| Hit rate collapse | < 0.01 over 20 draws | Model failure |

> **Corrected 2026-11-11.**
> - **§10.1 is accurate.** Definition and evaluation are implemented (`chapter_13_triggers.py:201-370`);
>   execution needs approval (`:369`). Thresholds are as tabled (`watcher_policies.json:19-27`).
> - **Hit rate collapse is a one-draw miss detector (B3, BROKEN; governed).** `current_hit_rate` is the
>   current draw's `exact_hits / pool_size` (`chapter_13_diagnostics.py:529-531`), compared once
>   (`chapter_13_triggers.py:268-271`); `hit_rate_collapse_window: 20` (`watcher_policies.json:27`) has no
>   reader. Already ruled structurally invalid: `docs/TB_RULING_S176_WATCHER_KPI.md` §2.1 and §9
>   "Approved" (`:795-801`). The authorised repair (`docs/TB_RULING_S179_IMPLEMENTATION_AUTH.md`
>   `:525-532`: BOOTSTRAP by default, all performance/cadence/regime triggers `audit_only`) **did not
>   land** — not in the tracked tree, not in git history, not in any untracked script on the host
>   (audit addendum, U3). **That is S2, SWITCHED-OFF-BY-RULING** as to activation: lifting it would let
>   these triggers request retrains on their present definitions; nothing in code holds them back
>   except the human gate.
> - **N draws accumulated is always true after the tenth cycle (B7, BROKEN).** `runs_since_retrain` is
>   reset only by `chapter_13_triggers.py:619-621` (unreachable, B5) and `:690-692` (`--execute`).
>   Host: `.chapter13_cooldown_state.json` has `last_retrain_at: null`, `runs_since_retrain: 236`, after
>   39 WATCHER-completed and 1 CLI-completed learning-loop requests.
> - **On host data every cycle fired**: the last request lists `hit_rate_collapse`,
>   `consecutive_misses`, `confidence_drift`, `n_draws_accumulated`, `llm_proposed` together
>   (`pending_approval.json`, 2026-02-15). With an empty pool (B1) that is the expected outcome.
> - All triggers are defensive; there is no opportunity trigger, and the one positive-signal flag the
>   diagnostics can emit, `NEW_STRONG_SURVIVORS` (`chapter_13_diagnostics.py:631-632`), has no reader
>   (D2).

### 10.3 LLM-Proposed Triggers

LLM may propose retraining when diagnostics show:
- Feature importance shift
- Regime change indicators
- Entropy anomalies

Proposals require WATCHER validation before execution.

### 10.4 Regime Shift (Full Pipeline)

Trigger Steps 1→6 only when:
- Window decay > 0.5
- Survivor churn > 0.4
- LLM flags structural drift with confidence > 0.8
- Manual operator override

> **Corrected 2026-11-11 (B4, BROKEN).** Window decay and survivor churn are not the quantities named
> (§8.3 note; `chapter_13_diagnostics.py:549-559`). `llm_confidence_threshold: 0.8`
> (`watcher_policies.json:36`) has no reader; the LLM's `recommended_action` / `retrain_scope` are
> logged (`chapter_13_orchestrator.py:388`) and the approval request carries the trigger's action,
> not the LLM's (`:415`). A regime-shift rerun would also execute through the range path (B6, §12).
> Partly governed: S176 lists churn and decay as having no empirical baseline. On host data
> `HIGH_WINDOW_DECAY` appeared in 1 of 233 diagnostics and `REGIME_SHIFT_POSSIBLE` in 1 of 233.

---

## 11. LLM Role: Strategist

The LLM is **advisory only**. It does not execute.

### 11.1 LLM Capabilities

| Action | Allowed |
|--------|---------|
| Interpret diagnostics | ✅ |
| Identify cross-run trends | ✅ |
| Propose parameter adjustments | ✅ |
| Flag regime shifts | ✅ |
| Explain performance changes | ✅ |

### 11.2 LLM Restrictions

| Action | Forbidden |
|--------|-----------|
| Modify files | ❌ |
| Execute code | ❌ |
| Apply parameters directly | ❌ |
| Override WATCHER | ❌ |
| Bypass validation | ❌ |

### 11.3 System Prompt

```
You are an analytical advisor for a probabilistic research system.

HARD CONSTRAINTS:
- You do NOT execute actions
- You do NOT modify parameters directly
- You do NOT assume stationarity
- You MUST express uncertainty

YOUR TASK:
Interpret diagnostic deltas from real-world outcomes and propose 
cautious, reversible adjustments.

If uncertainty is high, recommend NO CHANGE.
```

### 11.4 User Prompt Template

```
Given the following post-draw diagnostics:

{{ post_draw_diagnostics.json }}

Previous 5 run summaries:
{{ run_history[-5:] }}

Tasks:
1. Identify the most likely failure mode (if any)
2. Classify the issue:
   - Model calibration
   - Feature relevance
   - Window misalignment
   - Random variance
   - Regime shift
3. Propose parameter adjustments ONLY if justified
4. Assign confidence score (0.0-1.0) to each proposal
5. Recommend: RETRAIN / WAIT / ESCALATE
```

### 11.5 LLM Proposal Schema

```json
{
  "analysis_summary": "Model underconfident in recent regime",
  "failure_mode": "calibration_drift",
  "confidence": 0.78,
  
  "recommended_action": "RETRAIN",
  "retrain_scope": "steps_3_5_6",
  
  "parameter_proposals": [
    {
      "parameter": "confidence_threshold",
      "current_value": 0.7,
      "proposed_value": 0.65,
      "delta": "-0.05",
      "confidence": 0.82,
      "rationale": "Underconfidence pattern over last 8 draws"
    }
  ],
  
  "risk_level": "low",
  "requires_human_review": false,
  
  "alternative_hypothesis": "Random variance (30% probability)"
}
```

> **Verified / corrected 2026-11-11.** The LLM is advisory only: accurate (`chapter_13_orchestrator.py:373-441`;
> no apply path exists, S3). The system prompt in §11.3 matches `chapter_13_llm_advisor.py:95-104`.
> The call is grammar-constrained with a heuristic fallback and archived to `llm_proposals/`
> (`chapter_13_llm_advisor.py:299-344`, `:346-436`; host: 224 archived proposals). **§11.4 is not
> what is sent:** the orchestrator calls `analyze_diagnostics(diagnostics)` with no run history
> (`chapter_13_orchestrator.py:386`), and the prompt carries no parameter vocabulary or frozen list
> (TODO B3/B7; `chapter_13_parameter_vocabulary.py` has never been committed and is not on disk).
> DESCRIPTION-ONLY for the template; the missing vocabulary feeds B17.

---

## 12. WATCHER Role: Executor

WATCHER is the **sole authority** for execution decisions.

### 12.1 WATCHER Responsibilities

1. Detect new labeled data
2. Run Chapter 13 diagnostics
3. Invoke LLM for analysis (optional)
4. Validate LLM proposals against policies
5. Execute approved actions
6. Log all decisions

### 12.2 Execution Authority

```
LLM Proposal → WATCHER Validation → Execution (or Rejection)
```

WATCHER enforces:
- Parameter bounds
- Change magnitude limits
- Cooldown periods
- Safety constraints

### 12.3 Decision Matrix

| LLM Recommendation | WATCHER Validation | Result |
|--------------------|-------------------|--------|
| RETRAIN (high confidence) | Passes all checks | ✅ Execute |
| RETRAIN (low confidence) | Below threshold | ⏸️ Wait |
| Parameter change | Within bounds | ✅ Apply |
| Parameter change | Exceeds bounds | ❌ Reject |
| ESCALATE | Any | 🚨 Human review |

> **Corrected 2026-11-11 — who executes, and how.**
> - "Execute approved actions": the only approval path that executes is
>   `agents/watcher_agent.py --approve` (`:3741-3779`). In production the WATCHER daemon notifies and
>   waits (`:3354-3395`).
> - **B5, BROKEN.** `chapter_13_orchestrator.py --approve` (`:1078-1080`) and `chapter_13_triggers.py
>   --approve` (`:923-925`) run nothing: the trigger manager is built with `watcher_agent=None`
>   (`chapter_13_orchestrator.py:125-127`), so `execute_learning_loop` returns False at
>   `chapter_13_triggers.py:603` and the request is marked failed (`:540`). Both the approval text
>   (`:484-485`) and WATCHER's Telegram message (`agents/watcher_agent.py:3379`) recommend these
>   commands. Host: the last `pending_approval.json` was approved and marked `failed` 0.34 ms later.
> - **B6, BROKEN.** An approved `[3, 5, 6]` runs Step 4 too: both executors call
>   `run_pipeline(min(steps), max(steps))` (`agents/watcher_agent.py:3774`; daemon path `:3338-3341`), and
>   `run_pipeline` walks a contiguous range (`:2717`, loop `:2802`). `dispatch_learning_loop` handles a
>   list correctly (`agents/watcher_dispatch.py:268-283`, `:303`) but is not on this path.
> - **"Parameter change → Apply"** does not happen (S3, SWITCHED-OFF-BY-RULING; §4 note).
> - WATCHER's enforcement of bounds/limits/cooldown: magnitude and count limits are enforced at
>   validation (§13); cooldown is not effective (B7); parameter bounds are not checked against a
>   source of truth (B17).

---

## 13. Acceptance & Rejection Rules

### 13.1 Automatic Rejection

WATCHER **must reject** proposals that:

- Modify any parameter by > 30%
- Reverse a change made within last 3 runs
- Violate parameter bounds
- Have confidence < 0.60
- Affect more than 3 parameters simultaneously
- Touch frozen components (Steps 1, 2, 4)

### 13.2 Automatic Acceptance

WATCHER **may accept** proposals if:

- `risk_level == "low"`
- `confidence >= 0.75`
- ≤ 2 parameters affected
- No safety violations
- Cooldown period elapsed (≥ 3 runs since last change)

### 13.3 Mandatory Escalation

WATCHER **must escalate** to human review if:

- `risk_level >= "medium"`
- Conflicting proposals exist
- `CHAOTIC` or `REGIME_SHIFT` flag present
- 3+ consecutive failures
- LLM confidence < 0.50 on all options

### 13.4 Implementation

```python
def validate_proposal(proposal: LLMProposal) -> ValidationResult:
    """Validate LLM proposal against WATCHER policies."""
    
    # Hard rejections
    if proposal.confidence < 0.60:
        return ValidationResult(accepted=False, reason="Low confidence")
    
    if proposal.risk_level in ["medium", "high"]:
        return ValidationResult(accepted=False, reason="Escalate to human")
    
    for param in proposal.parameter_proposals:
        if abs(param.delta_percent) > 30:
            return ValidationResult(accepted=False, reason=f"Delta too large: {param.parameter}")
        
        if param.parameter in FROZEN_PARAMETERS:
            return ValidationResult(accepted=False, reason=f"Frozen: {param.parameter}")
    
    if len(proposal.parameter_proposals) > 3:
        return ValidationResult(accepted=False, reason="Too many parameters")
    
    # Passed all checks
    return ValidationResult(accepted=True, reason="Passed validation")
```

> **Corrected 2026-11-11 — the code is `Chapter13AcceptanceEngine.validate_proposal`
> (`chapter_13_acceptance.py:327`), not the sketch above.**
> - Accurate: reject on confidence < 0.60 (`:359-370`); on more than 3 parameters (`:372-383`); on
>   delta > 30 % (`:400-414`, `:570-596`; `watcher_policies.json:41` `max_parameter_delta: 0.3`). Accept
>   needs low risk, confidence ≥ 0.75, ≤ 2 parameters (`:527-552`). Escalate on medium/high risk,
>   regime/chaotic flags, ≥ 3 consecutive failures (`:457-478`). Decisions are logged to
>   `acceptance_decisions.jsonl` (`:57`; host: 195 accept, 29 reject, 5 escalate).
> - **"Touch frozen components (Steps 1, 2, 4)" (B17, BROKEN; latent).** The frozen set is eight
>   literal names (`:72-84`, checked at `:388`); Step-1 parameters such as `forward_threshold`,
>   `offset`, `skip_min` are not in it.
> - **"Reverse a change within the last 3 runs" and "cooldown ≥ 3 runs" are vacuous** (consequence of
>   S3): the change history is never written (`record_applied_changes`, `:635`, and `increment_run`,
>   `:663`, have no caller; host: `.parameter_change_history.json` does not exist).
> - "Violate parameter bounds" and "conflicting proposals" / "LLM confidence < 0.50 on all options"
>   have no dedicated check found in `validate_proposal` (DESCRIPTION-ONLY at the design level; nothing
>   is applied, S3).

---

## 14. Outputs

### 14.1 Diagnostic Outputs

| File | Purpose | Retention |
|------|---------|-----------|
| `post_draw_diagnostics.json` | Current run diagnostics | Permanent |
| `diagnostics_history/` | Historical diagnostics | 1 year |
| `llm_proposals/` | LLM analysis archive | Permanent |

### 14.2 Decision Outputs

| File | Purpose |
|------|---------|
| `watcher_decision_log.json` | All accept/reject decisions |
| `parameter_change_history.json` | Applied parameter changes |
| `retrain_history.json` | When/why retraining occurred |

### 14.3 Audit Trail

Every decision is logged with:
- Timestamp
- Input diagnostics hash
- LLM proposal (if any)
- Validation result
- Action taken
- Outcome (next run performance)

> **Corrected 2026-11-11 (D5, DESCRIPTION-ONLY; plus S3).** File names in code:
> acceptance decisions → `acceptance_decisions.jsonl` (`chapter_13_acceptance.py:57`); parameter history
> → `.parameter_change_history.json` (`:56`; never written, S3); cycle history →
> `chapter13_cycle_history.jsonl` (`chapter_13_orchestrator.py:83`). `watcher_policies.json:88-92` still
> names `watcher_decision_log.json`, `parameter_change_history.json`, `retrain_history.json`. Of these
> only `retrain_history.json` is used by tracked code (`chapter_13_triggers.py:55` `TRIGGER_HISTORY_FILE`;
> host copy 802 bytes, 2026-02-15); the other two names occur in no tracked non-backup `*.py` and do not
> exist on the host.
> No "retention: 1 year" pruning was looked for. Host: 233 files in `diagnostics_history/`, 224 in
> `llm_proposals/`, 231 cycles in `chapter13_cycle_history.jsonl` (outcomes: `auto_executed_test_mode`
> 120, `pending_approval` 77, `proposal_rejected` 29, `escalated` 5). "Outcome (next run performance)" is
> not recorded per decision.

---

## 15. Integration With Steps 1-6

### 15.1 No Code Changes Required

Steps 1-6 are **pure functions**:
```
Outputs = Pipeline(History, Config)
```

Chapter 13 modifies:
```
History ← History + New Draw
```

Then re-invokes the pipeline.

### 15.2 Step Interaction Matrix

| Step | Chapter 13 Interaction |
|------|----------------------|
| 1 | Re-invoke only on regime shift |
| 2 | Re-invoke only on regime shift |
| 3 | **Re-invoke** to refresh labels |
| 4 | Re-invoke only on architecture change |
| 5 | **Re-invoke** to retrain model |
| 6 | **Re-invoke** to generate new predictions |

### 15.3 Partial vs Full Reruns

```python
def execute_learning_loop(diagnostics: Diagnostics):
    """Execute appropriate pipeline subset based on diagnostics."""
    
    if diagnostics.requires_regime_reset():
        # Full pipeline
        run_steps([1, 2, 3, 4, 5, 6])
    
    elif diagnostics.requires_retrain():
        # Learning loop only
        run_steps([3, 5, 6])
    
    elif diagnostics.requires_prediction_refresh():
        # Inference only
        run_steps([6])
    
    else:
        # No action needed
        log("Diagnostics healthy, no rerun required")
```

> **Corrected 2026-11-11.**
> - **§15.1 "History ← History + New Draw, then re-invoke" (B2, BROKEN).** The history Chapter 13 grows
>   is not the one Steps 3/5/6 read (§9 note).
> - **§15.2–15.3 step lists are accurate as requests** (`chapter_13_triggers.py:450-458`
>   `_get_steps_for_action`: `LEARNING_LOOP` → `[3, 5, 6]`, `FULL_PIPELINE` → `[1, 2, 3, 4, 5, 6]`,
>   `STEP_6_ONLY` → `[6]`), **but not as executions (B6, BROKEN):** the approval paths run the
>   contiguous range `min..max`, so `[3, 5, 6]` also runs Step 4, which §5 lists as never per-draw.
>   The sketched `execute_learning_loop(diagnostics)` is not the code; the trigger manager's
>   `execute_learning_loop` is the B5 dead path.
> - Step 3 in this loop consumes Step 1's NPZ output (`agent_manifests/full_scoring.json:118`
>   `bidirectional_survivors_binary.npz`); see §23 for the NPZ contract.

---

## 16. Convergence Guarantees

### 16.1 Why This Doesn't Spiral

The system remains stable because:

1. **Data grows slowly** — One draw at a time; strong regularization
2. **Labels are sparse** — Hits are rare; noise naturally penalized
3. **Retraining is gated** — Not every draw triggers retrain
4. **LLM doesn't change weights directly** — Proposes only; WATCHER enforces
5. **Parameter bounds enforced** — No extreme jumps allowed
6. **Cooldown periods** — Prevents thrashing

### 16.2 Convergence Metrics

Monitor these for system health:

| Metric | Healthy Range | Action if Exceeded |
|--------|--------------|-------------------|
| Parameter volatility | < 10% per 10 runs | Increase cooldown |
| Model weight churn | < 20% per retrain | Reduce learning rate |
| Prediction variance | Decreasing trend | Continue |
| Hit rate | Stable or increasing | Continue |

### 16.3 Divergence Detection

If detected, WATCHER halts and escalates:
- 5+ consecutive performance drops
- Parameter oscillation (A→B→A pattern)
- Model metrics degrading after retrain

> **Corrected 2026-11-11.**
> - **§16.1 item 6 "Cooldown periods" (B7, BROKEN):** the cooldown test
>   (`chapter_13_triggers.py:384`) needs `last_retrain_at`, which no executed retrain sets (§10 note).
>   Items 4–5 hold because nothing is applied at all (S3), not because bounds are enforced (B17).
> - **§16.2:** no code computes parameter volatility, weight churn or prediction-variance trends
>   (DESCRIPTION-ONLY; monitoring targets, not a code path the chapter claims exists).
> - **§16.3 divergence detection (B9, BROKEN).** `_create_halt` (`chapter_13_orchestrator.py:241-245`)
>   has no caller; `escalate_on_consecutive_rejects` and `escalate_on_divergence`
>   (`watcher_policies.json:50-52`) have no reader. The nearest live behaviour is escalation at ≥ 3
>   consecutive misses (`chapter_13_acceptance.py:475-478`).

---

## 17. Safety & Ethics

### 17.1 Explicit Prohibitions

Chapter 13 **forbids**:
- Automated wagering
- External execution hooks
- Financial transactions
- Unlogged state changes
- Silent parameter mutation

### 17.2 Audit Requirements

- All decisions logged with full context
- All LLM interactions recorded
- All parameter changes reversible
- 25-year data retention capability

### 17.3 Human Override

At any time, human operator can:
- Set halt flag
- Override any decision
- Force specific parameters
- Trigger manual reruns
- Disable LLM entirely

> **Verified 2026-11-11.** Halt, approve/reject and LLM-disable controls exist
> (`chapter_13_orchestrator.py:1003-1019`, `:1078-1084`); halt files are checked each cycle (`:158-175`).
> Caveat: of the three advertised approve commands only `agents/watcher_agent.py --approve` executes
> (B5). "Force specific parameters" has no Chapter 13 code path (S3). §17.2 "all parameter changes
> reversible" is vacuous while none is applied.

### 17.4 Research Classification

This system is:
- ✅ Research-grade
- ✅ Auditable
- ✅ Non-exploitative
- ✅ Academically publishable

---

## 18. Configurable Parameter Reference

### 18.1 Step 1 — Window Optimizer

| Parameter | Location | Type | Bounds | Effect |
|-----------|----------|------|--------|--------|
| `window_size` | `optimal_window_config.json` | int | 100-500 | History window for sieve |
| `offset` | `optimal_window_config.json` | int | 0-50 | Starting offset |
| `skip_min` / `skip_max` | `optimal_window_config.json` | int | 1-500 | Skip range for PRNG |
| `forward_threshold` | `optimal_window_config.json` | float | 0.01-0.15 | Forward sieve tolerance |
| `reverse_threshold` | `optimal_window_config.json` | float | 0.01-0.15 | Reverse sieve tolerance |
| `trials` | `agent_manifests/window_optimizer.json` | int | 10-200 | Optuna trials |

### 18.2 Step 2 — Scorer Meta-Optimizer

| Parameter | Location | Type | Bounds | Effect |
|-----------|----------|------|--------|--------|
| `sample_size` | `optimal_scorer_config.json` | int | 10000-100000 | Survivors to score |
| `hidden_layers` | `optimal_scorer_config.json` | str | "64", "128_64" | NN architecture |
| `dropout` | `optimal_scorer_config.json` | float | 0.1-0.5 | Regularization |
| `learning_rate` | `optimal_scorer_config.json` | float | 1e-4 - 1e-2 | Training LR |
| `batch_size` | `optimal_scorer_config.json` | int | 32-256 | Batch size |

### 18.3 Step 3 — Full Scoring

| Parameter | Location | Type | Bounds | Effect |
|-----------|----------|------|--------|--------|
| `chunk_size` | `agent_manifests/full_scoring.json` | int | 1000-50000 | GPU batch size |
| `holdout_ratio` | Config | float | 0.1-0.3 | Train/holdout split |

### 18.4 Step 4 — ML Meta-Optimizer

| Parameter | Location | Type | Bounds | Effect |
|-----------|----------|------|--------|--------|
| `max_pool_size` | `reinforcement_engine_config.json` | int | 100-10000 | Survivor pool cap |
| `hidden_layers` | `reinforcement_engine_config.json` | list | [64]-[256,128,64] | NN depth |
| `epochs` | `reinforcement_engine_config.json` | int | 50-300 | Training epochs |

### 18.5 Step 5 — Anti-Overfit Training

| Parameter | Location | Type | Bounds | Effect |
|-----------|----------|------|--------|--------|
| `model_type` | `agent_manifests/reinforcement.json` | choice | xgboost/lightgbm/catboost/neural_net | Model family |
| `n_estimators` | `best_model.meta.json` | int | 100-1000 | Trees (boosting) |
| `max_depth` | `best_model.meta.json` | int | 3-10 | Tree depth |
| `learning_rate` | `best_model.meta.json` | float | 0.01-0.3 | Boosting LR |
| `k_folds` | `agent_manifests/reinforcement.json` | int | 3-10 | CV folds |
| `trials` | `agent_manifests/reinforcement.json` | int | 20-100 | Optuna trials |

### 18.6 Step 6 — Prediction Generator

| Parameter | Location | Type | Bounds | Effect |
|-----------|----------|------|--------|--------|
| `pool_size` | `agent_manifests/prediction.json` | int | 5-100 | Prediction pool size |
| `k` | `agent_manifests/prediction.json` | int | 10-50 | Top-K predictions |
| `confidence_threshold` | Script default | float | 0.5-0.95 | Min confidence |

### 18.7 Chapter 13 — Feedback Loop

| Parameter | Location | Type | Bounds | Effect |
|-----------|----------|------|--------|--------|
| `retrain_after_n_draws` | `watcher_policies.json` | int | 5-50 | Retrain trigger |
| `confidence_drift_threshold` | `watcher_policies.json` | float | 0.1-0.3 | Calibration alert |
| `max_consecutive_misses` | `watcher_policies.json` | int | 3-10 | Failure trigger |
| `parameter_change_limit` | `watcher_policies.json` | float | 0.1-0.3 | Max delta per change |
| `cooldown_runs` | `watcher_policies.json` | int | 2-5 | Runs between changes |

### 18.8 Frozen (Never Autonomous)

| Component | Location | Reason |
|-----------|----------|--------|
| Step ordering | `watcher_agent.py` | Causal dependency |
| Feature schema | `config_manifests/feature_registry.json` | Hash validation |
| PRNG algorithms | `prng_registry.py` | Mathematical invariant |
| Sieve math | `bidirectional_sieve_filter.py` | Core logic |
| Pydantic schemas | `models/*.py` | Contract enforcement |

> **Corrected 2026-11-11 — reference tables (D5, DESCRIPTION-ONLY unless marked).** These tables are a
> design-time reference and were not re-derived row by row. Points checked:
> - **§18.1 bounds disagree with the code's defaults.** `window_optimizer.py:72-79` (defaults used when
>   `distributed_config.json` has no `search_bounds`): `window_size` 2–50, `offset` 0–100, `skip_min`
>   0–10, `skip_max` 10–250, `forward_threshold` / `reverse_threshold` 0.40–0.75 (default 0.50). Live
>   values come from `distributed_config.json`, not read here.
> - **§18.1 `offset`:** after the window-anchor change (Brief I) the fused `offset` was split into
>   `window_anchor` + `generator_phase` inside the RANGE-MINER/coordinator path; `optimal_window_config.json`
>   still carries `offset`, and its migration is explicitly **Brief II** (`docs/S172_WINDOW_ANCHOR_BRIEF_I.md:360-368`).
>   See §23.
> - **§18.1 location:** `optimal_window_config.json` is gitignored and **absent on the host root** today.
> - **§18.7:** the delta key is `max_parameter_delta` (`watcher_policies.json:41`), not
>   `parameter_change_limit`; `cooldown_runs: 3` is `:45` (and ineffective, S3/B7).
> - **§18.4:** `reinforcement_engine_config.json` is not in the working tree root on the host.
> - **§18.8:** `bidirectional_sieve_filter.py` does not exist; the certifying sieve engine is
>   RANGE-MINER (`miner/`), with `sieve_filter.py` / `reverse_sieve_filter.py` on the legacy path (§23).
>   The "Pydantic schemas" are dataclasses (§2 note). `config_manifests/feature_registry.json` exists
>   (tracked).
> - The LLM can in fact name any of these as a free string; none is protected by the frozen check
>   except the eight names in `FROZEN_PARAMETERS` (B17).

---

## 19. Implementation Checklist

### Phase 1: Draw Ingestion
- [ ] `draw_ingestion_daemon.py` — Monitors for new draws
- [ ] `synthetic_draw_injector.py` — Test mode draw generation
  - Reads PRNG type from `optimal_window_config.json` (no hardcoding)
  - Uses `prng_registry.py` (same as Steps 1-6)
  - Modes: manual (`--inject-one`), daemon (`--daemon --interval 60`), flag-triggered
- [ ] Append-only history updates
- [ ] Fingerprint change detection
- [ ] `watcher_policies.json` — Includes test_mode and synthetic_injection settings

### Phase 2: Diagnostics Engine
- [ ] `chapter_13_diagnostics.py` — Core diagnostic generator
- [ ] Prediction vs reality comparison
- [ ] Confidence calibration metrics
- [ ] Survivor performance tracking
- [ ] Feature drift detection
- [ ] Generate `post_draw_diagnostics.json`
- [ ] Create `diagnostics_history/` archival

### Phase 3: LLM Integration
- [ ] `chapter_13_llm_advisor.py` — LLM analysis module
- [ ] `llm_proposal_schema.py` — Pydantic model for proposals
- [ ] `chapter_13.gbnf` — Grammar constraint
- [ ] System/user prompt templates
- [ ] Integration with existing LLM infrastructure

### Phase 4: WATCHER Policies
- [ ] Acceptance/rejection rules
- [ ] Retrain trigger thresholds
- [ ] Cooldown enforcement
- [ ] Escalation handlers

### Phase 5: Orchestration
- [ ] Partial rerun logic (Steps 3→5→6)
- [ ] Full rerun trigger (Steps 1→6)
- [ ] Decision logging
- [ ] Audit trail

### Phase 6: Testing
- [ ] Synthetic draw injection
- [ ] Proposal validation tests
- [ ] Convergence monitoring
- [ ] Divergence detection tests

> **SUPERSEDED (2026-11-11) as a status record (D8).** The boxes above were never ticked here; the
> status lives in `docs/CHAPTER_13_IMPLEMENTATION_PROGRESS_v3_9.md` and
> `docs/CHAPTER_13_SECTION_19_UPDATED.md` ("Full Autonomy Achieved", which this pass does not accept as
> written — D7). State at `a15b524`, by phase: **1** files exist and work for test/manual draws; no
> production draw source raises the flag (B2, B18). **2** diagnostics exist; inputs unconnected
> (B1, B19); survivor tracking empty (B8). **3** advisor exists; proposals are dataclasses (D6).
> **4** rules exist; cooldown ineffective (B7). **5** partial/full rerun requests exist but execute as
> ranges (B6); decision logging exists. **6** synthetic injection and soaks ran (host records; audit
> addendum U1, U7); divergence detection has no code (B9). Note also that `docs/CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md`
> is a near-duplicate of this file (one added note), left untouched by this pass.

---

## 20. Deferred Extensions (Roadmap)

The following extensions are **not required** for Chapter 13 v1 correctness, safety, or autonomy guarantees. They are optional enhancements for future implementation.

### 20.1 Extension #1: Step-6 Backtesting Hooks

**Status:** 🔲 Deferred

**What It Does:**
Adds an offline replay capability to Step 6 that allows historical draws to be replayed through prediction logic.

**Enables:**
- Controlled regression testing
- Confidence sanity checks
- Historical comparison without contaminating live learning

**Why Deferred:**
Chapter 13 v1 already validates live predictions correctly. Backtesting is diagnostic-only.

---

### 20.2 Extension #2: Confidence Calibration Curves (Rolling)

**Status:** 🔲 Deferred

**What It Does:**
Tracks whether predicted confidence values correspond to observed hit rates **over time**.

**Example:**
- Predictions with 0.80 confidence should succeed ~80% of the time
- Drift indicates miscalibration

**v1 vs Extension:**
| Aspect | v1 | Extension |
|--------|-----|-----------|
| Point metrics | ✅ | ✅ |
| Rolling curves | ❌ | ✅ |
| Longitudinal analysis | ❌ | ✅ |

**Why Deferred:**
Chapter 13 v1 already reports confidence deltas. Rolling curves are a long-horizon quality metric.

---

### 20.3 Extension #3: Autonomous Trigger Execution

**Status:** 🔲 Deferred

**What It Does:**
Allows WATCHER to **automatically execute** retrain triggers without human approval.

**v1 vs Extension:**
| Aspect | v1 | Extension |
|--------|-----|-----------|
| Trigger definition | ✅ | ✅ |
| Trigger evaluation | ✅ | ✅ |
| Trigger execution | ⚠️ Manual/Approved | ✅ Fully autonomous |
| Human-in-the-loop | Required | Optional |

**Why Deferred:**
Automated retraining is powerful and should only activate after stability is proven. v1 requires human approval as a safety gate.

> **Status 2026-11-11 — S1, SWITCHED-OFF-BY-RULING.** Held by this section (attributed to Team Beta in
> the version note), by `docs/PIPELINE_BEHAVIOUR_MODEL.md:954` ("approve a Chapter-13 retrain request —
> human, unless BOTH `test_mode` and `auto_approve_in_test_mode` are true"), and by
> `docs/TB_RULING_S179_IMPLEMENTATION_AUTH.md:525-532` ("no autonomous retraining enabled"). Code
> agrees (`chapter_13_triggers.py:369`). **What lifting it would enable:** a fired trigger could run
> Steps 3→5→6 (or 1→6) unattended. **What else would have to change for that to mean anything:** there
> is no production auto-execute path to switch on — setting `v1_approval_required.retrain_execution`
> false yields the label `auto_execute_disabled` and no run (`chapter_13_orchestrator.py:419-425`), the
> test-mode orchestrator route labels `auto_executed_test_mode` and runs nothing (`:410-412`, B16), and
> WATCHER's auto-dispatch is conditioned on `test_mode` (`agents/watcher_agent.py:3317-3331`); then B1,
> B2, B3/B4, B6, B7. Owner decision; no recommendation made here.

---

### 20.4 Extension #4: Convergence Dashboards

**Status:** 🔲 Deferred

**What It Does:**
Provides human-readable visualizations of:
- Survivor convergence
- Confidence stability
- Parameter drift
- Model improvement rate

**Key Constraint:**
- No control authority
- No parameter mutation
- Observability only

**Why Deferred:**
Autonomy does not require dashboards. Dashboards support trust, audit, and debugging—not execution.

---

### 20.5 Extension Summary Table

| Extension | Description | v1 Status | Deferred |
|-----------|-------------|-----------|----------|
| #1 | Backtesting Hooks | Not present | ✅ |
| #2 | Calibration Curves (rolling) | Point metrics only | ✅ |
| #3 | Autonomous Execution | Manual approval required | ✅ |
| #4 | Convergence Dashboards | Not present | ✅ |

**None of these extensions duplicate existing code.** They consume existing outputs and operate post-hoc.

---

## Summary

**Chapter 13 closes the autonomy loop.**

| Before Chapter 13 | After Chapter 13 |
|-------------------|------------------|
| Pipeline runs correctly | Pipeline **improves** continuously |
| Models train once | Models **retrain** on live data |
| Predictions generated | Predictions **validated** against reality |
| Learning stops at execution | Learning **never stops** |

**The architecture guarantees:**
- Structure is frozen
- Learning happens via data accumulation
- Adaptation happens via parameter space
- Strategy is suggested by LLMs
- Authority remains with WATCHER

You are not building a gambler.  
You are building a **self-correcting inference system**.

> **Corrected 2026-11-11 (D2, DESCRIPTION-ONLY).** "Improves continuously" / "learning never stops"
> describes the design goal. At `a15b524` every trigger is defensive (`chapter_13_triggers.py:246-319`),
> retraining needs a human (S1), the graded pool is empty on every recorded cycle (B1), and the draw
> history the loop grows does not reach the relearning steps (B2). "Authority remains with WATCHER":
> execution authority is with the human approver; WATCHER executes only on `--approve` or in test mode.

---

## Version History — addendum

```
Version 1.3.0 — 2026-11-11 (documentation correction pass, base a15b524)
- Added §1 framing block, status header, §0 (how produced), §22–§26
- Corrected in place, with visible notes: §1, §2, §3.3, §4, §6, §6.1, §7, §8.3, §9 (§9.1/§9.3 SUPERSEDED),
  §10.2, §10.4, §11, §12, §13.4, §14, §15, §16, §17.3, §18, §19 (SUPERSEDED as status), §20.3, Summary, §21
- No text removed. Authority: docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md REV1
```

---

## Version History

```
Version 1.1.0 — January 11, 2026
- Added Section 20: Deferred Extensions (Roadmap)
- Clarified v1 trigger execution requires human approval
- Added Team Beta extension proposals
- Updated Table of Contents

Version 1.0.0 — January 11, 2026
- Initial release
- Complete circular learning loop specification
- LLM strategist / WATCHER executor separation
- Acceptance/rejection policies
- Convergence guarantees
- Full integration with Steps 1-6
```

---

**END OF CHAPTER 13** *(v1.1.0 end marker, kept. §21 (v1.2.0, 2026-01-30) and §22–§26 (v1.3.0,
2026-11-11) follow it and are part of the chapter — D8.)*

---

## 21. Phase 9A: Selfplay Integration

**Added:** 2026-01-30 | **Version:** 1.2.0

### 21.1 Overview

Phase 9A connects Chapter 13 to the selfplay learning loop. Chapter 13 can:
- Validate selfplay candidates
- Promote policies to active
- Request selfplay retraining

> **Corrected 2026-11-11 (D1, DESCRIPTION-ONLY).** The governed wording for selfplay is
> "policy-conditioned evaluation harness" / "discovery front-end"
> (`docs/TB_UPDATE_SELFPLAY_REFRAMING_2026-07-28.md:20-40`), not a learning loop. Of the three
> capabilities: validate — works (`chapter_13_acceptance.py:945-946`, `validate_selfplay_candidate`
> `:733-780`); promote — works mechanically but drops the policy (B11); request — has no caller (B10).
> Selfplay's own limits are B13, B14 and S4 (§22).

### 21.2 New Files

| File | Purpose |
|------|---------|
| `learned_policy_candidate.json` | Selfplay emits candidates |
| `learned_policy_active.json` | Chapter 13 promotes approved policies |
| `watcher_requests/*.json` | Retrain request queue |
| `telemetry/learning_health_latest.json` | Selfplay health snapshot |

### 21.3 Acceptance Engine Extensions

**New CLI Commands:**
```bash
# Validate a candidate
python3 chapter_13_acceptance.py --validate-selfplay learned_policy_candidate.json

# Promote to active
python3 chapter_13_acceptance.py --promote learned_policy_candidate.json
```

**Validation Thresholds (watcher_policies.json):**
```json
{
  "selfplay": {
    "min_fitness": 0.50,
    "min_val_r2": 0.80,
    "max_train_val_gap": 5.0,
    "min_survivor_count": 1000
  }
}
```

> **Verified / corrected 2026-11-11.** The CLI and thresholds exist as stated
> (`chapter_13_acceptance.py:945-948`; constants `:64-67`; policy `watcher_policies.json:95-98`).
> - **B13, BROKEN.** Selfplay's fitness is `max(0, val_r2 − 0.05) − penalties`
>   (`inner_episode_trainer.py:136-159`, `:148`), trained on `holdout_hits` then `score`
>   (`selfplay_orchestrator.py:933`), while Step 5 trains on `holdout_quality`. At R² below 0.05,
>   fitness ≤ 0, so `fitness ≥ 0.50` / `val_r2 ≥ 0.80` cannot pass, and `--promote` re-validates
>   (`chapter_13_acceptance.py:997-1005`). Host: last candidate (2026-02-24) `val_r2 −0.00029`,
>   `fitness −0.0014`. (One promotion did pass on 2026-01-30, `val_r2 0.99999`, under the data and
>   target of that time.)
> - **B12, BROKEN.** The ratified contract requires real-draw (shadow) validation before
>   `learned_policy_active.json` changes (`docs/CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md:106-110`,
>   `:218-222`); `validate_selfplay_candidate` checks only source, status, fitness, `val_r2`, gap and
>   survivor count (`chapter_13_acceptance.py:733-780`). Depends on B1.
> - **B11, BROKEN (governed).** `SelfplayCandidate` (`chapter_13_acceptance.py:224`) has no
>   `transforms`, `parent_policy_id` or `fingerprint`, and `from_dict` (`:240`) discards them, although
>   selfplay writes them (`selfplay_orchestrator.py:303-307`, `:1011-1014`). `promote_candidate`
>   (`:818`, write at `:848`) therefore writes an active policy with no transforms. Host: the active
>   policy has no `transforms` key.

### 21.4 Diagnostics Engine Extensions

Chapter 13 diagnostics now include `selfplay_health`:
```json
{
  "selfplay_health": {
    "available": true,
    "models_trained_total": 26,
    "models_trained_last_hour": 0,
    "fitness_best": 0.8474,
    "health_warnings": [],
    "current_best_policy": "policy_selfplay_20260129_234039_ep001"
  }
}
```

> **Verified 2026-11-11.** The block exists (`chapter_13_diagnostics.py:871-881`) and is observational:
> telemetry is read (`:82-97`) and no trigger consumes it. Host value (2026-02-15): `fitness_best −0.0004`,
> warnings `LOW_THROUGHPUT`, `PREMATURE_CONVERGENCE: policy_entropy=0.000`.

### 21.5 Triggers Engine Extensions

**New Methods:**
- `request_selfplay(reason, source_policy, priority)` — Create retrain request
- `should_request_selfplay(diagnostics)` — Check explicit triggers

**Request Flow:**
```
Diagnostics → Triggers → watcher_requests/*.json → WATCHER → Selfplay
```

> **Corrected 2026-11-11 (B10, BROKEN).** The methods exist (`chapter_13_triggers.py:762`
> `request_selfplay`, `:819` `should_request_selfplay`) and have **no caller** in tracked code or in any
> commit (`git log -S 'request_selfplay('` → only the defining commit `358e615`). The diagnostics never
> emit `SELFPLAY_RECOMMENDED` or `recommended_actions.request_selfplay` (flags
> `chapter_13_diagnostics.py:599-667`, actions `:681-687`); the enum says "enum only, no auto-dispatch"
> (`chapter_13_triggers.py:69`). The WATCHER side is complete: `watcher_requests/*.json` is scanned each
> cycle (`agents/watcher_agent.py:3187`) and routed to `dispatch_selfplay` (`agents/watcher_dispatch.py:88`,
> `:403-548`). Host archive: 21 selfplay requests, 20 written by named tests/soaks and 1 with
> `requested_by: chapter_13_triggers` (2026-01-30, a manual invocation). S179 §2.1 requires an
> `audit_only` gate inside `request_selfplay` (governed; not implemented, U3).

### 21.6 Data Flow
```
Selfplay                     Chapter 13                    WATCHER
────────                     ──────────                    ───────
learned_policy_candidate.json
        │
        └──────────────────► validate_selfplay_candidate()
                                      │
                                      ▼
                             ACCEPT / REJECT / ESCALATE
                                      │
                                      ▼ (if ACCEPT)
                             learned_policy_active.json
                             telemetry.record_promotion()
                                      │
                                      ▼ (if retrain needed)
                             request_selfplay() ──────────► watcher_requests/*.json
```

### 21.7 Invariants (Phase 9A Contract)

| Invariant | Enforcement |
|-----------|-------------|
| Selfplay cannot promote policies | Only Chapter 13 writes `learned_policy_active.json` |
| Chapter 13 requests, WATCHER executes | Requests queued in `watcher_requests/` |
| Telemetry is observational only | No control path from telemetry |
| Audit trail preserved | Append-only request files |

> **Verified / corrected 2026-11-11.** Row 1: accurate — the sole writer of `learned_policy_active.json`
> is `chapter_13_acceptance.py:848`. Row 2: "Chapter 13 requests" — no code path requests (B10). Row 3:
> accurate (`chapter_13_diagnostics.py:82-97`). Row 4: requests are archived under
> `watcher_requests/archive/` (host: 61 files). The §21.6 diagram's last arrow (`request_selfplay()`) is
> B10. Further selfplay facts not in this section: outer (sieving) episodes are off "until coordinator
> interface is ready" (`selfplay_orchestrator.py:194-196`, `configs/selfplay_config.json:8`) and the
> coordinator has no counterpart interface (B14); `propose_transform_update` returns its input
> (`policy_conditioned_episode.py:373`, `:398-401`) by approved deferral (S4); selfplay may not see live
> outcomes or promote (S7, `docs/CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md`).

---

## 22. Known gaps at a15b524

*(Added 2026-11-11. IDs as in `docs/CHAPTER_13_AUDIT.md`; full anchors there. Each item has exactly one
label. "Governed" means a ruling or governed document already records it — a status, not a new finding.)*

### 22.1 BROKEN (19)

| ID | What the chapter says | What the code does at `a15b524` | Stops at |
|---|---|---|---|
| B1 | Chapter 13 grades Step 6's pool (§7) | Reads `prediction_pool.json`, which nothing writes; Step 6 writes `predictions/next_draw_prediction.json` (list of ints). Host: 233/233 empty pools. Step 6 side governed (`c6e3989`) | `chapter_13_diagnostics.py:114` |
| B2 | Appended draw drives relearning (§9, §15) | Draws go to `lottery_history.json`; Steps 3/5/6 read `train_history.json`/`holdout_history.json`/`daily3.json` | `agent_manifests/full_scoring.json:120` |
| B3 | Collapse "over 20 draws" (§10.2) | One-draw test; window key unread. *Governed* (S176) | `chapter_13_triggers.py:269` |
| B4 | Decay/churn/LLM regime inputs (§10.4) | Decay is 0/1 on one draw; churn is a count delta; LLM threshold unread | `chapter_13_diagnostics.py:552`, `:559` |
| B5 | Approve commands execute (§12) | `chapter_13_orchestrator.py --approve` and `chapter_13_triggers.py --approve` run nothing | `chapter_13_triggers.py:603` |
| B6 | "3→5→6" (§15.3) | Executes `min..max`, so Step 4 runs | `agents/watcher_agent.py:3774` |
| B7 | Cooldown prevents thrashing (§16.1) | No executed retrain resets cooldown; `N_DRAWS` always true after 10 | `chapter_13_triggers.py:384` |
| B8 | Survivor reinforcement/decay (§4) | Decay list never filled; hit match on fields Step 3 does not write | `chapter_13_diagnostics.py:396` |
| B9 | Divergence halt (§16.3) | No detector; `_create_halt` has no caller | `chapter_13_orchestrator.py:241` |
| B10 | Ch13 requests selfplay (§21.5) | `request_selfplay` has no caller | `chapter_13_triggers.py:762` |
| B11 | Promotion makes the candidate active (§21.6) | Transforms, lineage, fingerprint dropped. *Governed* (system map) | `chapter_13_acceptance.py:240` |
| B12 | Real-draw validation before promotion (contract) | No shadow evaluation | `chapter_13_acceptance.py:780` |
| B13 | Selfplay candidates can pass (§21.3) | R² objective vs `holdout_quality` pipeline; gate unreachable. *Governed as open* (fact file) | `inner_episode_trainer.py:148` |
| B14 | Selfplay explores (contract) | Outer episodes off; no coordinator interface; episodes identical in input | `selfplay_orchestrator.py:606-607` |
| B15 | Per-survivor attribution (tracker) | Gate never opens (0/233 on host); no ranked-predictions producer; wrong sidecar key; bypasses NN loader; no scaler; dead `request_diagnostics_analysis` call. Blockers 1–4 *governed* (system map) | `chapter_13_orchestrator.py:326` |
| B16 | Test-mode orchestrator route "auto-executes" (`docs/WATCHER_POLICIES_REFERENCE.md:105`) | Sets a label, runs nothing (host: 120 such cycles) | `chapter_13_orchestrator.py:412` |
| B17 | Frozen components rejected (§13.1) | Eight literal names only | `chapter_13_acceptance.py:388` |
| B18 | Four execution preconditions (§6) | Halt + flag only | `chapter_13_orchestrator.py:510` |
| **B19** *(new, this pass)* | Model provenance from Step 5's sidecar (§7, §8.2) | Reads root `best_model.meta.json` (absent; Step 5 writes `models/reinforcement/…`) and top-level keys the sidecar nests; hash always None, `schema_hash_match` defaults True | `chapter_13_diagnostics.py:52` |

### 22.2 SWITCHED-OFF-BY-RULING (9) — owner decisions; no recommendation made

| ID | Held back | Ruling / approved document | Lifting it would enable |
|---|---|---|---|
| S1 | Retrain without human approval | Chapter §10.1, §20.3; `docs/PIPELINE_BEHAVIOUR_MODEL.md:954`; `docs/TB_RULING_S179_IMPLEMENTATION_AUTH.md:525-532` | A fired trigger running Steps 3→5→6 / 1→6 unattended (no production auto-execute path exists yet; B16) |
| S2 | KPI triggers active before calibration | `docs/TB_RULING_S176_WATCHER_KPI.md:795-821`; `docs/TB_RULING_S179_IMPLEMENTATION_AUTH.md:513-547` | Triggers requesting retrains on their present definitions (no behaviour change unless S1 is also lifted) |
| S3 | Applying LLM parameter proposals | Beta position quoted in `docs/CLAUDE_CODE_INSTRUCTIONS_CHAIN_C_TRUTHFULNESS_HOTFIX.md:104-106`; chapter §11.2 | Accepted low-risk proposals changing parameters (needs an apply function, bounds, `record_applied_changes`/`increment_run` calls) |
| S4 | Selfplay proposing transform updates (9B.3) | `docs/PHASE_9B2_INTEGRATION_SPEC.md:5`, `:393-409` | `propose_transform_update` returning changed transforms |
| S5 | Acting on attribution | Tracker `v3_9:232-238`; system map `:222-228`, `:277-278` | Root-cause diagnosis feeding triggers or selfplay priors (after B15) |
| S6 | Activating attribution piecemeal | System map `:272`, `:275`, `:99` | Fixing blockers one at a time; running attribution before prediction parity |
| S7 | Selfplay seeing live outcomes / promoting | `docs/CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md:4-6`, `:106-123`, `:188-193` | Selfplay scored on graded draws, or promotion without a human |
| S8 | Sieve thresholds in the LLM application path | `docs/TODO_SELFPLAY_AND_LLM_AUTONOMY.md:92-103`, `:111-131` | `forward_threshold`/`reverse_threshold` proposed and applied |
| S9 | Advisor-chosen `search_strategy` / sampler | `docs/CLAUDE_CODE_INSTRUCTIONS_BOUNDED_PHASE_6.md:98`, `:121`; `docs/S172_RUN4_ROUTING_PATCH_BRIEF_ROUTE_A.md:84` | The Strategy Advisor selecting the Step-1 sampler |

### 22.3 DESCRIPTION-ONLY (9 + notes)

D1 selfplay as a "learning loop" (§21.1). D2 "improves continuously" vs defensive triggers (Summary).
D3 "without human intervention" (§1). D4 `holdout_hits` vs `holdout_quality` (§3.3, §4.1). D5 reference
tables and file names (§14, §18). D6 "Pydantic" (§2, §19). D7 tracker status words
("COMPLETE", "Full Autonomy Achieved"). D8 file hygiene (end marker before §21; `_v1_1` near-duplicate;
unchecked §19). D9 "Phase 7 complete" means WATCHER dispatch integration (Feb 2026), not S172 Phase 7.
Smaller wording notes made in place this pass (counted under D5/D8, not separately): §6.1
illustrative code and the non-existent "flag-triggered" mode; §8.2 key names; §11.4 template;
§16.2 monitoring targets; §13.3 checks with no code.

---

## 23. RANGE-MINER, the NPZ contract and the window-anchor change

*(Added 2026-11-11.)* None of the three changes Chapter 13's own code; all three change what the
chapter's re-invocations would run.

- **RANGE-MINER.** A regime-shift rerun (`FULL_PIPELINE` → `[1..6]`) re-runs Step 1. Whether Step 1 uses
  RANGE-MINER is the manifest flag `use_range_miner`, default **false**
  (`agent_manifests/window_optimizer.json:212-218`; recorded, not changed). The 3→5→6 loop does not run
  Step 1, so the miner is not on the learning-loop path. Selfplay's outer episode would call
  `coordinator.py --episode-id` (`selfplay_orchestrator.py:626-630`), an interface neither the legacy
  coordinator nor the miner provides (B14). The chapter's §18.8 "Sieve math —
  `bidirectional_sieve_filter.py`" names a file that does not exist; the certifying engine is
  `miner/` (see `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` §8).
- **The NPZ contract.** Step 3, the first step of the learning loop, reads Step 1's survivor NPZ
  (`agent_manifests/full_scoring.json:118` `bidirectional_survivors_binary.npz`). A learning-loop rerun
  therefore re-scores whatever NPZ the last Step 1 left; it does not refresh it. Chapter 13 itself reads no
  NPZ (it reads `survivors_with_scores.json`, `chapter_13_diagnostics.py:54`). The 22-array contract was not
  re-counted in this pass.
- **The window-anchor change (Brief I).** The fused `offset` was split into `window_anchor` +
  `generator_phase` in the miner/coordinator path (commit `48a8705`). `optimal_window_config.json`, which
  Chapter 13's §18.1 lists and from which the synthetic injector takes `prng_type`, still carries `offset`;
  its migration is Brief II (`docs/S172_WINDOW_ANCHOR_BRIEF_I.md:360-368`). §18.1's `offset` row therefore
  names the pre-split field; it is not a frozen parameter in `FROZEN_PARAMETERS` (B17).

---

## 24. Open items

1. **Repairs (B1–B19).** Recorded, not made. The repair statements are in `docs/CHAPTER_13_AUDIT.md` §4
   and addendum A.9. B1, B2, B12 and B18 share one prerequisite: a single prediction artifact and a single
   dataset authority that both the pipeline and Chapter 13 use.
2. **Owner decisions (S1–S9).** §22.2. Under the 2026-11-11 authority rulings these are the owner's to
   keep or lift.
3. **U5** (convergence target on synthetic data) remains UNVERIFIED; **U1** soak logs not opened.
4. **Host surfaces UNAVAILABLE this pass:** `.127` bare-metal fallback (not booted), rig CT100s, the private
   remote and ser8 archives, root and other-user crontabs (`/var/spool/cron/crontabs` permission denied).
5. **Host-state observations, not defects:** `daily3scraper.service` is disabled and its runner
   `run_daily3scraper.py` has never existed; `optimal_window_config.json` is absent from the project root;
   the loop's runtime artifacts were last written February–March 2026.
6. **Elsewhere (listed, not edited):** `docs/TFM_PROJECT_FACTS_SKILL.md:200-202` ("an enabled
   boot-triggered unit was on the host") is stale for VM 101; `watcher_policies.json:13` seed-finding note;
   `docs/CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md` near-duplicate; `docs/WATCHER_POLICIES_REFERENCE.md:105` (B16).

---

## 25. Verification declaration (VIR-1…6)

**execution proof (VIR-1).** Every `file:line` added to this chapter on 2026-11-11 was opened on VM 101 at
`a15b524` this session (`sed -n`, `Read`, `/bin/grep -n`), or is quoted from `docs/CHAPTER_13_AUDIT.md`
after its anchors were re-verified (tracked tree byte-identical to `a15b524`; ~200 anchors opened, all
matched). The files opened are listed in the audit addendum A.8. Host runtime facts come from files read on
the host and are dated.

**clean control (VIR-2).** `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` was read for shape only and not
modified. Sections of this chapter verified accurate and left without correction: §2 LLM
can/cannot lists (as policy), §6.1 dual-flag gating and metadata tagging, §8.1 output file, §10.1,
§11.1–11.3, §13.1 thresholds 0.60/3/30 %, §13.2 acceptance criteria, §13.3 escalation on risk/flags/failures,
§17.3 controls, §21.3 CLI and thresholds, §21.4 block, §21.7 rows 1 and 3.

**fault-injection control.** **Not applicable to a documentation pass.** No detector, gate or harness was
run whose silence could be vacuous, so none was injected. Stated rather than invented.

**completion sentinel.** §26.3.

**unavailable-observer behaviour.** Every surface that could not be read is listed as UNAVAILABLE (§24.4) and
is not counted as clean. The host answer "no bridge" is scoped to VM 101.

**audit claim scope.** What this chapter says against what the code at `a15b524` does. Not whether the design
is right, not a seed-recovery question (§1 framing), not a run.

**searched surfaces / unavailable surfaces.** Searched: tracked repo; untracked and gitignored files
(`git status --ignored`, `git ls-files -o -i --exclude-standard`, text search of files ≤ 2 MB); git history
(`git log --all -S`, `git log --all -- path`, `git show`); host state (system and user systemd units, timers,
journal, `crontab -l`, `/etc/cron*`, `/usr/local/bin`, `~/` scripts outside both projects, processes, runtime
artifacts, `agent_manifests/definitions.json`). Unavailable: §24.4. `~/piece_matcher_project` deliberately
not touched.

**governance trail searched.** `docs/TB_RULING_S176/S177/S178/S179*`, `docs/TB_UPDATE_SELFPLAY_REFRAMING_2026-07-28.md`,
`docs/CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md`, `docs/PHASE_9B2_INTEGRATION_SPEC.md`,
`docs/PROPOSAL_WATCHER_KPI_GOVERNANCE_STATES_v1_1.md`, `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md` §4.4.3,
`docs/S172_WINDOW_ANCHOR_BRIEF_I.md`, `docs/TEAM_ALPHA_AUTONOMY_CONTROL_SURFACE_SUBMISSION.md` (via the audit),
`docs/TB_*` and `docs/PROPOSAL_*` grep for `prediction_pool.json` and for "canonical output", commit messages
`eb44bec`, `c6e3989`, `833507e`. **Chapters searched:** this chapter and `_v1_1`,
`docs/CHAPTER_13_IMPLEMENTATION_PROGRESS_v3_9.md`, `docs/CHAPTER_13_SECTION_19_UPDATED.md` (via the audit),
`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` (shape), `docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1 and authority table.

---

## 26. Closure statement

### 26.1 Verified against

`a15b524` on VM 101, 2026-11-11, working tree (tracked files clean), plus host state as listed in §25.

### 26.2 What changed in this file

Added: framing block, status header, §0, TOC entries, §22–§26, a version-history addendum, and in-place
correction notes in §1, §2, §3.3, §4, §6, §6.1 (three), §7, §8.3, §9, §10.2, §10.4, §11, §12, §13.4, §14, §15,
§16, §17.3, §18, §19, §20.3, Summary, END marker, §21 (six). Marked **SUPERSEDED (2026-11-11)**: §9.1 steps
1–2 with §9.3's code block (→ §9 note and §22 B2), §19 as a status record (→ §19 note). **No text was removed;
every original heading still exists.** The chapter is longer, not shorter (1229 → 1843 lines), so no removal
list is needed. Six original lines were edited rather than appended to, and their previous text is kept
here: `**Version:** 1.1.0` (now 1.3.0 — note that §21 had already declared 1.2.0); `**Status:**
Architecture-Final` (now the 2026-11-11 status); `**Depends On:** Chapters 1-12`, `**Executes After:** Step 6
(Prediction Generator)` and `**Purpose:** Close the autonomy loop — convert predictions into learning`
(unchanged in wording, annotation appended); `**END OF CHAPTER 13**` (kept, annotation appended).

### 26.3 Completion sentinel

```
CHAPTER 13 CORRECTION (2026-11-11):  PASS
```

`PASS` is claimed for the declared scope: the existing audit was re-verified against the live host and
extended (addendum), the §5 host question is answered with evidence, and every substantive chapter claim
now either stands verified or carries a visible correction with one label. It is **not** a claim that
Chapter 13 works: 19 BROKEN items and 9 switched-off items stand, U5 is unverified, and the surfaces in §24.4
were unavailable.

**Files changed by this pass:** `docs/CHAPTER_13_LIVE_FEEDBACK_LOOP.md`, `docs/CHAPTER_13_AUDIT.md` (addendum
appended).
