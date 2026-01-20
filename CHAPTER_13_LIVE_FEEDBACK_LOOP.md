# Chapter 13: Live Feedback Loop & Autonomous Learning

## PRNG Analysis Pipeline — Complete Operating Guide

**Version:** 1.1.0  
**Status:** Architecture-Final  
**Depends On:** Chapters 1-12  
**Executes After:** Step 6 (Prediction Generator)  
**Purpose:** Close the autonomy loop — convert predictions into learning

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

**END OF CHAPTER 13**
