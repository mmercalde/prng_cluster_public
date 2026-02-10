# Prediction Rate Improvement Strategies
## Documented Approaches from Verified Sources

**Version:** 1.0.0  
**Date:** February 10, 2026  
**Status:** Extracted from Project Documentation

---

## Overview

The system uses **three complementary strategies** to improve prediction accuracy over time:

1. **Bidirectional Sieve Validation** (Triangle 1)
2. **Multi-Model ML Ensemble** (Triangle 3)
3. **Live Feedback Loop** (Chapter 13)

---

## Strategy 1: Bidirectional Sieve Validation

**Source:** Functional_Mimicry_via_Reinforcement.pdf, CANONICAL_PIPELINE_AND_CH13_WITH_STARTUP_COMPLETE.md

### Core Mechanism

```
Forward Sieve:  seed → [o₁, o₂, ..., oₙ]  (predict future from past)
Reverse Sieve:  seed → [oₙ, ..., o₂, o₁]  (validate from recent backward)
Intersection:   Seeds passing BOTH → High confidence
```

### How It Improves Predictions

**Step 1: Forward Sieve Filters**
- Tests millions of candidate seeds
- Keeps seeds matching historical pattern
- Typical: 10M seeds → 50K survivors

**Step 2: Reverse Sieve Validates**
- Tests forward survivors against reversed sequence
- Eliminates false positives
- Typical: 50K → 1.5K survivors (97% reduction)

**Step 3: Intersection = High Confidence**
- Only seeds passing BOTH directions survive
- False positive rate: ~10⁻¹⁸⁰ (theoretical)
- These become ML training candidates

**Documented Quote:**
> "The forward–reverse–ML ensemble produces a continuously learning PRNG emulator:
> Forward sieve filters potential seeds, Reverse sieve validates backward consistency,
> ML reinforcement prioritizes statistically coherent survivors"

---

## Strategy 2: Multi-Model ML Ensemble

**Source:** CHAPTER_5_ANTI_OVERFIT_TRAINING.md (documented), meta_prediction_optimizer_anti_overfit.py

### Four Model Types

**Why 4 models?** Each has different "inductive biases" - they learn patterns differently.

| Model | Strength | When It Wins |
|-------|----------|--------------|
| **XGBoost** | Tree-based splits, feature importance | 45% of time (most common) |
| **LightGBM** | Histogram-based, fast | 20% of time |
| **CatBoost** | Ordered boosting, robust | 10% of time |
| **Neural Network** | Non-linear patterns | 25% of time |

**Source:** TRIANGULATED_FUNCTIONAL_MIMICRY_VERIFIED_v1_0.md, Section 8

### Automatic Model Selection

```python
# Documented process from Step 5
For each model type:
    1. Optuna hyperparameter search (20-100 trials)
    2. K-fold cross-validation (k=5)
    3. Early stopping (patience=10 epochs)
    4. Holdout evaluation (final test set)

Select best model:
    criteria = min(holdout_MAE) AND overfit_ratio < 1.5
```

**Critical Design:**
- Models compete on **holdout data** (not training data)
- Winner determined by **unseen performance**
- Overfit models rejected automatically

### Feature Learning (47 Features)

**Source:** survivor_scorer.py documentation, CHAPTER_4_FULL_SCORING.md

**Feature Categories:**
- Lane agreement (8 features): Agreement across modulo lanes
- Pattern analysis (12 features): Skip entropy, temporal stability
- Statistical (15 features): Intersection weight, consistency
- PRNG-specific (15 features): Residue patterns, modulo checks
- Global context (12 features): Pool diversity, regime detection

**How Features Improve Predictions:**

1. **Intersection Weight:** Seeds strong in BOTH directions rank higher
2. **Skip Entropy:** Stable skip patterns predict better
3. **Temporal Stability:** Seeds consistent over time preferred
4. **Lane Agreement:** Multi-modulo validation confidence
5. **Survivor Overlap:** Forward/reverse consensus

**Example documented feature:**
```python
def compute_intersection_weight(survivor):
    """How strongly survivor appears in both forward and reverse."""
    forward_rate = survivor['forward_match_rate']
    reverse_rate = survivor['reverse_match_rate']
    
    # Geometric mean (penalizes imbalance)
    weight = sqrt(forward_rate * reverse_rate)
    
    return weight
```

---

## Strategy 3: Live Feedback Loop (Chapter 13)

**Source:** CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md, CANONICAL_PIPELINE_AND_CH13_WITH_STARTUP_COMPLETE.md

### The Learning Cycle

```
Step 6 → Predictions Emitted
           ↓
    New Draw Occurs
           ↓
    History Updated (append-only)
           ↓
    Chapter 13 Diagnostics
           ↓
    WATCHER Policy Check
           ↓
    Retrain Decision
           ↓
    Step 3: Refresh Labels (holdout_hits)
           ↓
    Step 5: Retrain Model
           ↓
    Step 6: Better Predictions
           ↓
        Repeat
```

### Critical Innovation: Holdout Hits

**Source:** CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md, Section 9

**Problem with naive approach:**
```
❌ BAD: Train on training data match rate
   → Model learns: "high match rate = good"
   → Circular! Training data used to CREATE survivors
   → No generalization
```

**Documented solution:**
```
✅ GOOD: Train on HOLDOUT data performance
   → holdout_hits = how many holdout draws did seed predict?
   → Holdout data NEVER seen during sieve
   → True predictive power measured
```

**Documented quote:**
> "Step 3 recomputes `holdout_hits` using expanded history.
> Step 5 retrains model weights on refreshed labels.
> Step 6 generates improved predictions from better model.
> No code changes to Steps 1-6. Only orchestration of when to re-invoke."

### Retrain Trigger Policies

**Source:** CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md, Section 10

**When retraining happens:**

| Trigger | Condition | Action |
|---------|-----------|--------|
| **Accumulation** | N new draws collected | Re-run Steps 3→5→6 |
| **Confidence Drift** | Prediction variance increasing | Retrain Step 5 |
| **Accuracy Drop** | Hit rate declining | Full diagnostic + retrain |
| **Regime Shift** | Pattern change detected | Re-run FULL pipeline (1→6) |

**Documented example:**
```python
# From CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md
if diagnostics.requires_regime_reset():
    run_steps([1, 2, 3, 4, 5, 6])  # Full pipeline
elif diagnostics.requires_retrain():
    run_steps([3, 5, 6])            # Learning loop only
elif diagnostics.requires_prediction_refresh():
    run_steps([6])                  # Inference only
else:
    log("Diagnostics healthy, no rerun required")
```

### What Gets Better Over Time

**1. Label Quality**
- More draws → more accurate holdout_hits
- Survivors get better y-labels
- Model learns from accumulating truth

**2. Model Weights**
- Retraining adjusts feature importance
- Good patterns reinforced
- Poor patterns down-weighted

**3. Confidence Calibration**
- System learns which confidence scores reliable
- Tight pool becomes more selective
- Wide pool maintains coverage

**4. Pattern Adaptation**
- If PRNG changes (reseeding, drift)
- System detects via diagnostics
- Triggers appropriate response (partial or full retrain)

---

## Strategy 4: LLM-Guided Parameter Tuning

**Source:** CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md, Section 11

### LLM Role: Strategist (NOT Executor)

**Documented architecture:**

```
Diagnostics → LLM Analysis → Parameter Proposal → WATCHER Validation → Execute or Reject
```

**LLM can propose:**
- Threshold adjustments
- Pool size changes
- Feature weight suggestions
- Retrain cadence modifications

**LLM CANNOT:**
- Execute code directly
- Change pipeline structure
- Bypass validation
- Mutate core algorithms

**Documented quote:**
> "The LLM cannot: Rewrite mathematical logic, Invent new features,
> Bypass validation, Mutate control flow, Change step ordering.
> The LLM can: Interpret diagnostics, Detect drift patterns,
> Propose parameter adjustments, Recommend retraining."

### Example LLM Analysis

**Source:** CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md

```json
{
  "observation": "Hit rate dropped from 0.15 to 0.08 over last 10 draws",
  "hypothesis": "Possible PRNG reseeding or skip pattern change",
  "proposal": {
    "action": "retrain",
    "steps": [3, 5, 6],
    "parameter_adjustments": {
      "window_size": 512,
      "confidence_threshold": 0.75
    }
  },
  "confidence": 0.82
}
```

**WATCHER then validates:**
- Are parameters within bounds?
- Does diagnosis make sense?
- Is action safe?

**If valid → Execute**  
**If invalid → Reject + log reason**

---

## Strategy 5: Autonomous Decision Tiers

**Source:** WATCHER_CONFIGURATION_AND_AGENT_MANIFESTS_v1_0.md

### Three-Tier Decision System

**~95% of decisions use Tier 1 (heuristics):**

```
Tier 1: Heuristic Rules
  - Threshold checks (< 1 second)
  - File validation
  - Simple logic: if accuracy > 0.8 → PROCEED

Tier 2: DeepSeek-R1-14B (Local LLM)
  - Grammar-constrained (GBNF)
  - 51 tok/sec
  - Complex pattern analysis
  - Escalates if confidence < 0.3

Tier 3: Claude Opus 4.6 (Advanced LLM)
  - 38 tok/sec
  - Strategic decisions
  - REGIME_SHIFT detection
  - Only when Tier 2 uncertain
```

**How this improves predictions:**
- Fast decisions for routine cases
- Deep analysis for complex patterns
- Strategic reasoning for major changes
- No human bottleneck (autonomous)

---

## Summary: How Predictions Improve

### Initial Predictions (After Step 6, First Run)

```
Quality: Based on forward/reverse intersection
Confidence: From bidirectional validation
Accuracy: Unknown (no feedback yet)
```

### After 10 Draws (Chapter 13 Active)

```
Quality: Model trained on 10 real outcomes
Confidence: Calibrated against actual hits
Accuracy: Measured, feedback incorporated
Labels: 10 more data points added to holdout_hits
```

### After 100 Draws (Continuous Learning)

```
Quality: Model seen 100 real outcomes
Confidence: Well-calibrated confidence scores
Accuracy: Converging to stable performance
Patterns: Adapted to any PRNG drift/changes
Parameters: LLM-optimized thresholds
```

### After 1000 Draws (Long-Term)

```
Quality: Rich training history
Confidence: Highly calibrated
Accuracy: Stable, mature predictions
Regime Changes: Automatically detected and adapted
Feature Importance: Optimized through experience
```

---

## What Is NOT Used (Clarification)

**NOT documented in the system:**
- ❌ Deep reinforcement learning (e.g., Q-learning, policy gradients)
- ❌ Online learning (streaming updates per draw)
- ❌ Transfer learning from other PRNGs
- ❌ Adversarial training
- ❌ Genetic algorithms for seed evolution

**What IS used:**
- ✅ Supervised learning (XGBoost, LightGBM, CatBoost, Neural Network)
- ✅ Bayesian optimization (Optuna for hyperparameters)
- ✅ Feedback loop (retrain on accumulated data)
- ✅ Ensemble learning (multi-model voting)
- ✅ LLM-guided parameter tuning (proposals, not execution)

---

## Empirical Evidence (What's Documented)

### Verified Improvements

**Step 2.5 Optimization:**
```
Before: 5000 samples @ 4 concurrent → 3.4 trials/min
After:  450 samples @ 12 concurrent → 15.41 trials/min
Speedup: 4.5×
```
**Source:** CHAPTER_3_SCORER_META_OPTIMIZER.md, Section 9.4

**Dynamic Work Distribution:**
```
Before: Static 26 chunks → 45 seconds
After:  Dynamic 500 chunks → 13.5 seconds
Speedup: 3.3×
```
**Source:** Verified in multiple session changelogs

### Unverified Claims

**Prediction accuracy improvements:** NOT documented with empirical data  
**Confidence calibration:** NOT validated against ground truth  
**Long-term convergence:** NOT demonstrated over 100+ draws  
**Hit rate improvements:** NOT measured in production

---

## Conclusion

The system improves predictions through:

1. **Bidirectional filtering** (exponential false positive reduction)
2. **Multi-model ensemble** (different inductive biases)
3. **Holdout-based learning** (true generalization target)
4. **Continuous feedback** (accumulating real outcomes)
5. **LLM-guided tuning** (strategic parameter optimization)
6. **Autonomous decision-making** (no human bottleneck)

**The key innovation:** Using `holdout_hits` as training target ensures the model learns which patterns predict FUTURE draws, not which patterns fit PAST draws used to find the seeds.

**Documented quote summarizing the approach:**
> "The result is a data-driven mimicry engine—an evolving model of the
> true PRNG's behavior that converges through iterative feedback."

---

**Document Status:** Verified from project documentation  
**Sources:** Chapter 13, Functional Mimicry whitepaper, WATCHER documentation, Session changelogs  
**Version:** 1.0.0

