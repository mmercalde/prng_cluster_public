# Chapter 5: Adaptive Meta-Optimizer (Step 4)

## PRNG Analysis Pipeline — Complete Operating Guide

**Version:** 2.0.0 (Corrected)  
**File:** `adaptive_meta_optimizer.py`  
**Purpose:** Capacity & architecture planning for ML training

---

## Table of Contents

1. [Overview](#1-overview)
2. [Pipeline Position](#2-pipeline-position)
3. [What Step 4 Does (And Does NOT)](#3-what-step-4-does-and-does-not)
4. [Data Sources & Weights](#4-data-sources--weights)
5. [Derived Parameters](#5-derived-parameters)
6. [Output Configuration](#6-output-configuration)
7. [CLI Interface](#7-cli-interface)
8. [Integration Points](#8-integration-points)
9. [Common Misconceptions](#9-common-misconceptions)

---

## 1. Overview

### 1.1 What the Adaptive Meta-Optimizer Does

The Adaptive Meta-Optimizer is **Step 4** of the 6-step pipeline. It is a **capacity and architecture planner** that determines:

1. **How many survivors** the model should train on (pool size)
2. **How deep** the network should be (architecture)
3. **How long** training should run (epochs)

### 1.2 Critical Design Principle

> **Step 4 is intentionally NOT data-aware.**
> 
> It derives capacity parameters from window optimization behavior and training-history complexity only. Survivor-level data (including `holdout_hits`) is first consumed in **Step 5**, where model selection and overfit control occur.

### 1.3 Pipeline Context

```
Step 3: Full Scoring
    ↓ (survivors_with_scores.json — 62 features per survivor)
    ↓ [NOT consumed by Step 4]
    ↓
Step 4: ADAPTIVE META-OPTIMIZER ← You are here
    │
    ├── Reads: optimal_window_config.json (Step 1 output)
    ├── Reads: train_history.json (lottery data)
    ├── Reads: reinforcement feedback (if available)
    │
    ↓ (reinforcement_engine_config.json — capacity parameters)
    ↓
Step 5: Anti-Overfit Training
    │
    └── FIRST step to consume survivors_with_scores.json
    └── FIRST step to use holdout_hits
    └── Model selection happens HERE
```

---

## 2. Pipeline Position

### 2.1 Why Step 4 Exists

Before training a model, we need to decide:
- How much data should it see?
- How complex should the model be?
- How long should training run?

Step 4 answers these questions **without looking at survivor-level data**, based purely on:
- How the sieve optimization performed (convergence speed, stability)
- How complex the lottery patterns are (entropy, regime changes)

### 2.2 Input Requirements

| Input | Source | Contains |
|-------|--------|----------|
| `optimal_window_config.json` | Step 1 | Sieve optimization results |
| `train_history.json` | Data source | Training lottery draws |
| `reinforcement_feedback.json` | Step 6 (optional) | Historical performance |

### 2.3 What Step 4 Does NOT Read

| File | Why Not |
|------|---------|
| `survivors_with_scores.json` | Would cause validation leakage |
| `holdout_history.json` | Would contaminate capacity decisions |
| `holdout_hits` values | Belong to Step 5 evaluation |

### 2.4 Output

| Output | Consumed By |
|--------|-------------|
| `reinforcement_engine_config.json` | Step 5 |

---

## 3. What Step 4 Does (And Does NOT)

### 3.1 ✅ Step 4 DOES

| Action | Purpose |
|--------|---------|
| Load window optimizer results | Understand sieve behavior |
| Load training lottery history | Analyze pattern complexity |
| Read reinforcement feedback | Learn from past runs |
| Derive survivor pool size | Capacity planning |
| Derive network architecture | Complexity planning |
| Derive training epochs | Duration planning |
| Write config JSON | Pass to Step 5 |

### 3.2 ❌ Step 4 DOES NOT

| Action | Why Not |
|--------|---------|
| Load `survivors_with_scores.json` | Causes validation leakage |
| Inspect `holdout_hits` | Contaminates decisions |
| Perform any evaluation | That's Step 5's job |
| Choose model type | That's Step 5's job |
| Touch holdout data | Would invalidate pipeline |

### 3.3 Why This Separation Matters

If Step 4 became data-aware:
- ❌ Hyperparameters would be tuned on validation data
- ❌ Capacity decisions would overfit to specific survivors
- ❌ Step 5's holdout evaluation would be compromised
- ❌ Pipeline's separation-of-concerns would be violated

---

## 4. Data Sources & Weights

### 4.1 Weighted Combination

```python
sources = {
    'window_optimizer_results': {
        'path': 'optimal_window_config.json',
        'weight': 0.60,    # PRIMARY (60%)
        'required': True
    },
    'lottery_history': {
        'path': 'train_history.json',
        'weight': 0.35,    # SECONDARY (35%)
        'required': True
    },
    'reinforcement_feedback': {
        'weight': 0.05,    # CONTINUOUS (5% → 25%)
        'max_weight': 0.25,
        'growth_rate': 'confidence_based'
    }
}
```

### 4.2 Window Optimizer Analysis (60%)

Extracts from Step 1 results:
- **Survivor count range** — min, optimal, max from trials
- **Convergence metrics** — how quickly good solutions were found
- **Stability metrics** — variance in top results

### 4.3 Historical Pattern Analysis (35%)

Analyzes training lottery data:
- **Entropy** — higher entropy = more complex = more capacity needed
- **Temporal stability** — unstable patterns need larger pools
- **Regime change frequency** — frequent changes need adaptability

### 4.4 Reinforcement Feedback (5-25%)

If previous runs exist:
- **Best performing configuration** — what worked before
- **Confidence score** — how much to trust historical data
- Weight grows from 5% to 25% based on confidence

---

## 5. Derived Parameters

### 5.1 Survivor Pool Size

```python
def derive_optimal_survivor_count():
    # Weighted combination of all sources
    optimal = (
        window_optimal * 0.60 +
        pattern_optimal * 0.35 +
        feedback_optimal * feedback_weight
    )
    
    return {
        'min': optimal * 0.5,
        'optimal': optimal,
        'max': optimal * 2.0
    }
```

### 5.2 Network Architecture

```python
# Heuristic based on problem complexity
options = [
    [32],              # Simple
    [64, 32],          # Medium
    [128, 64, 32],     # Complex (default)
    [256, 128, 64]     # Very complex
]
```

### 5.3 Training Epochs

```python
def estimate_training_epochs():
    # Based on convergence speed from window optimizer
    base_epochs = 100
    
    speed_factor = 1.0 / (convergence_speed + 0.5)
    stability_factor = 1.0 + (1.0 - stability)
    
    optimal = base_epochs * speed_factor * stability_factor
    return clamp(optimal, min=50, max=500)
```

---

## 6. Output Configuration

### 6.1 Config File Format

```json
{
  "survivor_count": 500,
  "survivor_count_range": {
    "min": 250,
    "max": 1000
  },
  "network_architecture": [128, 64, 32],
  "training_epochs": 100,
  "training_epochs_range": {
    "min": 50,
    "max": 500
  },
  "confidence": 0.75,
  "weights_used": {
    "window_optimizer": 0.60,
    "historical_patterns": 0.35,
    "reinforcement_feedback": 0.05
  },
  "timestamp": "2025-01-01T12:00:00",
  "calibration_type": "full",
  "_meta_optimizer": {
    "last_calibration": "2025-01-01T12:00:00",
    "calibration_type": "full",
    "confidence": 0.75
  }
}
```

### 6.2 Key Fields

| Field | Description | Used By Step 5 |
|-------|-------------|----------------|
| `survivor_count` | Recommended pool size | Sample size selection |
| `network_architecture` | Layer sizes | Neural net config |
| `training_epochs` | Duration | Early stopping baseline |
| `confidence` | Trust in recommendations | Adjustment decisions |

### 6.3 What's NOT in Output

| Field | Why Not |
|-------|---------|
| `model_type` | Decided by Step 5 |
| `hyperparameters` | Optimized by Step 5 |
| `holdout_metrics` | Step 5 computes these |

---

## 7. CLI Interface

### 7.1 Basic Usage

```bash
python3 adaptive_meta_optimizer.py \
    --mode full \
    --window-results optimal_window_config.json \
    --lottery-data train_history.json \
    --apply
```

### 7.2 CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--mode` | No | full | Optimization mode |
| `--window-results` | No | optimal_window_config.json | Step 1 output |
| `--lottery-data` | Yes | - | Training lottery history |
| `--apply` | No | False | Write to reinforcement_engine_config.json |
| `--config` | No | - | Custom config file |

### 7.3 Arguments NOT Used

These arguments exist for backward compatibility but are **intentionally ignored**:

| Argument | Status | Reason |
|----------|--------|--------|
| `--survivor-data` | Ignored | Would cause validation leakage |
| `--holdout-history` | Ignored | Step 4 must not see holdout |

---

## 8. Integration Points

### 8.1 Pipeline Integration

```
Step 1: Window Optimizer
    ↓ optimal_window_config.json
Step 4: adaptive_meta_optimizer.py
    │
    ├── Analyze window optimization behavior
    ├── Analyze lottery pattern complexity
    ├── Derive capacity parameters
    └── Output: reinforcement_engine_config.json
    ↓
Step 5: meta_prediction_optimizer_anti_overfit.py
    │
    ├── Load survivors_with_scores.json (FIRST time!)
    ├── Use holdout_hits for evaluation (FIRST time!)
    ├── Select model type (neural_net/xgboost/lightgbm/catboost)
    └── Output: best_model.* + best_model.meta.json
```

### 8.2 Agent Framework Integration

```python
# In agent_manifests/ml_meta.json
{
  "agent_name": "ml_meta_agent",
  "pipeline_step": 4,
  "script": "adaptive_meta_optimizer.py",
  "inputs": {
    "window_results": "${STEP1_OUTPUT}",
    "lottery_data": "${TRAIN_HISTORY}"
  },
  "outputs": {
    "config": "reinforcement_engine_config.json"
  },
  "next_agent": "reinforcement_agent"
}
```

Note: The manifest does NOT include `survivors` or `holdout_history` because Step 4 must not consume them.

---

## 9. Common Misconceptions

### 9.1 "Step 4 should analyze survivor distributions"

**Wrong.** That would cause validation leakage. Step 4 derives capacity from sieve behavior, not survivor data.

### 9.2 "Step 4 should recommend model type"

**Wrong.** Model selection requires evaluating models on data, which happens in Step 5 with proper holdout separation.

### 9.3 "Step 4 should use Optuna for hyperparameters"

**Wrong.** Step 4 derives capacity heuristically. Step 5 uses Optuna to optimize actual model hyperparameters with proper cross-validation.

### 9.4 "The --survivor-data argument means Step 4 uses survivors"

**Wrong.** The argument exists for backward compatibility but is intentionally ignored. Removing it from workflow calls is recommended.

---

## Summary

### What Step 4 Does

1. **Analyzes** sieve optimization behavior (from Step 1)
2. **Analyzes** lottery pattern complexity (entropy, stability)
3. **Derives** capacity parameters (pool size, depth, epochs)
4. **Outputs** configuration for Step 5

### What Step 4 Does NOT Do

- Load survivor-level data
- Inspect holdout_hits
- Select model type
- Optimize hyperparameters
- Perform any evaluation

### Key Principle

> **Step 4 is the last step before survivor data is touched.**
> **Step 5 is the first step to consume survivors_with_scores.json and holdout_hits.**

This separation is **intentional** and **critical** for pipeline integrity.

---

## Version History

```
Version 2.0.0 - January 1, 2026
- MAJOR: Corrected documentation to reflect actual design
- MAJOR: Clarified that Step 4 is capacity planner, not data-aware optimizer
- MAJOR: Documented intentional exclusion of survivor-level data
- MAJOR: Removed misleading references to model selection
- Added "Common Misconceptions" section

Version 1.0.0 - December 30, 2025
- Initial documentation (contained incorrect expectations)
```

---

**END OF CHAPTER 5**
