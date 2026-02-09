# Chapter 6: Anti-Overfit Training (Step 5)

## PRNG Analysis Pipeline — Complete Operating Guide

**Version:** 3.1.0 (Holdout Integration)  
**File:** `meta_prediction_optimizer_anti_overfit.py`  
**Lines:** ~900  
**Purpose:** ML model training with anti-overfitting measures and multi-model comparison

---

## Table of Contents

1. [Why ML Matters for Survivors](#1-why-ml-matters-for-survivors)
2. [The Training Target Problem](#2-the-training-target-problem)
3. [Correct Training Architecture](#3-correct-training-architecture)
4. [Feature Categories](#4-feature-categories)
5. [Multi-Model Architecture](#5-multi-model-architecture)
6. [AntiOverfitMetaOptimizer Class](#6-antioverfit-metaoptimizer-class)
7. [K-Fold Cross-Validation](#7-k-fold-cross-validation)
8. [Hyperparameter Sampling](#8-hyperparameter-sampling)
9. [Sidecar Metadata](#9-sidecar-metadata)
10. [CLI Interface](#10-cli-interface)
11. [Integration Points](#11-integration-points)

---

## 1. Why ML Matters for Survivors

### 1.1 Survivors Are Mathematically Significant

The bidirectional sieves eliminate noise with astronomical precision:

```
P(random seed survives bidirectional) ≈ 10⁻¹¹⁹¹

Expected false positives from 4.3 billion seeds ≈ 0
```

**Every bidirectional survivor is significant** — they exist because they actually match PRNG behavior, not by chance.

### 1.2 But Survivors Aren't All Equal

Real-world complications mean survivors may represent:

| Scenario | Implication |
|----------|-------------|
| True seed | Will continue to predict correctly |
| One of multiple seeds | Lottery uses different seeds for different sessions |
| Partial match | Valid before a reseed event, may fail afterward |

### 1.3 What ML Learns

ML doesn't determine IF a survivor is significant — the sieves already proved that.

**ML learns WHICH survivors will continue to perform** based on:
- Feature patterns that indicate robustness
- Temporal stability across different windows
- Quality of bidirectional agreement
- Skip hypothesis consistency

---

## 2. The Training Target Problem

### 2.1 The Bug (CRITICAL)

The original implementation had a **circular training target**:

```python
# BROKEN: y is computed from features that define y
X = [all 62 features including exact_matches, residue_1000_match_rate]
y = score  # Which IS exact_matches / total × 100
```

**What this means:**

```python
# survivor_scorer.py line 198
base_score = matches.float().mean().item()
features = {
    'score': base_score * 100,
    'exact_matches': matches.sum().item(),
    'residue_1000_match_rate': matches.sum().item() / total,
    ...
}
```

The score IS the exact_matches / total — mathematically equivalent to `residue_1000_match_rate × 100`.

### 2.2 What the Model Learned (Wrong)

```
Feature Importance (BROKEN):
  residue_1000_match_rate:  65.01%
  exact_matches:            34.99%
  all other 60 features:     0.00%  ← COMPLETELY IGNORED
```

The model learned a **tautology**:
```
y ≈ 0.65 × residue_1000_match_rate + 0.35 × exact_matches
```

It predicts the score from the components that **define** the score. Circular. Useless.

### 2.3 Why This Defeats the Purpose

Per the whitepapers:
> "ML models learn optimal weighting across features to maximize Hit@K metrics"

The current ML **cannot generalize** because:
- It doesn't use structural features (intersection, lane agreement, temporal)
- It predicts **training** performance, not **future** performance
- It's mathematically circular — no learning required

---

## 3. Correct Training Architecture

### 3.1 The Fix

| Aspect | BROKEN (Current) | CORRECT (Fixed) |
|--------|------------------|-----------------|
| **y (target)** | `score` (training match rate) | `holdout_hits` (holdout Hit@K) |
| **Learning** | Tautology | Generalization |

### 3.2 Correct Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│ STEP 2: SIEVES                                               │
│   Input: Seed space + lottery history                        │
│   Output: Bidirectional survivors (P ≈ 10⁻¹¹⁹¹)              │
├──────────────────────────────────────────────────────────────┤
│ STEP 3: FEATURE EXTRACTION                                   │
│   Input: Survivors + TRAINING history                        │
│   Output: 47 features characterizing WHY each survived       │
│                                                              │
│   ALSO: Compute HOLDOUT performance                          │
│   ┌────────────────────────────────────────────────────────┐ │
│   │ offset = len(train_history)  ← DERIVED, NOT CONFIGURED │ │
│   │ predictions = prng_func(seed, n_holdout, skip=offset)  │ │
│   │ holdout_hits = matches / n_holdout                     │ │
│   └────────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────────┤
│ STEP 5: ML TRAINING                                          │
│   X = structural features (exclude circular ones)            │
│   y = holdout_hits (NOT training score!)                     │
│   Learn: which feature patterns → FUTURE success             │
├──────────────────────────────────────────────────────────────┤
│ STEP 6: PREDICTION                                           │
│   Rank new survivors by predicted holdout quality            │
│   Top-K form prediction pool                                 │
└──────────────────────────────────────────────────────────────┘
```

### 3.2.1 Why Offset Must Be Derived (CRITICAL)

Per Team Beta's guidance:

> "Remove offset as a choice. Keep offset as a law."

```python
# CORRECT (immutable):
offset = len(train_history)

# WRONG (configurable):
offset = args.holdout_offset  # ❌ NEVER DO THIS
```

**Rationale:**
- PRNGs are state machines
- Holdout is contiguous future data
- Validation requires continuing from post-training state
- Any manual offset introduces ambiguity and defeats verification

The offset is recorded in metadata for debugging but NEVER configurable:
```json
"holdout_eval": {
  "train_len": 4000,
  "holdout_len": 1000,
  "offset": 4000,
  "derived_from": "train_history_length"
}
```

### 3.3 Implementation Change

```python
# meta_prediction_optimizer_anti_overfit.py

# OLD (broken):
y = [s['features']['score'] for s in survivors]

# NEW (correct):
y = [s['holdout_hits'] for s in survivors]
```

### 3.4 Features to Consider Excluding from X

These features measure training performance and may dominate:

```python
POTENTIALLY_CIRCULAR = {
    'score',                    # The old target
    'confidence',               # Derived from score
    'exact_matches',            # Defines score
    'residue_1000_match_rate',  # Equivalent to score
}
```

**Recommendation:** Test both approaches:
1. Keep all features in X, just change y
2. Exclude circular features from X, change y

Compare which produces better feature distribution.

### 3.5 Expected Outcomes After Fix

**Before (BROKEN):**
```
Feature Importance:
  residue_1000_match_rate:  65%
  exact_matches:            35%
  [60 other features]:       0%
```

**After (FIXED):**
```
Feature Importance:
  intersection_weight:      ~15-20%
  lane_agreement_*:         ~10-15%
  skip_entropy:             ~10-15%
  temporal_stability_*:     ~10-15%
  residue_*_coherence:      ~5-10%
  forward_count:            ~5-10%
  [distributed across structural features]
```

### 3.6 Success Criteria

| Metric | BROKEN | TARGET |
|--------|--------|--------|
| Features with >1% importance | 2 | >10 |
| Top feature importance | 65% | <25% |
| Holdout Hit@K | Unknown | Measurable lift vs random |

### 3.7 Verification Test (Synthetic Data)

Before running on real data, verify the pipeline with known ground truth:

**Test Setup:**
- PRNG: `java_lcg` (from `prng_registry`)
- TRUE_SEED: `12345`
- Training: `synthetic_train_v2.json` (4000 draws, positions 0-3999)
- Holdout: `synthetic_holdout_v2.json` (1000 draws, positions 4000-4999)

**Expected Result:**
```
Seed 12345 → holdout_hits = 1.0 (100%)
All other seeds → holdout_hits ≈ 0.001 (random chance)
```

**What This Proves:**
- ✅ Temporal offset logic works (`offset = len(train_history)`)
- ✅ PRNG registry alignment correct
- ✅ holdout_hits is a valid ML y-label
- ✅ Distributed pipeline works end-to-end

**Verification Command:**
```bash
python3 -c "
import json
from prng_registry import get_cpu_reference

with open('synthetic_holdout_v2.json') as f:
    holdout = json.load(f)

prng = get_cpu_reference('java_lcg')
outputs = prng(12345, 5000)
predicted = [v % 1000 for v in outputs[4000:]]  # offset = 4000

hits = sum(1 for p, a in zip(predicted, holdout) if p == a)
print(f'Seed 12345: {hits}/{len(holdout)} = {hits/len(holdout):.4f}')
# Expected: 1000/1000 = 1.0000
"
```

---

## 4. Feature Categories

### 4.1 Why Features Matter

Since every survivor passed the ~10⁻¹¹⁹¹ sieve, features characterize **WHY**:

### 4.2 Intersection Features

| Feature | Meaning | Predictive Value |
|---------|---------|------------------|
| `intersection_count` | Seeds in BOTH forward AND reverse | Higher = more robust |
| `intersection_ratio` | Quality of bidirectional overlap | Higher = better |
| `intersection_weight` | Strength of agreement | Higher = more confident |
| `forward_only_count` | Passed forward but not reverse | Lower = better |
| `reverse_only_count` | Passed reverse but not forward | Lower = better |
| `bidirectional_selectivity` | Precision of intersection | Higher = better |

### 4.3 Skip/Gap Features

| Feature | Meaning | Predictive Value |
|---------|---------|------------------|
| `skip_min` | Minimum gap that worked | Lower range = stronger |
| `skip_max` | Maximum gap that worked | Tighter = better |
| `skip_range` | Hypothesis flexibility | Smaller = more confident |
| `skip_entropy` | Distribution of successful gaps | Lower = more consistent |
| `skip_mean`, `skip_std` | Central tendency | Lower std = better |

**Tight skip range = stronger hypothesis** (only one gap pattern works)

### 4.4 Lane Agreement Features (CRT-derived)

| Feature | Meaning | Predictive Value |
|---------|---------|------------------|
| `lane_agreement_8` | Bit-level consistency | Higher = better |
| `lane_agreement_125` | Decimal structure consistency | Higher = better |
| `lane_consistency` | Overall CRT coherence | Higher = better |

### 4.5 Residue Features

| Feature | Meaning | Note |
|---------|---------|------|
| `residue_8_match_rate` | Direct matches at mod 8 | |
| `residue_125_match_rate` | Direct matches at mod 125 | |
| `residue_1000_match_rate` | Direct matches at mod 1000 | ⚠️ CIRCULAR with score |
| `residue_*_coherence` | Distribution similarity | Good predictor |
| `residue_*_kl_divergence` | Information-theoretic distance | Good predictor |

### 4.6 Temporal Features

| Feature | Meaning | Predictive Value |
|---------|---------|------------------|
| `temporal_stability_mean` | Consistency over time windows | Higher = more robust |
| `temporal_stability_std` | Variance in performance | Lower = better |
| `temporal_stability_trend` | Improving or degrading | Positive = better |
| `survivor_velocity` | Population change rate | Stable = better |

### 4.7 Global Features

| Feature | Meaning | Predictive Value |
|---------|---------|------------------|
| `global_residue_*_entropy` | Distribution uniformity | Signals PRNG type |
| `global_regime_change` | Lottery behavior shift | Warning flag |
| `global_marker_*_variance` | Tracking specific numbers | Bias detection |

---

## 5. Multi-Model Architecture

### 5.1 Supported Models

| Model | Library | Strengths |
|-------|---------|-----------|
| **neural_net** | PyTorch | Non-linear patterns, feature interactions |
| **xgboost** | XGBoost | High variance features, sparse data |
| **lightgbm** | LightGBM | Fast training, large datasets |
| **catboost** | CatBoost | Categorical features, robust defaults |
| **random_forest** | scikit-learn | Interpretability, feature importance |

### 5.2 Model Selection

```bash
# Train single model type
python3 meta_prediction_optimizer_anti_overfit.py \
    --model-type xgboost \
    --survivors survivors_with_scores.json

# Compare all model types
python3 meta_prediction_optimizer_anti_overfit.py \
    --compare-models \
    --survivors survivors_with_scores.json
```

### 5.3 Subprocess Isolation

Each model type runs in a **separate subprocess** to prevent GPU context conflicts:

```python
class SubprocessTrialCoordinator:
    """Coordinates model training in isolated subprocesses"""
    
    def train_model(self, model_type: str, config: dict):
        # Each model gets clean GPU context
        result = subprocess.run([
            'python3', 'train_single_trial.py',
            '--model-type', model_type,
            '--config', json.dumps(config)
        ])
        return result
```

This resolves OpenCL/CUDA conflicts when comparing PyTorch neural nets with XGBoost/LightGBM/CatBoost.

### 5.4 CRITICAL: GPU Isolation Design Invariant

> ⚠️ **MANDATORY INVARIANT (Session 72, Feb 2026)**
> 
> **GPU-accelerated code must NEVER run in the coordinating process when using `--compare-models`.**

**Why this matters:**
- LightGBM uses OpenCL, CatBoost/XGBoost/PyTorch use CUDA
- These runtimes do NOT coordinate VRAM ownership
- Once CUDA initializes in parent process, OpenCL fails with error -9999
- Cleanup APIs (`gc.collect()`, cache clears) are **ineffective**

**Implementation:**
```python
# At module level - DO NOT initialize GPU
CUDA_INITIALIZED = False  # Deferred to main()

# In main() - conditional based on mode
if args.compare_models:
    CUDA_INITIALIZED = False  # Parent stays GPU-clean!
else:
    CUDA_INITIALIZED = initialize_cuda_early()
```

**Verification:** When `--compare-models` is active, you MUST see:
```
⚡ Mode: Multi-Model Comparison (Subprocess Isolation)
   GPU initialization DEFERRED to subprocesses
✅ CUDA initialized: False
```

**Key Files:**
| File | Role |
|------|------|
| `meta_prediction_optimizer_anti_overfit.py` | Coordinator (NO GPU imports) |
| `subprocess_trial_coordinator.py` | Subprocess orchestration |
| `train_single_trial.py` | Single model worker (HAS GPU) |

See `docs/DESIGN_INVARIANT_GPU_ISOLATION.md` for full documentation.




## 6. AntiOverfitMetaOptimizer Class

### 6.1 Initialization

```python
class AntiOverfitMetaOptimizer:
    """ML optimizer with anti-overfitting measures"""
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,           # MUST be holdout_hits, NOT score!
        n_splits: int = 5,
        test_size: float = 0.2,
        study_name: str = 'anti_overfit',
        storage: str = None
    ):
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.test_size = test_size
        
        # Optuna study for hyperparameter optimization
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction='minimize',
            load_if_exists=True
        )
```

### 6.2 Key Methods

| Method | Purpose |
|--------|---------|
| `optimize(n_trials)` | Run Optuna optimization |
| `_objective(trial)` | Single trial with K-fold CV |
| `_evaluate_fold(...)` | Train and evaluate one fold |
| `final_evaluation()` | Holdout test set evaluation |
| `save_best_model()` | Save model + sidecar metadata |

---

## 7. K-Fold Cross-Validation

### 7.1 Why K-Fold

K-Fold prevents overfitting by:
- Training on K-1 folds, validating on 1 fold
- Rotating which fold is validation
- Averaging metrics across all folds

### 7.2 Implementation

```python
def _objective(self, trial: optuna.Trial) -> float:
    """Optuna objective with K-fold cross-validation"""
    
    # Sample hyperparameters
    params = self._sample_hyperparameters(trial)
    
    # K-fold cross-validation
    kf = KFold(n_splits=self.n_splits, shuffle=True)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train)):
        X_fold_train = self.X_train[train_idx]
        y_fold_train = self.y_train[train_idx]
        X_fold_val = self.X_train[val_idx]
        y_fold_val = self.y_train[val_idx]
        
        # Train and evaluate
        metrics = self._evaluate_fold(
            X_fold_train, y_fold_train,
            X_fold_val, y_fold_val,
            params
        )
        fold_scores.append(metrics.val_mae)
    
    return np.mean(fold_scores)
```

### 7.3 Overfitting Detection

```python
def is_overfitting(metrics: ValidationMetrics) -> bool:
    """Detect if model is overfitting"""
    return (
        metrics.overfit_ratio > 1.5 or      # Test error 50% higher than train
        metrics.test_mae > metrics.val_mae * 1.3 or  # Test 30% higher than val
        metrics.p_value > 0.05              # Not statistically significant
    )
```

---

## 8. Hyperparameter Sampling

### 8.1 Neural Network Parameters

```python
def _sample_neural_net_params(self, trial: optuna.Trial) -> dict:
    return {
        'hidden_layers': trial.suggest_int('hidden_layers', 1, 4),
        'hidden_size': trial.suggest_int('hidden_size', 32, 256),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'learning_rate': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'epochs': trial.suggest_int('epochs', 50, 200)
    }
```

### 8.2 XGBoost Parameters

```python
def _sample_xgboost_params(self, trial: optuna.Trial) -> dict:
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
    }
```

---

## 9. Sidecar Metadata

### 9.1 Purpose

The sidecar file (`best_model.meta.json`) contains everything needed to:
- Load and use the model
- Reproduce the training
- Track provenance

### 9.2 Sidecar Format

```json
{
  "model_type": "xgboost",
  "checkpoint_path": "models/reinforcement/best_model.xgb",
  "feature_schema_hash": "a1b2c3d4...",
  "feature_count": 62,
  "feature_names": ["intersection_weight", "lane_agreement_8", ...],
  "training_metrics": {
    "train_mae": 0.15,
    "val_mae": 0.18,
    "test_mae": 0.19,
    "overfit_ratio": 1.05
  },
  "hyperparameters": {
    "n_estimators": 500,
    "max_depth": 6,
    ...
  },
  "provenance": {
    "training_target": "holdout_hits",
    "survivor_count": 831672,
    "train_history_draws": 18000,
    "holdout_history_draws": 225,
    "cli_args": "--model-type xgboost --trials 50",
    "git_commit": "abc123...",
    "timestamp": "2025-12-30T14:30:00Z"
  },
  "agent_metadata": {
    "run_id": "anti_overfit_20251230_143000",
    "parent_run_id": "full_scoring_20251230_120000",
    "pipeline_step": 5
  }
}
```

### 9.3 Feature Schema Hash

The feature schema hash ensures the model receives features in the expected order:

```python
def compute_feature_hash(feature_names: List[str]) -> str:
    """Compute hash of feature schema for validation"""
    schema_str = ','.join(sorted(feature_names))
    return hashlib.sha256(schema_str.encode()).hexdigest()[:16]
```

Step 6 validates this hash before making predictions.

---

## 10. CLI Interface

### 10.1 Basic Usage

```bash
# Train single model
python3 meta_prediction_optimizer_anti_overfit.py \
    --survivors survivors_with_scores.json \
    --lottery-data train_history.json \
    --model-type neural_net \
    --trials 50 \
    --k-folds 5 \
    --output-dir models/reinforcement

# Compare all models
python3 meta_prediction_optimizer_anti_overfit.py \
    --survivors survivors_with_scores.json \
    --lottery-data train_history.json \
    --compare-models \
    --trials 50 \
    --output-dir models/reinforcement
```

### 10.2 CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--survivors` | Yes | - | Path to survivors with features |
| `--lottery-data` | Yes | - | Path to training history |
| `--model-type` | No | neural_net | Model type to train |
| `--compare-models` | No | False | Train all 4 model types |
| `--trials` | No | 50 | Number of Optuna trials |
| `--k-folds` | No | 5 | K-fold cross-validation splits |
| `--output-dir` | No | models/reinforcement | Output directory |
| `--study-name` | No | anti_overfit | Optuna study name |
| `--save-all-models` | No | False | Save all 4 models for analysis |

---

## 11. Integration Points

### 11.1 Pipeline Position

```
Step 3: Full Scoring
    ↓ (survivors_with_scores.json)
Step 4: Adaptive Meta-Optimizer
    ↓ (reinforcement_engine_config.json)
Step 5: ANTI-OVERFIT TRAINING ← You are here
    ↓ (best_model.* + best_model.meta.json)
Step 6: Prediction Generator
```

### 11.2 Input Requirements

| File | Source | Contains |
|------|--------|----------|
| `survivors_with_scores.json` | Step 3 | 62 features per survivor + holdout_hits |
| `train_history.json` | Step 1 | Training lottery draws |
| `holdout_history.json` | Step 1 | Holdout lottery draws |
| `reinforcement_engine_config.json` | Step 4 | Derived training parameters |

### 11.3 Output Files

```
models/reinforcement/
├── best_model.pth          # (if neural_net)
├── best_model.xgb          # (if xgboost)
├── best_model.lgb          # (if lightgbm)
├── best_model.cbm          # (if catboost)
└── best_model.meta.json    # Sidecar metadata (ALWAYS)
```

### 11.4 Consumed By

- **prediction_generator.py** — Uses model to rank survivors
- **watcher_agent.py** — Monitors for training completion

---

## Summary: Getting ML Right

### The Problem

Bidirectional survivors are mathematically significant (~10⁻¹¹⁹¹ false positive rate), but:
- Multiple seeds may exist (different sessions)
- Reseeding events may invalidate some
- Need to predict which will **continue** to perform

### The Wrong Approach (BROKEN)

```python
y = training_score  # Circular: score = f(features that define score)
```

Model learns tautology, ignores 60/62 features.

### The Right Approach (FIXED)

```python
y = holdout_hits  # Generalization: predict future from structural features
```

Model learns which feature patterns predict future success:
- Intersection quality
- Skip hypothesis consistency
- Temporal stability
- Lane agreement strength

### Success Metrics

| Metric | BROKEN | FIXED |
|--------|--------|-------|
| Features with >1% importance | 2 | >10 |
| Top feature importance | 65% | <25% |
| Generalization | None | Measurable |

---

## Version History

```
Version 3.0.0 - December 30, 2025
- MAJOR: Documented training target bug
- MAJOR: Documented correct y = holdout_hits
- MAJOR: Added feature importance expectations
- MAJOR: Explained why ML matters for survivors

Version 2.0.0 - December 22, 2025
- NEW: Multi-model support (--model-type, --compare-models)
- NEW: Subprocess isolation for OpenCL/CUDA compatibility
- NEW: Sidecar metadata generation

Version 1.x - November 9, 2025
- Original implementation
- K-fold cross-validation
```

---

**END OF CHAPTER 4**
