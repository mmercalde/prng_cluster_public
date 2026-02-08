# Chapter 14 Implementation Proposal — Session 69

**Date:** February 8, 2026  
**Status:** PROPOSAL — Awaiting Team Beta Approval  
**Scope:** Phases 1, 3, 6 of Chapter 14 Training Diagnostics  
**Estimated Lines:** ~695 new/modified

---

## Executive Summary

Chapter 14 Training Diagnostics has **specification complete** (3,134 lines in `CHAPTER_14_TRAINING_DIAGNOSTICS.md`) but **zero implementation**. This proposal covers the foundational phases required for autonomous operation:

| Phase | Component | Est. Lines | Priority |
|-------|-----------|-----------|----------|
| **1** | `training_diagnostics.py` — Core module | ~400 | **P0** |
| **3** | Pipeline wiring (`--enable-diagnostics`) | ~70 | **P1** |
| **6** | WATCHER integration (`check_training_health()`) | ~225 | **P2** |

**Deferred to future sessions:** Phases 2, 4, 5, 7, 8 (per-survivor attribution, dashboard, TensorBoard, LLM integration, selfplay wiring)

---

## 1. PyTorch Dynamic Computational Graph Integration

### 1.1 Core Mechanism

PyTorch's eager mode (`torch.autograd`) rebuilds the computational graph on every forward pass. This enables **passive observation** via hooks that fire automatically:

```python
# Hooks capture data WITHOUT modifying training behavior
model.layer.register_forward_hook(capture_activations)      # Output of layer
model.layer.register_full_backward_hook(capture_gradients)  # Gradients flowing through
```

### 1.2 What Hooks Capture (Neural Net)

| Metric | Hook Type | Purpose |
|--------|-----------|---------|
| Activation mean/std | Forward hook | Are neurons producing diverse outputs? |
| Dead neuron % | Forward hook (output == 0) | Is ReLU killing capacity? |
| Gradient norm per layer | Backward hook | Are gradients reaching early layers? |
| Gradient per feature | Backward hook (input layer) | Which features drive learning? |
| Weight norm | Direct param access | Are weights growing/collapsing? |

### 1.3 Tree Model Equivalents

Tree models don't use hooks but have native callback mechanisms:

| Model | Mechanism | Data Available |
|-------|-----------|----------------|
| XGBoost | `model.evals_result()` | Per-round train/val loss, feature importance |
| LightGBM | `lgb.record_evaluation()` callback | Per-round metrics, split/gain importance |
| CatBoost | `model.get_evals_result()` | Per-iteration metrics, PredictionValuesChange |

### 1.4 Key Design Principle

**Diagnostics are PASSIVE OBSERVERS.** They:
- ✅ Capture data during training
- ✅ Write JSON for downstream consumers
- ❌ NEVER modify gradients, weights, or training behavior
- ❌ NEVER fail training if diagnostics fail (best-effort, non-fatal)

---

## 2. Phase 1: `training_diagnostics.py` (~400 lines)

### 2.1 File Location

```
distributed_prng_analysis/
├── training_diagnostics.py          ← NEW FILE
└── diagnostics_outputs/             ← NEW DIRECTORY (auto-created)
    └── training_diagnostics.json
```

### 2.2 Class Structure

```python
training_diagnostics.py
├── class TrainingDiagnostics (base)
│   ├── attach(model, model_type)     # Register hooks/callbacks
│   ├── on_round_end(round, ...)      # Capture per-epoch/round snapshot
│   ├── detach()                      # Clean up hooks
│   ├── get_report()                  # Analyze + return diagnostics dict
│   ├── save(path)                    # Write JSON
│   └── _analyze(), _diagnose()       # Internal analysis methods
│
├── class NNDiagnostics(TrainingDiagnostics)
│   └── PyTorch hook registration (register_forward_hook, register_full_backward_hook)
│
├── class XGBDiagnostics(TrainingDiagnostics)
│   └── Wraps evals_result() + feature_importances_
│
├── class LGBDiagnostics(TrainingDiagnostics)
│   └── Wraps record_evaluation callback + split/gain importance
│
└── class CatBoostDiagnostics(TrainingDiagnostics)
    └── Wraps get_evals_result() + get_feature_importance(type='PredictionValuesChange')
```

### 2.3 Unified JSON Output Schema

```json
{
  "schema_version": "1.0.0",
  "model_type": "neural_net|xgboost|lightgbm|catboost",
  "generated_at": "2026-02-08T10:30:00Z",
  "training_config": {
    "epochs_or_rounds": 200,
    "early_stopping_round": 150,
    "feature_count": 47
  },
  "round_data": [
    {
      "round": 0,
      "train_loss": 0.0234,
      "val_loss": 0.0312,
      "layers": {  /* NN only */
        "fc1": {"activation_mean": 0.45, "dead_pct": 12.3, "gradient_norm": 0.0023}
      }
    }
  ],
  "analysis": {
    "loss": {
      "final_train": 0.0089,
      "final_val": 0.0156,
      "best_val": 0.0145,
      "best_val_round": 142,
      "overfit_gap": 0.0067,
      "is_plateau": false
    },
    "gradient_health": {  /* NN only */
      "vanishing": false,
      "exploding": false,
      "dead_neuron_pct": 8.5
    },
    "feature_importance": {
      "top_10": [{"feature": "intersection_weight", "importance": 0.234}, ...],
      "concentration_ratio": 0.45
    }
  },
  "diagnosis": {
    "severity": "ok|warning|critical",
    "issues": ["Dead neuron percentage high (12.3%)", "Feature scale imbalance"],
    "suggested_fixes": ["Switch ReLU → LeakyReLU", "Add BatchNorm to input layer"]
  }
}
```

### 2.4 Implementation Checklist (Phase 1)

| # | Task | Est. Lines |
|---|------|-----------|
| 1.1 | Create `training_diagnostics.py` with base `TrainingDiagnostics` class | ~80 |
| 1.2 | Implement `NNDiagnostics` — PyTorch hooks, per-epoch capture | ~120 |
| 1.3 | Implement `XGBDiagnostics` — evals_result wrapper | ~50 |
| 1.4 | Implement `LGBDiagnostics` — record_evaluation wrapper | ~50 |
| 1.5 | Implement `CatBoostDiagnostics` — get_evals_result wrapper | ~50 |
| 1.6 | Implement `_analyze()` — loss plateau, gradient flow, dead neurons | ~30 |
| 1.7 | Implement `_diagnose()` — severity classification, issue detection | ~20 |

**Total Phase 1: ~400 lines**

---

## 3. Phase 3: Pipeline Wiring (~70 lines)

### 3.1 Files Modified

| File | Change | Lines |
|------|--------|-------|
| `meta_prediction_optimizer_anti_overfit.py` | Add `--enable-diagnostics` CLI flag | ~10 |
| `reinforcement_engine.py` | Wire hooks into training loop | ~25 |
| `models/wrappers/*.py` | Add diagnostics capture to each wrapper | ~15 each (×4) |
| `reinforcement_engine_config.json` | Add `diagnostics` config block | ~10 |

### 3.2 CLI Interface

```bash
# Default: diagnostics OFF (zero overhead)
python3 meta_prediction_optimizer_anti_overfit.py --model-type catboost --trials 10

# Enable diagnostics for investigation
python3 meta_prediction_optimizer_anti_overfit.py --model-type catboost --trials 10 --enable-diagnostics

# Compare all models with diagnostics
python3 meta_prediction_optimizer_anti_overfit.py --compare-models --enable-diagnostics
```

### 3.3 Config Block Addition

```json
// reinforcement_engine_config.json
{
  "diagnostics": {
    "enabled": false,
    "capture_every_n": 5,
    "output_dir": "diagnostics_outputs",
    "nn_attribution_method": "grad_x_input",
    "top_survivors_for_attribution": 5
  }
}
```

### 3.4 Implementation Checklist (Phase 3)

| # | Task | Est. Lines |
|---|------|-----------|
| 3.1 | Add `--enable-diagnostics` CLI flag to anti_overfit.py | ~10 |
| 3.2 | Wire NNDiagnostics into `reinforcement_engine.py` epoch loop | ~25 |
| 3.3 | Wire tree diagnostics into each wrapper's `fit()` method | ~15 ×4 |
| 3.4 | Add config block to `reinforcement_engine_config.json` | ~10 |

**Total Phase 3: ~70 lines** (plus ~60 across wrappers)

---

## 4. Phase 6: WATCHER Integration (~225 lines)

### 4.1 Files Modified

| File | Change | Lines |
|------|--------|-------|
| `agents/watcher_agent.py` | Add `check_training_health()`, skip registry | ~180 |
| `watcher_policies.json` | Add training diagnostics policy entries | ~45 |

### 4.2 WATCHER Flow

```
Step 5 Completes
       ↓
check_training_health()
       ↓
Read diagnostics_outputs/training_diagnostics.json
       ↓
┌─────────────────────────────────────────┐
│ severity == "ok"     → PROCEED to Step 6│
│ severity == "warning"→ PROCEED + LOG    │
│ severity == "critical"                  │
│   ├─ 1st critical → RETRY with params   │
│   ├─ 2nd critical → RETRY with params   │
│   └─ 3rd critical → SKIP_MODEL + PROCEED│
└─────────────────────────────────────────┘
```

### 4.3 Policy Entries

```json
// watcher_policies.json additions
{
  "training_diagnostics": {
    "severity_thresholds": {
      "dead_neuron_pct": {"warning": 10, "critical": 25},
      "gradient_norm_min": {"warning": 1e-6, "critical": 1e-8},
      "overfit_gap": {"warning": 0.3, "critical": 0.5}
    },
    "retry_policy": {
      "max_retries_per_model": 2,
      "retry_with_param_adjustment": true
    },
    "skip_registry": {
      "skip_duration_hours": 24,
      "skip_file": "diagnostics_outputs/model_skip_state.json"
    }
  }
}
```

### 4.4 Implementation Checklist (Phase 6)

| # | Task | Est. Lines |
|---|------|-----------|
| 6.1 | Add policy entries to `watcher_policies.json` | ~45 |
| 6.2 | Implement `check_training_health()` in watcher_agent.py | ~80 |
| 6.3 | Implement `_check_skip_registry()` + `reset_skip_registry()` | ~40 |
| 6.4 | Implement `_archive_diagnostics()` | ~20 |
| 6.5 | Wire health check into pipeline (Step 5 → health → Step 6) | ~40 |

**Total Phase 6: ~225 lines**

---

## 5. Dependencies & Prerequisites

### 5.1 Already Installed (per pip_list.txt)

```
torch          # PyTorch — hooks, autograd
xgboost        # eval_set, evals_result
lightgbm       # record_evaluation
catboost       # get_evals_result
pydantic       # Schema validation
```

### 5.2 No New Dependencies Required

TensorBoard is optional (Phase 5, deferred) — not needed for core functionality.

---

## 6. Testing Plan

### 6.1 Unit Tests (Phase 1)

```bash
# Test NN diagnostics
python3 -c "
from training_diagnostics import NNDiagnostics
import torch.nn as nn
model = nn.Sequential(nn.Linear(47, 64), nn.ReLU(), nn.Linear(64, 1))
diag = NNDiagnostics()
diag.attach(model)
# Simulate one forward/backward
x = torch.randn(32, 47)
y = model(x)
y.sum().backward()
diag.on_round_end(0, train_loss=0.5, val_loss=0.6)
diag.detach()
report = diag.get_report()
print(f'✅ NN Diagnostics: {len(report[\"round_data\"])} rounds captured')
"
```

### 6.2 Integration Test (Phase 3)

```bash
# Run Step 5 with diagnostics enabled
python3 meta_prediction_optimizer_anti_overfit.py \
  --model-type catboost \
  --trials 1 \
  --enable-diagnostics

# Verify output
cat diagnostics_outputs/training_diagnostics.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'Model: {d[\"model_type\"]}')
print(f'Severity: {d[\"diagnosis\"][\"severity\"]}')
print(f'Issues: {d[\"diagnosis\"][\"issues\"]}')
"
```

### 6.3 WATCHER Integration Test (Phase 6)

```bash
# Test health check with ok severity
echo '{"diagnosis":{"severity":"ok"}}' > diagnostics_outputs/training_diagnostics.json
PYTHONPATH=. python3 -c "
from agents.watcher_agent import WatcherAgent
agent = WatcherAgent()
result = agent.check_training_health()
print(f'Action: {result[\"action\"]}')  # Expected: PROCEED
"

# Test health check with critical severity
echo '{"diagnosis":{"severity":"critical","issues":["test"]}}' > diagnostics_outputs/training_diagnostics.json
# Run same test — expected: RETRY
```

---

## 7. Summary

| Phase | Files | Lines | Deliverable |
|-------|-------|-------|-------------|
| **1** | 1 new | ~400 | `training_diagnostics.py` with 4 model-type backends |
| **3** | 5 modified | ~70 | `--enable-diagnostics` CLI, config, wrapper wiring |
| **6** | 2 modified | ~225 | `check_training_health()`, policy entries, skip registry |
| **Total** | **8 files** | **~695** | Core diagnostics + autonomous WATCHER integration |

---

## 8. Not In Scope (Deferred)

| Phase | What | Why Deferred |
|-------|------|--------------|
| 2 | `per_survivor_attribution.py` | Foundation first |
| 4 | `/training` dashboard route | Foundation first |
| 5 | TensorBoard integration | Optional, human-only |
| 7 | LLM `DiagnosticsBundle` | Requires Phase 1 data first |
| 8 | Selfplay + Chapter 13 wiring | Requires Phases 1, 6, 7 |

---

## 9. Approval Request

**Team Beta:** Please review and approve this proposal for Session 69 implementation.

- [ ] Phase 1 scope approved
- [ ] Phase 3 scope approved  
- [ ] Phase 6 scope approved
- [ ] JSON schema approved
- [ ] Testing plan approved

---

*End of Proposal*
