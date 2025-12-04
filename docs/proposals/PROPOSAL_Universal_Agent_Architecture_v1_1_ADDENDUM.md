# Universal Agent Architecture Proposal v1.1 — ADDENDUM

**Document Version:** 1.1 (Addendum)  
**Date:** December 2, 2025  
**Author:** Claude (AI Assistant)  
**Status:** DRAFT - Pending Review  
**Compatibility:** Schema v1.0.3+, Dual-LLM Infrastructure  
**Relationship:** ADDENDUM to Universal Agent Architecture Proposal v1.0  

---

## Document Relationship

This document is an **ADDENDUM** to the Universal Agent Architecture Proposal v1.0. It provides:

1. **Complete parameter specifications** that v1.0 references but doesn't define
2. **Optuna Agent Bridge** implementation for cross-run learning
3. **PRNG Registry** documentation (44 algorithms)
4. **Search Strategy** documentation (6 strategies)

**This document does NOT replace v1.0** — it supplements it with the detailed data needed for implementation.

---

## Implementation Status Overview

### Completed ✅

| Component | Document | Status | Evidence |
|-----------|----------|--------|----------|
| Schema v1.0.3 | Schema Extension Proposal | ✅ DONE | `metadata_writer.py` deployed |
| agent_metadata injection | Schema Extension Proposal | ✅ DONE | `results_manager.py` patched |
| Dual-LLM Servers | Schema v1.0.4 Proposal | ✅ DONE | Qwen2.5-Coder-14B + Math-7B running |
| LLM Router | Schema v1.0.4 Proposal | ✅ DONE | `llm_services/` deployed |
| Pipeline Steps 1 & 2 | Workflow Guide | ✅ DONE | bidirectional_survivors.json exists |

### Proposed 📝

| Component | Document | Status | This Addendum |
|-----------|----------|--------|---------------|
| BaseAgent class | v1.0 Proposal | 📝 PROPOSED | — |
| agent_manifests/ | v1.0 Proposal | 📝 PROPOSED | — |
| Step agents (1-6) | v1.0 Proposal | 📝 PROPOSED | — |
| config_manifests/ values | v1.0 Proposal | 📝 PROPOSED | ✅ DEFINED HERE |
| optuna_agent_bridge.py | — | 📝 PROPOSED | ✅ DEFINED HERE |
| PRNG Registry docs | — | 📝 PROPOSED | ✅ DEFINED HERE |

---

## Visual Status Map

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         IMPLEMENTATION ROADMAP                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Phase 1        Phase 2         Phase 3         Phase 3.5       Phase 4      │
│  Schema ───────► LLM ──────────► Agent ─────────► Config ───────► Watcher    │
│  v1.0.3          Infra           Architecture    Data            Agent       │
│                                                                               │
│  [  DONE  ]     [  DONE  ]      [ PROPOSED ]    [  THIS   ]     [ FUTURE ]   │
│  [   ✅   ]     [   ✅   ]      [   📝    ]     [ ADDENDUM]     [   📝   ]   │
│                                                                               │
│  Delivered:     Delivered:      Proposed in:    Proposed in:    Proposed in: │
│  • metadata_    • Qwen2.5-      • v1.0:         • v1.1:         • v1.0:      │
│    writer.py      Coder-14B       BaseAgent       71 params       Watcher    │
│  • results_     • Qwen2.5-      • v1.0:         • v1.1:           Agent      │
│    manager        Math-7B         6 step          44 PRNGs                   │
│    patch        • LLM Router      agents        • v1.1:                      │
│                                 • v1.0:           Optuna                     │
│                                   manifests       bridge                     │
│                                                                               │
│                                              ◄── WE ARE HERE                 │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

1. [What This Addendum Provides](#1-what-this-addendum-provides)
2. [Complete Parameter Registry](#2-complete-parameter-registry)
3. [PRNG Registry (44 Algorithms)](#3-prng-registry-44-algorithms)
4. [Search Strategies (6 Available)](#4-search-strategies-6-available)
5. [Config Manifest Specifications](#5-config-manifest-specifications)
6. [Optuna Agent Bridge Design](#6-optuna-agent-bridge-design)
7. [Integration with v1.0 Architecture](#7-integration-with-v10-architecture)
8. [Implementation Plan](#8-implementation-plan)

---

## 1. What This Addendum Provides

### v1.0 References → v1.1 Defines

| v1.0 Reference | v1.0 Says | v1.1 Provides |
|----------------|-----------|---------------|
| `config_manifests/step1_window_optimizer.json` | "What each step CAN TUNE" | Exact parameter names, types, ranges |
| Optuna parameters | Not specified | All 26 Optuna parameters with bounds |
| CLI arguments | Not specified | All 45 CLI arguments with defaults |
| PRNG types | "prng_type parameter" | Complete list of 44 algorithms |
| Search strategies | "bayesian" mentioned | All 6 strategies documented |
| Cross-run learning | Not addressed | `optuna_agent_bridge.py` implementation |

### New Components Defined

| Component | Lines of Code | Purpose |
|-----------|---------------|---------|
| `optuna_agent_bridge.py` | ~250 | Cross-run learning, agent-adjustable bounds |
| `config_manifests/*.json` | ~600 | Complete parameter specifications |
| `shared/prng_registry.json` | ~100 | Machine-readable PRNG list |

---

## 2. Complete Parameter Registry

### Summary

| Step | CLI Parameters | Optuna Parameters | Total |
|------|----------------|-------------------|-------|
| 1 - Window Optimizer | 7 | 5 | 12 |
| 2.5 - Scorer Meta | 9 | 9 | 18 |
| 3 - Full Scoring | 6 | 0 | 6 |
| 4 - ML Meta | 5 | 2 | 7 |
| 5 - Anti-Overfit | 7 | 10 | 17 |
| 6 - Prediction | 6 | 0 | 6 |
| **TOTAL** | **45** | **26** | **71** |

---

### Step 1: Window Optimizer

**Scripts:** `window_optimizer.py`, `window_optimizer_bayesian.py`, `window_optimizer_integration_final.py`

#### CLI Arguments

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `--strategy` | str | 'bayesian' | No | Optimization strategy |
| `--lottery-file` | str | — | Yes | Path to lottery data JSON |
| `--trials` | int | 50 | No | Number of Bayesian trials |
| `--output` | str | 'optimal_window_config.json' | No | Output config file |
| `--max-seeds` | int | 10,000,000 | No | Max seeds per trial |
| `--prng-type` | str | 'java_lcg' | No | Base PRNG type (see Section 3) |
| `--test-both-modes` | bool | False | No | Test constant AND variable skip |

#### Optuna Search Space

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `window_size` | int | 256-2048 | Analysis window size |
| `offset` | int | 0-500 | Window offset |
| `session_idx` | int | 0-2 | Session filter (both/midday/evening) |
| `skip_min` | int | 0-3 | Minimum skip value |
| `skip_max` | int | 10-200 | Maximum skip value |

#### Sieve Execution Modes

| Flag | Forward PRNG | Reverse PRNG |
|------|--------------|--------------|
| (default) | `{prng_type}` | `{prng_type}_reverse` |
| `--test-both-modes` | + `{prng_type}_hybrid` | + `{prng_type}_hybrid_reverse` |

---

### Step 2.5: Scorer Meta-Optimizer

**Scripts:** `generate_scorer_jobs.py`, `scorer_trial_worker.py`

#### CLI Arguments

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `--trials` | int | — | Yes | Number of trials to generate |
| `--survivors` | str | — | Yes | Path to survivors JSON |
| `--train-history` | str | — | Yes | Path to training history JSON |
| `--holdout-history` | str | — | Yes | Path to holdout history JSON |
| `--study-name` | str | — | Yes | Optuna study name |
| `--study-db` | str | — | Yes | Optuna storage URL |
| `--output` | str | 'scorer_jobs.json' | No | Output jobs file |
| `--legacy-scoring` | bool | False | No | Use legacy batch_score() |
| `--sample-size` | int | 25000 | No | Training sample size |

#### Optuna Search Space

| Parameter | Type | Range | Scale | Description |
|-----------|------|-------|-------|-------------|
| `residue_mod_1` | int | 5-20 | linear | First residue modulus |
| `residue_mod_2` | int | 50-150 | linear | Second residue modulus |
| `residue_mod_3` | int | 500-1500 | linear | Third residue modulus |
| `max_offset` | int | 1-15 | linear | Maximum offset value |
| `temporal_window_size` | int | 50-100 | linear | Temporal window size |
| `temporal_num_windows` | int | 3-10 | linear | Number of temporal windows |
| `min_confidence_threshold` | float | 0.05-0.25 | linear | Minimum confidence |
| `dropout` | float | 0.1-0.5 | linear | Dropout rate |
| `learning_rate` | float | 1e-4 to 1e-2 | log | Learning rate |

---

### Step 3: Full Scoring

**Scripts:** `generate_full_scoring_jobs.py`

#### CLI Arguments

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `--survivors` | str | — | Yes | Path to bidirectional_survivors.json |
| `--config` | str | 'optimal_scorer_config.json' | No | Path to optimal scorer config |
| `--train-history` | str | 'train_history.json' | No | Path to train history |
| `--holdout-history` | str | 'holdout_history.json' | No | Path to holdout history |
| `--chunk-size` | int | 1000 | No | Number of survivors per job |
| `--jobs-file` | str | 'scoring_jobs.json' | No | Output JSON file |

**Note:** No Optuna parameters — uses optimal config from Step 2.5

---

### Step 4: ML Meta-Optimizer

**Scripts:** `generate_ml_jobs.py`, `adaptive_meta_optimizer.py`

#### CLI Arguments

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `--trials` | int | 30 | No | Number of Optuna trials |
| `--survivors` | str | — | Yes | Path to survivors JSON |
| `--scores` | str | — | Yes | Path to scores JSON |
| `--study-name` | str | — | Yes | Optuna study name |
| `--study-db` | str | — | Yes | Optuna study database URL |

#### Optuna Search Space

| Parameter | Type | Range | Scale | Description |
|-----------|------|-------|-------|-------------|
| `dropout` | float | 0.1-0.5 | linear | Dropout rate |
| `epochs` | int | 30-100 | linear | Training epochs |

---

### Step 5: Anti-Overfit Training

**Scripts:** `meta_prediction_optimizer_anti_overfit.py`, `generate_anti_overfit_jobs.py`

#### CLI Arguments

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `--survivors` | str | — | Yes | Path to survivors JSON |
| `--lottery-data` | str | — | Yes | Path to lottery data |
| `--trials` | int | 50 | No | Number of trials |
| `--k-folds` | int | 5 | No | Number of K-Folds |
| `--test-holdout` | float | 0.2 | No | Test holdout ratio |
| `--study-name` | str | auto | No | Optuna study name |
| `--storage` | str | 'sqlite:///optuna_studies.db' | No | Optuna storage URL |

#### Optuna Search Space

| Parameter | Type | Range | Scale | Description |
|-----------|------|-------|-------|-------------|
| `n_layers` | int | 2-4 | linear | Number of hidden layers |
| `layer_{i}` | int | 32-256 | linear | Size of layer i (dynamic) |
| `dropout` | float | 0.2-0.5 | linear | Dropout rate |
| `weight_decay` | float | 1e-5 to 1e-2 | log | Weight decay |
| `learning_rate` | float | 1e-5 to 1e-3 | log | Learning rate |
| `epochs` | int | 50-150 | linear | Training epochs |
| `patience` | int | 5-15 | linear | Early stopping patience |
| `min_delta` | float | 1e-4 to 1e-2 | log | Early stopping min delta |
| `gradient_clip` | float | 0.5-5.0 | linear | Gradient clipping value |
| `batch_size` | int | 32-128 | categorical | Batch size |

---

### Step 6: Prediction

**Scripts:** `prediction_generator.py`, `concrete_prediction_test.py`

#### CLI Arguments

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `--config` | str | — | No | Path to config file |
| `--survivors-forward` | str | — | No | Path to forward survivors |
| `--survivors-reverse` | str | — | No | Path to reverse survivors |
| `--lottery-history` | str | — | No | Path to lottery history |
| `--k` | int | — | No | Number of predictions |
| `--test` | bool | False | No | Run in test mode |

**Note:** No Optuna parameters — uses trained model from Step 5

---

## 3. PRNG Registry (44 Algorithms)

### Organization

| Category | Count | Pattern |
|----------|-------|---------|
| Base PRNGs | 11 | `{family}` |
| Hybrid (Variable Skip) | 11 | `{family}_hybrid` |
| Reverse | 11 | `{family}_reverse` |
| Hybrid Reverse | 11 | `{family}_hybrid_reverse` |
| **Total** | **44** | |

### Complete PRNG Table

| Family | Constant | Hybrid | Reverse | Hybrid Reverse |
|--------|----------|--------|---------|----------------|
| `java_lcg` | ✅ | ✅ | ✅ | ✅ |
| `mt19937` | ✅ | ✅ | ✅ | ✅ |
| `xorshift32` | ✅ | ✅ | ✅ | ✅ |
| `xorshift64` | ✅ | ✅ | ✅ | ✅ |
| `xorshift128` | ✅ | ✅ | ✅ | ✅ |
| `pcg32` | ✅ | ✅ | ✅ | ✅ |
| `lcg32` | ✅ | ✅ | ✅ | ✅ |
| `minstd` | ✅ | ✅ | ✅ | ✅ |
| `xoshiro256pp` | ✅ | ✅ | ✅ | ✅ |
| `philox4x32` | ✅ | ✅ | ✅ | ✅ |
| `sfc64` | ✅ | ✅ | ✅ | ✅ |

### Machine-Readable Format

```json
{
  "prng_registry_version": "1.0",
  "total_count": 44,
  "families": [
    "java_lcg", "mt19937", "xorshift32", "xorshift64", "xorshift128",
    "pcg32", "lcg32", "minstd", "xoshiro256pp", "philox4x32", "sfc64"
  ],
  "variants": ["", "_hybrid", "_reverse", "_hybrid_reverse"],
  "all_prngs": [
    "java_lcg", "java_lcg_hybrid", "java_lcg_reverse", "java_lcg_hybrid_reverse",
    "mt19937", "mt19937_hybrid", "mt19937_reverse", "mt19937_hybrid_reverse",
    "xorshift32", "xorshift32_hybrid", "xorshift32_reverse", "xorshift32_hybrid_reverse",
    "xorshift64", "xorshift64_hybrid", "xorshift64_reverse", "xorshift64_hybrid_reverse",
    "xorshift128", "xorshift128_hybrid", "xorshift128_reverse", "xorshift128_hybrid_reverse",
    "pcg32", "pcg32_hybrid", "pcg32_reverse", "pcg32_hybrid_reverse",
    "lcg32", "lcg32_hybrid", "lcg32_reverse", "lcg32_hybrid_reverse",
    "minstd", "minstd_hybrid", "minstd_reverse", "minstd_hybrid_reverse",
    "xoshiro256pp", "xoshiro256pp_hybrid", "xoshiro256pp_reverse", "xoshiro256pp_hybrid_reverse",
    "philox4x32", "philox4x32_hybrid", "philox4x32_reverse", "philox4x32_hybrid_reverse",
    "sfc64", "sfc64_hybrid", "sfc64_reverse", "sfc64_hybrid_reverse"
  ]
}
```

### PRNG Selection Logic

```
User specifies: --prng-type java_lcg

Without --test-both-modes (2 sieves):
  Forward sieve → java_lcg
  Reverse sieve → java_lcg_reverse

With --test-both-modes (4 sieves):
  Forward constant → java_lcg
  Reverse constant → java_lcg_reverse
  Forward variable → java_lcg_hybrid
  Reverse variable → java_lcg_hybrid_reverse
```

---

## 4. Search Strategies (6 Available)

### Strategy Classes

| Strategy | Class | File | Status |
|----------|-------|------|--------|
| `bayesian` | `BayesianOptimization` | `window_optimizer.py` | ✅ CLI enabled |
| `random` | `RandomSearch` | `window_optimizer.py` | Code exists, CLI disabled |
| `grid` | `GridSearch` | `window_optimizer.py` | Code exists, CLI disabled |
| `evolutionary` | `EvolutionarySearch` | `window_optimizer.py` | Code exists, CLI disabled |
| `optuna_tpe` | `OptunaBayesianSearch` | `window_optimizer_bayesian.py` | Used by bayesian |
| `gaussian_process` | `GaussianProcessBayesianSearch` | `window_optimizer_bayesian.py` | Fallback |

### Current CLI Configuration

```python
# window_optimizer.py line 697 (current)
parser.add_argument('--strategy', type=str, choices=['bayesian'],
                   help='Optimization strategy')
```

### Proposed: Enable All Strategies

```python
# Proposed change
parser.add_argument('--strategy', type=str, 
                   choices=['bayesian', 'random', 'grid', 'evolutionary'],
                   default='bayesian',
                   help='Optimization strategy')
```

### Strategy Selection Guide

| Use Case | Recommended Strategy |
|----------|---------------------|
| Production runs | `bayesian` (TPE) |
| Quick testing | `random` |
| Small parameter space | `grid` |
| Escape local optima | `evolutionary` |

---

## 5. Config Manifest Specifications

### Purpose

Config manifests provide **machine-readable parameter definitions** that:

1. Enable agents to know what parameters exist
2. Define valid ranges for agent adjustment
3. Support the `optuna_agent_bridge.py` module
4. Document the system for future development

### Directory Structure

```
config_manifests/
├── step1_window_optimizer.json
├── step2_scorer_meta.json
├── step3_full_scoring.json
├── step4_ml_meta.json
├── step5_anti_overfit.json
├── step6_prediction.json
└── shared/
    ├── prng_registry.json
    └── cluster_config.json
```

### Step 1 Config Manifest (Complete)

```json
{
  "manifest_version": "1.0.0",
  "step": 1,
  "step_name": "window_optimizer",
  "description": "Bayesian window parameter optimization with bidirectional sieving",
  
  "scripts": {
    "primary": "window_optimizer.py",
    "bayesian": "window_optimizer_bayesian.py",
    "integration": "window_optimizer_integration_final.py"
  },
  
  "cli_arguments": {
    "strategy": {
      "type": "string",
      "default": "bayesian",
      "options": ["bayesian", "random", "grid", "evolutionary"],
      "enabled": ["bayesian"],
      "description": "Optimization strategy",
      "agent_adjustable": true
    },
    "lottery_file": {
      "type": "string",
      "required": true,
      "description": "Path to lottery data JSON"
    },
    "trials": {
      "type": "integer",
      "default": 50,
      "range": [10, 500],
      "description": "Number of Bayesian trials",
      "agent_adjustable": true
    },
    "output": {
      "type": "string",
      "default": "optimal_window_config.json",
      "description": "Output config file path"
    },
    "max_seeds": {
      "type": "integer",
      "default": 10000000,
      "range": [100000, 1000000000],
      "description": "Maximum seeds per trial",
      "agent_adjustable": true
    },
    "prng_type": {
      "type": "string",
      "default": "java_lcg",
      "options": "$ref:shared/prng_registry.json#/families",
      "description": "Base PRNG type",
      "agent_adjustable": true
    },
    "test_both_modes": {
      "type": "boolean",
      "default": false,
      "description": "Test both constant and variable skip patterns"
    }
  },
  
  "optuna_parameters": {
    "window_size": {
      "type": "integer",
      "range": [256, 2048],
      "default_range": [256, 2048],
      "step": 1,
      "description": "Analysis window size",
      "agent_adjustable": true
    },
    "offset": {
      "type": "integer",
      "range": [0, 500],
      "default_range": [0, 500],
      "step": 1,
      "description": "Window offset",
      "agent_adjustable": true
    },
    "session_idx": {
      "type": "integer",
      "range": [0, 2],
      "options_map": {
        "0": ["midday", "evening"],
        "1": ["midday"],
        "2": ["evening"]
      },
      "description": "Session filter index"
    },
    "skip_min": {
      "type": "integer",
      "range": [0, 3],
      "default_range": [0, 3],
      "step": 1,
      "description": "Minimum skip value",
      "agent_adjustable": true
    },
    "skip_max": {
      "type": "integer",
      "range": [10, 200],
      "default_range": [10, 200],
      "step": 1,
      "description": "Maximum skip value",
      "agent_adjustable": true
    }
  },
  
  "optuna_study": {
    "name_pattern": "step1_window_opt",
    "storage_dir": "optuna_studies",
    "persistent": true,
    "n_startup_trials": 5,
    "sampler": "TPESampler"
  },
  
  "inputs": [],
  
  "outputs": [
    "optimal_window_config.json",
    "bidirectional_survivors.json",
    "train_history.json",
    "holdout_history.json"
  ],
  
  "success_criteria": {
    "min_survivors": 1000,
    "min_confidence": 0.7
  },
  
  "follow_up_agent": "scorer_meta_agent"
}
```

### Step 2.5 Config Manifest (Complete)

```json
{
  "manifest_version": "1.0.0",
  "step": 2,
  "step_name": "scorer_meta_optimizer",
  "description": "Distributed scorer parameter optimization",
  
  "scripts": {
    "job_generator": "generate_scorer_jobs.py",
    "worker": "scorer_trial_worker.py",
    "launcher": "run_scorer_meta_optimizer.sh"
  },
  
  "cli_arguments": {
    "trials": {
      "type": "integer",
      "required": true,
      "range": [10, 500],
      "description": "Number of trials to generate",
      "agent_adjustable": true
    },
    "survivors": {
      "type": "string",
      "required": true,
      "description": "Path to survivors JSON"
    },
    "train_history": {
      "type": "string",
      "required": true,
      "description": "Path to training history JSON"
    },
    "holdout_history": {
      "type": "string",
      "required": true,
      "description": "Path to holdout history JSON"
    },
    "study_name": {
      "type": "string",
      "required": true,
      "description": "Optuna study name"
    },
    "study_db": {
      "type": "string",
      "required": true,
      "description": "Optuna storage URL"
    },
    "output": {
      "type": "string",
      "default": "scorer_jobs.json",
      "description": "Output jobs file"
    },
    "legacy_scoring": {
      "type": "boolean",
      "default": false,
      "description": "Use legacy batch_score()"
    },
    "sample_size": {
      "type": "integer",
      "default": 25000,
      "range": [1000, 100000],
      "description": "Training sample size",
      "agent_adjustable": true
    }
  },
  
  "optuna_parameters": {
    "residue_mod_1": {
      "type": "integer",
      "range": [5, 20],
      "default_range": [5, 20],
      "description": "First residue modulus",
      "agent_adjustable": true
    },
    "residue_mod_2": {
      "type": "integer",
      "range": [50, 150],
      "default_range": [50, 150],
      "description": "Second residue modulus",
      "agent_adjustable": true
    },
    "residue_mod_3": {
      "type": "integer",
      "range": [500, 1500],
      "default_range": [500, 1500],
      "description": "Third residue modulus",
      "agent_adjustable": true
    },
    "max_offset": {
      "type": "integer",
      "range": [1, 15],
      "default_range": [1, 15],
      "description": "Maximum offset value",
      "agent_adjustable": true
    },
    "temporal_window_size": {
      "type": "integer",
      "range": [50, 100],
      "default_range": [50, 100],
      "description": "Temporal window size",
      "agent_adjustable": true
    },
    "temporal_num_windows": {
      "type": "integer",
      "range": [3, 10],
      "default_range": [3, 10],
      "description": "Number of temporal windows",
      "agent_adjustable": true
    },
    "min_confidence_threshold": {
      "type": "float",
      "range": [0.05, 0.25],
      "default_range": [0.05, 0.25],
      "description": "Minimum confidence threshold",
      "agent_adjustable": true
    },
    "dropout": {
      "type": "float",
      "range": [0.1, 0.5],
      "default_range": [0.1, 0.5],
      "description": "Dropout rate",
      "agent_adjustable": true
    },
    "learning_rate": {
      "type": "float",
      "range": [0.0001, 0.01],
      "default_range": [0.0001, 0.01],
      "scale": "log",
      "description": "Learning rate",
      "agent_adjustable": true
    }
  },
  
  "optuna_study": {
    "name_pattern": "step2_scorer_meta",
    "storage_dir": "optuna_studies",
    "persistent": true
  },
  
  "inputs": [
    "bidirectional_survivors.json",
    "train_history.json",
    "holdout_history.json"
  ],
  
  "outputs": [
    "optimal_scorer_config.json",
    "scorer_meta_results.json"
  ],
  
  "success_criteria": {
    "min_validation_score": 0.7,
    "min_confidence": 0.8
  },
  
  "follow_up_agent": "full_scoring_agent"
}
```

**Note:** Steps 3-6 manifests follow the same pattern with their respective parameters from Section 2.

---

## 6. Optuna Agent Bridge Design

### Purpose

The `optuna_agent_bridge.py` module provides a **reusable interface** for AI agents to:

1. **Load/create persistent studies** — Cross-run learning
2. **Analyze historical trials** — Inform agent decisions
3. **Adjust search bounds** — Agent-driven parameter tuning
4. **Seed starting points** — `enqueue_trial()` with known-good params
5. **Export study summaries** — For `agent_metadata` injection

### Current Optuna Usage (from code audit)

| Script | Study Creation | Loads Existing? | Uses enqueue? |
|--------|----------------|-----------------|---------------|
| `generate_scorer_jobs.py` | `create_study()` | ❌ No | ❌ No |
| `meta_prediction_optimizer_anti_overfit.py` | `create_study()` | ❌ No | ❌ No |
| `generate_ml_jobs.py` | `load_study()` | ✅ Yes | ❌ No |
| `generate_anti_overfit_jobs.py` | `create_study()` | ❌ No | ❌ No |

**Problem:** Most scripts create fresh studies — no cross-run learning!

### Bridge Solution

```python
# agents/optuna_agent_bridge.py

import optuna
from optuna.samplers import TPESampler
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import numpy as np


class OptunaAgentBridge:
    """
    Unified Optuna integration for AI-driven hyperparameter optimization.
    
    Provides cross-run learning and agent-adjustable search spaces
    for all 6 pipeline steps.
    
    Usage:
        bridge = OptunaAgentBridge("config_manifests/step2_scorer_meta.json")
        study = bridge.create_or_load_study()
        
        # Agent analyzes history
        history = bridge.analyze_history()
        if history['total_trials'] > 50:
            # Narrow search to promising region
            bridge.suggest_bounds('residue_mod_1', (8, 15))
        
        # Seed with best historical params
        if history['best_params']:
            bridge.enqueue_priors([history['best_params']])
        
        # Use in objective function
        def objective(trial):
            params = {
                'residue_mod_1': bridge.suggest_int(trial, 'residue_mod_1'),
                'learning_rate': bridge.suggest_float(trial, 'learning_rate'),
            }
            return evaluate(params)
        
        study.optimize(objective, n_trials=100)
        
        # Export for agent_metadata
        optuna_metadata = bridge.get_study_summary()
    """
    
    def __init__(self, 
                 config_manifest_path: str,
                 storage_dir: str = "optuna_studies"):
        """
        Initialize bridge with config manifest.
        
        Args:
            config_manifest_path: Path to step's config manifest JSON
            storage_dir: Directory for Optuna SQLite databases
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Load config manifest
        with open(config_manifest_path) as f:
            self.manifest = json.load(f)
        
        self.step_name = self.manifest['step_name']
        self.optuna_params = self.manifest.get('optuna_parameters', {})
        self.study_config = self.manifest.get('optuna_study', {})
        
        # Current bounds (can be modified by agent)
        self._bounds = self._load_default_bounds()
        
        # Study reference
        self._study: Optional[optuna.Study] = None
    
    def _load_default_bounds(self) -> Dict[str, Dict]:
        """Load default bounds from manifest."""
        bounds = {}
        for param_name, param_config in self.optuna_params.items():
            bounds[param_name] = {
                'range': list(param_config.get('default_range', param_config.get('range', [0, 100]))),
                'type': param_config['type'],
                'scale': param_config.get('scale', 'linear'),
                'step': param_config.get('step', 1)
            }
        return bounds
    
    def create_or_load_study(self, 
                             study_name: Optional[str] = None,
                             sampler: Optional[optuna.samplers.BaseSampler] = None
                             ) -> optuna.Study:
        """
        Load existing study or create new one for cross-run learning.
        
        Args:
            study_name: Override study name (defaults to manifest pattern)
            sampler: Custom sampler (defaults to TPESampler)
            
        Returns:
            Optuna Study object
        """
        if study_name is None:
            study_name = self.study_config.get('name_pattern', f"study_{self.step_name}")
        
        storage_path = self.storage_dir / f"{study_name}.db"
        storage_url = f"sqlite:///{storage_path}"
        
        if sampler is None:
            n_startup = self.study_config.get('n_startup_trials', 5)
            sampler = TPESampler(n_startup_trials=n_startup)
        
        self._study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,  # KEY: Cross-run learning
            direction="maximize",
            sampler=sampler
        )
        
        print(f"📊 Optuna study '{study_name}': {len(self._study.trials)} existing trials")
        return self._study
    
    def analyze_history(self) -> Dict[str, Any]:
        """
        Analyze past trials for LLM context.
        
        Returns:
            Dictionary with trial statistics and recommendations
        """
        if self._study is None:
            return {"error": "No study loaded"}
        
        trials = self._study.trials
        completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if not completed:
            return {
                "total_trials": 0,
                "best_params": None,
                "best_value": None,
                "ready_for_optimization": False
            }
        
        # Best trial info
        best_trial = self._study.best_trial
        
        # Calculate parameter importance (if enough trials)
        importance = {}
        if len(completed) >= 10:
            try:
                importance = optuna.importance.get_param_importances(self._study)
            except Exception:
                pass
        
        # Convergence trend (last 10 trials)
        recent_values = [t.value for t in completed[-10:] if t.value is not None]
        if len(recent_values) >= 2:
            trend = "improving" if recent_values[-1] > np.mean(recent_values[:-1]) else "stable"
        else:
            trend = "insufficient_data"
        
        # Parameter distributions for top 20% trials
        good_trials = sorted(completed, key=lambda t: t.value or 0, reverse=True)
        top_n = max(1, len(good_trials) // 5)
        top_trials = good_trials[:top_n]
        
        param_distributions = {}
        for param_name in self.optuna_params.keys():
            values = [t.params.get(param_name) for t in top_trials if param_name in t.params]
            if values:
                param_distributions[param_name] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)) if len(values) > 1 else 0
                }
        
        return {
            "total_trials": len(completed),
            "best_params": best_trial.params,
            "best_value": best_trial.value,
            "best_trial_number": best_trial.number,
            "param_importance": dict(importance),
            "convergence_trend": trend,
            "param_distributions": param_distributions,
            "ready_for_optimization": True
        }
    
    def suggest_bounds(self, 
                       param_name: str, 
                       new_range: Tuple[float, float]) -> bool:
        """
        Agent adjusts search bounds for a parameter.
        
        Args:
            param_name: Parameter name from manifest
            new_range: New (min, max) range
            
        Returns:
            True if bounds updated successfully
        """
        if param_name not in self._bounds:
            print(f"⚠️  Unknown parameter: {param_name}")
            return False
        
        old_range = self._bounds[param_name]['range']
        self._bounds[param_name]['range'] = list(new_range)
        print(f"📐 Adjusted {param_name}: {old_range} → {list(new_range)}")
        return True
    
    def enqueue_priors(self, prior_params: List[Dict[str, Any]]) -> int:
        """
        Seed study with known-good starting points.
        
        Args:
            prior_params: List of parameter dictionaries to try first
            
        Returns:
            Number of trials enqueued
        """
        if self._study is None:
            return 0
        
        count = 0
        for params in prior_params:
            try:
                self._study.enqueue_trial(params)
                count += 1
            except Exception as e:
                print(f"⚠️  Failed to enqueue: {e}")
        
        if count > 0:
            print(f"🎯 Enqueued {count} prior trial(s)")
        return count
    
    def suggest_int(self, trial: optuna.Trial, param_name: str) -> int:
        """
        Suggest integer parameter using agent-adjusted bounds.
        """
        bounds = self._bounds.get(param_name, {})
        range_vals = bounds.get('range', [0, 100])
        step = bounds.get('step', 1)
        
        return trial.suggest_int(param_name, int(range_vals[0]), int(range_vals[1]), step=step)
    
    def suggest_float(self, trial: optuna.Trial, param_name: str) -> float:
        """
        Suggest float parameter using agent-adjusted bounds.
        """
        bounds = self._bounds.get(param_name, {})
        range_vals = bounds.get('range', [0.0, 1.0])
        scale = bounds.get('scale', 'linear')
        
        return trial.suggest_float(
            param_name, 
            float(range_vals[0]), 
            float(range_vals[1]),
            log=(scale == 'log')
        )
    
    def get_study_summary(self) -> Dict[str, Any]:
        """
        Get summary for agent_metadata injection.
        
        Returns:
            Dictionary for optuna_metadata field
        """
        if self._study is None:
            return {}
        
        return {
            "study_name": self._study.study_name,
            "total_trials": len(self._study.trials),
            "best_trial_number": self._study.best_trial.number if self._study.best_trial else None,
            "best_value": self._study.best_value if self._study.best_trial else None,
            "cross_run_learning": True,
            "current_bounds": {k: v['range'] for k, v in self._bounds.items()}
        }
    
    def archive_study(self, reason: str = "manual") -> str:
        """
        Archive current study and start fresh.
        
        Args:
            reason: Reason for archiving
            
        Returns:
            Path to archived study
        """
        if self._study is None:
            return ""
        
        import shutil
        from datetime import datetime
        
        current_path = self.storage_dir / f"{self._study.study_name}.db"
        archive_name = f"{self._study.study_name}_archived_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        archive_path = self.storage_dir / "archives" / archive_name
        
        archive_path.parent.mkdir(exist_ok=True)
        shutil.copy2(current_path, archive_path)
        
        print(f"📦 Archived study to: {archive_path}")
        print(f"   Reason: {reason}")
        
        # Delete current study
        optuna.delete_study(
            study_name=self._study.study_name,
            storage=f"sqlite:///{current_path}"
        )
        
        self._study = None
        return str(archive_path)


def create_bridge(step: int) -> OptunaAgentBridge:
    """
    Factory function to create bridge for a pipeline step.
    
    Args:
        step: Pipeline step number (1-6)
        
    Returns:
        Configured OptunaAgentBridge
    """
    manifest_map = {
        1: "config_manifests/step1_window_optimizer.json",
        2: "config_manifests/step2_scorer_meta.json",
        3: "config_manifests/step3_full_scoring.json",
        4: "config_manifests/step4_ml_meta.json",
        5: "config_manifests/step5_anti_overfit.json",
        6: "config_manifests/step6_prediction.json"
    }
    
    manifest_path = manifest_map.get(step)
    if not manifest_path:
        raise ValueError(f"Invalid step: {step}")
    
    return OptunaAgentBridge(manifest_path)
```

---

## 7. Integration with v1.0 Architecture

### How This Addendum Connects

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    v1.0 ARCHITECTURE (from v1.0 Proposal)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  BaseAgent (v1.0)                                                            │
│  ├── load_config()          ──────────► config_manifests/*.json (v1.1)      │
│  ├── consult_llm()          ──────────► Already deployed ✅                  │
│  ├── adjust_parameters()    ──────────► Uses optuna_agent_bridge.py (v1.1)  │
│  └── execute()                                                               │
│                                                                              │
│  Step Agents (v1.0)                                                          │
│  ├── WindowOptimizerAgent   ──────────► step1_window_optimizer.json (v1.1)  │
│  ├── ScorerMetaAgent        ──────────► step2_scorer_meta.json (v1.1)       │
│  └── ... (Steps 3-6)        ──────────► Remaining manifests (v1.1)          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### v1.0 Code That Uses v1.1 Data

```python
# From v1.0 BaseAgent.load_config()
def load_config(self) -> Dict:
    """Load configuration from manifest."""
    manifest_path = f"config_manifests/{self.step_name}.json"  # ← v1.1 defines this
    with open(manifest_path) as f:
        return json.load(f)

# From v1.0 BaseAgent.adjust_parameters()  
def adjust_parameters(self, analysis: Dict) -> Dict:
    """Use LLM to adjust parameters based on analysis."""
    # v1.1's optuna_agent_bridge.py provides:
    # - analyze_history() for LLM context
    # - suggest_bounds() for range adjustment
    # - enqueue_priors() for seeding good params
    pass
```

---

## 8. Implementation Plan

### What This Addendum Requires

| Task | Time | Dependency |
|------|------|------------|
| Create `config_manifests/` directory | 5 min | None |
| Create `step1_window_optimizer.json` | 15 min | None |
| Create `step2_scorer_meta.json` | 15 min | None |
| Create `step3_full_scoring.json` | 10 min | None |
| Create `step4_ml_meta.json` | 10 min | None |
| Create `step5_anti_overfit.json` | 15 min | None |
| Create `step6_prediction.json` | 10 min | None |
| Create `shared/prng_registry.json` | 10 min | None |
| Create `agents/optuna_agent_bridge.py` | 30 min | Manifests |
| **Total** | **~2 hours** | |

### What v1.0 Requires (Separate)

| Task | Time | Dependency |
|------|------|------------|
| Create `agents/base_agent.py` | 2 hours | v1.1 manifests |
| Create step agents (6) | 3-4 hours | base_agent.py |
| Create `agents/agent_manager.py` | 1 hour | step agents |
| Integration testing | 2 hours | All above |
| **Total** | **~8-10 hours** | |

### Recommended Order

1. **v1.1 First** — Create manifests and bridge (this addendum)
2. **v1.0 Second** — Implement BaseAgent using v1.1 data
3. **Testing** — Verify end-to-end

---

## Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Author | Claude (AI) | 2025-12-02 | ✓ |
| Technical Review | | | |
| Team Lead | | | |
| Final Approval | | | |

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-02 | Universal Agent Architecture (BaseAgent, step agents) |
| 1.1 | 2025-12-02 | **ADDENDUM**: Parameter registry (71), PRNG registry (44), search strategies (6), config manifests, optuna_agent_bridge.py |

---

## Quick Reference

### Parameter Counts

| Step | CLI | Optuna | Total |
|------|-----|--------|-------|
| 1 | 7 | 5 | 12 |
| 2.5 | 9 | 9 | 18 |
| 3 | 6 | 0 | 6 |
| 4 | 5 | 2 | 7 |
| 5 | 7 | 10 | 17 |
| 6 | 6 | 0 | 6 |
| **Total** | **45** | **26** | **71** |

### PRNG Summary

- **11 families**: java_lcg, mt19937, xorshift32/64/128, pcg32, lcg32, minstd, xoshiro256pp, philox4x32, sfc64
- **4 variants each**: base, hybrid, reverse, hybrid_reverse
- **44 total PRNGs**

### Document Hierarchy

```
Schema v1.0.3 ✅ → Schema v1.0.4 ✅ → v1.0 Proposal 📝 → v1.1 Addendum 📝
                                           │                    │
                                           │                    └── Parameter data
                                           └── Architecture pattern
```

---
┌─────────────────────────────────────────────────────────────────────┐
│           WINDOW OPTIMIZER - STRATEGY SUPPORT COMPLETE              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Strategy      │ Status    │ Description                           │
│  ─────────────────────────────────────────────────────────────────  │
│  bayesian      │ ✅ READY  │ Optuna TPE - learns, most efficient   │
│  random        │ ✅ READY  │ Random sampling - good baseline       │
│  grid          │ ✅ READY  │ Exhaustive grid search                │
│  evolutionary  │ ✅ READY  │ Genetic algorithm approach            │
│                                                                     │
│  Manifest: agent_manifests/window_optimizer.json v1.2.0             │
│  CLI: python3 window_optimizer.py --strategy {bayesian|random|...}  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
**End of Addendum**
