# Chapter 5: Adaptive Meta-Optimizer (Step 4)

> **What TFM is.** TFM filters candidate generators against observed draws in two directions,
> keeps a population of survivors on purpose, ranks them, and lets the top candidates vote their
> next value into a prediction pool. A survivor is a scored candidate, not a verdict. The
> generator family is a substrate for mimicry, not a claim about the machine. The target is a
> pool hit rate. TFM does not recover a seed or reconstruct state, and an objection that assumes
> it does is out of scope.

**Correction-pass status (2026-11-11):** **CORRECTED AGAINST CODE at `a15b524`**, not closed.
The v2.0.0 text below is kept in full. Where it disagrees with the code, an inline **"Corrected
2026-11-11"** note or a **SUPERSEDED (2026-11-11)** marker says what the code at `a15b524` does,
with an anchor. **Read §10–§15 (end of file) first:** §10 is what Step 4 actually produces and who
reads it, §11 covers RANGE-MINER / the 22-array NPZ / the window anchor, §12 lists the known gaps,
§13 the open items, §14 the VIR-1…6 declaration, §15 the closure statement and completion sentinel.
**Authority for this pass:** `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md` (REV1).
**Base:** `a15b5247336255801de8cf67d52653701b1e8c80` (`a15b524`) on VM 101. The working tree was at
`741822b`, which differs from `a15b524` only under `docs/` (`git diff --name-only a15b524 741822b`),
so the code read is the code at `a15b524`.
**Audit:** `docs/CHAPTER_5_AUDIT.md` (41 claims: 19 accurate, 5 BROKEN, 1 SWITCHED-OFF-BY-RULING,
10 DESCRIPTION-ONLY, 1 UNVERIFIED, 5 absorbed into other items).
**Numbering:** bare "Step N" in this chapter means the **executable** scheme
(`docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1). This chapter is **executable Step 4**:
`agents/watcher_agent.py:392` `STEP_SCRIPTS[4] = "adaptive_meta_optimizer.py"`, `:403`
`STEP_MANIFESTS[4] = "ml_meta.json"`, `agent_manifests/ml_meta.json:4` `"pipeline_step": 4`.
Chapter number 5 is not a step number.
**Canonical version:** this file is the only Chapter 5 ever committed (`git log --all` over every
`CHAPTER_5*` path). The "Version 1.0.0" in the history below was never committed (audit U1).

### How this chapter was produced (2026-11-11 pass)

1. Every substantive claim was checked against the code at `a15b524`: `adaptive_meta_optimizer.py`
   read in full, its manifest and config, the WATCHER dispatch and freshness code, Step 1's writer
   of `optimal_window_config.json`, and Step 5's inputs. Results are in `docs/CHAPTER_5_AUDIT.md`.
2. The governance trail was searched before anything was called a finding. The one governed item
   (S1, TB Q4 "preserve the drift") is labelled SWITCHED-OFF-BY-RULING, not reported as new.
3. Host state was read, never changed: the 27 archived Step 4 results in `optimization_results/`,
   `train_history.json`, the live model sidecar, the skip registry, `agent_manifests/definitions.json`
   (gitignored; no Step 4 entry), `crontab -l`, `systemctl list-units`. Nothing was run. No
   training, Optuna or pipeline step was started.
4. The chapter was then corrected **in place, section by section**. No section was deleted. Stale
   sections carry SUPERSEDED markers and keep their text. Every disagreement carries exactly one
   label (BROKEN, SWITCHED-OFF-BY-RULING or DESCRIPTION-ONLY); all are listed in §12.
5. Claim scope: what the chapter says against what the code at `a15b524` does. This pass does not
   judge whether the design is right.

## PRNG Analysis Pipeline — Complete Operating Guide

**Version:** 2.0.0 (Corrected), with a 2026-11-11 correction overlay (see Version History)  
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
10. [What Step 4 actually produces at a15b524, and who reads it](#10-what-step-4-actually-produces-at-a15b524-and-who-reads-it) *(added 2026-11-11)*
11. [RANGE-MINER, the 22-array NPZ and the window anchor](#11-range-miner-the-22-array-npz-and-the-window-anchor) *(added 2026-11-11)*
12. [Known gaps at a15b524](#12-known-gaps-at-a15b524) *(added 2026-11-11)*
13. [Open items](#13-open-items) *(added 2026-11-11)*
14. [Verification declaration (VIR-1…6)](#14-verification-declaration-vir-16) *(added 2026-11-11)*
15. [Closure statement](#15-closure-statement) *(added 2026-11-11)*

---

## 1. Overview

### 1.1 What the Adaptive Meta-Optimizer Does

The Adaptive Meta-Optimizer is **Step 4** of the 6-step pipeline. It is a **capacity and architecture planner** that determines:

1. **How many survivors** the model should train on (pool size)
2. **How deep** the network should be (architecture)
3. **How long** training should run (epochs)

> **Corrected 2026-11-11 (DESCRIPTION-ONLY, D1):** previously said "the 6-step pipeline". At
> `a15b524` `STEP_SCRIPTS` has **seven** entries, 0–6. Step 0 is TRSE regime segmentation
> (`agents/watcher_agent.py:387-395`). Step 4 is still Step 4.
>
> **Added 2026-11-11:** Step 4 computes these three values, but at `a15b524` none of them reaches
> Step 5 as data (§10, B5). On the host, every archived run produced the same three values (§10.2).

### 1.2 Critical Design Principle

> **Step 4 is intentionally NOT data-aware.**
> 
> It derives capacity parameters from window optimization behavior and training-history complexity only. Survivor-level data (including `holdout_hits`) is first consumed in **Step 5**, where model selection and overfit control occur.

> **Verified 2026-11-11 (accurate):** `adaptive_meta_optimizer.py` opens only its config (`:167`),
> the window results (`:226`), the training history (`:321`), the feedback file (`:482`) and its
> own outputs (`:972-1078`). It opens no survivor or holdout file. Step 5's training target is
> `holdout_quality`, and `holdout_hits` is excluded from its features
> (`meta_prediction_optimizer_anti_overfit.py:612`, `:629`, `:1481`). See D8.

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

> **Corrected 2026-11-11:**
> - *(DESCRIPTION-ONLY, D2)* "62 features per survivor" is stale. The live trained sidecar
>   `models/reinforcement/best_model.meta.json` (2026-03-06) records `feature_schema.feature_count`
>   = **89**. Step 4 does not read the file, so this does not change Step 4.
> - *(BROKEN, B4)* "Reads: reinforcement feedback (if available)". The reader exists
>   (`adaptive_meta_optimizer.py:471`), but nothing in the tracked tree, git history or host writes
>   `optimization_results/reinforcement_feedback.json`. See §12.
> - *(BROKEN, B5)* The arrow "reinforcement_engine_config.json — capacity parameters ↓ Step 5" is a
>   file-existence and freshness dependency only. Step 5 does not read the values (§10).
> - *(DESCRIPTION-ONLY, D8)* "FIRST step to use holdout_hits": Step 5 is the first step to consume
>   holdout-derived fields. Its target is `holdout_quality`, and `holdout_hits` is excluded from
>   features (`meta_prediction_optimizer_anti_overfit.py:612`, `:629`). Chapter 6 owns the detail.

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

> **Corrected 2026-11-11:**
> - *(SWITCHED-OFF-BY-RULING, S1)* "How the sieve optimization performed" does not reach the
>   calculation. Step 4 reads `best_result` / `all_results` keys that Step 1 does not write, so it
>   uses constants. TB Q4 rules that this drift is preserved (§12.2).
> - *(BROKEN, B2/B3)* Of "entropy, regime changes", entropy is computed and discarded, and
>   regime-change frequency is never computed (§4.3 note).
>
> Only one history metric, temporal stability of entropy across 100-draw windows, shapes the output.

### 2.2 Input Requirements

| Input | Source | Contains |
|-------|--------|----------|
| `optimal_window_config.json` | Step 1 | Sieve optimization results |
| `train_history.json` | Data source | Training lottery draws |
| `reinforcement_feedback.json` | Step 6 (optional) | Historical performance |

> **Corrected 2026-11-11:** the first two rows are accurate (`agent_manifests/ml_meta.json:6-9`;
> `adaptive_meta_optimizer.py:83-93`, `:1140-1143`). The host `train_history.json` is a flat list
> of ints, which `:332-333` parses. The third row is **BROKEN (B4)**. The path read is
> `optimization_results/reinforcement_feedback.json` (`:471`), and **no producer exists**. Step 6
> (`prediction_generator.py`) does not write it, and neither does any other tracked file. Git
> history (`git log --all -S`) and the host (`find / -xdev`) have no writer either.

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

> **Corrected 2026-11-11:**
> - *(DESCRIPTION-ONLY, D4)* There are two outputs, not one. With `--apply` in full mode Step 4
>   merges into `reinforcement_engine_config.json` (`adaptive_meta_optimizer.py:1171-1172`,
>   `:994-1053`). In every mode it writes `optimization_results/meta_optimization_results.json` plus
>   a timestamped copy (`:1057-1081`). The manifest lists both (`agent_manifests/ml_meta.json:16-19`).
> - *(BROKEN, B5)* "Consumed By Step 5": WATCHER requires the file to exist and to be older than
>   Step 5's output (`agent_manifests/reinforcement.json:6-10` → `check_output_freshness`,
>   `agents/watcher_agent.py:803-834`). Step 5 itself never opens it (§10.1).

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

> **Corrected 2026-11-11:** each row of this table is checked once, elsewhere. Loading the window
> results and the history is accurate (§2.2 note). The window results feed only constants (S1).
> The feedback row has no producer (B4). The pool size is derived (`:628-738`), but its primary
> input is constant (S1). The architecture is derived but fixed (B1, §5.2 note). The epochs formula
> is accurate (`:814-824`). The config JSON is written (`:994-1053`), and Step 5 does not read it
> (B5).

### 3.2 ❌ Step 4 DOES NOT

| Action | Why Not |
|--------|---------|
| Load `survivors_with_scores.json` | Causes validation leakage |
| Inspect `holdout_hits` | Contaminates decisions |
| Perform any evaluation | That's Step 5's job |
| Choose model type | That's Step 5's job |
| Touch holdout data | Would invalidate pipeline |

> **Verified 2026-11-11 (accurate):** no survivor or holdout read (§1.2 note). No model-type field
> appears in either output (`adaptive_meta_optimizer.py:861-877`, `:1031-1049`). There is no
> evaluation: the architecture "trials" are a closed-form score (`:757-771`).

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

> **Verified 2026-11-11 (accurate):** this is `MetaOptimizerConfig.sources` verbatim
> (`adaptive_meta_optimizer.py:83-101`). The tracked `adaptive_meta_optimizer_config.json`, the
> default `--config`, carries the same weights. Its `lottery_history.path` is `lottery_data.json`,
> which WATCHER overrides with `--lottery-data`.

### 4.2 Window Optimizer Analysis (60%)

Extracts from Step 1 results:
- **Survivor count range** — min, optimal, max from trials
- **Convergence metrics** — how quickly good solutions were found
- **Stability metrics** — variance in top results

> **SWITCHED-OFF-BY-RULING (2026-11-11, S1).** The extraction code exists, but its input does not.
> `WindowOptimizerAnalyzer` reads `best_result.bidirectional_count`, `best_result.precision` and
> `all_results[]` (`adaptive_meta_optimizer.py:244-263`, `:271-295`). Step 1 writes a **flat**
> `optimal_window_config.json` with a top-level `bidirectional_count` and neither key
> (`window_optimizer.py:951-965`).
>
> With a real Step 1 file the analyzer therefore returns constants (`:247`, `:255-257`, `:263`,
> `:272-273`):
>
> | value | constant |
> |---|---|
> | optimal | 500 |
> | min | 250 |
> | max | 1000 |
> | confidence | 0.0 |
> | convergence speed | 0.5 |
> | stability | 0.5 |
>
> The `{100, 500, 2000}` defaults at `:242` fire only for an empty file.
>
> **Ruling:** TB Q4, "do NOT add `best_result`, `all_results`, or `mod` to
> `optimal_window_config.json` inside S172 … Miner inherits the silent-fallback behavior unchanged"
> (`docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:304`, §13 Q4 `:723`, §11.F `:561-567`).
>
> **Lifting it would enable:** the pool size, epoch count and confidence would follow what Step 1
> actually found, instead of being the same on every run (§10.2). This pass neither recommends
> nor opposes that.

### 4.3 Historical Pattern Analysis (35%)

Analyzes training lottery data:
- **Entropy** — higher entropy = more complex = more capacity needed
- **Temporal stability** — unstable patterns need larger pools
- **Regime change frequency** — frequent changes need adaptability

> **Corrected 2026-11-11.** One of the three bullets reaches the output:
> - **Temporal stability: accurate.** `estimate_optimal_survivor_count` (`:375-413`) sets the
>   pattern optimum, which enters the weighted result (`:649`, `:691-694`).
> - **Entropy: BROKEN (B2).** `estimate_min_survivor_count` (`:345-373`) computes it. The result
>   is logged (`:648`, `:651`) and enters nothing in the returned result (`:713-728`).
> - **Regime-change frequency: BROKEN (B3).** `get_regime_change_frequency` (`:415-454`) is
>   called nowhere. `handle_regime_change` (`:946`) cannot be reached from the CLI (`:1156-1165`).
>
> Entropy also assumes mod 1000 (`:363`).

### 4.4 Reinforcement Feedback (5-25%)

If previous runs exist:
- **Best performing configuration** — what worked before
- **Confidence score** — how much to trust historical data
- Weight grows from 5% to 25% based on confidence

> **Corrected 2026-11-11.**
> - **The logic is accurate.** The best past configuration comes from `:491-510`, confidence from
>   `:512-552`, and the weight `initial + (max − initial) · confidence` from `:554-563`.
> - **The source is BROKEN (B4): "if previous runs exist" never comes true.** Nothing writes
>   `optimization_results/reinforcement_feedback.json`, the file read at `:471`, which expects a
>   list of records with `performance`, `survivor_count`, `network_architecture` and
>   `training_epochs`.
> - **Consequence: the derived optimum sits 5 % low.** The unfilled 0.05 weight stays in the
>   denominator (`:686-699`), so the optimum is 5 % below the weighted mean of the other two sources.

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

> **Corrected 2026-11-11 (DESCRIPTION-ONLY, D3).** The code differs from the sketch above in three
> places:
> - **It normalises.** `weighted_optimal = int(Σ value·w / Σ w)` (`adaptive_meta_optimizer.py:699`).
> - **The minimum has a floor.** `min = max(window_min, int(optimal · 0.5))` (`:705`), and
>   `max = int(optimal · 2.0)` (`:706`).
> - **Feedback takes weight from the other two sources.** When feedback is present, the window and
>   pattern weights each drop by `fw/2` (`:680-684`).
>
> In practice `window_optimal` is the constant 500 (S1), and the feedback term never appears (B4).

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

> **Corrected 2026-11-11 (BROKEN, B1).** Previously: "Heuristic based on problem complexity",
> `[128, 64, 32]` the default. In fact the choice depends on no input.
>
> `optimize_network_architecture` (`adaptive_meta_optimizer.py:740-792`) scores each option by its
> own size:
>
> ```
> score = 0.7 · min(1, sum(arch)/256) + 0.3 · (1 − len(arch)/4)        (:765-771)
> ```
>
> Its `survivor_count` argument is only logged (`:752`), and the code calls the loop a simulation
> (`:757`, `:764`). Over the configured option list (`:120-125`; the same list is in
> `adaptive_meta_optimizer_config.json`) the scores are:
>
> | option | score |
> |---|---|
> | `[32]` | 0.3125 |
> | `[64,32]` | 0.4125 |
> | `[128,64,32]` | 0.6875 |
> | `[256,128,64]` | **0.775** |
>
> So **`[256, 128, 64]` is always selected.** On the host, 27 of 27 archived results show it. No
> governing document records this. A repair would have to make the choice depend on an input the
> chapter names, or say that the option list fixes the architecture.

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

> **Verified 2026-11-11 (accurate):** the formula is `estimate_training_epochs`
> (`adaptive_meta_optimizer.py:814-824`), with the clamp bounds from config (`:128-133`). Its
> inputs are the constant speed 0.5 and stability 0.5 (S1), so it returns
> `100 · 1.0 · 1.5 = 150` on every run.

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

> **Corrected 2026-11-11 (DESCRIPTION-ONLY, D4).** The JSON above mixes two files.
>
> - **The top-level fields** (`survivor_count` … `calibration_type`) are what `full_calibration()`
>   returns (`adaptive_meta_optimizer.py:861-877`). They are saved to
>   `optimization_results/meta_optimization_results.json` and to a timestamped copy (`:1057-1081`).
> - **`reinforcement_engine_config.json` receives a merge instead** (`:1031-1049`):
>   `model.hidden_layers` ← architecture, `training.epochs` ← epochs,
>   `survivor_pool.max_pool_size` ← survivor count, and `_meta_optimizer` =
>   `{last_calibration, calibration_type, confidence, weights_used}`. The example omits
>   `weights_used` from that block.
>
> The example values (500, `[128,64,32]`, 100, 0.75) are illustrative. The host's actual
> `meta_optimization_results.json` (2026-03-07) reads: survivor count 476, range 250–952,
> `[256, 128, 64]`, 150 epochs, confidence 0.245 (B1, S1).

### 6.2 Key Fields

| Field | Description | Used By Step 5 |
|-------|-------------|----------------|
| `survivor_count` | Recommended pool size | Sample size selection |
| `network_architecture` | Layer sizes | Neural net config |
| `training_epochs` | Duration | Early stopping baseline |
| `confidence` | Trust in recommendations | Adjustment decisions |

> **Corrected 2026-11-11 (BROKEN, B5).** The "Used By Step 5" column does not hold at `a15b524`.
>
> **No Step 5 code reads these fields.**
> - `meta_prediction_optimizer_anti_overfit.py` has no reference to `reinforcement_engine_config`,
>   `ReinforcementConfig` or `reinforcement_engine`.
> - Its CLI takes none of these values (`:3040-3105`).
> - Its NN hidden layers come from Optuna, or from the default `[256, 128, 64]` (`:908`, `:2710`).
> - `subprocess_trial_coordinator.py` and `train_single_trial.py` do not reference the file either.
>
> **The only coupling is WATCHER's existence-and-freshness gate**
> (`agent_manifests/reinforcement.json:6-10`, `agents/watcher_agent.py:803-834`).
> `reinforcement_engine.py:1045` reads the file, but that script is not in `STEP_SCRIPTS`.
>
> No governance document says this was intended. A repair would have to make Step 5 read the named
> fields, or state that Step 4's output is an ordering precondition only.

### 6.3 What's NOT in Output

| Field | Why Not |
|-------|---------|
| `model_type` | Decided by Step 5 |
| `hyperparameters` | Optimized by Step 5 |
| `holdout_metrics` | Step 5 computes these |

> **Verified 2026-11-11 (accurate):** neither output carries these fields
> (`adaptive_meta_optimizer.py:861-877`, `:1031-1049`).

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

> **Verified 2026-11-11 (accurate):** all four flags exist (`adaptive_meta_optimizer.py:1104-1113`).
> This is what WATCHER sends from `default_params` (`agent_manifests/ml_meta.json:104-109`). A
> `True` value becomes the bare `--apply` flag (`agents/watcher_agent.py:2143-2148`). Step 4 runs
> locally, with `"distributed": false` and a 10-minute timeout (`agent_manifests/ml_meta.json:30-31`).

### 7.2 CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--mode` | No | full | Optimization mode |
| `--window-results` | No | optimal_window_config.json | Step 1 output |
| `--lottery-data` | Yes | - | Training lottery history |
| `--apply` | No | False | Write to reinforcement_engine_config.json |
| `--config` | No | - | Custom config file |

> **Corrected 2026-11-11.**
>
> **Accurate:** `--mode` (choices `full`/`survivor`/`network`/`epochs`, default `full`,
> `adaptive_meta_optimizer.py:1104-1107`); `--window-results` (default comes from the config path
> `optimal_window_config.json`, `:85`, `:1140-1141`); `--apply` (`:1112-1113`).
>
> **DESCRIPTION-ONLY:**
> - *D5.* `--lottery-data` is not `required=True` (`:1110-1111`). Without it the path comes from
>   the config: `lottery_data.json`, which is absent on the host. The run then fails with
>   `FileNotFoundError` and exits 1 (`:1146-1153`). It is required in practice.
> - *D6.* `--config` defaults to `adaptive_meta_optimizer_config.json` (`:1101-1103`). The table
>   omits `--test` (`:1114-1128`, a message-only self-test that returns 0). `--apply` takes effect
>   only with `--mode full` (`:1171`).

### 7.3 Arguments NOT Used

> **SUPERSEDED (2026-11-11), DESCRIPTION-ONLY (D7).** Replaced by the §7.2 correction above; the
> text below is kept as history.
>
> **Neither argument exists at `a15b524`.**
> - `--survivor-data` existed at `185e755` (2025-12-25) and was removed at `6e905e7` (2026-01-01,
>   "Step 4 contract fix").
> - `--holdout-history` exists in no committed version (`0101306`, `c9f7f07`, `185e755`,
>   `a15b524`).
> - `parse_args()` (`adaptive_meta_optimizer.py:1117`) rejects unknown flags, so passing either
>   one makes the run exit at argument parsing.
>
> **Two legacy callers still pass `--survivor-data`:** `complete_whitepaper_workflow.py:198-203`
> and `complete_whitepaper_workflow_with_meta_optimizer.py:309-314`. Neither is the WATCHER path.
> `complete_whitepaper_workflow_with_meta_optimizer_v3.py:350-357` does not pass it.

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

> **Corrected 2026-11-11.**
>
> **Accurate:** the order Step 1 → … → Step 4 → Step 5 (`agents/watcher_agent.py:389-393`).
> Step 5 is the first step to load `survivors_with_scores.json`: `--survivors` is required
> (`meta_prediction_optimizer_anti_overfit.py:3041`). It chooses among the four families
> (`:3057-3062`; compare list `:148`). It writes `best_model.*` and `best_model.meta.json`
> (`:1322`).
>
> **DESCRIPTION-ONLY (D8):** "Use holdout_hits for evaluation". The target is `holdout_quality`
> (`:612`, `:1481`).
>
> **BROKEN (B5):** the step between the two boxes. `reinforcement_engine_config.json` gates Step 5
> by existence and freshness only, and Step 5 does not read it (§6.2 note).
>
> **Chapter 14 cross-references (not re-reported here):**
> - `docs/CHAPTER_14_AUDIT.md` B3: final tree models get no diagnostics.
> - B6: the compare-models summary filename and key do not match between writer and reader.
> - The skip registry is read inside `_s88_run_compare_models`
>   (`meta_prediction_optimizer_anti_overfit.py:141`, path `:150`, threshold test `:173-174`). On the
>   host, `neural_net` has `consecutive_critical` 1 and `skip_active` false, so it is not skipped.
>
> None of these interacts with Step 4, because no Step 4 value reaches Step 5.

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

> **Corrected 2026-11-11.**
>
> **DESCRIPTION-ONLY (D9): the snippet is not the manifest's shape.** The real
> `agent_manifests/ml_meta.json` (v2.0.0) has:
> - `required_inputs` and `inputs` lists (`:6-15`);
> - an `outputs` list naming `reinforcement_engine_config.json` and
>   `optimization_results/meta_optimization_results.json` (`:16-19`);
> - `actions[0].args_map` (`:20-33`);
> - `follow_up_agents: ["reinforcement_agent"]` (`:34-36`);
> - `success_condition` / `evaluation_type: "file_exists"` (`:37-39`, `:122`).
>
> `agent_name`, `pipeline_step` 4 and the script match.
>
> **The note above it is accurate:** there is no survivor or holdout input (`:6-15`).
>
> **Three other surfaces, none of which WATCHER uses:**
> - A root-level `ml_meta.json` exists on the host. It is **untracked**, still v1.0.0, and still
>   describes Optuna over `survivors_with_scores` (`ml_meta.json:84`). WATCHER loads
>   `agent_manifests/` only (`agents/watcher_agent.py:776`).
> - The gitignored `agent_manifests/definitions.json` has **no Step 4 entry**.
> - `preflight_check.py:286` lists `survivors_with_scores.json` as a Step 4 input. That is an
>   existence check only; Step 4 does not open the file.

---

## 9. Common Misconceptions

### 9.1 "Step 4 should analyze survivor distributions"

**Wrong.** That would cause validation leakage. Step 4 derives capacity from sieve behavior, not survivor data.

### 9.2 "Step 4 should recommend model type"

**Wrong.** Model selection requires evaluating models on data, which happens in Step 5 with proper holdout separation.

### 9.3 "Step 4 should use Optuna for hyperparameters"

**Wrong.** Step 4 derives capacity heuristically. Step 5 uses Optuna to optimize actual model hyperparameters with proper cross-validation.

> **Verified 2026-11-11 (accurate):**
> - **§9.1 and §9.2 hold.** Step 4 reads no survivor data and chooses no model type (§1.2 and
>   §3.2 notes).
> - **§9.3 holds.** Step 4 imports no Optuna (`adaptive_meta_optimizer.py:48-66`). Step 5 does,
>   with K-fold cross-validation (`meta_prediction_optimizer_anti_overfit.py:42-45`, `--k-folds`
>   `:3047`).
> - **One caveat.** `agents/registry/parameter_registry.py:437-478` still describes Step 4 as
>   "Optimize ML model architecture using Optuna", with `--trials`, `--min-layers` and
>   `--max-layers`. None of those flags exists. This is recorded for Chapter 10's pass.

### 9.4 "The --survivor-data argument means Step 4 uses survivors"

> **SUPERSEDED (2026-11-11), DESCRIPTION-ONLY (absorbed into D7).** The argument no longer exists.
> It was removed at `6e905e7`, and `parse_args()` rejects it (`adaptive_meta_optimizer.py:1117`).
> The conclusion still holds: Step 4 does not use survivors. See the §7.3 marker. The text below is
> kept as history.

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

> **Corrected 2026-11-11.**
>
> **Accurate:** the "Does NOT Do" list holds (§1.2, §3.2, §9 notes).
>
> **"What Step 4 Does" needs qualifying:**
> - Item 1 analyses constants, not sieve behaviour (S1).
> - Item 2 reaches the output through stability only (B2, B3).
> - Item 3 yields a fixed architecture (B1).
> - Item 4's configuration is not read by Step 5 (B5).
>
> **DESCRIPTION-ONLY (D10):** "the last step before survivor data is touched". Steps 1–3 produce and
> score survivors. The accurate form is that Step 4 is the last step before survivor-level data is
> used to *train a model*.

---

## 10. What Step 4 actually produces at a15b524, and who reads it

*(Added 2026-11-11.)*

### 10.1 The path, as built

1. **Dispatch.** WATCHER dispatches Step 4 locally (`agents/watcher_agent.py:392`;
   `agent_manifests/ml_meta.json:30`) with
   `--mode full --window-results optimal_window_config.json --lottery-data train_history.json --apply`
   (`:104-109`).
2. **Calculation.** `full_calibration()` (`adaptive_meta_optimizer.py:838-891`) runs three
   derivations:
   - the survivor count (`:628-738`);
   - the architecture (`:740-792`);
   - the epochs (`:794-836`).
3. **Outputs.**
   - `save_results()` writes `optimization_results/meta_optimization_<ts>.json` and
     `meta_optimization_results.json` (`:1057-1081`).
   - `update_target_config()` merges into `reinforcement_engine_config.json` (`:994-1053`).
4. **Success test.** WATCHER checks that `reinforcement_engine_config.json` exists
   (`agent_manifests/ml_meta.json:37-39`, `:122`).
5. **Step 5 gate.** Step 5 lists that file in `required_inputs` (`agent_manifests/reinforcement.json:6-10`).
   `check_output_freshness` (`agents/watcher_agent.py:803-834`) hard-fails if the file is missing,
   and calls Step 5's output stale if the file is newer.
   **That is the whole downstream effect.** No Step 5 code reads the values (B5).

### 10.2 What the host shows

**27 of 27** archived Step 4 results in `optimization_results/` (2025-11-30 → 2026-03-07) have:

| field | value |
|---|---|
| architecture | `[256, 128, 64]` |
| epochs | 150 |
| confidence | 0.245 |
| survivor count | 476 (23 runs) or 475 (4 runs) |
| range | 250–952 or 250–950 |

**The code produces this from three causes**, traced line by line, not inferred from the data:

1. **The window analyzer returns constants (S1, governed).** Optimal 500, min 250, confidence 0,
   speed and stability 0.5. Hence epochs `100 · 1/(0.5+0.5) · (1+0.5) = 150`, and confidence
   `0·0.60 + 0.7·0.35 + 0·0.05 = 0.245` (`:709-711`).
2. **The architecture score ignores its input (B1).**
3. **The feedback file has no producer (B4).** Its 0.05 weight dilutes the result:
   `int(500·0.60 + p·0.35)` (denominator 1.0, `:699`) gives 476 for a pattern optimum `p` = 503
   and 475 for `p` ≤ 502. `p` is 500 for a history shorter than 500 draws (`:382-383`), and
   otherwise `int(500·(1 + 2·(1 − stability)))` (`:408-411`). Which history file each archived run
   used is not recorded in the artifact.

**The pattern optimum is the only quantity that moves**, and it moves by one survivor.

Host state on 2026-11-11 (point-in-time):
- `reinforcement_engine_config.json` and `optimal_window_config.json` are **absent**
  (`find / -xdev`).
- `train_history.json` is present: 14,454 draws, 2026-08-21.
- The last Step 4 run was 2026-03-07.

---

## 11. RANGE-MINER, the 22-array NPZ and the window anchor

*(Added 2026-11-11.)*

- **RANGE-MINER** changed nothing in Step 4, and is required to change nothing.
  - Proposal §11.F (`docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:561-567`) makes Step 4 output
    identity an acceptance item.
  - TB Q4 keeps the missing `best_result`/`all_results` keys missing (`:304`, `:723`). Both engines
    write the same flat `optimal_window_config.json`, so Step 4's constants are the same under
    either (S1).
- **The 22-array NPZ contract** does not reach Step 4. Step 4 reads no NPZ and no per-survivor
  record (`/bin/grep -n "npz\|range_miner"` on `adaptive_meta_optimizer.py` → 0 hits). It is
  Step 3's and Step 5's concern.
- **The window-anchor change** (Brief I, `69ca910`: `window_anchor` + `generator_phase` split)
  does not touch Step 4.
  - `adaptive_meta_optimizer.py` reads no `offset`, `window_anchor` or `generator_phase`
    (`/bin/grep -n` → 0 hits).
  - The change edited `utils/canonical_records.py`, the miner and
    `window_optimizer_integration_final.py`.
  - Final acceptance of Brief I is pending a completed production-shape run. That does not affect
    this chapter.

---

## 12. Known gaps at a15b524

*(Added 2026-11-11.)* Each item carries exactly one label. Full evidence: `docs/CHAPTER_5_AUDIT.md`
§4–§6.

### 12.1 BROKEN (5)

None of these was found in the governance trail.

| # | chapter says | code at `a15b524` | a repair would have to |
|---|---|---|---|
| B1 | architecture by problem complexity; `[128,64,32]` default (§5.2, §6.1) | score depends on option size only (`adaptive_meta_optimizer.py:765-771`); input only logged (`:752`) → always `[256,128,64]`; host 27/27 | make the choice depend on a named input, or state the option list fixes it |
| B2 | entropy drives capacity (§4.3) | computed (`:345-373`), logged (`:648`, `:651`), absent from result (`:713-728`) | feed it into the derivation, or state it is a log line only |
| B3 | regime-change frequency (§4.3) | `get_regime_change_frequency` (`:415-454`) called nowhere | call it from the derivation, or state it is unused |
| B4 | reinforcement feedback from Step 6, weight 5 %→25 % (§2.2, §3.1, §4.4) | reader at `:471`; no writer in tracked tree, git history or host | name one producer writing the schema at `:502-510`, or state the source is unconnected |
| B5 | Step 5 uses survivor count, architecture, epochs, confidence (§2.4, §6.2) | Step 5 never opens the file; coupling is WATCHER existence/freshness only (`agent_manifests/reinforcement.json:6-10`, `agents/watcher_agent.py:803-834`) | make Step 5 read the fields, or state Step 4's output is an ordering precondition only |

### 12.2 SWITCHED-OFF-BY-RULING (1)

| # | chapter says | held back by | lifting it would enable |
|---|---|---|---|
| S1 | window-optimizer analysis is the 60 % primary source: survivor range, convergence, stability from Step 1 (§2.1, §4.2) | TB Q4, "preserve the drift": `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:304`, §13 Q4 `:723`, §11.F `:561-567`; also `docs/PIPELINE_BEHAVIOUR_MODEL.md:519-524`. Under owner ruling §0.1 this is a recorded review position the owner may keep or lift | Step 4's pool size, epochs and confidence would follow Step 1's actual survivor counts, convergence and stability instead of the constants 500 / 150 / 0.245 |

### 12.3 DESCRIPTION-ONLY (10)

| # | chapter | code at `a15b524` |
|---|---|---|
| D1 | "6-step pipeline" (§1.1) | seven steps, 0–6 (`agents/watcher_agent.py:387-395`) |
| D2 | 62 features per survivor (§1.3) | live sidecar: 89 |
| D3 | survivor formula (§5.1) | normalised; floored minimum (`:699`, `:705`) |
| D4 | §6.1 format is `reinforcement_engine_config.json`; one output (§2.4) | two files; config gets `model.hidden_layers`, `training.epochs`, `survivor_pool.max_pool_size`, `_meta_optimizer` (`:1031-1049`) |
| D5 | `--lottery-data` required (§7.2) | optional flag; falls back to `lottery_data.json` (`:1110-1111`, `:1146-1153`) |
| D6 | `--config` default "-"; five flags (§7.2) | default `adaptive_meta_optimizer_config.json`; `--test` exists (`:1101-1128`) |
| D7 | `--survivor-data`, `--holdout-history` exist and are ignored (§7.3, §9.4) | neither exists; `parse_args()` rejects them (`:1117`); removed at `6e905e7` / never added |
| D8 | Step 5 uses `holdout_hits` (§1.3, §8.1) | target `holdout_quality` (`meta_prediction_optimizer_anti_overfit.py:612`, `:1481`) |
| D9 | manifest snippet (§8.2) | real shape at `agent_manifests/ml_meta.json:6-39` |
| D10 | "last step before survivor data is touched" (Summary) | last step before survivor-level data trains a model |

---

## 13. Open items

1. **U1 — chapter v1.0.0 (2025-12-30).** Not in git history or on the host. Would verify: the ser8
   pre-repository archive.
2. **What removed `reinforcement_engine_config.json` from the host** after the 2026-03-07 run is
   not established. While it is absent, a Step-5-only WATCHER run would hard-fail its freshness
   check (`agents/watcher_agent.py:816-818`). Recorded, not investigated further. No run was made.
3. **Whether the 27 archived runs were WATCHER- or hand-launched** is not established. WATCHER run
   logs were not opened. It changes no label.
4. **B1–B5 have no governance record.** They go to the owner as new findings. S1 is governed and
   is not re-reported.

---

## 14. Verification declaration (VIR-1…6)

**execution proof (VIR-1).** Every `file:line` added in this pass was opened on VM 101 in this
session, with Read, `sed -n`, `awk 'NR==n'` or `/bin/grep -n`. The code base is `a15b524`; the
working tree is `741822b`, which is docs-only on top of it. **One exception:** another session
modified `agents/watcher_agent.py` in the working tree during this pass. Every
`agents/watcher_agent.py` anchor here was re-checked against
`git show a15b524:agents/watcher_agent.py` and holds at `a15b524`. Host artifacts were opened with
read-only `python3 -I` one-liners. The one computation is the B1 score arithmetic, a closed-form
formula evaluated over four lists; no project code was executed. Files opened are listed in
`docs/CHAPTER_5_AUDIT.md` §12.

**clean control (VIR-2).** `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` was read as the model and was not
edited. Verified **correct and unchanged** by this pass: §1.2 (not data-aware), §3.2, §4.1 weights,
§5.3 epoch formula, §6.3, §7.1, §9.1–§9.3 (19 accurate claims, audit §9).

**fault-injection control.** **Not applicable** to a documentation pass. No detector, gate or harness
was run that could pass vacuously, and none was invented.

**completion sentinel.** §15.5.

**unavailable-observer behaviour.** Surfaces that could not be read are recorded as UNAVAILABLE,
never as clean:
- WATCHER run logs for the 27 Step 4 runs;
- the deleted `reinforcement_engine_config.json`;
- the ser8 archives, the private remote and the rigs (no SSH by brief).

Each absence claim lists its surfaces:
- tracked tree (`/bin/grep -rn` over `*.py`/`*.sh`/`*.json`);
- gitignored files (`agent_manifests/definitions.json`, root `ml_meta.json`, `optimization_results/`);
- git history (`git log --all -S`, `git log --all -- path`, `git show`);
- host (`find / -xdev`, `crontab -l`, `systemctl list-units --all`).

**audit claim scope.** What this chapter says against what the code at `a15b524` does. This pass
does not judge whether the design is right, and it proposes no component for removal or
simplification.

**governance trail searched.**
- By `/bin/grep -l -i` for Step 4 / `adaptive_meta` / `ml_meta`: `docs/TB_RULING_*`,
  `docs/TEAM_ALPHA_*`, `docs/PROPOSAL_*`, `docs/proposals/*`.
- Read in the relevant sections: `PROPOSAL_S172_RANGE_MINER_v1_4_4.md` (§4.4.3, §11.F, §13 Q4);
  `TB_RULING_S176_WATCHER_KPI.md` (Q4 D1); `WATCHER_KPI_CALIBRATION_FINDINGS_S176.md`;
  `SESSION_CHANGELOG_20260720_S176.md`; `PIPELINE_BEHAVIOUR_MODEL.md` §1.1 and §6;
  `TFM_PROJECT_FACTS_SKILL.md`; `PROJECT_FILE_CATALOG.md`; `BACKLOG.md` §1;
  `TODO_CONSOLIDATED.md`.

**chapters searched.**
- This chapter, every line.
- `CHAPTER_2_BIDIRECTIONAL_SIEVE.md` (the model).
- `CHAPTER_6_ANTI_OVERFIT_TRAINING.md`, `CHAPTER_12_WATCHER_AGENT.md`: Step 4 lines.
- `CHAPTER_13_AUDIT.md` (layout).
- `CHAPTER_14_AUDIT.md` (B3, B6) and `CHAPTER_14_TRAINING_DIAGNOSTICS.md` §16–§19 (form).

---

## 15. Closure statement

### 15.1 Verified against

VM 101, `/home/michael/distributed_prng_analysis`, code at
`a15b5247336255801de8cf67d52653701b1e8c80`, 2026-11-11, read-only.

### 15.2 What this pass did

- Added the TFM framing block, the status header, the numbering statement and "How this chapter
  was produced".
- Annotated §1–§9 and the Summary with dated corrections.
- Marked §7.3 and §9.4 **SUPERSEDED (2026-11-11)**, with pointers; their text is kept.
- Added §10 (actual output and readers), §11 (RANGE-MINER / 22-array NPZ / window anchor), §12
  Known gaps (5 BROKEN, 1 SWITCHED-OFF-BY-RULING, 10 DESCRIPTION-ONLY), §13 Open items and §14
  VIR-1…6.
- Answered the §5 host question: `agent_manifests/definitions.json` has no Step 4 entry.

### 15.3 What this chapter is NOT

- **Not a repair.** No code, test, config or manifest was changed.
- **Not a design review.**
- **Not a claim that Step 4 is useless.** It is a record that, at `a15b524`, its values are
  constant and unread.

### 15.4 Length

The chapter started at 423 lines and is longer after this pass. **No heading was removed and no
text was deleted.** One original line was extended in place, with its original text kept as the
prefix: the `**Version:** 2.0.0 (Corrected)` header line, which now also names the 2026-11-11
overlay.

### 15.5 Completion sentinel

```
CHAPTER 5 CORRECTION:  PASS
```

**`PASS` is claimed for the declared scope only:**
- every substantive claim in the chapter (41, audit §10) was checked against the code at `a15b524`;
- each was labelled once and reflected here with an anchor;
- the §5 host question is answered.

**One `[UNVERIFIED]` item remains (U1).**

**Files changed by this pass:** `docs/CHAPTER_5_ML_ARCHITECTURE_OPTIMIZER_v2.md` and the new
`docs/CHAPTER_5_AUDIT.md`.

---

## Version History

```
Version 2.1.0 (correction overlay) - November 11, 2026
- Correction pass against code at a15b524 (docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md REV1)
- Added TFM framing block, status header, numbering statement (executable Step 4)
- Inline corrections in §1-§9 and Summary; §7.3 and §9.4 marked SUPERSEDED (text kept)
- Added §10-§15: actual output and readers, RANGE-MINER/NPZ/window anchor,
  known gaps (5 BROKEN, 1 SWITCHED-OFF-BY-RULING, 10 DESCRIPTION-ONLY), open items, VIR, closure
- Audit: docs/CHAPTER_5_AUDIT.md

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
