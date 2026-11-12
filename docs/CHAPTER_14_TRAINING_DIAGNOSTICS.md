# Chapter 14: Training Diagnostics & Model Introspection

> **What TFM is.** TFM filters candidate generators against observed draws in two directions,
> keeps a population of survivors on purpose, ranks them, and lets the top candidates vote their
> next value into a prediction pool. A survivor is a scored candidate, not a verdict. The
> generator family is a substrate for mimicry, not a claim about the machine. The target is a
> pool hit rate. TFM does not recover a seed or reconstruct state, and an objection that assumes
> it does is out of scope.

**Correction-pass status (2026-11-11):** **CORRECTED AGAINST CODE at `a15b524`** — not closed.
The design text below (v1.2.0, February 2026) is retained in full; where it disagrees with the
code, an inline **"Corrected 2026-11-11"** note or a **SUPERSEDED (2026-11-11)** marker says what
the code at `a15b524` does, with an anchor. **Read §15–§19 (end of file) first:** §15 is what the
diagnostics actually see, §16 lists the known gaps, §17 open items, §18 the VIR-1…6 declaration,
§19 the closure statement and completion sentinel.
**Authority for this pass:** `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md` (REV1).
**Base:** `a15b5247336255801de8cf67d52653701b1e8c80` (`a15b524`) on VM 101.
**Audit:** `docs/CHAPTER_14_AUDIT.md` (§1–§8 from the public mirror; "Addendum — live-host
re-verification, 2026-11-11" from VM 101).
**Numbering:** bare "Step N" in this chapter means the **executable** scheme
(`docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1): Step 5 = `meta_prediction_optimizer_anti_overfit.py`,
Step 6 = `prediction_generator.py` (`agents/watcher_agent.py` `STEP_SCRIPTS`). "Phase N" means
this chapter's own implementation phases (§13), not S172 phases.

### How this chapter was produced (2026-11-11 pass)

1. The existing audit `docs/CHAPTER_14_AUDIT.md` was re-verified anchor by anchor against the live
   tree at `a15b524`, and its `[UNVERIFIED]` items were settled from host artifacts where the host
   allowed (audit Addendum A.1–A.4).
2. Host state was read, never changed: `diagnostics_outputs/` (111 archived health checks, the
   live skip registry, the NN report, 60 compare-models summaries), `agent_manifests/definitions.json`,
   `configs/selfplay_config.json`, cron and systemd listings. Nothing was run.
3. The chapter was then corrected **in place, section by section**. No section was deleted; stale
   sections carry SUPERSEDED markers and keep their text. Every disagreement carries exactly one
   label — BROKEN, SWITCHED-OFF-BY-RULING or DESCRIPTION-ONLY — listed together in §16.
4. Claim scope: what the chapter says against what the code at `a15b524` does. Whether the design
   is right is not judged here.

## PRNG Analysis Pipeline — Complete Operating Guide

**Version:** 1.2.0 (design text) · 1.3.0 correction overlay 2026-11-11  
**Status:** ACTIVE — Phases 1, 3, 5, 6 Complete (S69-S73)  
> **Corrected 2026-11-11 (DESCRIPTION-ONLY, D8):** the status line above disagrees with this
> chapter's own checklist (§14 marks 1, 2, 3, 6, 7 ✅ and 8 partial; 4, 5, 9 not started), with
> `CHAPTER_14_HEADER_PATCH.md:24`, `:49` ("Phases 1-3 Complete", "Phase 2 DEFERRED") and with
> `CHAPTER_13_IMPLEMENTATION_PROGRESS_v3_9.md:72` (Phase 8 "COMPLETE"). Code at `a15b524`:
> Phases 1, 2, 6, 7 exist as modules; Phase 3 is wired **for the neural net only** (§16 B3);
> Phase 5 (TensorBoard) does not exist in any tracked `*.py`; Phase 8 code exists but is off or
> unreachable (§9 notes). "Phase 5" means TensorBoard here and FIFO pruning in the header patch.  
**Date:** February 8, 2026 (v1.2.0 update: Session 71)  
**Author:** Team Beta  
**Depends On:** Chapter 6 (Anti-Overfit Training), Chapter 11 (Feature Importance), Contract: Strategy Advisor v1.0  
**Extends:** Chapter 6 Sections 5-8, Chapter 11 Sections 4-7

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Capability Overview](#2-capability-overview)
3. [Live Training Introspection](#3-live-training-introspection)
4. [Per-Survivor Attribution](#4-per-survivor-attribution)
5. [Training Dynamics Web Dashboard](#5-training-dynamics-web-dashboard)
6. [TensorBoard Integration](#6-tensorboard-integration)
7. [WATCHER Integration](#7-watcher-integration)
8. [LLM Integration — DiagnosticsBundle & Grammar](#8-llm-integration--diagnosticsbundle--grammar)
9. [Selfplay & Live Feedback Wiring](#9-selfplay--live-feedback-wiring)
10. [TensorBoard Automation Boundary](#10-tensorboard-automation-boundary)
11. [Neural Net Evaluation & Repurposing Path](#11-neural-net-evaluation--repurposing-path)
12. [File Inventory](#12-file-inventory)
13. [Implementation Plan](#13-implementation-plan)
14. [Implementation Checklist](#14-implementation-checklist)
15. [What the diagnostics actually see at a15b524 — RANGE-MINER, the 22-array NPZ, the window anchor](#15-what-the-diagnostics-actually-see-at-a15b524--range-miner-the-22-array-npz-the-window-anchor) *(added 2026-11-11)*
16. [Known gaps at a15b524](#16-known-gaps-at-a15b524) *(added 2026-11-11)*
17. [Open items](#17-open-items) *(added 2026-11-11)*
18. [Verification declaration (VIR-1…6)](#18-verification-declaration-vir-16) *(added 2026-11-11)*
19. [Closure statement](#19-closure-statement) *(added 2026-11-11)*

---

## 1. Motivation

### 1.1 The Problem

December 21, 2025 test results (95K survivors, `--compare-models`):

| Model | MSE | Rank |
|-------|-----|------|
| CatBoost | 1.77e-9 | #1 |
| XGBoost | 9.32e-9 | #2 |
| LightGBM | 1.06e-8 | #3 |
| Neural Net | 9.32e-4 | #4 |

Neural net MSE is **100,000x worse** than CatBoost. Current tooling tells us the result
but not **why**. We know neural_net lost. We don't know if it's:

- Feature scale imbalance (likely — forward_count ranges 0-50K, intersection_weight 0-1)
- Dead ReLU neurons (likely — 128→64→32 with ReLU is classic dying neuron architecture)
- Vanishing gradients (possible — 3 linear layers with no skip connections)
- Wrong learning rate (possible — stuck in local minimum)
- Fundamentally wrong architecture for tabular data (possible — trees are often better)

Each diagnosis leads to a different fix. Without diagnostics, we're guessing.

> **Corrected 2026-11-11 (DESCRIPTION-ONLY, D1/D11; audit §7 items 1–2, 5):** the table above is
> a December 2025 measurement on PWC-produced survivors, before the `holdout_quality` target
> (`meta_prediction_optimizer_anti_overfit.py:1479-1482`, default `:612`) and before RANGE-MINER
> and the window-anchor split. It describes a population a current run does not produce. Two of
> the hypotheses name features that are **not per-seed**: `forward_count` and
> `intersection_weight` are trial-level counts stamped identically on every seed of a trial
> (`utils/canonical_records.py:216-244`, copied per seed `:246-257`). The scaling concern itself
> was addressed later: the NN input is standardised (`train_single_trial.py:484-492`) and `y` is
> z-scored (`:502-510`). See §15.

### 1.2 Why This Matters for Autonomy

The pipeline currently selects the best model via `--compare-models` and moves on.
But with the Strategy Advisor (Contract v1.0) and selfplay running autonomously:

- **Strategy Advisor needs training telemetry** to classify MODEL_DIVERSITY focus areas
- **Selfplay episodes train models** — diagnostics tell us if training is healthy mid-episode
- **Chapter 13 evaluates predictions** — if confidence calibration is poor, training diagnostics
  explain whether it's a model problem or a data problem
- **LLM analysis** can consume structured diagnostics JSON for deeper interpretation

A well-diagnosed neural_net, if fixable, gains a unique capability no tree model has:
**per-survivor gradient attribution** — explaining why THIS specific seed is ranked high,
not just global feature importance. That feeds directly into pool precision strategy.

> **Corrected 2026-11-11 (DESCRIPTION-ONLY, D17):** per-sample attribution is not unique to the
> NN. `per_survivor_attribution.py` gives all four families a per-sample backend
> (`per_survivor_attribution_xgb` `:134`, `_lgb` `:184`, `_catboost` `:231`, dispatcher `:289`);
> this chapter's own §4.3–4.5 say the same; the system map corrected it
> (`TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:237`). "Why THIS specific seed is ranked
> high" is also limited by what the model is fed: no per-seed sieve evidence reaches it (§15).

### 1.3 Scope

Four capabilities, each applicable across all four model types:

| # | Capability | NN | XGBoost | LightGBM | CatBoost |
|---|-----------|-----|---------|----------|----------|
| 1 | Live Training Introspection | PyTorch hooks | eval_set callbacks | record_evaluation | get_evals_result |
| 2 | Per-Survivor Attribution | Input gradients | pred_contribs | pred_contrib | ShapValues |
| 3 | Web Dashboard Charts | Loss + dead neurons + gradients | Loss + gain importance | Loss + split/gain | Loss + PredictionValuesChange |
| 4 | TensorBoard | add_graph + add_histogram | add_scalars from evals | add_scalars from evals | add_scalars from evals |

> **Corrected 2026-11-11:** at `a15b524` row 1 runs on the production path **for the NN only**
> (`train_single_trial.py:607-660`, reached via `meta_prediction_optimizer_anti_overfit.py:2587`);
> final tree models train inline with no diagnostics hook (`mpo:1832`, `:2761`) — **BROKEN, B3**.
> Row 2 exists as a module with no production caller (Chapter 13 is its only caller, and that path
> is unreachable — §9.3). Rows 3 and 4 were deferred and do not exist — **SWITCHED-OFF, S9**.

---

## 2. Capability Overview

### 2.1 Current State (What Chapter 6 + 11 Already Do)

```
Chapter 6: Trains models, reports final metrics (MAE, overfit ratio, R²)
           → Knows the SCORE but not the WHY

Chapter 11: Post-training feature importance (permutation, gradient, SHAP)
            → Knows WHICH features matter globally, but not:
              - When they started mattering during training
              - Why they matter for specific survivors
              - What's going wrong during training
```

### 2.2 What Chapter 14 Adds

```
Capability 1 (Live Introspection):
    DURING training → epoch-by-epoch / round-by-round diagnostics
    See the model learning in real-time
    Detect problems as they develop, not after training ends

Capability 2 (Per-Survivor Attribution):
    AFTER training → per-seed feature explanations
    "This survivor is ranked #3 because intersection_weight=0.89
     and skip_entropy=0.72 drove 68% of its prediction"
    Not available from global importance methods
    [Corrected 2026-11-11, D11: intersection_weight is trial-level and
     skip_entropy has no producer (constant 0.0) — neither can explain
     one seed against another. See §15.]

Capability 3 (Dashboard):
    VISUAL → Plotly charts on existing web_dashboard.py
    Diagnosis at a glance for all 4 model types
    Side-by-side comparison when using --compare-models

Capability 4 (TensorBoard):
    DEEP INVESTIGATION → full interactive exploration
    Weight histograms, gradient flow visualization, model graph
    Separate UI (port 6006) for research sessions
```

### 2.3 Design Invariant: Diagnostics Are Best-Effort and Non-Fatal

**Diagnostics generation is best-effort and non-fatal. Failure to produce
diagnostics must never fail Step 5, block pipeline progression, or alter
training outcomes.**

All diagnostics code paths are wrapped in try/except. If hooks fail to attach,
JSON fails to write, or the diagnostics module itself raises an exception, the
training run continues normally and WATCHER receives an "absent diagnostics"
signal (which maps to PROCEED). This invariant prevents diagnostics from ever
becoming a hard dependency in the pipeline.

> **Verified 2026-11-11 (accurate):** the invariant holds in code — diagnostics are wrapped in
> try/except in the trial worker (`train_single_trial.py:119-158`, `:610-618`, `:738-755`) and the
> health check (`training_health_check.py:80-120`), and an absent file maps to PROCEED
> (`_check_training_health_impl`, `:126-135`). **What the invariant does not cover:** when a
> diagnostics file *is* present, the health check's verdict does alter the run — RETRY re-runs
> Step 5 and the skip registry can drop `neural_net` from `--compare-models`
> (`meta_prediction_optimizer_anti_overfit.py:151-177`). Both are driven by a grading defect
> (§16 B1).

---

## 3. Live Training Introspection

### 3.1 Neural Net — PyTorch Dynamic Graph Hooks

**Mechanism:** `register_forward_hook()` and `register_full_backward_hook()` on
each `nn.Linear` layer. These fire automatically on every forward/backward pass because
PyTorch rebuilds the computational graph dynamically each pass (eager mode via `torch.autograd`).

**What hooks capture:**

| Metric | Source | What It Tells You |
|--------|--------|-------------------|
| Activation mean/std | Forward hook output | Are neurons producing diverse outputs? |
| Dead neuron % | Forward hook (output == 0) | Is ReLU killing neurons? |
| Gradient norm per layer | Backward hook grad_output | Are gradients reaching early layers? |
| Gradient per feature | Backward hook grad_input (layer 0) | Which features drive learning? |
| Weight norm | Direct param access | Are weights growing or collapsing? |

**Insertion point:** `reinforcement_engine.py` — inside the training epoch loop.

> **Corrected 2026-11-11.**
> - **Insertion point (DESCRIPTION-ONLY, D2):** production wiring is in the Step-5 trial worker,
>   `train_single_trial.py:607-660` (`NNDiagnostics` built at `:614`, saved `:738-746`).
>   `reinforcement_engine.py:697-712` also has hooks, on a path Step 5 does not use.
> - **Dead neuron % (DESCRIPTION-ONLY, D12):** the code measures exact zeros on the `nn.Linear`
>   output (`training_diagnostics.py:404-412`), which sits before BatchNorm and the activation
>   (`models/wrappers/neural_net_wrapper.py:66-69`), and production forces LeakyReLU
>   (`meta_prediction_optimizer_anti_overfit.py:245-246`). It cannot see ReLU deaths. **Host
>   observation:** 242 of 242 `layer_health` entries in the real NN reports on VM 101 read
>   `dead_pct` 0.0 (audit Addendum A.2).
> - **Gradient per feature — BROKEN (B18, audit UNVERIFIED #1, now resolved):** the code does
>   not do the extra `requires_grad` pass shown below; it reads `grad_input[0]` from the first
>   layer's backward hook (`training_diagnostics.py:429-431`) while the training input is a plain
>   tensor (`train_single_trial.py:515`). On VM 101, 0 of 75 real schema-1.1.0 NN reports contain
>   `feature_gradients`; every complete one has `gradient_spread_ratio` 1.0, the initial value at
>   `training_diagnostics.py:563`. Per-feature NN gradients are not captured in production. No
>   governance record of this was found; `S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md:426`
>   describes the artifact as carrying `feature_gradients.top_10`.
> - **Weight norm:** not captured as a separate metric; the `norm` field in a layer snapshot is the
>   gradient norm from the backward hook (`training_diagnostics.py:417-427`, `_make_backward_hook`).

```python
# ─── HOOK REGISTRATION (before epoch loop) ────────────────────────

# Hook factories — closures capture layer name
def make_forward_hook(name, storage):
    def hook(module, inp, out):
        storage[name] = {
            'activation_mean': out.detach().mean().item(),
            'activation_std': out.detach().std().item(),
            'dead_pct': (out.detach() == 0).float().mean().item() * 100,
            'neuron_count': out.shape[-1],
        }
    return hook

def make_backward_hook(name, storage):
    def hook(module, grad_input, grad_output):
        if grad_output[0] is not None:
            storage[name] = {
                'gradient_norm': grad_output[0].detach().norm().item(),
                'gradient_mean': grad_output[0].detach().abs().mean().item(),
                'gradient_max': grad_output[0].detach().abs().max().item(),
            }
    return hook

# Attach to SurvivorQualityNet layers
activations = {}
gradients = {}
handles = []

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        handles.append(module.register_forward_hook(
            make_forward_hook(name, activations)))
        handles.append(module.register_full_backward_hook(
            make_backward_hook(name, gradients)))

# ─── INSIDE EPOCH LOOP (after loss.backward()) ───────────────────

epoch_snapshots = []

for epoch in range(epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss = validate(model, val_loader, criterion)

    # Hooks already fired during train_one_epoch — data is in activations/gradients
    snapshot = {
        'epoch': epoch,
        'train_loss': float(train_loss),
        'val_loss': float(val_loss),
        'layers': {},
    }
    for name in activations:
        snapshot['layers'][name] = {**activations.get(name, {}), **gradients.get(name, {})}

    # Per-feature gradient attribution (input layer)
    # Requires one extra forward+backward pass with requires_grad on input
    model.eval()
    sample_batch = next(iter(val_loader))
    x_sample = sample_batch[0].to(device).requires_grad_(True)
    pred = model(x_sample)
    pred.sum().backward()
    feat_grads = x_sample.grad.abs().mean(dim=0)  # [62] — one value per feature
    snapshot['feature_gradients'] = {
        'top_10': sorted(
            [{'feature': feature_names[i], 'magnitude': feat_grads[i].item()}
             for i in range(len(feature_names))],
            key=lambda x: x['magnitude'], reverse=True
        )[:10],
        'spread_ratio': float(feat_grads.max() / (feat_grads.min() + 1e-10)),
    }
    model.train()

    epoch_snapshots.append(snapshot)

# ─── CLEANUP (after epoch loop) ──────────────────────────────────

for h in handles:
    h.remove()
```

**Key principle:** Hooks are passive observers. They detach tensors immediately
(`out.detach()`) so they never interfere with gradient computation or training behavior.

### 3.2 XGBoost — Native eval_set Callbacks

> **Corrected 2026-11-11 — applies to §3.2, §3.3 and §3.4 (BROKEN, B3 and B4).** None of the
> three wrapper insertion points below is the production path. Step 5's final tree models train
> inline with no diagnostics hook (`meta_prediction_optimizer_anti_overfit.py:1832` Optuna path,
> `:2761` single-shot); `_cli_enable_diagnostics` is consumed only for the NN (`:2587`). Where the
> trial worker's `_emit_tree_diagnostics` does run (`train_single_trial.py:112-158`), only
> CatBoost rounds are read (`:126-141`), importance needs `feature_importances_` (`:144`), which
> the `lgb.train` / `xgb.train` Booster objects (`:216`, `:309`, `:323`) are not given
> [UNVERIFIED — library behaviour not executed], and features are keyed `f{i}` (`:147`, B5).
> `XGBDiagnostics` / `LGBDiagnostics` are empty subclasses (`training_diagnostics.py:683-690`).
> **Host:** no xgboost or lightgbm diagnostics report exists on VM 101; the four CatBoost archives
> predate S88 (2026-02-09) and name features `f4`, `f6`, … (audit Addendum A.2). The library
> mechanisms described below are accurate as library facts; they are not what Step 5 does.

**Mechanism:** XGBoost natively logs per-boosting-round metrics when you pass `eval_set`.
No hooks, no wrappers. The library does it for you.

**Insertion point:** `models/wrappers/xgboost_wrapper.py` — inside `train()` method.

```python
# XGBoost records per-round metrics automatically via eval_set
model = xgb.XGBRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    tree_method='gpu_hist',         # GPU acceleration
    eval_metric=['mae', 'rmse'],    # Track both
    early_stopping_rounds=early_stopping_patience,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False,
)

# After training — extract full round-by-round history
evals_result = model.evals_result()
# evals_result = {
#     'validation_0': {'mae': [0.45, 0.43, 0.41, ...], 'rmse': [...]},  ← train
#     'validation_1': {'mae': [0.52, 0.49, 0.47, ...], 'rmse': [...]},  ← val
# }

# Best round (early stopping point) — tells you exactly when overfitting started
best_round = model.best_iteration  # e.g., 127 out of 500

# Three types of feature importance (unique to XGBoost)
importance_gain = model.get_booster().get_score(importance_type='gain')
importance_weight = model.get_booster().get_score(importance_type='weight')
importance_cover = model.get_booster().get_score(importance_type='cover')
# gain: how much a feature improves predictions when used in a split
# weight: how many times a feature appears in all trees
# cover: how many samples a feature affects
# KEY INSIGHT: If weight is high but gain is low → feature is used often but adds little value

round_snapshots = []
for i in range(len(evals_result['validation_0']['mae'])):
    round_snapshots.append({
        'round': i,
        'train_mae': evals_result['validation_0']['mae'][i],
        'val_mae': evals_result['validation_1']['mae'][i],
        'train_rmse': evals_result['validation_0']['rmse'][i],
        'val_rmse': evals_result['validation_1']['rmse'][i],
    })
```

### 3.3 LightGBM — record_evaluation Callback

**Mechanism:** LightGBM uses a callback pattern. `lgb.record_evaluation()` captures
per-round metrics into a dict you provide.

**Insertion point:** `models/wrappers/lightgbm_wrapper.py` — inside `train()` method.

```python
eval_results = {}

model = lgb.LGBMRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    device='gpu',
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_metric=['mae', 'rmse'],
    callbacks=[
        lgb.record_evaluation(eval_results),  # Captures per-round metrics
        lgb.log_evaluation(0),                 # Suppress per-round stdout
        lgb.early_stopping(early_stopping_patience),
    ],
)

# eval_results = {
#     'training': {'l1': [...], 'rmse': [...]},       ← train metrics per round
#     'valid_1':  {'l1': [...], 'rmse': [...]},        ← val metrics per round
# }

best_round = model.best_iteration_

# LightGBM has two importance types
importance_split = dict(zip(
    model.feature_name_, model.feature_importances_  # default: split frequency
))
# For gain importance:
importance_gain = dict(zip(
    model.feature_name_,
    model.booster_.feature_importance(importance_type='gain')
))
# KEY INSIGHT: If split importance >> gain importance for a feature,
# the model splits on it often but doesn't improve much — possible noise feature

round_snapshots = []
for i in range(len(eval_results['training']['l1'])):
    round_snapshots.append({
        'round': i,
        'train_mae': eval_results['training']['l1'][i],
        'val_mae': eval_results['valid_1']['l1'][i],
        'train_rmse': eval_results['training']['rmse'][i],
        'val_rmse': eval_results['valid_1']['rmse'][i],
    })
```

### 3.4 CatBoost — Built-in Metrics History

**Mechanism:** CatBoost stores the full training history internally. No callbacks needed,
just call `get_evals_result()` after training.

**Insertion point:** `models/wrappers/catboost_wrapper.py` — inside `train()` method.

```python
from catboost import CatBoostRegressor, Pool

model = CatBoostRegressor(
    iterations=n_estimators,
    depth=max_depth,
    learning_rate=learning_rate,
    eval_metric='MAE',
    task_type='GPU',
    devices='0:1',              # Both 3080 Ti's
    use_best_model=True,
    verbose=50,
)

train_pool = Pool(X_train, y_train, feature_names=feature_names)
val_pool = Pool(X_val, y_val, feature_names=feature_names)

model.fit(
    train_pool,
    eval_set=val_pool,
    early_stopping_rounds=early_stopping_patience,
)

# Full training history — no callbacks needed
evals_result = model.get_evals_result()
# evals_result = {
#     'learn':      {'MAE': [0.45, 0.43, ...]},   ← train
#     'validation': {'MAE': [0.52, 0.49, ...]},    ← val
# }

best_round = model.get_best_iteration()

# CatBoost has the MOST importance types of any tree library
importance_prediction = model.get_feature_importance(
    train_pool, type='PredictionValuesChange'
)
# PredictionValuesChange: Most accurate — measures actual prediction change
# when feature is removed. Unique to CatBoost.

importance_loss = model.get_feature_importance(
    train_pool, type='LossFunctionChange'
)
# LossFunctionChange: How much worse the loss gets without each feature

round_snapshots = []
for i in range(len(evals_result['learn']['MAE'])):
    round_snapshots.append({
        'round': i,
        'train_mae': evals_result['learn']['MAE'][i],
        'val_mae': evals_result['validation']['MAE'][i],
    })
```

### 3.5 Unified Diagnostics Output

> **SUPERSEDED (2026-11-11) — schema below replaced by schema 1.1.0** (`training_diagnostics.py:68`
> `SCHEMA_VERSION`; report built in `TrainingDiagnostics.get_report`, `:191-210`; summary
> `_compute_training_summary`, `:224-241`). DESCRIPTION-ONLY (D3), but it matters: two consumers
> were written to the old shape and read keys the producer never writes (§16 B2, B7). What a real
> report on VM 101 contains (`diagnostics_outputs/neural_net_diagnostics.json`, generated
> 2026-03-06, read this session):
>
> ```
> schema_version "1.1.0", model_type, generated_at, status
> training_summary: rounds_captured, final_train_loss, final_val_loss,
>                   best_val_loss, best_val_round, overfit_gap
> diagnosis:        severity, issues, suggested_fixes, confidence
> model_specific:   gradient_health {vanishing, exploding, dead_neuron_pct,
>                   gradient_spread_ratio}, layer_health {<layer>: dead_pct,
>                   dead_trend, gradient_norm}
> round_data_truncated, round_data_sample (last 10 rounds)
> ```
>
> No `training_rounds`, `feature_importance.values` or `nn_specific`. **File path** (below) is
> the canonical copy made by `train_single_trial.py:101-107` from
> `diagnostics_outputs/{model}_diagnostics.json`. **Consumed by** (below): at `a15b524` the
> consumers are the WATCHER health check (`training_health_check.py:80`) and the LLM prompt
> builder (`diagnostics_llm_analyzer.py:112`); there is no `/training` route (S9), and the
> Strategy Advisor reads `diagnostics_history/`, not this file (`parameter_advisor.py:210`, `:500`).
> The original text is kept below.

All four model types write to the same JSON schema:

```json
{
  "schema_version": "1.0.0",
  "model_type": "xgboost",
  "generated_at": "2026-02-10T14:30:00Z",
  "training_rounds": {
    "total": 500,
    "best": 127,
    "train_loss": [0.45, 0.43, 0.41, "..."],
    "val_loss": [0.52, 0.49, 0.47, "..."]
  },
  "feature_importance": {
    "primary_method": "gain",
    "values": {
      "intersection_weight": 0.182,
      "skip_entropy": 0.156,
      "lane_agreement_8": 0.134,
      "...": "..."
    },
    "secondary_method": "weight",
    "secondary_values": {"...": "..."}
  },
  "nn_specific": {
    "layer_health": {
      "fc1": {"dead_pct": 47, "gradient_norm": 0.0003},
      "fc2": {"dead_pct": 12, "gradient_norm": 0.012},
      "fc3": {"dead_pct": 3, "gradient_norm": 0.45}
    },
    "feature_gradient_spread": 12847
  },
  "diagnosis": {
    "severity": "critical",
    "issues": ["47% dead neurons in fc1", "Feature gradient spread 12847x"],
    "suggested_fixes": ["Replace ReLU with LeakyReLU", "Add input BatchNorm"],
    "summary": "Feature scaling + dead ReLU neurons explain poor NN performance"
  }
}
```

**File path:** `diagnostics_outputs/training_diagnostics.json`  
**Created by:** Whichever model type runs with `--enable-diagnostics`  
**Consumed by:** Web dashboard `/training` route, Strategy Advisor, LLM analysis

---

## 4. Per-Survivor Attribution

### 4.1 What It Is

Global feature importance (Chapter 11) answers: "Which features matter on average?"
Per-survivor attribution answers: "For THIS specific seed, which features drove its prediction?"

This is critical for pool strategy: if Top 20 survivors are driven by different features
than Top 300 survivors, they're structurally different populations requiring different
optimization strategies.

> **Corrected 2026-11-11 — applies to all of §4.**
> - **The module exists and matches the design** (accurate): `per_survivor_attribution.py` has
>   the four backends, the dispatcher and `compare_pool_tiers` (`:49`, `:134`, `:184`, `:231`,
>   `:289`, `:323`), default `method="grad_x_input"` (`:53`).
> - **It never runs on real predictions — SWITCHED-OFF-BY-RULING, S2.** "Real attribution run —
>   **Blocked** until schema, scaler and prediction-parity gates pass" and "One-line Ch13
>   feature-name patch alone — Not sufficient"
>   (`TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:272`, `:275`; Gate A-NORM `:99`). Lifting
>   it alone would change nothing: the only caller is the unreachable Chapter 13 path (§9.3),
>   whose four blockers (`…SYSTEM_MAP…:91-97`) are recorded there. Step 5 never imports this module
>   (§16 B11).
> - **Feature count (DESCRIPTION-ONLY, D1):** "62" throughout §4 is stale. Step 5 trains **89**
>   columns of **91** extracted (exclude list `meta_prediction_optimizer_anti_overfit.py:1482`).
> - **NN attribution omits the training scaler** (`per_survivor_attribution.py:98-101` feeds the
>   raw vector; training standardises, `train_single_trial.py:484-492`) — governed as system-map
>   Blocker 4 (`…SYSTEM_MAP…:97`). With `grad_x_input` (`:105-106`) multiplying by raw values,
>   large raw trial counts would dominate if the path ran (a reading of the code, not observed).
> - **"Which features drove THIS seed":** the attribution is assigned positionally to every name
>   (`:117`), including 13 trial-level and 17 run-level columns that are identical across the seeds
>   of a trial (§15). A trial-level name in a seed's top-3 is not evidence distinguishing seeds
>   (`…SYSTEM_MAP…:191-197`). No diagnostic module separates the namespaces.

### 4.2 Neural Net — Input Gradient per Sample

```python
def per_survivor_attribution_nn(model, features, feature_names, device='cuda',
                                method='grad_x_input'):
    """
    Compute per-seed feature attribution via input gradients.

    Uses PyTorch dynamic graph: forward pass builds graph,
    backward pass computes d(prediction)/d(each_input_feature).

    Two methods available:
        'grad'         — |∂y/∂x|  (raw gradient magnitude)
        'grad_x_input' — |x * ∂y/∂x|  (gradient weighted by input value)

    grad_x_input is the default because it handles differently-scaled features
    more stably (avoids over-emphasizing small-magnitude features with large
    gradients). No extra graph cost — same backward pass, one extra multiply.

    Args:
        model: Trained SurvivorQualityNet
        features: np.ndarray shape (62,) — one survivor's features
        feature_names: List[str] of 62 feature names
        device: 'cuda' or 'cpu'
        method: 'grad' or 'grad_x_input' (default)

    Returns:
        dict mapping feature_name → attribution_score
    """
    model.eval()
    x = torch.tensor(features, dtype=torch.float32, device=device)
    x = x.unsqueeze(0).requires_grad_(True)  # [1, 62]

    # Forward pass — dynamic graph built
    prediction = model(x)

    # Backward pass — graph traversed, gradients computed
    prediction.backward()

    # Attribution method selection
    if method == 'grad_x_input':
        # |x * ∂y/∂x| — gradient weighted by input value
        # More stable when features are differently scaled
        grads = (x.grad * x).squeeze().abs()
    else:
        # |∂y/∂x| — raw gradient magnitude
        grads = x.grad.squeeze().abs()

    # Normalize to sum to 1 for interpretability
    total = grads.sum()
    if total > 0:
        grads = grads / total

    return {name: grads[i].item() for i, name in enumerate(feature_names)}
```

**Example output for a Top 5 survivor:**
```python
{
    'intersection_weight':  0.23,   # ← 23% of prediction driven by this
    'skip_entropy':         0.18,
    'lane_agreement_8':     0.12,
    'temporal_stability':   0.09,
    'forward_count':        0.08,   # ← if NN was scaling-broken, this would be 0.85
    # ... remaining 57 features with smaller values
}
```

### 4.3 XGBoost — pred_contribs

```python
def per_survivor_attribution_xgb(model, features, feature_names):
    """
    XGBoost native per-sample contribution.
    Built into the library — one function call.

    Each tree votes on the prediction. pred_contribs decomposes
    the final prediction into per-feature contributions.

    Returns:
        dict mapping feature_name → attribution_score
    """
    import xgboost as xgb

    dmatrix = xgb.DMatrix(
        features.reshape(1, -1),
        feature_names=feature_names
    )
    contributions = model.get_booster().predict(dmatrix, pred_contribs=True)
    # contributions shape: [1, 63] — 62 features + 1 bias term

    raw = contributions[0][:-1]  # Drop bias
    total = np.abs(raw).sum()
    if total > 0:
        normalized = np.abs(raw) / total
    else:
        normalized = raw

    return {name: float(normalized[i]) for i, name in enumerate(feature_names)}
```

### 4.4 LightGBM — pred_contrib

```python
def per_survivor_attribution_lgb(model, features, feature_names):
    """
    LightGBM native per-sample contribution.
    Same concept as XGBoost — decomposes prediction into feature contributions.

    Returns:
        dict mapping feature_name → attribution_score
    """
    contributions = model.predict(
        features.reshape(1, -1),
        pred_contrib=True
    )
    # contributions shape: [1, 63] — 62 features + 1 bias

    raw = contributions[0][:-1]
    total = np.abs(raw).sum()
    if total > 0:
        normalized = np.abs(raw) / total
    else:
        normalized = raw

    return {name: float(normalized[i]) for i, name in enumerate(feature_names)}
```

### 4.5 CatBoost — Native SHAP (C++ Implementation)

```python
def per_survivor_attribution_catboost(model, features, feature_names):
    """
    CatBoost native SHAP values — implemented in C++, much faster
    than the Python shap library.

    SHAP provides theoretically grounded attribution:
    each feature's contribution is its average marginal
    contribution across all possible feature coalitions.

    Returns:
        dict mapping feature_name → attribution_score
    """
    from catboost import Pool

    pool = Pool(
        features.reshape(1, -1),
        feature_names=feature_names
    )
    shap_values = model.get_feature_importance(pool, type='ShapValues')
    # shap_values shape: [1, 63] — 62 features + 1 expected value

    raw = shap_values[0][:-1]
    total = np.abs(raw).sum()
    if total > 0:
        normalized = np.abs(raw) / total
    else:
        normalized = raw

    return {name: float(normalized[i]) for i, name in enumerate(feature_names)}
```

### 4.6 Unified Interface

```python
def per_survivor_attribution(model, model_type, features, feature_names, device='cuda'):
    """
    Unified per-survivor attribution across all model types.

    Args:
        model: Trained model (any type)
        model_type: str — 'neural_net', 'xgboost', 'lightgbm', 'catboost'
        features: np.ndarray shape (62,)
        feature_names: List[str] of 62 names
        device: for neural_net only

    Returns:
        dict mapping feature_name → normalized attribution score
    """
    if model_type == 'neural_net':
        return per_survivor_attribution_nn(model, features, feature_names, device)
    elif model_type == 'xgboost':
        return per_survivor_attribution_xgb(model, features, feature_names)
    elif model_type == 'lightgbm':
        return per_survivor_attribution_lgb(model, features, feature_names)
    elif model_type == 'catboost':
        return per_survivor_attribution_catboost(model, features, feature_names)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
```

### 4.7 Pool Tier Comparison (Strategic Application)

```python
def compare_pool_tiers(model, model_type, survivors_with_scores, feature_names):
    """
    Compare what drives predictions for tight vs wide pool.
    This is the data the Strategy Advisor uses for POOL_PRECISION focus.
    """
    # Sort by prediction score
    ranked = sorted(survivors_with_scores, key=lambda s: s['prediction'], reverse=True)

    top_20 = ranked[:20]
    top_100 = ranked[:100]
    top_300 = ranked[:300]

    def avg_attribution(pool):
        attrs = [per_survivor_attribution(model, model_type,
                 np.array(s['features']), feature_names) for s in pool]
        return {f: np.mean([a[f] for a in attrs]) for f in feature_names}

    tier_comparison = {
        'top_20': avg_attribution(top_20),
        'top_100': avg_attribution(top_100),
        'top_300': avg_attribution(top_300),
    }

    # Which features differentiate tight from wide pool?
    divergence = {}
    for f in feature_names:
        divergence[f] = tier_comparison['top_20'][f] - tier_comparison['top_300'][f]
    # Positive = feature concentrates in top tier
    # Negative = feature more important in wide tier

    tier_comparison['divergence'] = divergence
    return tier_comparison
```

---

## 5. Training Dynamics Web Dashboard

> **SUPERSEDED (2026-11-11) as a description of the system — never built. SWITCHED-OFF-BY-RULING
> (deferral), S9.** Deferred in the approved-scope proposal
> (`CHAPTER_14_IMPLEMENTATION_PROPOSAL_S69.md:374-382`), the header patch
> (`CHAPTER_14_HEADER_PATCH.md:63`) and the tracker ("Web Dashboard — Future — Lower priority",
> `CHAPTER_13_IMPLEMENTATION_PROGRESS_v3_9.md:74`). At `a15b524` `web_dashboard.py` has no
> `/training` route (routes `:1943-2225`). **Lifting it would enable** human-readable loss curves
> and importances; it would need B3–B5 and B18 fixed first, otherwise three of four families have
> nothing to plot, features are unnamed and NN gradient charts are empty. The design text below is
> kept as the plan.

### 5.1 New Route: `/training`

Add to existing `web_dashboard.py`.

```python
@app.route('/training')
def training_diagnostics():
    """Training diagnostics dashboard — all model types."""
    diag_path = "diagnostics_outputs/training_diagnostics.json"
    if not os.path.exists(diag_path):
        return base_template(
            "<h2>No Training Diagnostics Available</h2>"
            "<p>Run Step 5 with <code>--enable-diagnostics</code> to generate data.</p>"
            "<pre>python3 meta_prediction_optimizer_anti_overfit.py \\\n"
            "    --model-type neural_net --trials 1 --enable-diagnostics</pre>",
            active_tab="training"
        )

    with open(diag_path) as f:
        report = json.load(f)

    charts_html = generate_training_charts(report)
    return base_template(charts_html, active_tab="training")
```

### 5.2 Five Charts (All Model Types)

#### Chart 1: Loss Curves (Universal)

Every model type produces train/val loss per round. This is the most important chart.

```python
def chart_loss_curves(report):
    """Works for ALL model types — they all produce round-by-round loss."""
    import plotly.graph_objects as go

    rounds = report['training_rounds']
    train_loss = rounds['train_loss']
    val_loss = rounds['val_loss']
    best = rounds.get('best', len(val_loss) - 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(train_loss))), y=train_loss,
        name="Train Loss", line=dict(color="#3b82f6")))
    fig.add_trace(go.Scatter(
        x=list(range(len(val_loss))), y=val_loss,
        name="Val Loss", line=dict(color="#ef4444")))
    fig.add_vline(x=best, line_dash="dash", line_color="#10b981",
                  annotation_text=f"Best: round {best}")
    fig.update_layout(
        title=f"{report['model_type']} — Loss Curve",
        xaxis_title="Round/Epoch", yaxis_title="Loss (MAE)",
        plot_bgcolor="#2a2e33", paper_bgcolor="#1a1d21",
        font=dict(color="#e8e8e8"))
    return fig
```

**What you see → What it means:**

| Pattern | Meaning | Action |
|---------|---------|--------|
| Train drops, val flat | Memorizing, not generalizing | More dropout, early stopping |
| Both flat from round 5 | Stuck in bad local minimum | Lower LR, different optimizer |
| Both decreasing together | Healthy training | Continue |
| Val rises after round N | Classic overfitting | Best round = N |

#### Chart 2: Feature Importance (Global, All Types)

```python
def chart_feature_importance(report):
    """Top 20 features by primary importance method."""
    import plotly.graph_objects as go

    importance = report.get('feature_importance', {}).get('values', {})
    if not importance:
        return None

    sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
    names = [f[0] for f in sorted_feats]
    values = [f[1] for f in sorted_feats]

    fig = go.Figure()
    fig.add_trace(go.Bar(y=names, x=values, orientation='h',
                         marker=dict(color='#10b981')))
    method = report.get('feature_importance', {}).get('primary_method', 'importance')
    fig.update_layout(
        title=f"{report['model_type']} — Top 20 Features ({method})",
        xaxis_title=f"Importance ({method})",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#2a2e33", paper_bgcolor="#1a1d21",
        font=dict(color="#e8e8e8"))
    return fig
```

#### Chart 3: Per-Survivor Attribution (Top 3 Seeds)

```python
def chart_survivor_attribution(report):
    """Side-by-side attribution for top 3 ranked survivors."""
    import plotly.graph_objects as go

    attributions = report.get('top_survivor_attributions', [])
    if not attributions:
        return None

    fig = go.Figure()
    for attr in attributions[:3]:
        sorted_feats = sorted(attr['features'].items(), key=lambda x: x[1], reverse=True)[:15]
        fig.add_trace(go.Bar(
            name=f"Seed {attr['seed']} (rank #{attr['rank']})",
            y=[f[0] for f in sorted_feats],
            x=[f[1] for f in sorted_feats],
            orientation='h'))

    fig.update_layout(
        title="Per-Survivor Feature Attribution (Top 3 Seeds)",
        xaxis_title="Normalized Attribution",
        barmode='group',
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#2a2e33", paper_bgcolor="#1a1d21",
        font=dict(color="#e8e8e8"))
    return fig
```

#### Chart 4: Neural Net Only — Dead Neurons + Gradient Flow

```python
def chart_nn_health(report):
    """NN-specific: dead neurons and gradient norms per layer over training."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if report['model_type'] != 'neural_net':
        return None

    nn_data = report.get('nn_specific', {})
    epoch_data = report.get('epoch_snapshots', [])
    if not epoch_data:
        return None

    layer_names = list(epoch_data[0].get('layers', {}).keys())
    epochs = [e['epoch'] for e in epoch_data]

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=["Dead Neurons (%) Per Layer",
                                        "Gradient Norms Per Layer (log)"])

    # Dead neurons
    for layer in layer_names:
        dead_pcts = [e.get('layers', {}).get(layer, {}).get('dead_pct', 0)
                     for e in epoch_data]
        fig.add_trace(go.Scatter(x=epochs, y=dead_pcts, name=f"{layer} dead%"),
                      row=1, col=1)
    fig.add_hline(y=25, line_dash="dash", line_color="yellow", row=1, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="red", row=1, col=1)

    # Gradient norms
    for layer in layer_names:
        grad_norms = [e.get('layers', {}).get(layer, {}).get('gradient_norm', 0)
                      for e in epoch_data]
        fig.add_trace(go.Scatter(x=epochs, y=grad_norms, name=f"{layer} grad"),
                      row=2, col=1)
    fig.update_yaxes(type="log", row=2, col=1)

    fig.update_layout(
        height=700,
        plot_bgcolor="#2a2e33", paper_bgcolor="#1a1d21",
        font=dict(color="#e8e8e8"))
    return fig
```

#### Chart 5: Diagnosis Panel

```python
def chart_diagnosis_panel(report):
    """Color-coded severity box with issues and suggested fixes."""
    diagnosis = report.get('diagnosis', {})
    severity = diagnosis.get('severity', 'ok')
    color = {'ok': '#10b981', 'warning': '#f59e0b', 'critical': '#ef4444'}.get(
        severity, '#8a9099')

    issues_html = "".join(f"<li>{i}</li>" for i in diagnosis.get('issues', []))
    fixes_html = "".join(f"<li>{f}</li>" for f in diagnosis.get('suggested_fixes', []))

    return f"""
    <div style="border: 2px solid {color}; border-radius: 8px;
                padding: 16px; margin: 16px 0; background: #1a1d21;">
        <h3 style="color: {color};">
            {report['model_type']} Diagnosis: {severity.upper()}
        </h3>
        <h4 style="color: #e8e8e8;">Issues Found:</h4>
        <ul style="color: #ccc;">{issues_html}</ul>
        <h4 style="color: #e8e8e8;">Suggested Fixes:</h4>
        <ul style="color: #ccc;">{fixes_html}</ul>
    </div>
    """
```

#### Assembler Function

```python
def generate_training_charts(report):
    """Generate all applicable charts for any model type."""
    parts = []

    # Diagnosis panel first (most important at-a-glance info)
    parts.append(chart_diagnosis_panel(report))

    # Loss curves (all model types)
    fig1 = chart_loss_curves(report)
    parts.append(fig1.to_html(full_html=False, include_plotlyjs="cdn"))

    # Feature importance (all model types)
    fig2 = chart_feature_importance(report)
    if fig2:
        parts.append(fig2.to_html(full_html=False, include_plotlyjs=False))

    # Per-survivor attribution (all model types)
    fig3 = chart_survivor_attribution(report)
    if fig3:
        parts.append(fig3.to_html(full_html=False, include_plotlyjs=False))

    # NN-specific health charts
    fig4 = chart_nn_health(report)
    if fig4:
        parts.append(fig4.to_html(full_html=False, include_plotlyjs=False))

    return "".join(f'<div style="margin: 16px 0;">{p}</div>' for p in parts)
```

### 5.3 Navigation Tab Addition

Add "Training" to the existing dashboard tab bar alongside Overview, Workers, Stats, Plots, Settings.

---

## 6. TensorBoard Integration

> **SUPERSEDED (2026-11-11) as a description of the system — never built. SWITCHED-OFF-BY-RULING
> (deferral), S9.** Same deferral record as §5. No `tensorboard` or `SummaryWriter` string occurs
> in any tracked `*.py`, and `--enable-tensorboard` does not exist (`meta_prediction_optimizer_anti_overfit.py`
> argparse defines `--enable-diagnostics` only, `:3080`). §6.5's config block has no file on
> VM 101 (`reinforcement_engine_config.json` absent). **Lifting it would enable** weight/gradient
> histograms and graph views for human sessions. Text kept as the plan.

### 6.1 What TensorBoard Provides Beyond Our Dashboard

| Feature | Our Plotly Dashboard | TensorBoard |
|---------|---------------------|-------------|
| Loss curves | ✅ | ✅ |
| Feature importance | ✅ | Via matplotlib images |
| Dead neurons | ✅ (NN only) | Via custom scalars |
| Weight histograms | ❌ | ✅ — interactive per-epoch |
| Model computational graph | ❌ | ✅ — visual graph of all layers |
| Gradient distributions | ❌ | ✅ — histogram per layer per epoch |
| Hyperparameter comparison | ❌ | ✅ — built-in HParams dashboard |
| Embedding projector | ❌ | ✅ — t-SNE/PCA of hidden layers |

**Decision:** TensorBoard is complementary, not a replacement. Plotly dashboard for
monitoring and at-a-glance status. TensorBoard for deep research sessions when
investigating neural_net behavior.

### 6.2 Neural Net — Full TensorBoard Support

```python
from torch.utils.tensorboard import SummaryWriter

# Create writer — one per training run
writer = SummaryWriter(f'runs/{model_type}_{study_name}_{datetime.now():%Y%m%d_%H%M%S}')

# ─── BEFORE TRAINING ─────────────────────────────────────────────

# Log the computational graph — visualize the full network
dummy_input = torch.randn(1, 62).to(device)  # 62 features
writer.add_graph(model, dummy_input)

# ─── INSIDE EPOCH LOOP ───────────────────────────────────────────

for epoch in range(epochs):
    train_loss = train_one_epoch(...)
    val_loss = validate(...)

    # Scalar metrics
    writer.add_scalars('Loss', {
        'train': train_loss,
        'val': val_loss
    }, epoch)

    # Weight and gradient distributions per layer
    for name, param in model.named_parameters():
        writer.add_histogram(f'weights/{name}', param.data, epoch)
        if param.grad is not None:
            writer.add_histogram(f'gradients/{name}', param.grad, epoch)

    # Learning rate (if using scheduler)
    writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

# ─── AFTER TRAINING ──────────────────────────────────────────────

# Feature importance as bar chart image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
ax.barh([f[0] for f in sorted_importance], [f[1] for f in sorted_importance])
ax.set_title("Feature Importance")
writer.add_figure('Feature Importance', fig)
plt.close(fig)

writer.close()
```

### 6.3 Tree Models — Scalar Logging

```python
# XGBoost
writer = SummaryWriter(f'runs/xgboost_{study_name}')
evals = model.evals_result()
for i in range(len(evals['validation_0']['mae'])):
    writer.add_scalars('Loss', {
        'train': evals['validation_0']['mae'][i],
        'val': evals['validation_1']['mae'][i],
    }, i)
writer.close()

# LightGBM
writer = SummaryWriter(f'runs/lightgbm_{study_name}')
for i in range(len(eval_results['training']['l1'])):
    writer.add_scalars('Loss', {
        'train': eval_results['training']['l1'][i],
        'val': eval_results['valid_1']['l1'][i],
    }, i)
writer.close()

# CatBoost
writer = SummaryWriter(f'runs/catboost_{study_name}')
evals = model.get_evals_result()
for i in range(len(evals['learn']['MAE'])):
    writer.add_scalars('Loss', {
        'train': evals['learn']['MAE'][i],
        'val': evals['validation']['MAE'][i],
    }, i)
writer.close()
```

### 6.4 Launch TensorBoard on Zeus

```bash
# After running any training with TensorBoard enabled
tensorboard --logdir=runs/ --host=0.0.0.0 --port=6006

# Access from browser:
# http://zeus:6006

# View all model type runs side by side
# TensorBoard auto-discovers all runs/ subdirectories
```

### 6.5 Activation via Config

```json
{
    "diagnostics": {
        "enabled": false,
        "tensorboard": false,
        "capture_every_n": 5,
        "nn_attribution_method": "grad_x_input",
        "output_dir": "diagnostics_outputs",
        "tensorboard_dir": "runs"
    }
}
```

**Snapshot throttling (`capture_every_n`):** Hooks fire every epoch/round but only
append to the snapshot list on multiples of this value:

```python
# Inside epoch/round loop — gated capture
if epoch % config.get('capture_every_n', 5) == 0:
    epoch_snapshots.append(snapshot)
```

Default 5 keeps diagnostic JSON under 50KB for typical runs. Set to 1 only for
targeted short-run investigation. This prevents JSON bloat, dashboard overload,
and LLM context inflation.

```bash
# CLI activation
python3 meta_prediction_optimizer_anti_overfit.py \
    --model-type neural_net --trials 1 \
    --enable-diagnostics --enable-tensorboard
```

---

## 7. WATCHER Integration

### 7.1 Architectural Decision: When Does WATCHER Read Diagnostics?

**Question posed earlier:** Can the WATCHER monitor training as it happens and abort
mid-training, or does everything wait until training finishes?

**Answer: Post-training only.** Here's why:

| Approach | Pros | Cons |
|----------|------|------|
| Mid-training abort | Save GPU time on doomed runs | Requires polling loop, couples WATCHER to training internals, false positives kill good runs early |
| Post-training read | Clean separation, diagnostics file is complete, no polling overhead | Wastes time on a bad run (but Step 5 runs are 2-8 minutes, not hours) |

Step 5 training runs are short enough that aborting mid-run saves negligible time.
The risk of false-positive abort (killing a run that would have recovered) outweighs the
benefit. WATCHER reads `training_diagnostics.json` AFTER Step 5 completes, decides
whether to proceed to Step 6 or intervene.

Exception: If we later add multi-hour training runs (e.g., large neural_net with 10K+
epochs), we revisit this decision and add a `WatcherTrainingMonitor` polling class.

> **Verified 2026-11-11 (accurate):** the code matches this decision. WATCHER calls
> `check_training_health()` once, after Step 5 returns "proceed" and before Step 6
> (`agents/watcher_agent.py:2861-2868`); there is no mid-training abort and no
> `WatcherTrainingMonitor`. This is a design decision written by Team Beta inside the chapter,
> not a separate ruling. Note: the "2-8 minutes" premise predates S91, which recorded the NN
> taking 2+ hours for 20 Optuna trials (`PROPOSAL_CATEGORY_B_NN_TRAINING_ENHANCEMENTS_v1_0.md:357`);
> the chapter's own exception clause is the place that premise would be revisited.

### 7.2 New WATCHER Policy Entries

> **Corrected 2026-11-11.**
> - **The policy block exists with these values** (accurate): `watcher_policies.json`
>   `training_diagnostics`, read on VM 101 this session, matches the block below value for value.
> - **`overfit_ratio` description (DESCRIPTION-ONLY, D6):** the chapter's text says "Train loss /
>   val loss"; both the live policy file and the code use **val / train**
>   (`training_health_check.py:415`, `overfit_ratio = final_val / final_train`).
> - **Two threshold layers, not one (DESCRIPTION-ONLY, D16):** the producer applies its own
>   bounds before WATCHER sees the file — `training_diagnostics.py:75-80` `SEVERITY_THRESHOLDS`
>   (dead 15/25, gradient spread 1000/10000, gap ratio 0.3/0.5) — and the health check starts from
>   the producer's severity (`training_health_check.py:198`) before applying the WATCHER bounds.
> - **`early_stop_ratio` is mis-read — BROKEN, B1.** See §7.3 note.
> - **`nn_gradient_spread_ratio` is never evaluated — BROKEN, B2.** See §7.3 note.
> - **`unused_feature_pct` is declared and evaluated nowhere — BROKEN, B16.** It is in this
>   block, in `training_health_check.py:67`, and has no branch in `_evaluate_metrics`
>   (`:331-430`); the producer computes no such metric.
> - **`skip_duration_hours` is read nowhere — BROKEN, B8.** Defined at `training_health_check.py:72`
>   and in the policy file; no code expires a skip. A skip lasts until the counter is reset by a
>   non-critical check (`reset_skip_registry`, `:518-535`) or by hand.

Add to `watcher_policies.json` — follows existing policy structure:

```json
{
    "training_diagnostics": {
        "enabled": true,
        "description": "Post-Step-5 training health check before proceeding to Step 6",

        "severity_thresholds": {
            "ok": {
                "action": "PROCEED",
                "log_level": "info",
                "description": "Training healthy — proceed to Step 6"
            },
            "warning": {
                "action": "PROCEED_WITH_NOTE",
                "log_level": "warning",
                "description": "Training has minor issues — proceed but log for Strategy Advisor"
            },
            "critical": {
                "action": "RETRY_OR_SKIP",
                "log_level": "error",
                "max_retries": 2,
                "description": "Training fundamentally broken — retry with different config or skip model type"
            }
        },

        "metric_bounds": {
            "nn_dead_neuron_pct": {
                "warning": 25.0,
                "critical": 50.0,
                "description": "Dead ReLU neurons in any layer"
            },
            "nn_gradient_spread_ratio": {
                "warning": 100.0,
                "critical": 1000.0,
                "description": "Max/min feature gradient ratio — indicates scaling issue"
            },
            "overfit_ratio": {
                "warning": 1.3,
                "critical": 1.5,
                "description": "Train loss / val loss ratio"
            },
            "early_stop_ratio": {
                "warning": 0.3,
                "critical": 0.15,
                "description": "best_round / total_rounds — low = peaked early and overfit rest"
            },
            "unused_feature_pct": {
                "warning": 40.0,
                "critical": 70.0,
                "description": "Features with zero importance (trees) or zero gradient (NN)"
            }
        },

        "model_skip_rules": {
            "max_consecutive_critical": 3,
            "skip_duration_hours": 24,
            "description": "If a model type hits critical N times consecutively, skip it for N hours"
        }
    }
}
```

### 7.3 `check_training_health()` — The WATCHER Integration Function

> **Corrected 2026-11-11 — the live function, and what it does with a real file.**
>
> **Location (DESCRIPTION-ONLY, D4):** the function and its helpers live in
> `training_health_check.py` (`check_training_health` `:80`, `_evaluate_single_model` `:189`,
> `_evaluate_metrics` `:331`, `_severity_to_action` `:433`, `_check_skip_registry` `:467`,
> `reset_skip_registry` `:518`, `_archive_diagnostics` `:539`), imported into WATCHER at
> `agents/watcher_agent.py:103-108`. The sample code below is the design; the live code differs
> in the places listed next.
>
> **B1 — early-stop ratio reads a key the producer never writes (BROKEN).** The consumer reads
> `training_summary.get('best_round', 0)` (`training_health_check.py:392`); the producer writes
> `best_val_round` (`training_diagnostics.py:239`). The ratio is therefore `0 / rounds` = 0.00,
> which is ≤ the critical bound 0.15 (`:396-398`), so **every report that reaches the grader is
> graded critical**. Both lines date from 2026-02-08 (`881ec3b`, `51e74b7`).
> **Host evidence (VM 101, read 2026-11-11):** every real producer report on the host carries
> `best_val_round` and none carries `best_round` (64 complete schema-1.1.0 reports; the only two
> archived files with `best_round` are the health check's own self-test fixture,
> `training_health_check.py:783-800`). All 107 archived health checks in
> `diagnostics_outputs/history/` are `critical`; 105 carry "Early stop ratio 0.00". Computed from
> those artifacts with the rules in `:331-430` (not executed): with the producer's key, 26 of the
> 64 real reports would not have been critical; 38 would still be critical on other grounds.
> No governance document names the key; `SESSION_CHANGELOG_20260215_S92.md:188` names the
> symptom ("'Early stop ratio 0.00' as automatic critical needs revisiting").
>
> **B2 — gradient-spread cross-check never fires (BROKEN).** The consumer reads
> `model_specific.feature_gradient_spread` (`training_health_check.py:373`); the producer writes
> `model_specific.gradient_health.gradient_spread_ratio` (`training_diagnostics.py:595-598`).
> (Independently, the producer's value is always the initial 1.0 because NN feature gradients are
> not captured — B18, §3.1 note.)
>
> **B6 — compare-models winner override never fires (BROKEN).** Looks for
> `diagnostics_outputs/compare_models_summary.json` with key `winner_model_type`/`winner`
> (`training_health_check.py:160-166`); the writer emits `compare_models_summary_{run_id}.json`
> with `winner_best_effort` (`meta_prediction_optimizer_anti_overfit.py:311`, `:325`). On VM 101
> there is no `compare_models_summary.json` and 60 `compare_models_summary_S88_*.json`. The gate
> judges whatever model wrote the canonical file — in practice the NN, the only family that writes
> one on the production path (B3).
>
> **Stale file.** WATCHER deletes `diagnostics_outputs/training_diagnostics.json` before each
> RETRY (`agents/watcher_agent.py:2897-2901`), but not after PROCEED, PROCEED_WITH_NOTE or
> SKIP_MODEL, so a later run that writes no new file is graded on the old one. Host: of 78
> archived checks with a producer timestamp, 35 re-graded a report already graded (e.g. the check
> of 2026-03-07T04:11Z graded a report generated 2026-02-25). The canonical file is absent on
> VM 101 now, so the next check would return PROCEED / `absent` unless Step 5 writes a new one.
> This is a consequence of B1 + B3, not a separate label.

Add to `agents/watcher_agent.py`. Called between Step 5 and Step 6 in the pipeline.

```python
# ─── agents/watcher_agent.py (new method) ─────────────────────────

import json
import os
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Path constants — same convention as existing WATCHER paths
DIAGNOSTICS_PATH = "diagnostics_outputs/training_diagnostics.json"
DIAGNOSTICS_HISTORY_DIR = "diagnostics_outputs/history/"
WATCHER_POLICIES_PATH = "watcher_policies.json"
SKIP_REGISTRY_PATH = "diagnostics_outputs/model_skip_registry.json"


def check_training_health(diagnostics_path=DIAGNOSTICS_PATH):
    """
    Post-Step-5 training health check.

    Called by WATCHER between Step 5 and Step 6 dispatch.
    Reads training_diagnostics.json, evaluates against watcher_policies.json
    thresholds, returns an action decision.

    Returns:
        dict: {
            'action': 'PROCEED' | 'PROCEED_WITH_NOTE' | 'RETRY' | 'SKIP_MODEL',
            'model_type': str,
            'severity': 'ok' | 'warning' | 'critical',
            'issues': list[str],
            'suggested_fixes': list[str],
            'confidence': float,
        }

    If diagnostics file doesn't exist (--enable-diagnostics not used),
    returns PROCEED with a note. Absence of diagnostics is NOT a failure.
    """
    # ── No diagnostics file → proceed normally ──────────────────────
    if not os.path.isfile(diagnostics_path):
        logger.info("No training diagnostics found — proceeding without health check")
        return {
            'action': 'PROCEED',
            'model_type': 'unknown',
            'severity': 'ok',
            'issues': [],
            'suggested_fixes': [],
            'confidence': 0.5,
            'note': 'Diagnostics not enabled for this run',
        }

    # ── Load diagnostics ────────────────────────────────────────────
    try:
        with open(diagnostics_path) as f:
            diag = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error("Failed to read diagnostics: %s", e)
        return {
            'action': 'PROCEED_WITH_NOTE',
            'model_type': 'unknown',
            'severity': 'warning',
            'issues': [f'Diagnostics file unreadable: {e}'],
            'suggested_fixes': ['Check diagnostics_outputs/ for corruption'],
            'confidence': 0.3,
        }

    # ── Load policies ───────────────────────────────────────────────
    policies = {}
    if os.path.isfile(WATCHER_POLICIES_PATH):
        with open(WATCHER_POLICIES_PATH) as f:
            all_policies = json.load(f)
        policies = all_policies.get('training_diagnostics', {})

    metric_bounds = policies.get('metric_bounds', {})
    model_type = diag.get('model_type', 'unknown')
    diagnosis = diag.get('diagnosis', {})
    severity = diagnosis.get('severity', 'ok')
    issues = diagnosis.get('issues', [])
    fixes = diagnosis.get('suggested_fixes', [])

    # ── Evaluate against policy thresholds ──────────────────────────
    # Cross-check diagnostics severity against WATCHER's own metric bounds
    # This prevents a buggy diagnostics module from under-reporting severity
    watcher_issues = []

    # Check NN-specific metrics
    nn_data = diag.get('nn_specific', {})
    if nn_data:
        for layer_name, layer_data in nn_data.get('layer_health', {}).items():
            dead_pct = layer_data.get('dead_pct', 0)
            bounds = metric_bounds.get('nn_dead_neuron_pct', {})
            if dead_pct >= bounds.get('critical', 50):
                watcher_issues.append(
                    f"WATCHER: {layer_name} dead neurons {dead_pct:.1f}% >= critical threshold"
                )
                severity = 'critical'
            elif dead_pct >= bounds.get('warning', 25):
                watcher_issues.append(
                    f"WATCHER: {layer_name} dead neurons {dead_pct:.1f}% >= warning threshold"
                )
                if severity == 'ok':
                    severity = 'warning'

        gradient_spread = diag.get('nn_specific', {}).get('feature_gradient_spread', 1)
        bounds = metric_bounds.get('nn_gradient_spread_ratio', {})
        if gradient_spread >= bounds.get('critical', 1000):
            watcher_issues.append(
                f"WATCHER: Feature gradient spread {gradient_spread:.0f}x >= critical"
            )
            severity = 'critical'
        elif gradient_spread >= bounds.get('warning', 100):
            watcher_issues.append(
                f"WATCHER: Feature gradient spread {gradient_spread:.0f}x >= warning"
            )
            if severity == 'ok':
                severity = 'warning'

    # Check universal metrics (all model types)
    rounds = diag.get('training_rounds', {})
    best = rounds.get('best', 0)
    total = rounds.get('total', 1)
    early_stop_ratio = best / max(total, 1)

    bounds = metric_bounds.get('early_stop_ratio', {})
    if early_stop_ratio <= bounds.get('critical', 0.15):
        watcher_issues.append(
            f"WATCHER: Early stop ratio {early_stop_ratio:.2f} "
            f"(peaked at round {best}/{total}) — severe overfitting"
        )
        severity = 'critical'
    elif early_stop_ratio <= bounds.get('warning', 0.3):
        watcher_issues.append(
            f"WATCHER: Early stop ratio {early_stop_ratio:.2f} — possible overfitting"
        )
        if severity == 'ok':
            severity = 'warning'

    # Combine diagnostics issues with WATCHER's own findings
    all_issues = issues + watcher_issues

    # ── Map severity to action ──────────────────────────────────────
    severity_map = policies.get('severity_thresholds', {})
    policy = severity_map.get(severity, {})
    base_action = policy.get('action', 'PROCEED')

    # Check skip registry for consecutive failures
    action = base_action
    if severity == 'critical':
        action = _check_skip_registry(model_type, policies)

    # ── Archive diagnostics for Strategy Advisor history ────────────
    _archive_diagnostics(diag, severity, all_issues)

    result = {
        'action': action,
        'model_type': model_type,
        'severity': severity,
        'issues': all_issues,
        'suggested_fixes': fixes,
        'confidence': _severity_to_confidence(severity),
    }

    logger.info(
        "Training health check: model=%s severity=%s action=%s issues=%d",
        model_type, severity, action, len(all_issues)
    )
    return result


def _check_skip_registry(model_type, policies):
    """
    Track consecutive critical failures per model type.
    If threshold exceeded, return SKIP_MODEL instead of RETRY.

    Skip registry file format:
    {
        "neural_net": {"consecutive_critical": 2, "last_critical": "2026-02-10T..."},
        "catboost": {"consecutive_critical": 0, "last_critical": null}
    }
    """
    skip_rules = policies.get('model_skip_rules', {})
    max_consecutive = skip_rules.get('max_consecutive_critical', 3)

    registry = {}
    if os.path.isfile(SKIP_REGISTRY_PATH):
        with open(SKIP_REGISTRY_PATH) as f:
            registry = json.load(f)

    entry = registry.get(model_type, {'consecutive_critical': 0, 'last_critical': None})
    entry['consecutive_critical'] += 1
    entry['last_critical'] = datetime.now(timezone.utc).isoformat()

    registry[model_type] = entry

    os.makedirs(os.path.dirname(SKIP_REGISTRY_PATH), exist_ok=True)
    with open(SKIP_REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=2)

    if entry['consecutive_critical'] >= max_consecutive:
        logger.warning(
            "Model %s hit %d consecutive critical failures — SKIP_MODEL",
            model_type, entry['consecutive_critical']
        )
        return 'SKIP_MODEL'

    return 'RETRY'


def _archive_diagnostics(diag, severity, issues):
    """
    Archive each diagnostics run to history/ for Strategy Advisor consumption.
    One JSON per run, timestamped filename.
    """
    os.makedirs(DIAGNOSTICS_HISTORY_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_type = diag.get('model_type', 'unknown')
    filename = f"{model_type}_{timestamp}.json"

    archive = {
        'diagnostics': diag,
        'watcher_severity': severity,
        'watcher_issues': issues,
        'archived_at': datetime.now(timezone.utc).isoformat(),
    }

    with open(os.path.join(DIAGNOSTICS_HISTORY_DIR, filename), 'w') as f:
        json.dump(archive, f, indent=2)


def _severity_to_confidence(severity):
    """Map severity to WATCHER confidence for proceed decision."""
    return {'ok': 0.90, 'warning': 0.65, 'critical': 0.30}.get(severity, 0.5)


def reset_skip_registry(model_type):
    """
    Reset consecutive critical count for a model type.
    Called when a model type succeeds (severity != critical).
    """
    if not os.path.isfile(SKIP_REGISTRY_PATH):
        return

    with open(SKIP_REGISTRY_PATH) as f:
        registry = json.load(f)

    if model_type in registry:
        registry[model_type]['consecutive_critical'] = 0
        with open(SKIP_REGISTRY_PATH, 'w') as f:
            json.dump(registry, f, indent=2)
        logger.info("Reset skip registry for %s", model_type)
```

### 7.4 WATCHER Pipeline Dispatch Wiring

> **Corrected 2026-11-11.**
> - **Wiring exists** (accurate): call site `agents/watcher_agent.py:2861-2868`, guarded by
>   step 5, `recommended_action == "proceed"`, `TRAINING_HEALTH_CHECK_AVAILABLE` and Step 5 not
>   freshness-skipped; action mapping `_handle_training_health` (`:2262-2309`); retries capped at
>   2 from policy (`_get_max_training_retries`, `:2535-2545`; loop `:2871`).
> - **A RETRY re-runs the same Step 5 — SWITCHED-OFF-BY-RULING, S1.** `_build_retry_params`
>   proposes `dropout`, `normalize_features`, `use_leaky_relu` (`:2375-2386`) and the LLM
>   whitelist proposes six fields (`:2336-2343`), but WATCHER merges only keys in the Step-5
>   manifest's `default_params` (`:1813-1820`), and none of them is there
>   (`agent_manifests/reinforcement.json:173-184`). Ruling: "The step-scoped filter is a deliberate
>   executable-interface boundary… The six proposals must not be wired through"
>   (quoted in `CLAUDE_CODE_INSTRUCTIONS_CHAIN_C_TRUTHFULNESS_HOTFIX.md:33-44`; upheld
>   `PIPELINE_BEHAVIOUR_MODEL.md:993-996`). **Lifting it would enable** a retry that differs from
>   the failed run; it would also need the names declared in the manifest, a CLI route for each in
>   Step 5, and B1 fixed first (otherwise every run triggers adaptation). The three heuristic
>   `[WATCHER][RETRY]` log lines still claim a change (`:2378`, `:2382`, `:2386`; `BACKLOG.md:96-100`).
> - **Before each RETRY WATCHER deletes the canonical diagnostics file** (`:2897-2901`, `[S99]`) —
>   not shown in the sample below.

Where `check_training_health()` gets called in the existing WATCHER pipeline:

```python
# ─── agents/watcher_agent.py — inside run_pipeline() ──────────────

def run_pipeline(self, start_step=1, end_step=6, params=None):
    """Existing pipeline dispatch — add diagnostics check between Step 5 and 6."""

    for step in range(start_step, end_step + 1):

        # ... existing step dispatch logic ...

        if step == 5:
            # Existing: run Step 5 (meta_prediction_optimizer_anti_overfit.py)
            result = self.dispatch_step(5, params)
            evaluation = self.evaluate_step(5, result)

            if evaluation['action'] == 'PROCEED':
                # ── NEW: Post-Step-5 diagnostics check ──────────────
                health = check_training_health()

                if health['action'] == 'PROCEED':
                    # Reset skip registry — model is healthy
                    reset_skip_registry(health['model_type'])
                    logger.info("Training health OK — proceeding to Step 6")

                elif health['action'] == 'PROCEED_WITH_NOTE':
                    # Log warning for Strategy Advisor, continue to Step 6
                    reset_skip_registry(health['model_type'])
                    logger.warning(
                        "Training health WARNING for %s: %s",
                        health['model_type'],
                        "; ".join(health['issues'])
                    )
                    # Append to open_incidents for next LLM bundle
                    self._record_incident(
                        f"Step 5 training warning ({health['model_type']}): "
                        f"{'; '.join(health['issues'][:3])}"
                    )

                elif health['action'] == 'RETRY':
                    logger.warning(
                        "Training health CRITICAL for %s — retrying Step 5",
                        health['model_type']
                    )
                    # ── Request LLM analysis on CRITICAL ────────────
                    llm_analysis = self._request_diagnostics_llm(health)

                    # Retry with modified params (e.g., different model type)
                    retry_params = self._build_retry_params(params, health, llm_analysis)
                    result = self.dispatch_step(5, retry_params)
                    # Re-evaluate after retry
                    evaluation = self.evaluate_step(5, result)
                    if evaluation['action'] != 'PROCEED':
                        self.escalate("Step 5 retry failed")
                        return

                elif health['action'] == 'SKIP_MODEL':
                    logger.error(
                        "Model %s skipped — %d consecutive critical failures",
                        health['model_type'],
                        health.get('consecutive_critical', 0)
                    )
                    # ── Request LLM analysis on SKIP ────────────────
                    llm_analysis = self._request_diagnostics_llm(health)

                    # Remove this model type from --compare-models for next run
                    self._record_model_skip(health['model_type'])
                    # Still proceed to Step 6 with remaining models
                    logger.info("Proceeding to Step 6 with remaining model types")

            # Continue to step 6 ...

    def _build_retry_params(self, original_params, health, llm_analysis=None):
        """Build modified params for Step 5 retry based on diagnostics."""
        retry_params = dict(original_params or {})

        # If NN failed, retry with tree models only
        if health['model_type'] == 'neural_net':
            retry_params['model_type'] = 'catboost'
            logger.info("Retry: switching from neural_net to catboost")

        # If feature scaling issue, add normalization flag
        if any('scaling' in i.lower() or 'spread' in i.lower()
               for i in health['issues']):
            retry_params['normalize_features'] = True
            logger.info("Retry: enabling feature normalization")

        # Apply LLM parameter proposals if available
        if llm_analysis and hasattr(llm_analysis, 'parameter_proposals'):
            for proposal in llm_analysis.parameter_proposals:
                param_name = proposal.parameter
                # Only apply proposals for params within WATCHER policy bounds
                if self._is_within_policy_bounds(param_name, proposal.proposed_value):
                    retry_params[param_name] = proposal.proposed_value
                    logger.info(
                        "LLM proposal applied: %s = %s (rationale: %s)",
                        param_name, proposal.proposed_value, proposal.rationale
                    )
                else:
                    logger.warning(
                        "LLM proposal REJECTED (out of bounds): %s = %s",
                        param_name, proposal.proposed_value
                    )

        return retry_params

    def _record_incident(self, description):
        """
        Append incident to watcher_failures.jsonl for Tier 2 retrieval.

        Next LLM bundle assembly reads this via _retrieve_open_incidents()
        and injects it into the prompt as an open incident.
        """
        incident = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'source': 'training_diagnostics',
            'description': description,
            'resolved': False,
        }
        failures_path = os.path.join(self.config.get('state_dir', '.'), 'watcher_failures.jsonl')
        with open(failures_path, 'a') as f:
            f.write(json.dumps(incident) + '\n')
        logger.info("Recorded incident: %s", description[:100])

    def _record_model_skip(self, model_type):
        """
        Record a model type skip so next --compare-models excludes it.

        Writes to model_skip_state.json. The Step 5 dispatcher reads this
        file and removes skipped model types from the comparison list.
        Skip expires after skip_duration_hours (default 24h) from policy.
        """
        skip_state_path = os.path.join(
            self.config.get('state_dir', '.'), 'model_skip_state.json'
        )
        skip_state = {}
        if os.path.isfile(skip_state_path):
            with open(skip_state_path) as f:
                skip_state = json.load(f)

        skip_state[model_type] = {
            'skipped_at': datetime.now(timezone.utc).isoformat(),
            'expires_hours': self.config.get(
                'training_diagnostics', {}
            ).get('model_skip_rules', {}).get('skip_duration_hours', 24),
            'reason': 'consecutive_critical_failures',
        }

        with open(skip_state_path, 'w') as f:
            json.dump(skip_state, f, indent=2)
        logger.warning("Model %s added to skip list", model_type)

    def _request_diagnostics_llm(self, health):
        """
        Request LLM diagnostics analysis. Called on CRITICAL or SKIP_MODEL.

        Returns DiagnosticsAnalysis or None if LLM unavailable.
        Failure is non-fatal — WATCHER continues with heuristic-only decision.
        """
        try:
            from strategy_advisor import request_llm_diagnostics_analysis

            diag_path = "diagnostics_outputs/training_diagnostics.json"
            tier_path = "diagnostics_outputs/tier_comparison.json"

            if not os.path.isfile(diag_path):
                return None

            analysis = request_llm_diagnostics_analysis(
                diagnostics_path=diag_path,
                tier_comparison_path=tier_path if os.path.isfile(tier_path) else None,
            )

            if analysis:
                # Archive LLM analysis alongside diagnostics history
                archive_path = os.path.join(
                    "diagnostics_outputs/history/",
                    f"llm_analysis_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
                )
                os.makedirs(os.path.dirname(archive_path), exist_ok=True)
                with open(archive_path, 'w') as f:
                    json.dump(analysis.model_dump(), f, indent=2)

                # Feed to Strategy Advisor for next selfplay cycle
                self._update_strategy_advisor(analysis)

            return analysis

        except Exception as e:
            logger.warning("LLM diagnostics analysis failed (non-fatal): %s", e)
            return None

    def _update_strategy_advisor(self, analysis):
        """
        Write LLM analysis to strategy_recommendation.json for
        the Strategy Advisor to consume on its next cycle.

        Strategy Advisor reads this file to set selfplay focus area
        and apply parameter proposals to episode configuration.
        """
        rec_path = os.path.join(
            self.config.get('state_dir', '.'), 'strategy_recommendation.json'
        )
        recommendation = {
            'source': 'chapter_14_diagnostics',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'focus_area': analysis.focus_area.value,
            'root_cause': analysis.root_cause,
            'root_cause_confidence': analysis.root_cause_confidence,
            'model_verdicts': {
                r.model_type.value: r.verdict.value
                for r in analysis.model_recommendations
            },
            'parameter_proposals': [
                {
                    'parameter': p.parameter,
                    'current_value': p.current_value,
                    'proposed_value': p.proposed_value,
                    'rationale': p.rationale,
                }
                for p in analysis.parameter_proposals
            ],
            'selfplay_guidance': analysis.selfplay_guidance,
            'requires_human_review': analysis.requires_human_review,
        }
        with open(rec_path, 'w') as f:
            json.dump(recommendation, f, indent=2)
        logger.info(
            "Strategy recommendation written: focus=%s, proposals=%d",
            analysis.focus_area.value, len(analysis.parameter_proposals)
        )

    def _is_within_policy_bounds(self, param_name, proposed_value):
        """
        Validate LLM parameter proposal against watcher_policies.json bounds.

        Returns True if proposed value is within allowed range.
        Unknown parameters are rejected by default (whitelist approach).
        """
        allowed_params = self.config.get('training_diagnostics', {}).get(
            'llm_adjustable_params', {
                'normalize_features': {'type': 'bool'},
                'nn_activation': {'type': 'enum', 'values': [0, 1, 2]},
                'learning_rate': {'type': 'float', 'min': 1e-6, 'max': 0.1},
                'dropout': {'type': 'float', 'min': 0.0, 'max': 0.5},
                'n_estimators': {'type': 'int', 'min': 50, 'max': 2000},
                'max_depth': {'type': 'int', 'min': 3, 'max': 15},
            }
        )
        if param_name not in allowed_params:
            return False

        bounds = allowed_params[param_name]
        if bounds.get('type') == 'bool':
            return proposed_value in (0, 1, True, False)
        if bounds.get('type') == 'enum':
            return proposed_value in bounds.get('values', [])
        if 'min' in bounds and proposed_value < bounds['min']:
            return False
        if 'max' in bounds and proposed_value > bounds['max']:
            return False
        return True
```

### 7.5 WATCHER Summary — What Gets Wired

| Component | File | Lines Added | What It Does |
|-----------|------|-------------|-------------|
| `check_training_health()` | `agents/watcher_agent.py` | ~130 | Read diagnostics JSON, evaluate against policies, return action |
| `_check_skip_registry()` | `agents/watcher_agent.py` | ~35 | Track consecutive failures, decide RETRY vs SKIP_MODEL |
| `_archive_diagnostics()` | `agents/watcher_agent.py` | ~20 | Save each run to history/ for Strategy Advisor |
| `reset_skip_registry()` | `agents/watcher_agent.py` | ~15 | Reset failure counter when model succeeds |
| `_record_incident()` | `agents/watcher_agent.py` | ~15 | Append to watcher_failures.jsonl for Tier 2 retrieval |
| `_record_model_skip()` | `agents/watcher_agent.py` | ~25 | Write model_skip_state.json, expires after N hours |
| `_request_diagnostics_llm()` | `agents/watcher_agent.py` | ~35 | Trigger LLM analysis on CRITICAL/SKIP, archive result |
| `_update_strategy_advisor()` | `agents/watcher_agent.py` | ~30 | Write strategy_recommendation.json from LLM analysis |
| `_is_within_policy_bounds()` | `agents/watcher_agent.py` | ~25 | Whitelist validation for LLM parameter proposals |
| Policy entries | `watcher_policies.json` | ~50 | Thresholds for NN dead neurons, gradient spread, overfit ratio |
| Pipeline wiring | `agents/watcher_agent.py` | ~40 | Call health check between Step 5→6, handle RETRY/SKIP |

**Total: ~420 lines added to existing files. Zero new files for WATCHER integration.**

> **Corrected 2026-11-11 — row by row against `a15b524`.**
>
> | Component | What the code at `a15b524` has | Label |
> |---|---|---|
> | `check_training_health()`, `_check_skip_registry()`, `_archive_diagnostics()`, `reset_skip_registry()` | Exist, in **`training_health_check.py`** (`:80`, `:467`, `:539`, `:518`), not `agents/watcher_agent.py`; a new file was created | DESCRIPTION-ONLY (D4) |
> | `_record_incident()`, `_record_model_skip()`, `_request_diagnostics_llm()`, `_update_strategy_advisor()` | Not defined in any tracked module (`_record_model_skip` only in the unimported `watcher_integration_snippet.py:158`). No `watcher_failures.jsonl`, `model_skip_state.json` or `state/` exists on VM 101 | BROKEN (B8, B9) |
> | `_is_within_policy_bounds()` | Exists, `agents/watcher_agent.py:2318` (used at `:2484`); proposals validated, then dropped at the step boundary | accurate; effect held by S1 |
> | LLM on CRITICAL | `request_llm_diagnostics_analysis` (`diagnostics_llm_analyzer.py:308`) called on RETRY only (`agents/watcher_agent.py:2400-2432`); results archived to `diagnostics_outputs/llm_proposals/` (`diagnostics_llm_analyzer.py:470-484`; 44 files on VM 101), which no code reads | accurate as a call; output unconsumed |
> | Policy entries, pipeline wiring | Exist | accurate |
>
> **The skip registry, live (VM 101, 2026-11-11):** `diagnostics_outputs/model_skip_registry.json`
> holds `neural_net: consecutive_critical 1` (last critical 2026-03-08T03:07:51Z). The exclusion
> rule is `consecutive_critical >= 3` (`meta_prediction_optimizer_anti_overfit.py:174`, filter `:178`;
> `training_health_check.py:667-689` `is_model_skipped`), so **`neural_net` is not excluded
> now**. The file's `skip_active`, `consecutive_failures` and `reset_reason` keys are read by no
> code. The registry is a tracked file written at runtime; its committed history shows `neural_net`
> at or above 3 from S91 to S111, each drop below 3 being a hand reset (S90, S111, S121).

---

## 8. LLM Integration — DiagnosticsBundle & Grammar

> **Corrected 2026-11-11 — applies to all of §8.**
> - **Grammar and schema exist** (accurate): `grammars/diagnostics_analysis.gbnf` (not repo root)
>   and `diagnostics_analysis_schema.py` (Pydantic models from `:63`).
> - **§8.1 is SUPERSEDED (2026-11-11) — DESCRIPTION-ONLY, D5.** There is no `DiagnosticsBundle`
>   in `bundle_factory.py` (its only "diagnos" strings are at `:51` and `:141`, both unrelated). The
>   prompt is built by `build_diagnostics_prompt` in `diagnostics_llm_analyzer.py:112`, and the
>   call is `request_llm_diagnostics_analysis` (`:308`). The §8.1 text is kept as the original
>   design.
> - **§8.4 — the LLM sees almost none of the diagnostics — BROKEN, B7.** The prompt builder reads
>   `training_rounds`, `feature_importance.values` and `nn_specific.*`
>   (`diagnostics_llm_analyzer.py:183-191`); the producer writes `training_summary` and
>   `model_specific.*` (schema 1.1.0, §3.5 note). The LLM gets model type, severity and issue
>   strings only. The worked example in §8.4–§8.6 is built on the superseded schema and on
>   features that are trial-level or constant (D11, §15).
> - **§8.5 — when it is called:** on RETRY only (`agents/watcher_agent.py:2400-2432`), not on
>   every warning. Proposals are validated against the whitelist and then dropped at the step
>   boundary (S1). The tier-comparison input (`diagnostics_llm_analyzer.py:215-226`) is never
>   produced (B11), so it is always absent.

### 8.1 DiagnosticsBundle — New Bundle Type for bundle_factory.py

Follows the same pattern as the existing 7 bundle types (Steps 1-6 + Chapter 13).
The WATCHER builds this bundle and sends it to the LLM when diagnostics reveal
issues that need analytical interpretation.

```python
# ─── bundle_factory.py additions ──────────────────────────────────

# Add to STEP_MISSIONS dict:
DIAGNOSTICS_MISSION = (
    "Training Diagnostics Analyst: Evaluate training health data from Step 5 "
    "across up to 4 model types (neural_net, xgboost, lightgbm, catboost). "
    "Diagnose root causes of model underperformance. Classify into focus areas: "
    "MODEL_DIVERSITY, FEATURE_RELEVANCE, POOL_PRECISION, CONFIDENCE_CALIBRATION, "
    "REGIME_SHIFT. Recommend actionable parameter changes for selfplay exploration. "
    "Your recommendations are PROPOSALS — WATCHER validates against policy bounds. "
    "You do NOT have execution authority."
)

# Add to STEP_SCHEMA_EXCERPTS dict:
DIAGNOSTICS_SCHEMA_EXCERPT = (
    "DiagnosticsAnalysis: key_fields=[focus_area, root_cause, model_recommendations[], "
    "parameter_proposals[], confidence]. "
    "focus_area enum: MODEL_DIVERSITY, FEATURE_RELEVANCE, POOL_PRECISION, "
    "CONFIDENCE_CALIBRATION, REGIME_SHIFT. "
    "model_recommendations[]: per-model-type verdict (viable/fixable/skip). "
    "parameter_proposals[]: specific changes with rationale. "
    "confidence: 0.0-1.0 in your analysis."
)

# Add to STEP_GRAMMAR_NAMES dict:
DIAGNOSTICS_GRAMMAR = "diagnostics_analysis.gbnf"

# Add to STEP_GUARDRAILS dict:
DIAGNOSTICS_GUARDRAILS = [
    "You are analyzing training TELEMETRY, not making execution decisions.",
    "All parameter proposals must include specific numeric values, not vague directions.",
    "If multiple model types were diagnosed, compare them — do not analyze in isolation.",
    "Neural net dead neurons > 50% is ALWAYS critical — do not downplay.",
    "Feature gradient spread > 1000x indicates preprocessing failure, not model failure.",
    "Trees outperforming NN on tabular data is EXPECTED, not a defect.",
    "Your focus_area recommendation directly drives selfplay episode planning.",
]


def build_diagnostics_bundle(
    diagnostics_path: str,
    tier_comparison_path: str = None,
    history_paths: list = None,
    budgets: TokenBudget = None,
) -> StepAwarenessBundle:
    """
    Build an LLM context bundle from training diagnostics data.

    This is called by the Strategy Advisor or WATCHER when diagnostics
    reveal issues that need LLM interpretation.

    NOT called on every run — only when:
    1. WATCHER severity >= warning, OR
    2. Strategy Advisor scheduled analysis cycle, OR
    3. Chapter 13 requests root cause analysis after hit rate drop

    Args:
        diagnostics_path: Path to training_diagnostics.json
        tier_comparison_path: Path to tier_comparison.json (per-survivor attribution)
        history_paths: Paths to previous diagnostics in history/ dir
        budgets: Token budget override

    Returns:
        StepAwarenessBundle ready for render_prompt_from_bundle()
    """
    if budgets is None:
        budgets = TokenBudget()

    run_id = f"diag_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    # ── Load current diagnostics ────────────────────────────────────
    with open(diagnostics_path) as f:
        current_diag = json.load(f)

    # ── Build Tier 1: Current diagnostics summary ───────────────────
    inputs_summary = {
        'model_type': current_diag.get('model_type', 'unknown'),
        'severity': current_diag.get('diagnosis', {}).get('severity', 'unknown'),
        'issues': current_diag.get('diagnosis', {}).get('issues', []),
        'training_rounds_total': current_diag.get('training_rounds', {}).get('total', 0),
        'training_rounds_best': current_diag.get('training_rounds', {}).get('best', 0),
        'top_5_features': list(
            current_diag.get('feature_importance', {}).get('values', {}).keys()
        )[:5],
    }

    # Add NN-specific metrics if present
    nn_data = current_diag.get('nn_specific', {})
    if nn_data:
        inputs_summary['nn_layer_health'] = nn_data.get('layer_health', {})
        inputs_summary['nn_gradient_spread'] = nn_data.get('feature_gradient_spread', 0)

    # ── Build Tier 1: Tier comparison (if available) ────────────────
    evaluation_summary = {}
    if tier_comparison_path and os.path.isfile(tier_comparison_path):
        with open(tier_comparison_path) as f:
            tier_data = json.load(f)
        # Only include top divergent features to save tokens
        divergence = tier_data.get('divergence', {})
        sorted_div = sorted(divergence.items(), key=lambda x: abs(x[1]), reverse=True)
        evaluation_summary = {
            'tier_comparison_available': True,
            'top_divergent_features': [
                {'feature': name, 'divergence': round(val, 4)}
                for name, val in sorted_div[:10]
            ],
            'interpretation': (
                "Positive divergence = feature concentrated in top tier. "
                "Negative divergence = feature more important in wide tier. "
                "Large absolute values indicate structurally different populations."
            ),
        }

    # ── Build Tier 2: Historical diagnostics (trend detection) ──────
    recent_outcomes = []
    if history_paths:
        for hp in history_paths[-5:]:  # Last 5 runs
            if os.path.isfile(hp):
                with open(hp) as f:
                    hist = json.load(f)
                recent_outcomes.append(OutcomeRecord(
                    step=5,
                    run_id=os.path.basename(hp).replace('.json', ''),
                    result=hist.get('watcher_severity', 'unknown'),
                    metric_delta=0.0,
                    key_metric='severity',
                    timestamp=hist.get('archived_at', ''),
                ))

    # ── Assemble bundle ─────────────────────────────────────────────
    provenance = [ProvenanceRecord.from_file(diagnostics_path)]
    if tier_comparison_path and os.path.isfile(tier_comparison_path):
        provenance.append(ProvenanceRecord.from_file(tier_comparison_path))

    bundle_context = BundleContext(
        mission=DIAGNOSTICS_MISSION,
        schema_excerpt=DIAGNOSTICS_SCHEMA_EXCERPT,
        grammar_name=DIAGNOSTICS_GRAMMAR,
        contracts=list(AUTHORITY_CONTRACTS),
        guardrails=DIAGNOSTICS_GUARDRAILS,
        inputs_summary=inputs_summary,
        evaluation_summary=evaluation_summary,
        recent_outcomes=recent_outcomes,
    )

    bundle = StepAwarenessBundle(
        step_id=14,             # Chapter 14 diagnostics
        step_name="training_diagnostics_analysis",
        run_id=run_id,
        is_chapter_13=False,
        context=bundle_context,
        budgets=budgets,
        provenance=provenance,
    )

    logger.info(
        "Built diagnostics bundle: model=%s, severity=%s, "
        "has_tier_comparison=%s, history_entries=%d",
        inputs_summary['model_type'],
        inputs_summary['severity'],
        bool(evaluation_summary),
        len(recent_outcomes),
    )
    return bundle
```

### 8.2 GBNF Grammar: `diagnostics_analysis.gbnf`

Constrains the LLM to produce valid JSON matching the diagnostics analysis schema.
Follows the same pattern as `chapter_13.gbnf`.

```
# Training Diagnostics Analysis Grammar
# GBNF grammar for llama.cpp grammar-constrained decoding
#
# Constrains LLM output to valid DiagnosticsAnalysis JSON.
# Used when WATCHER or Strategy Advisor requests LLM interpretation
# of training_diagnostics.json data.
#
# VERSION: 1.0.0
# DATE: 2026-02-03
#
# Usage with llama.cpp:
#   --grammar-file diagnostics_analysis.gbnf
#
# Usage with LLMRouter:
#   router.evaluate_with_grammar(prompt, grammar_file="diagnostics_analysis.gbnf")

# Root rule
root ::= analysis

# Main analysis object
analysis ::= "{" ws
    "\"focus_area\"" ws ":" ws focus-area "," ws
    "\"root_cause\"" ws ":" ws string "," ws
    "\"root_cause_confidence\"" ws ":" ws confidence-value "," ws
    "\"model_recommendations\"" ws ":" ws model-recommendations "," ws
    "\"parameter_proposals\"" ws ":" ws parameter-proposals "," ws
    "\"selfplay_guidance\"" ws ":" ws string "," ws
    "\"requires_human_review\"" ws ":" ws boolean
    ws "}"

# Focus area enum — maps directly to Strategy Advisor focus areas
focus-area ::= "\"MODEL_DIVERSITY\"" |
               "\"FEATURE_RELEVANCE\"" |
               "\"POOL_PRECISION\"" |
               "\"CONFIDENCE_CALIBRATION\"" |
               "\"REGIME_SHIFT\""

# Model recommendations array (1-4 items, one per model type)
model-recommendations ::= "[" ws model-recommendation
                          (ws "," ws model-recommendation)* ws "]"

model-recommendation ::= "{" ws
    "\"model_type\"" ws ":" ws model-type "," ws
    "\"verdict\"" ws ":" ws model-verdict "," ws
    "\"rationale\"" ws ":" ws string
    ws "}"

model-type ::= "\"neural_net\"" |
               "\"xgboost\"" |
               "\"lightgbm\"" |
               "\"catboost\""

model-verdict ::= "\"viable\"" |
                  "\"fixable\"" |
                  "\"skip\"" |
                  "\"not_evaluated\""

# Parameter proposals (0-5 items)
parameter-proposals ::= "[]" |
                        "[" ws parameter-proposal
                        (ws "," ws parameter-proposal)* ws "]"

parameter-proposal ::= "{" ws
    "\"parameter\"" ws ":" ws string "," ws
    "\"current_value\"" ws ":" ws (number | "null") "," ws
    "\"proposed_value\"" ws ":" ws number "," ws
    "\"rationale\"" ws ":" ws string
    ws "}"

# Confidence value (0.0 to 1.0)
confidence-value ::= "0" ("." [0-9]+)? |
                     "1" (".0")?  |
                     "0." [0-9]+

# Boolean
boolean ::= "true" | "false"

# Number (integer or float)
number ::= "-"? [0-9]+ ("." [0-9]+)?

# String (JSON string)
string ::= "\"" string-content "\""
string-content ::= ([^"\\] | "\\" ["\\/bfnrt])*

# Whitespace
ws ::= [ \t\n\r]*
```

### 8.3 Pydantic Schema: `diagnostics_analysis_schema.py`

Mirrors the GBNF grammar for Python-side validation of LLM output.

```python
# ─── diagnostics_analysis_schema.py ───────────────────────────────

"""
Pydantic models for training diagnostics LLM analysis output.

Mirrors diagnostics_analysis.gbnf exactly.
Used to validate and parse LLM responses after grammar-constrained decoding.
"""

from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class FocusArea(str, Enum):
    MODEL_DIVERSITY = "MODEL_DIVERSITY"
    FEATURE_RELEVANCE = "FEATURE_RELEVANCE"
    POOL_PRECISION = "POOL_PRECISION"
    CONFIDENCE_CALIBRATION = "CONFIDENCE_CALIBRATION"
    REGIME_SHIFT = "REGIME_SHIFT"


class ModelType(str, Enum):
    NEURAL_NET = "neural_net"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"


class ModelVerdict(str, Enum):
    VIABLE = "viable"           # Model is working well, continue using
    FIXABLE = "fixable"         # Model has issues but diagnostics suggest specific fixes
    SKIP = "skip"               # Model is not suitable for this data
    NOT_EVALUATED = "not_evaluated"  # No diagnostics data for this model type


class ModelRecommendation(BaseModel):
    model_type: ModelType
    verdict: ModelVerdict
    rationale: str = Field(..., max_length=500)


class ParameterProposal(BaseModel):
    parameter: str = Field(..., max_length=100)
    current_value: Optional[float] = None
    proposed_value: float
    rationale: str = Field(..., max_length=300)


class DiagnosticsAnalysis(BaseModel):
    """
    LLM-produced analysis of training diagnostics.

    This is the OUTPUT of LLM inference, not the input.
    The LLM receives a DiagnosticsBundle prompt and produces
    this structured response via GBNF-constrained decoding.
    """
    focus_area: FocusArea
    root_cause: str = Field(..., max_length=500)
    root_cause_confidence: float = Field(..., ge=0.0, le=1.0)
    model_recommendations: List[ModelRecommendation] = Field(
        ..., min_length=1, max_length=4
    )
    parameter_proposals: List[ParameterProposal] = Field(
        default_factory=list, max_length=5
    )
    selfplay_guidance: str = Field(..., max_length=500)
    requires_human_review: bool = False

    @field_validator('model_recommendations')
    @classmethod
    def validate_unique_models(cls, v):
        model_types = [r.model_type for r in v]
        if len(model_types) != len(set(model_types)):
            raise ValueError("Duplicate model types in recommendations")
        return v
```

### 8.4 LLM Prompt Template

What the rendered bundle actually looks like when sent to DeepSeek-R1-14B:

```
=== STEP MISSION ===
Training Diagnostics Analyst: Evaluate training health data from Step 5
across up to 4 model types (neural_net, xgboost, lightgbm, catboost).
Diagnose root causes of model underperformance. Classify into focus areas:
MODEL_DIVERSITY, FEATURE_RELEVANCE, POOL_PRECISION, CONFIDENCE_CALIBRATION,
REGIME_SHIFT. Recommend actionable parameter changes for selfplay exploration.
Your recommendations are PROPOSALS — WATCHER validates against policy bounds.
You do NOT have execution authority.

=== EVALUATION SCHEMA ===
DiagnosticsAnalysis: key_fields=[focus_area, root_cause, model_recommendations[],
parameter_proposals[], confidence].
focus_area enum: MODEL_DIVERSITY, FEATURE_RELEVANCE, POOL_PRECISION,
CONFIDENCE_CALIBRATION, REGIME_SHIFT.
model_recommendations[]: per-model-type verdict (viable/fixable/skip).
parameter_proposals[]: specific changes with rationale.
confidence: 0.0-1.0 in your analysis.

=== OUTPUT FORMAT ===
Respond using grammar: diagnostics_analysis.gbnf
Your output MUST conform to this grammar exactly.

=== GUARDRAILS ===
- You are analyzing training TELEMETRY, not making execution decisions.
- All parameter proposals must include specific numeric values, not vague directions.
- If multiple model types were diagnosed, compare them — do not analyze in isolation.
- Neural net dead neurons > 50% is ALWAYS critical — do not downplay.
- Feature gradient spread > 1000x indicates preprocessing failure, not model failure.
- Trees outperforming NN on tabular data is EXPECTED, not a defect.
- Your focus_area recommendation directly drives selfplay episode planning.

=== AUTHORITY CONTRACTS ===
Active contracts: CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md
You cannot override these. Propose within their bounds only.

=== STEP OUTPUT METRICS ===
{
  "model_type": "neural_net",
  "severity": "critical",
  "issues": ["47% dead neurons in fc1", "Feature gradient spread 12847x"],
  "training_rounds_total": 200,
  "training_rounds_best": 23,
  "top_5_features": ["intersection_weight", "skip_entropy", "lane_agreement_8",
                      "temporal_stability", "forward_count"],
  "nn_layer_health": {
    "fc1": {"dead_pct": 47, "gradient_norm": 0.0003},
    "fc2": {"dead_pct": 12, "gradient_norm": 0.012},
    "fc3": {"dead_pct": 3, "gradient_norm": 0.45}
  },
  "nn_gradient_spread": 12847
}

=== CURRENT EVALUATION ===
{
  "tier_comparison_available": true,
  "top_divergent_features": [
    {"feature": "intersection_weight", "divergence": 0.0891},
    {"feature": "skip_entropy", "divergence": 0.0634},
    {"feature": "forward_count", "divergence": -0.0412}
  ],
  "interpretation": "Positive divergence = feature concentrated in top tier..."
}
```

### 8.5 End-to-End LLM Call

How the WATCHER or Strategy Advisor actually calls the LLM with diagnostics:

```python
# ─── Called from Strategy Advisor or WATCHER when severity >= warning ─

from bundle_factory import (
    build_diagnostics_bundle, render_prompt_from_bundle, TokenBudget
)
from llm_router import LLMRouter
from diagnostics_analysis_schema import DiagnosticsAnalysis
import json
import glob

def request_llm_diagnostics_analysis(diagnostics_path, tier_comparison_path=None):
    """
    Full end-to-end LLM diagnostics analysis.

    1. Build bundle (assemble context with token budgets)
    2. Render prompt (tier-ordered sections)
    3. Call LLM with GBNF grammar constraint
    4. Parse and validate response with Pydantic
    5. Return structured DiagnosticsAnalysis

    Returns:
        DiagnosticsAnalysis or None if LLM call fails
    """
    # Gather history files for trend context
    history_paths = sorted(glob.glob("diagnostics_outputs/history/*.json"))

    # Step 1: Build bundle
    bundle = build_diagnostics_bundle(
        diagnostics_path=diagnostics_path,
        tier_comparison_path=tier_comparison_path,
        history_paths=history_paths,
        budgets=TokenBudget(ctx_max_tokens=32768),  # DeepSeek-R1-14B context
    )

    # Step 2: Render prompt
    prompt = render_prompt_from_bundle(bundle)

    # Step 3: Call LLM with grammar constraint
    router = LLMRouter()
    raw_response = router.evaluate_with_grammar(
        prompt=prompt,
        grammar_file="diagnostics_analysis.gbnf",
        max_tokens=2000,
    )

    if not raw_response:
        logger.error("LLM returned empty response for diagnostics analysis")
        return None

    # Step 4: Parse with Pydantic
    try:
        # Clean any markdown fencing the LLM might add despite grammar
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]

        analysis = DiagnosticsAnalysis.model_validate_json(cleaned)
        logger.info(
            "LLM diagnostics analysis: focus=%s, root_cause_confidence=%.2f, "
            "models=%d recommendations, params=%d proposals",
            analysis.focus_area.value,
            analysis.root_cause_confidence,
            len(analysis.model_recommendations),
            len(analysis.parameter_proposals),
        )
        return analysis

    except Exception as e:
        logger.error("Failed to parse LLM diagnostics response: %s", e)
        logger.debug("Raw response: %s", raw_response[:500])
        return None
```

### 8.6 Example LLM Response (Grammar-Constrained)

What DeepSeek-R1-14B produces given the prompt above, constrained by `diagnostics_analysis.gbnf`:

```json
{
    "focus_area": "FEATURE_RELEVANCE",
    "root_cause": "Neural net feature gradient spread of 12847x indicates severe input scaling imbalance. forward_count (range 0-50000) dominates gradient computation while normalized features (0-1 range) are effectively invisible. Dead neurons in fc1 (47%) are a secondary effect — fc1 receives unscaled inputs and ReLU kills neurons that see negative values from the dominant feature. Tree models are unaffected because decision splits are scale-invariant.",
    "root_cause_confidence": 0.85,
    "model_recommendations": [
        {
            "model_type": "neural_net",
            "verdict": "fixable",
            "rationale": "Add BatchNorm after input layer or StandardScaler preprocessing. Replace ReLU with LeakyReLU. Expected to reduce MSE by 2-3 orders of magnitude."
        },
        {
            "model_type": "catboost",
            "verdict": "viable",
            "rationale": "CatBoost MSE 1.77e-9 is excellent. No issues detected. Continue as primary model."
        },
        {
            "model_type": "xgboost",
            "verdict": "viable",
            "rationale": "XGBoost MSE 9.32e-9 is strong. Gain importance well distributed."
        },
        {
            "model_type": "lightgbm",
            "verdict": "viable",
            "rationale": "LightGBM MSE 1.06e-8 is competitive. Split/gain ratio healthy."
        }
    ],
    "parameter_proposals": [
        {
            "parameter": "normalize_features",
            "current_value": null,
            "proposed_value": 1,
            "rationale": "Add StandardScaler to preprocessing pipeline — zero mean, unit variance"
        },
        {
            "parameter": "nn_activation",
            "current_value": null,
            "proposed_value": 0,
            "rationale": "Switch from ReLU (current) to LeakyReLU(0.01) to prevent dead neurons"
        }
    ],
    "selfplay_guidance": "Next selfplay episode should focus FEATURE_RELEVANCE: test normalized vs unnormalized features with catboost as baseline. If normalization improves neural_net to within 10x of catboost MSE, add neural_net per-survivor attribution to pool precision strategy.",
    "requires_human_review": false
}
```

### 8.7 LLM Integration Summary

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| `DIAGNOSTICS_MISSION` | `bundle_factory.py` | ~10 | Mission statement for diagnostics LLM context |
| `DIAGNOSTICS_GUARDRAILS` | `bundle_factory.py` | ~10 | Safety rails for diagnostics analysis |
| `build_diagnostics_bundle()` | `bundle_factory.py` | ~100 | Assemble token-budgeted context from diagnostics JSON |
| `diagnostics_analysis.gbnf` | new file | ~70 | GBNF grammar constraining LLM output |
| `diagnostics_analysis_schema.py` | new file | ~80 | Pydantic models mirroring GBNF grammar |
| `request_llm_diagnostics_analysis()` | `strategy_advisor.py` or inline | ~50 | End-to-end LLM call with parsing |

**Total: ~320 lines across existing + 2 new files.**

---

## 9. Selfplay & Live Feedback Wiring

### 9.1 Selfplay Episode Diagnostics

> **Corrected 2026-11-11 (Task 8.1).**
> - **Code exists** (accurate): `train_single_fold` fits with `eval_set`
>   (`inner_episode_trainer.py:271-293`), builds a `TrainingDiagnostics` and feeds rounds
>   (`:315-376`); `InnerEpisodeTrainer.train` collects them (`:549-598`).
> - **Names (DESCRIPTION-ONLY, D13):** the sample's `train_episode_model` / `run_inner_episodes`
>   / config key `episode_diagnostics` are `train_single_fold` / `InnerEpisodeTrainer` /
>   `SelfplayOrchestrator.run` / `SelfplayConfig.enable_diagnostics`.
> - **Off on the host.** `SelfplayConfig.enable_diagnostics` defaults to `False`
>   (`selfplay_orchestrator.py:213`); there is no CLI flag, WATCHER's selfplay dispatch passes none
>   (`agents/watcher_dispatch.py:141-147`), and the live `configs/selfplay_config.json` on VM 101
>   (identical to the tracked copy) has no such key. Episode diagnostics are not produced.
> - **Tree losses:** XGBoost records the validation loss as the training loss (`:339`); LightGBM
>   falls back to the same because only a validation set is passed (`:273-278`, `:325-328`). Their
>   overfit gap is always 0; only CatBoost folds can leave `ok`.
> - **Different model from Step 5:** selfplay trains on `holdout_hits` (`selfplay_orchestrator.py:933`)
>   with the record's whole `features` dict (`:924-925`), not on `holdout_quality` with 89 columns.

`inner_episode_trainer.py` trains models during selfplay episodes. Diagnostics
can capture per-episode training health without changing the trainer's interface.

```python
# ─── inner_episode_trainer.py additions ───────────────────────────

from training_diagnostics import TreeDiagnostics

def train_episode_model(survivors, features, config, enable_diagnostics=False):
    """
    Train a model for one selfplay episode.
    Existing function — add diagnostics capture at the end.
    """
    model_type = config.get('model_type', 'catboost')

    # ... existing training code ...

    # ── NEW: Capture episode diagnostics if enabled ─────────────────
    episode_diag = None
    if enable_diagnostics and hasattr(model, 'evals_result'):
        diag = TreeDiagnostics(model_type=model_type)
        diag.capture_from_model(model, feature_names)
        diag.analyze()
        episode_diag = diag.to_dict()

        # Don't write to disk for every episode — return for orchestrator to handle
        logger.debug(
            "Episode diagnostics: model=%s severity=%s best_round=%d/%d",
            model_type,
            episode_diag.get('diagnosis', {}).get('severity', 'unknown'),
            episode_diag.get('training_rounds', {}).get('best', 0),
            episode_diag.get('training_rounds', {}).get('total', 0),
        )

    return {
        'model': model,
        'fitness': fitness_score,
        'metrics': metrics,
        'diagnostics': episode_diag,  # None if diagnostics disabled
    }
```

### 9.2 Selfplay Orchestrator Consumption

> **Corrected 2026-11-11 (Tasks 8.2, 8.3).**
> - **Code exists** (accurate): `EpisodeResult.episode_diagnostics` (`selfplay_orchestrator.py:268`),
>   history capped at 20 (`:523`), `_check_episode_training_trend` at `:826` — in
>   `selfplay_orchestrator.py`, not `chapter_13_orchestrator.py` as the tracker says
>   (`CHAPTER_13_IMPLEMENTATION_PROGRESS_v3_9.md:81`; DESCRIPTION-ONLY, D10). Production rule:
>   three strictly declining ratios with the last < 0.2 (`:840`).
> - **Observe-only — SWITCHED-OFF-BY-RULING, S8:** "Observe-only — no automatic intervention"
>   (`SESSION_CHANGELOG_20260213_S83.md:14`, `:46`; docstring `:829`). A session decision; no
>   separate TB ruling text was found. **Lifting it would enable** a degrading trend to stop or
>   adjust a selfplay run; it would need episode diagnostics switched on (§9.1), B12, and the caller
>   using the return value (it discards it, `:524`).
> - **The incident is never recorded — BROKEN, B12:** the trend branch calls
>   `self.telemetry.record_event(...)` (`:845`); `LearningTelemetry` has no such method
>   (`modules/learning_telemetry.py` defines `record_inner_episode` `:299`,
>   `record_policy_emission` `:376`, `record_promotion` `:420`), and the `AttributeError` is
>   swallowed (`:852-853`). Present since `87c5bf1` (S83).
> - **Strategy Advisor feed:** not passed to telemetry or the Advisor (`inner_episode_trainer.py:581-592`).

The orchestrator collects episode diagnostics and uses them for two purposes:
(a) detect degrading training quality across episodes, and (b) feed the Strategy
Advisor with episode-level telemetry.

```python
# ─── selfplay_orchestrator.py additions ───────────────────────────

def run_inner_episodes(self, survivors, config):
    """
    Run N inner episodes. Existing method — add diagnostics aggregation.
    """
    episode_results = []
    diagnostics_history = []

    for episode_num in range(config.get('inner_episodes', 10)):
        result = train_episode_model(
            survivors=self._apply_policy_transforms(survivors),
            features=self.feature_matrix,
            config=config,
            enable_diagnostics=config.get('episode_diagnostics', False),
        )
        episode_results.append(result)

        # ── NEW: Track episode diagnostics ──────────────────────────
        if result.get('diagnostics'):
            diagnostics_history.append({
                'episode': episode_num,
                'severity': result['diagnostics'].get(
                    'diagnosis', {}).get('severity', 'ok'),
                'best_round_ratio': (
                    result['diagnostics'].get('training_rounds', {}).get('best', 0) /
                    max(result['diagnostics'].get(
                        'training_rounds', {}).get('total', 1), 1)
                ),
                'fitness': result['fitness'],
            })
            # Cap history — trend detection only needs recent window, not full archive
            diagnostics_history = diagnostics_history[-20:]

    # ── NEW: Detect degrading training quality ──────────────────────
    if diagnostics_history:
        self._check_episode_training_trend(diagnostics_history)

    return episode_results


def _check_episode_training_trend(self, diagnostics_history):
    """
    Detect if training quality is degrading across episodes.

    Pattern: If last 3 episodes have worsening best_round_ratio,
    training is degrading and we should stop early or adjust.
    """
    if len(diagnostics_history) < 3:
        return

    recent = diagnostics_history[-3:]
    ratios = [d['best_round_ratio'] for d in recent]

    # All three declining — training quality degrading
    if ratios[0] > ratios[1] > ratios[2] and ratios[2] < 0.2:
        logger.warning(
            "Episode training quality degrading: "
            "best_round_ratios=[%.2f, %.2f, %.2f] — "
            "possible data quality issue or parameter exhaustion",
            *ratios
        )
        # Record as incident for WATCHER
        self.telemetry.record_event(
            event_type='training_quality_degrading',
            details={
                'recent_ratios': ratios,
                'recent_severities': [d['severity'] for d in recent],
            },
        )

    # Count critical episodes
    critical_count = sum(1 for d in diagnostics_history if d['severity'] == 'critical')
    if critical_count >= len(diagnostics_history) * 0.5:
        logger.error(
            "Over 50%% of episodes have critical training issues (%d/%d)",
            critical_count, len(diagnostics_history)
        )
```

### 9.3 Chapter 13 Consumption — Post-Draw Root Cause Analysis

> **Corrected 2026-11-11 (Task 8.4).** `post_draw_root_cause_analysis()` exists and implements
> the hit/miss top-3 divergence rule below (`chapter_13_orchestrator.py:652-810`, rule `:690-762`)
> and is invoked from `run_cycle` (`:321-347`). Its governed status is **"Implemented, invoked,
> unreachable, unconsumed"** (`TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:105`). In call
> order it stops at:
>
> | # | stop | anchor | label |
> |---|---|---|---|
> | 1 | The gate tests `summary_flags` containing "hit_rate"+"drop" or `hit_at_20 < previous_hit_at_20`; no producer emits either | `c13:546-557`; `chapter_13_diagnostics.py:585-668`, `:855-870` | BROKEN, B14 (status governed: "has never fired in production", `SESSION_CHANGELOG_20260217_S95.md:178-180`) |
> | 2 | No writer for `predictions/ranked_predictions.json`; absent on VM 101 | reader `c13:876-937` | BROKEN, B13 (Blocker 1, `…SYSTEM_MAP…:91`) |
> | 3 | Reads `meta["feature_names"]`; Step 5 writes it under `feature_schema` | `c13:582`; `mpo:1293`, `:2903` | BROKEN, B13 (Blocker 2, `:93`) |
> | 4 | `torch.load(...)` then `.eval()` on a checkpoint dict | `c13:596-598`; `tst:708`, `:725` | BROKEN, B13 (Blocker 3, `:95`) |
> | 5 | NN attribution omits the training scaler | `psa:98-101`; `tst:484-492` | BROKEN, B13 (Blocker 4, `:97`) |
> | 6 | On `regime_shift` calls `StrategyAdvisor.request_diagnostics_analysis`, which is defined nowhere; exception caught | `c13:842`, `:854-855` | BROKEN, B15 (no governance record found) |
>
> Repair order is governed as P0-C (`…SYSTEM_MAP…:255-256`); a one-line patch is ruled
> insufficient (`:272`). The classification is **observe-only — SWITCHED-OFF-BY-RULING, S3**
> ("Recommended observe-only before granting trigger authority", `SESSION_CHANGELOG_20260214_S85.md:85`,
> `:95`; code comment `c13:321-323`). **Lifting S3 would enable** a `regime_shift` result to raise
> a retrain trigger; it would also need S2 lifted, a trigger type in `chapter_13_triggers.py`, and
> the multi-draw evidence conditions the ruling lists. Attribution-driven autonomous parameter
> changes are separately **"Not authorized"** (`…SYSTEM_MAP…:277`; S4). Chapter 13's triggers do
> not read training diagnostics at all (`chapter_13_triggers.py:174`, `:216`); the one reference to
> `training_diagnostics.json` is inside the unreachable branch (`c13:837`).

When Chapter 13 detects a hit rate drop after a new draw, it can request diagnostics
analysis to determine if the problem is model-related or data-related.

```python
# ─── chapter_13_orchestrator.py additions ─────────────────────────

from per_survivor_attribution import per_survivor_attribution, compare_pool_tiers

def post_draw_root_cause_analysis(self, draw_result, predictions, model, model_type):
    """
    Called when Chapter 13 detects Hit@20 dropped after a new draw.

    Uses per-survivor attribution to determine if:
    A) Top survivors relied on features that are now stale (regime shift)
    B) Model training was poor (training diagnostics issue)
    C) Random variance (wait for more draws)

    Args:
        draw_result: Actual draw outcome
        predictions: Model predictions that were evaluated
        model: Trained model from last Step 5
        model_type: str model type identifier
    """
    # ── Step 1: Get attribution for Top 20 survivors that MISSED ────
    missed_top = [p for p in predictions[:20] if not p['hit']]

    if not missed_top:
        return  # All top 20 hit — no analysis needed

    missed_attributions = []
    for survivor in missed_top:
        attr = per_survivor_attribution(
            model=model,
            model_type=model_type,
            features=survivor['features'],
            feature_names=self.feature_names,
        )
        missed_attributions.append({
            'seed': survivor['seed'],
            'rank': survivor['rank'],
            'top_3_features': sorted(
                attr.items(), key=lambda x: x[1], reverse=True
            )[:3],
        })

    # ── Step 2: Compare with Top 20 that HIT ────────────────────────
    hit_top = [p for p in predictions[:20] if p['hit']]

    hit_attributions = []
    for survivor in hit_top:
        attr = per_survivor_attribution(
            model=model,
            model_type=model_type,
            features=survivor['features'],
            feature_names=self.feature_names,
        )
        hit_attributions.append({
            'seed': survivor['seed'],
            'rank': survivor['rank'],
            'top_3_features': sorted(
                attr.items(), key=lambda x: x[1], reverse=True
            )[:3],
        })

    # ── Step 3: Classify the miss pattern ───────────────────────────
    # If missed survivors relied on DIFFERENT features than hits → regime shift
    # If missed survivors relied on SAME features → random variance
    missed_features = set()
    for m in missed_attributions:
        for feat_name, _ in m['top_3_features']:
            missed_features.add(feat_name)

    hit_features = set()
    for h in hit_attributions:
        for feat_name, _ in h['top_3_features']:
            hit_features.add(feat_name)

    overlap = missed_features & hit_features
    divergence_ratio = 1.0 - (len(overlap) / max(len(missed_features | hit_features), 1))

    analysis = {
        'type': 'post_draw_root_cause',
        'draw_id': draw_result.get('draw_id'),
        'missed_count': len(missed_top),
        'hit_count': len(hit_top),
        'feature_divergence_ratio': divergence_ratio,
        'missed_relied_on': list(missed_features),
        'hits_relied_on': list(hit_features),
        'diagnosis': 'regime_shift' if divergence_ratio > 0.5 else 'random_variance',
        'missed_details': missed_attributions[:5],
        'hit_details': hit_attributions[:5],
    }

    logger.info(
        "Post-draw root cause: divergence=%.2f diagnosis=%s "
        "missed_features=%s hit_features=%s",
        divergence_ratio,
        analysis['diagnosis'],
        missed_features,
        hit_features,
    )

    # ── Step 4: If regime shift detected, trigger LLM analysis ──────
    if analysis['diagnosis'] == 'regime_shift':
        logger.warning("REGIME SHIFT detected — requesting LLM diagnostics analysis")

        # Write tier comparison for LLM context
        tier_comparison = compare_pool_tiers(
            model, model_type,
            predictions,
            self.feature_names,
        )
        tier_path = "diagnostics_outputs/tier_comparison.json"
        os.makedirs("diagnostics_outputs", exist_ok=True)
        with open(tier_path, 'w') as f:
            json.dump(tier_comparison, f, indent=2)

        # Request LLM analysis (if diagnostics file exists)
        diag_path = "diagnostics_outputs/training_diagnostics.json"
        if os.path.isfile(diag_path):
            from strategy_advisor import request_llm_diagnostics_analysis
            llm_analysis = request_llm_diagnostics_analysis(
                diagnostics_path=diag_path,
                tier_comparison_path=tier_path,
            )
            if llm_analysis:
                analysis['llm_analysis'] = llm_analysis.model_dump()

    # Archive analysis
    self._archive_post_draw_analysis(analysis)
    return analysis
```

### 9.4 Data Flow Diagram — Complete Automation Loop

> **SUPERSEDED (2026-11-11) as a description of the running system.** The diagram below is the
> design. What the code at `a15b524` actually does, arrow by arrow (audit Q7):
>
> ```
> Step 5 (WATCHER manifest enable_diagnostics: true, agent_manifests/reinforcement.json:179)
>   ├── NN final model only → diagnostics_outputs/neural_net_diagnostics.json
>   │     → copied to training_diagnostics.json (train_single_trial.py:101-107)   [trees: none, B3]
>   ├── tier_comparison.json                                     NOT written by Step 5 (B11)
>   ▼
> WATCHER check_training_health() (agents/watcher_agent.py:2866)
>   ├── early-stop key mismatch → every report critical (B1)
>   ├── critical, counter < 3 → RETRY: canonical file deleted, Step 5 re-run unchanged (S1)
>   │       └── LLM analysis requested; proposals archived to llm_proposals/, no reader
>   ├── critical, counter ≥ 3 → SKIP_MODEL: next --compare-models drops that family
>   │       (meta_prediction_optimizer_anti_overfit.py:174-178); no 24 h expiry (B8)
>   ├── every check archived → diagnostics_outputs/history/ (training_health_check.py:539-569)
>   │       read only by the LLM prompt builder (diagnostics_llm_analyzer.py:368); never pruned (B10)
>   └── watcher_failures.jsonl, strategy_recommendation bridge         NOT implemented (B9)
>   ▼
> Step 6 — does not read training diagnostics
>   ▼
> Chapter 13 — triggers do not read training diagnostics; RCA unreachable (§9.3: B13–B15, S2, S3)
>   ▼
> Strategy Advisor — reads diagnostics_history/ (parameter_advisor.py:210), not
>   diagnostics_outputs/history/
>   ▼
> Selfplay — enable_diagnostics False by default and on VM 101 (§9.1); trend check observe-only
>   and its incident call fails (S8, B12)
> ```
>
> The only automatic effects of training diagnostics on the pipeline are the RETRY loop and the
> skip registry; model selection itself uses each sidecar's score (`mpo:94-139`, `:288-296`).

```
Step 5 training (--enable-diagnostics)
    │
    ├── training_diagnostics.json written
    │   (loss curves, feature importance, NN health, diagnosis)
    │
    ├── tier_comparison.json written
    │   (per-survivor attribution across pool tiers)
    │
    ▼
WATCHER: check_training_health()
    │
    ├── severity=ok → PROCEED → reset_skip_registry → Step 6
    │
    ├── severity=warning → PROCEED_WITH_NOTE
    │   │
    │   ├── Archive to diagnostics_outputs/history/
    │   ├── Record incident in watcher_failures.jsonl
    │   ├── [Optional] request_llm_diagnostics_analysis()
    │   │       ├── build_diagnostics_bundle()
    │   │       ├── render_prompt_from_bundle()
    │   │       ├── LLM call with diagnostics_analysis.gbnf
    │   │       └── Parse DiagnosticsAnalysis response
    │   └── Proceed to Step 6
    │
    ├── severity=critical → RETRY or SKIP_MODEL
    │   │
    │   ├── Check _check_skip_registry()
    │   │   ├── consecutive_critical < 3 → RETRY with modified params
    │   │   └── consecutive_critical >= 3 → SKIP_MODEL for 24 hours
    │   │
    │   ├── ALWAYS request_llm_diagnostics_analysis()
    │   │       └── LLM produces DiagnosticsAnalysis with fix recommendations
    │   │
    │   └── Feed LLM analysis to Strategy Advisor for next selfplay cycle
    │
    ▼
Step 6 (Prediction Generator)
    │
    ▼
Chapter 13 (Live Feedback — new draw arrives)
    │
    ├── Evaluate predictions vs actual
    │
    ├── If Hit@20 drops → post_draw_root_cause_analysis()
    │       ├── Per-survivor attribution for missed vs hit seeds
    │       ├── Feature divergence ratio calculation
    │       ├── If regime_shift → request LLM analysis
    │       └── Feed results to Strategy Advisor
    │
    ▼
Strategy Advisor (next cycle)
    │
    ├── Reads diagnostics_outputs/history/*.json
    ├── Reads LLM DiagnosticsAnalysis if available
    ├── Produces strategy_recommendation.json
    │   └── focus_area drives selfplay episode planning
    │
    ▼
Selfplay (next episode)
    │
    ├── Applies Strategy Advisor focus area
    ├── enable_diagnostics=True for episode training
    ├── _check_episode_training_trend() across episodes
    └── Emits candidate for Chapter 13 review
```

---

## 10. TensorBoard Automation Boundary

### 10.1 Explicit Decision: TensorBoard is Human-Only

> **Note 2026-11-11:** the boundary stands as a design decision, but TensorBoard and
> `agents/tensorboard_extractor.py` do not exist at `a15b524` (no such file; S9, §6 note). §10.3's
> rows for "Root cause analysis — LLM — DiagnosticsBundle" and "Per-survivor attribution — on
> demand" are subject to D5 and S2 respectively.

TensorBoard is a research UI designed for human investigation sessions.
The WATCHER and LLM **cannot** interact with the TensorBoard web UI.

However, TensorBoard stores data in a parseable format (`tfevents` files)
that we CAN read programmatically when needed.

### 10.2 What the WATCHER Can Extract from TensorBoard Logs

TensorBoard writes `tfevents` files to `runs/`. These are protobuf files
readable with `tensorboard.backend.event_processing.event_accumulator`:

```python
# ─── agents/tensorboard_extractor.py (OPTIONAL utility) ──────────

"""
Extract scalar data from TensorBoard tfevents files.

NOT part of the main automation loop. Utility for post-hoc analysis
when a human wants to compare across many runs programmatically.

The WATCHER uses training_diagnostics.json (our own format).
This module is for HUMAN-TRIGGERED deep dives only.
"""

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import json


def extract_scalars_from_run(run_dir):
    """
    Read all scalar data from a TensorBoard run directory.

    Args:
        run_dir: e.g., 'runs/neural_net_study_20260210_143000'

    Returns:
        dict mapping tag_name → list of (step, value) tuples
    """
    ea = EventAccumulator(run_dir)
    ea.Reload()

    scalars = {}
    for tag in ea.Tags().get('scalars', []):
        events = ea.Scalars(tag)
        scalars[tag] = [(e.step, e.value) for e in events]

    return scalars


def compare_runs(runs_dir='runs/', metric='Loss/val'):
    """
    Compare a specific metric across all TensorBoard runs.

    Returns:
        dict mapping run_name → final_value for the metric
    """
    comparison = {}
    for run_name in os.listdir(runs_dir):
        run_path = os.path.join(runs_dir, run_name)
        if not os.path.isdir(run_path):
            continue
        try:
            scalars = extract_scalars_from_run(run_path)
            if metric in scalars and scalars[metric]:
                final_value = scalars[metric][-1][1]  # Last recorded value
                comparison[run_name] = final_value
        except Exception:
            continue

    return comparison
```

### 10.3 Boundary Summary

| Task | Who Does It | Tool |
|------|-------------|------|
| Post-Step-5 health check | WATCHER (automated) | `training_diagnostics.json` |
| Root cause analysis | LLM (automated, triggered) | `DiagnosticsBundle` + GBNF |
| Per-survivor attribution | Code (automated, on demand) | `per_survivor_attribution.py` |
| Weight histogram inspection | Human (manual) | TensorBoard UI |
| Gradient distribution analysis | Human (manual) | TensorBoard UI |
| Cross-run comparison | Human (manual, script-assisted) | `tensorboard_extractor.py` |
| Model graph visualization | Human (manual) | TensorBoard UI |

**Rule: If the WATCHER needs data, it reads `training_diagnostics.json`.
It never reads TensorBoard files. TensorBoard is a human debugging tool.**

---

## 11. Neural Net Evaluation & Repurposing Path

> **Corrected 2026-11-11.** Phase B fixes were applied through other proposals, not through
> diagnostics output: production forces input normalisation and LeakyReLU for the NN
> (`meta_prediction_optimizer_anti_overfit.py:245-246`; `PROPOSAL_CATEGORY_B_NN_TRAINING_ENHANCEMENTS_v1_0.md:27`
> says "Phase A (Diagnose) is complete"), and S121 z-scored `y` (`train_single_trial.py:502-510`).
> The 9.32e-4 baseline in Phase C predates the `holdout_quality` target and is not comparable.
> Phase D's "information no tree model can" is D17 (§1.2 note). Phase A's diagnosis inputs are
> limited: the NN never captures per-feature gradients (B18) and the dead-neuron metric cannot see
> activation deaths (D12).

If diagnostics reveal fixable issues (not fundamentally wrong architecture):

```
Phase A: Diagnose (this chapter)
    → Run --enable-diagnostics on neural_net
    → Read diagnosis: feature scaling? dead neurons? vanishing gradients?

Phase B: Fix (informed by diagnosis)
    → If scaling: add BatchNorm to first layer (already in Optuna search space)
    → If dead neurons: switch ReLU → LeakyReLU(0.01) or GELU
    → If vanishing gradients: add residual connections or reduce depth to 2 layers
    → These are config/architecture changes, not code rewrites

Phase C: Validate
    → Re-run --compare-models --enable-diagnostics
    → Check if NN MSE improved from 9.32e-4
    → If competitive with trees → gain unique per-survivor gradient attribution
    → If still far behind → trees are the right tool, use NN only for attribution

Phase D: Deploy Attribution
    → Even if NN isn't the best predictor, its per-sample gradients provide
      information no tree model can: continuous-valued feature influence per seed
    → Trees give binary splits; NN gives smooth gradients
    → Use NN attribution to INFORM tree model training strategy
```

---

## 12. File Inventory

### 12.1 New Files

> **Corrected 2026-11-11 — as-built inventory at `a15b524` (line counts by `wc -l`).** New files
> that exist: `training_diagnostics.py` (1,068), `per_survivor_attribution.py` (502),
> `grammars/diagnostics_analysis.gbnf`, `diagnostics_analysis_schema.py`, plus two not planned
> here: `training_health_check.py` (827 — where §7's functions actually live) and
> `diagnostics_llm_analyzer.py` (662 — where §8's bundle/prompt actually lives). Not created:
> `agents/tensorboard_extractor.py`. Modified as planned: `agents/watcher_agent.py`,
> `watcher_policies.json`, `inner_episode_trainer.py`, `selfplay_orchestrator.py`,
> `chapter_13_orchestrator.py`, `meta_prediction_optimizer_anti_overfit.py` (`--enable-diagnostics`
> only), and `train_single_trial.py` (the real NN hook site, not listed). Not modified for
> diagnostics: `bundle_factory.py` (D5), `web_dashboard.py` (S9); the three tree wrappers are not
> the production path (B3). `reinforcement_engine.py` has hooks on a path Step 5 does not use (D2).

| File | Est. Lines | Purpose |
|------|-----------|---------|
| `training_diagnostics.py` | ~400 | Core diagnostics module — hooks, callbacks, analysis, diagnosis |
| `per_survivor_attribution.py` | ~200 | Unified per-seed attribution across all 4 model types |
| `diagnostics_analysis.gbnf` | ~70 | GBNF grammar for LLM diagnostics analysis output |
| `diagnostics_analysis_schema.py` | ~80 | Pydantic models mirroring GBNF grammar |
| `agents/tensorboard_extractor.py` | ~60 | Optional utility for human-triggered run comparison |

### 12.2 Modified Files

| File | Est. Lines Added | Change |
|------|-----------------|--------|
| `agents/watcher_agent.py` | ~420 | `check_training_health()`, skip registry, pipeline wiring, LLM trigger, incident recording |
| `watcher_policies.json` | ~50 | Training diagnostics policy entries + thresholds |
| `bundle_factory.py` | ~120 | `DIAGNOSTICS_MISSION`, guardrails, `build_diagnostics_bundle()` |
| `reinforcement_engine.py` | ~25 | Wire NN hooks into training loop |
| `models/wrappers/xgboost_wrapper.py` | ~15 | Capture evals_result + importance types |
| `models/wrappers/lightgbm_wrapper.py` | ~15 | Capture eval_results + split/gain |
| `models/wrappers/catboost_wrapper.py` | ~15 | Capture evals_result + PredictionValuesChange |
| `meta_prediction_optimizer_anti_overfit.py` | ~10 | `--enable-diagnostics`, `--enable-tensorboard` CLI flags |
| `web_dashboard.py` | ~100 | New `/training` route + 5 chart functions + nav tab |
| `inner_episode_trainer.py` | ~25 | Optional episode diagnostics capture |
| `selfplay_orchestrator.py` | ~50 | Episode diagnostics aggregation + trend detection |
| `chapter_13_orchestrator.py` | ~100 | `post_draw_root_cause_analysis()` with per-survivor attribution |

### 12.3 New Config Block

> **SUPERSEDED (2026-11-11) — DESCRIPTION-ONLY, D7 and D18.** There is no
> `reinforcement_engine_config.json` on VM 101 (searched the working tree; gitignored-file listing
> includes none). What exists: `ReinforcementConfig.diagnostics` defaults in code
> (`reinforcement_engine.py:272-276`, on the non-Step-5 path), a separate
> `SelfplayConfig.enable_diagnostics = False` (`selfplay_orchestrator.py:213`), and the Step-5
> manifest default **`enable_diagnostics: true`** (`agent_manifests/reinforcement.json:179`, set
> in S90) — so under WATCHER, diagnostics are **on** by default, not off. `capture_every_n`
> defaults to 5 in code (`training_diagnostics.py:100`; accurate). No code reads
> `nn_attribution_method` or `top_survivors_for_attribution`; the `tensorboard` keys have no
> consumer (S9). `agent_manifests/definitions.json` (gitignored, read on VM 101) declares no
> diagnostics configuration either. The block below is kept as the original design.

```json
// In reinforcement_engine_config.json
{
    "diagnostics": {
        "enabled": false,
        "tensorboard": false,
        "capture_every_n": 5,
        "nn_attribution_method": "grad_x_input",
        "output_dir": "diagnostics_outputs",
        "tensorboard_dir": "runs",
        "top_survivors_for_attribution": 5,
        "episode_diagnostics": false
    }
}
```

**Config notes:**
- `capture_every_n`: Controls hook snapshot frequency. Default 5 keeps JSON compact for
  typical 200-round tree models (~40 snapshots) and 100-epoch NNs (~20 snapshots). Set to
  1 only for targeted short-run investigation sessions.
- `nn_attribution_method`: `"grad_x_input"` (default) is more stable with differently-scaled
  features. `"grad"` is raw gradient magnitude — use for debugging feature scaling issues.

### 12.4 New Directories

> **Corrected 2026-11-11 — what is actually on VM 101 under `diagnostics_outputs/`** (listed
> this session; all gitignored by `.gitignore:41` `*.json` except the skip registry, which is
> tracked): `neural_net_diagnostics.json` (the producer file; the canonical
> `training_diagnostics.json` is **absent** — deleted by WATCHER before the last RETRY,
> `agents/watcher_agent.py:2897-2901`); `model_skip_registry.json`; `history/` (111 files: 104
> `neural_net_*`, 4 `catboost_*`, 7 `root_cause_*` — no `llm_analysis_*`); `llm_proposals/` (44
> files — LLM output goes here, not into `history/`); 60 `compare_models_summary_S88_*.json`;
> `holdout_feature_autocorr.json`; `regression_trigger_s87.json`; `soak_report_real.json`.
> **Never written by any tracked code:** `tier_comparison.json` by Step 5 (B11),
> `model_skip_state.json` (B8), `state/watcher_failures.jsonl` and
> `state/strategy_recommendation.json` (B9 — a `strategy_recommendation.json` at the repo root,
> dated 2026-02-07, comes from the Strategy Advisor). `runs/` does not exist (S9).

```
diagnostics_outputs/              ← auto-created when diagnostics enabled
├── training_diagnostics.json     ← unified output (all model types)
├── tier_comparison.json          ← per-survivor attribution across pool tiers
├── model_skip_registry.json      ← WATCHER consecutive failure tracking
├── model_skip_state.json         ← active model type skips (expires after N hours)
└── history/                      ← archived diagnostics per run
    ├── neural_net_20260210_143000.json
    ├── catboost_20260210_143500.json
    ├── llm_analysis_20260210_144000.json
    └── ...

state/                            ← existing WATCHER state directory
├── watcher_failures.jsonl        ← incident log (Tier 2 retrieval source)
└── strategy_recommendation.json  ← LLM analysis → Strategy Advisor bridge

runs/                             ← TensorBoard log directory (optional)
├── neural_net_study_20260210/
├── xgboost_study_20260210/
├── catboost_study_20260210/
└── lightgbm_study_20260210/
```

### 12.5 Dependencies

```
# Already installed (pip_list.txt confirms):
torch                  # PyTorch — hooks, autograd
xgboost                # eval_set, pred_contribs
lightgbm               # record_evaluation, pred_contrib
catboost               # get_evals_result, ShapValues
plotly                  # Dashboard charts
pydantic               # Schema validation

# May need:
tensorboard            # pip install tensorboard --break-system-packages
                       # Only needed if --enable-tensorboard used
```

### 12.6 Line Count Summary

| Category | Estimated Lines |
|----------|----------------|
| New files (5 files) | ~810 |
| Modified files (12 files) | ~945 |
| **Total new/modified** | **~1,755** |

---

## 13. Implementation Plan

### 13.1 Prerequisites

- ✅ Soak Test B PASSED (10/10)
- ⬜ Soak Test A (idle endurance) — COMPLETE FIRST
- ⬜ Soak Test C (full autonomous loop) — COMPLETE FIRST
- ⬜ No code changes until soak tests pass (Team Beta directive)

> **SUPERSEDED (2026-11-11) — DESCRIPTION-ONLY, D9.** §14 marks Soak A and C ✅ at S76/S77,
> after Phases 1, 3, 6 had already landed in S69–S73 (`git log`: `51e74b7` Phase 1 and `881ec3b`
> Phase 6, both 2026-02-08). Under the 2026-11-11 owner rulings
> (`docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md` §0) a Team Beta "directive" is a
> recorded review position, not a binding gate. The §13.2–§13.10 phase plan below is kept as the
> original plan; its status lives in §14 and §16.

### 13.2 Phase 1: Core Diagnostics Module (~2 hours)

**Goal:** Build `training_diagnostics.py` with unified interface for all 4 model types.

| Task | Details |
|------|---------|
| 1.1 | Create `training_diagnostics.py` with base `TrainingDiagnostics` class |
| 1.2 | Implement `NNDiagnostics` — PyTorch hook registration, per-epoch capture |
| 1.3 | Implement `TreeDiagnostics` — wraps eval_result extraction for XGB/LGB/CB |
| 1.4 | Implement `_analyze()` — loss plateau, gradient flow, dead neurons, feature scale |
| 1.5 | Implement `_diagnose()` — severity classification, issue detection, fix suggestions |
| 1.6 | Implement `save()` — write unified JSON schema |
| 1.7 | Unit test: create small NN, train 10 epochs, verify diagnostics JSON |

### 13.3 Phase 2: Per-Survivor Attribution (~1 hour)

**Goal:** Build `per_survivor_attribution.py` with all 4 model type backends.

| Task | Details |
|------|---------|
| 2.1 | Create `per_survivor_attribution.py` |
| 2.2 | Implement NN gradient attribution |
| 2.3 | Implement XGBoost pred_contribs |
| 2.4 | Implement LightGBM pred_contrib |
| 2.5 | Implement CatBoost ShapValues |
| 2.6 | Implement unified interface + pool tier comparison |
| 2.7 | Unit test: train catboost, get attribution for top 3 seeds |

### 13.4 Phase 3: Wire into Training Pipeline (~1 hour)

**Goal:** Add `--enable-diagnostics` flag and wire into existing training code.

| Task | Details |
|------|---------|
| 3.1 | Add CLI flags to `meta_prediction_optimizer_anti_overfit.py` |
| 3.2 | Wire NN hooks into `reinforcement_engine.py` epoch loop |
| 3.3 | Wire eval_result capture into each wrapper's `train()` method |
| 3.4 | Add diagnostics config block to `reinforcement_engine_config.json` |
| 3.5 | Test: `--model-type neural_net --trials 1 --enable-diagnostics` |
| 3.6 | Test: `--model-type catboost --trials 1 --enable-diagnostics` |
| 3.7 | Verify both produce valid `training_diagnostics.json` |

### 13.5 Phase 4: Web Dashboard (~1.5 hours)

**Goal:** Add `/training` route with 5 charts + diagnosis panel.

| Task | Details |
|------|---------|
| 4.1 | Add "Training" tab to dashboard navigation |
| 4.2 | Implement `/training` route |
| 4.3 | Implement `chart_loss_curves()` |
| 4.4 | Implement `chart_feature_importance()` |
| 4.5 | Implement `chart_survivor_attribution()` |
| 4.6 | Implement `chart_nn_health()` |
| 4.7 | Implement `chart_diagnosis_panel()` |
| 4.8 | Test: run dashboard, verify all charts render |

### 13.6 Phase 5: TensorBoard (Optional, ~30 min)

**Goal:** Add `--enable-tensorboard` flag for deep investigation sessions.

| Task | Details |
|------|---------|
| 5.1 | Install tensorboard on Zeus |
| 5.2 | Add SummaryWriter creation gated on config flag |
| 5.3 | Wire add_scalars for all model types |
| 5.4 | Wire add_histogram for NN weights/gradients |
| 5.5 | Wire add_graph for NN model visualization |
| 5.6 | Test: launch TensorBoard, verify data appears |

### 13.7 Phase 6: WATCHER Integration (~1.5 hours)

**Goal:** Wire diagnostics into autonomous WATCHER pipeline.

| Task | Details |
|------|---------|
| 6.1 | Add policy entries to `watcher_policies.json` |
| 6.2 | Implement `check_training_health()` in `watcher_agent.py` |
| 6.3 | Implement `_check_skip_registry()` + `reset_skip_registry()` |
| 6.4 | Implement `_archive_diagnostics()` |
| 6.5 | Wire health check into pipeline between Step 5 → Step 6 |
| 6.6 | Test: run pipeline with severity=ok, verify PROCEED |
| 6.7 | Test: inject critical diagnostics, verify RETRY action |
| 6.8 | Test: inject 3 consecutive criticals, verify SKIP_MODEL |

### 13.8 Phase 7: LLM Integration (~2 hours)

**Goal:** Build DiagnosticsBundle + GBNF grammar + end-to-end LLM analysis.

| Task | Details |
|------|---------|
| 7.1 | Create `diagnostics_analysis.gbnf` |
| 7.2 | Create `diagnostics_analysis_schema.py` |
| 7.3 | Add `DIAGNOSTICS_MISSION`, `DIAGNOSTICS_GUARDRAILS` to `bundle_factory.py` |
| 7.4 | Implement `build_diagnostics_bundle()` in `bundle_factory.py` |
| 7.5 | Implement `request_llm_diagnostics_analysis()` |
| 7.6 | Test: build bundle from real diagnostics JSON, verify prompt renders correctly |
| 7.7 | Test: call DeepSeek-R1-14B with grammar, verify valid JSON response |
| 7.8 | Test: parse response with Pydantic, verify all fields validate |

### 13.9 Phase 8: Selfplay + Chapter 13 Wiring (~1.5 hours)

**Goal:** Wire diagnostics into selfplay episodes and Chapter 13 post-draw analysis.

| Task | Details |
|------|---------|
| 8.1 | Add episode diagnostics to `inner_episode_trainer.py` |
| 8.2 | Add episode aggregation to `selfplay_orchestrator.py` |
| 8.3 | Implement `_check_episode_training_trend()` |
| 8.4 | Implement `post_draw_root_cause_analysis()` in `chapter_13_orchestrator.py` |
| 8.5 | Test: run 10 selfplay episodes, verify diagnostics collected |
| 8.6 | Test: inject declining best_round_ratio, verify warning logged |
| 8.7 | Test: simulate hit rate drop, verify root cause analysis runs |

### 13.10 Phase 9: First Diagnostic Investigation (~1 hour)

**Goal:** Run real diagnostics on Zeus and diagnose neural_net performance.

| Task | Details |
|------|---------|
| 9.1 | Run `--compare-models --enable-diagnostics` with real survivor data |
| 9.2 | Read `training_diagnostics.json` — identify root cause |
| 9.3 | View dashboard `/training` — verify charts match raw data |
| 9.4 | Document findings: is it scaling? dead neurons? architecture? |
| 9.5 | If fixable: plan Phase B fix (BatchNorm, LeakyReLU, etc.) |
| 9.6 | If not fixable: document NN as attribution-only tool |
| 9.7 | Run `request_llm_diagnostics_analysis()` — verify LLM agrees with manual analysis |

**Total estimated time: ~11 hours across 9 phases**

---

## 14. Implementation Checklist

> **Corrected 2026-11-11 — ✅ marks below that the code at `a15b524` does not bear out.** The
> marks are kept as historical record; read them with these corrections.
>
> | item | mark | what the code does | label |
> |---|---|---|---|
> | 1.3 TreeDiagnostics | ✅ | Exists; `XGBDiagnostics`/`LGBDiagnostics` empty subclasses (`training_diagnostics.py:683-690`) | B4 |
> | 2.1–2.7 attribution | ✅ | Module exists; no production run | S2 |
> | 3.1 `--enable-tensorboard` | ✅ | Flag does not exist; only `--enable-diagnostics` (`mpo:3080`, `tst:864`) | D15 |
> | 3.2 hooks in `reinforcement_engine.py` | ✅ | Production hooks are in `train_single_trial.py:607-660` | D2 |
> | 3.3 eval_result capture in wrappers | ✅ | Tree final models get no diagnostics on the production path | B3 |
> | 3.4 config block | ✅ | No `reinforcement_engine_config.json` on VM 101 | D7 |
> | 3.5 / 3.7 NN test, JSON output | ✅ | Real NN reports exist on VM 101 (schema 1.1.0) | accurate |
> | 6.6 severity=ok → PROCEED | ✅ | No real report can grade `ok` while B1 stands; 0 of 107 archived checks did | B1 |
> | 6.8 3× critical → SKIP_MODEL | ✅ | Works (`fc1cf2b`, S91); no expiry | accurate; B8 |
> | 7.3–7.4 bundle_factory | ✅ | No diagnostics code in `bundle_factory.py`; prompt built in `diagnostics_llm_analyzer.py:112` | D5 |
> | 8.1–8.3 | ✅ | Code exists; off by default and on VM 101; trend incident call fails | §9.1–9.2; B12 |
> | 8.4 | 🔧 | Implemented, invoked, unreachable | B13–B15 |
> | 8.5–8.7 | ⬜ (tracker: "validated") | Harness `test_phase_8_soak.py` — synthetic 8.5/8.6 assert on literals and call no production code (`:137-196`) | B17 |
>
> The tracker's "All Tasks 8.1-8.7 validated" (`CHAPTER_13_IMPLEMENTATION_PROGRESS_v3_9.md:12`,
> `:72`) and "FIFO History Pruning — Complete" (`:68`) are also not borne out: `_prune_history_fifo`
> is called only from `MultiModelDiagnostics.save` (`training_diagnostics.py:834`) and a self-test
> (`:1060`); nothing instantiates `MultiModelDiagnostics`, and the live archive writer
> (`training_health_check.py:562`) does not prune (B10; 111 files on VM 101). Phases 4, 5 and 9
> remain ⬜ and are tracked in `docs/TODO_CONSOLIDATED.md` (rows for Chapter 14 Phases 4, 5, 8
> remainder, 9 and task 3.6).

| # | Task | Phase | Status |
|---|------|-------|--------|
| | **PREREQUISITES** | | |
| P.1 | Soak Test A (idle endurance) complete | Pre | ✅ S76 |
| P.2 | Soak Test C (full autonomous loop) complete | Pre | ✅ S77 |
| P.3 | Team Beta approval to begin Chapter 14 | Pre | ✅ S69 |
| | **PHASE 1: Core Diagnostics** | | |
| 1.1 | Create `training_diagnostics.py` — base class | 1 | ✅ S69 |
| 1.2 | NNDiagnostics — PyTorch hooks | 1 | ✅ S69 |
| 1.3 | TreeDiagnostics — eval_result wrappers | 1 | ✅ S69 |
| 1.4 | Analysis engine (plateau, gradient flow, dead neurons) | 1 | ✅ S69 |
| 1.5 | Diagnosis engine (severity, issues, fixes) | 1 | ✅ S69 |
| 1.6 | JSON save/load | 1 | ✅ S69 |
| 1.7 | Unit test | 1 | ✅ S69 |
| | **PHASE 2: Per-Survivor Attribution** | | |
| 2.1 | Create `per_survivor_attribution.py` | 2 | ✅ S84 |
| 2.2 | NN gradient attribution | 2 | ✅ S84 |
| 2.3 | XGBoost pred_contribs | 2 | ✅ S84 |
| 2.4 | LightGBM pred_contrib | 2 | ✅ S84 |
| 2.5 | CatBoost ShapValues | 2 | ✅ S84 |
| 2.6 | Unified interface + pool tier comparison | 2 | ✅ S84 |
| 2.7 | Unit test | 2 | ✅ S84 |
| | **PHASE 3: Pipeline Wiring** | | |
| 3.1 | CLI flags (--enable-diagnostics, --enable-tensorboard) | 3 | ✅ S73 |
| 3.2 | Wire NN hooks into reinforcement_engine.py | 3 | ✅ S73 |
| 3.3 | Wire eval_result capture into wrappers | 3 | ✅ S73 |
| 3.4 | Config block in reinforcement_engine_config.json | 3 | ✅ S73 |
| 3.5 | Test: neural_net with diagnostics | 3 | ✅ S73 |
| 3.6 | Test: catboost with diagnostics | 3 | ⬜ |
| 3.7 | Verify JSON output | 3 | ✅ S73 |
| | **PHASE 4: Web Dashboard** | | |
| 4.1 | Navigation tab | 4 | ⬜ |
| 4.2 | /training route | 4 | ⬜ |
| 4.3 | Chart: Loss curves | 4 | ⬜ |
| 4.4 | Chart: Feature importance | 4 | ⬜ |
| 4.5 | Chart: Survivor attribution | 4 | ⬜ |
| 4.6 | Chart: NN health (dead neurons + gradients) | 4 | ⬜ |
| 4.7 | Chart: Diagnosis panel | 4 | ⬜ |
| 4.8 | Dashboard integration test | 4 | ⬜ |
| | **PHASE 5: TensorBoard (Optional)** | | |
| 5.1 | Install tensorboard on Zeus | 5 | ⬜ |
| 5.2 | SummaryWriter creation (gated on config) | 5 | ⬜ |
| 5.3 | add_scalars for all model types | 5 | ⬜ |
| 5.4 | add_histogram for NN | 5 | ⬜ |
| 5.5 | add_graph for NN | 5 | ⬜ |
| 5.6 | TensorBoard launch test | 5 | ⬜ |
| | **PHASE 6: WATCHER Integration** | | |
| 6.1 | Policy entries in watcher_policies.json | 6 | ✅ S72 |
| 6.2 | `check_training_health()` | 6 | ✅ S72 |
| 6.3 | `_check_skip_registry()` + `reset_skip_registry()` | 6 | ✅ S72 |
| 6.4 | `_archive_diagnostics()` | 6 | ✅ S71 |
| 6.5 | Pipeline wiring (Step 5 → health check → Step 6) | 6 | ✅ S73 |
| 6.6 | Test: severity=ok → PROCEED | 6 | ✅ S73 |
| 6.7 | Test: severity=critical → RETRY | 6 | ✅ S73 |
| 6.8 | Test: 3x critical → SKIP_MODEL | 6 | ✅ S73 |
| | **PHASE 7: LLM Integration** | | |
| 7.1 | Create `diagnostics_analysis.gbnf` | 7 | ✅ S75 |
| 7.2 | Create `diagnostics_analysis_schema.py` | 7 | ✅ S75 |
| 7.3 | Add mission + guardrails to bundle_factory.py | 7 | ✅ S75 |
| 7.4 | Implement `build_diagnostics_bundle()` | 7 | ✅ S75 |
| 7.5 | Implement `request_llm_diagnostics_analysis()` | 7 | ✅ S75 |
| 7.6 | Test: bundle renders correct prompt | 7 | ✅ S75 |
| 7.7 | Test: LLM + grammar → valid JSON | 7 | ✅ S75 |
| 7.8 | Test: Pydantic parses LLM response | 7 | ✅ S75 |
| | **PHASE 8: Selfplay + Chapter 13 Wiring** | | |
| 8.1 | Episode diagnostics in inner_episode_trainer.py | 8 | ✅ S83 |
| 8.2 | Episode aggregation in selfplay_orchestrator.py | 8 | ✅ S83 |
| 8.3 | `_check_episode_training_trend()` | 8 | ✅ S83 |
| 8.4 | `post_draw_root_cause_analysis()` in chapter_13 | 8 | 🔧 S84 |
| 8.5 | Test: 10 episodes with diagnostics | 8 | ⬜ |
| 8.6 | Test: declining trend detection | 8 | ⬜ |
| 8.7 | Test: hit rate drop → root cause analysis | 8 | ⬜ |
| | **PHASE 9: First Investigation** | | |
| 9.1 | Run --compare-models --enable-diagnostics on Zeus | 9 | ⬜ |
| 9.2 | Analyze training_diagnostics.json | 9 | ⬜ |
| 9.3 | View /training dashboard | 9 | ⬜ |
| 9.4 | Document root cause of NN underperformance | 9 | ⬜ |
| 9.5 | Plan fix (if fixable) or document as attribution-only | 9 | ⬜ |
| 9.6 | Verify LLM analysis matches manual analysis | 9 | ⬜ |

---

## Version History

```
Version 1.0.0 — February 3, 2026
    - Initial chapter creation
    - All 4 capabilities documented with code for all 4 model types
    - Integration points with Strategy Advisor, Selfplay, Chapter 13
    - Phased implementation plan (~7 hours total)
    - Deferred until Soak Tests A, B, C complete

Version 1.1.0 — February 3, 2026
    - Added Section 7: WATCHER Integration with check_training_health(),
      skip registry, policy entries, and pipeline dispatch wiring
    - Added Section 8: LLM Integration with DiagnosticsBundle,
      diagnostics_analysis.gbnf, Pydantic schema, end-to-end LLM call
    - Added Section 9: Selfplay episode diagnostics, Chapter 13
      post_draw_root_cause_analysis(), complete data flow diagram
    - Added Section 10: TensorBoard automation boundary (human-only)
    - Expanded implementation plan from 6 phases to 9 phases (~11 hours total)
    - Added 23 new checklist items for Phases 6-8

Version 1.1.1 — February 4, 2026
    - Fixed header version (was still showing 1.0.0)
    - Added _record_incident() — appends to watcher_failures.jsonl
    - Added _record_model_skip() — writes model_skip_state.json with expiry
    - Added _request_diagnostics_llm() — triggers LLM call on CRITICAL/SKIP
    - Added _update_strategy_advisor() — bridges LLM analysis to Strategy Advisor
    - Added _is_within_policy_bounds() — whitelist validation for LLM proposals
    - Updated _build_retry_params() to apply LLM parameter proposals
    - All 4 previously identified gaps now filled with sample code
    - Total estimated new/modified code: ~1,755 lines across 17 files

Version 1.1.2 — February 4, 2026 (Team Alpha review recommendations)
    - Rec 1: Added grad_x_input attribution method (|x * ∂y/∂x|) as default
      for NN per-survivor attribution. Config: nn_attribution_method flag.
    - Rec 2: Set capture_every_n default to 5 (was 1). Added throttle guard
      code and documentation note on snapshot frequency trade-offs.
    - Rec 3: Added Section 2.3 Design Invariant: "Diagnostics Are Best-Effort
      and Non-Fatal" — explicit policy statement preventing future regressions.
    - Rec 4: Added diagnostics_history[-20:] cap in selfplay episode loop
      to prevent unbounded memory growth during long selfplay runs.
    - Rec 5: Renamed Section 11 "Neural Net Rehabilitation Path" →
      "Neural Net Evaluation & Repurposing Path" — aligns with actual strategy
      where NN may correctly remain attribution-only.

Version 1.3.0 — November 11, 2026 (correction pass, base a15b524)
    - Authority: docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md REV1
    - Added the TFM framing block, correction-pass status header,
      numbering statement and "How this chapter was produced"
    - Inline "Corrected 2026-11-11" notes and SUPERSEDED markers in
      §1-§14 and the appendices; no design text deleted
    - Added §15 (what the diagnostics see under RANGE-MINER / 22-array
      NPZ / window anchor), §16 Known gaps at a15b524, §17 Open items,
      §18 VIR-1…6, §19 Closure statement
    - Host evidence from VM 101 recorded (real reports carry
      best_val_round; neural_net not currently skipped)
```

---

**END OF CHAPTER 14** *(design text, v1.2.0. The session appendices follow, then the 2026-11-11
correction sections §15–§19.)*

---

## Session 73 Bugfix: Subprocess Sidecar Consistency

### Issue Discovered
When using `--compare-models` (subprocess isolation), Step 5 wrote a degenerate sidecar despite successful training. Root cause: code checked `self.best_model` (in-memory) instead of disk artifacts.

### Fix Applied (Team Beta v1.3)
- **Principle:** Disk artifacts are authoritative, not parent process memory
- **Implementation:** Check `best_checkpoint_path` before declaring degenerate
- **Invariant:** `prediction_allowed=True` ⇒ `checkpoint_path` must exist

### Verification
```
model_type: lightgbm ✅
checkpoint_path: models/reinforcement/best_model.txt ✅
outcome: SUCCESS ✅
```

### Files
- `meta_prediction_optimizer_anti_overfit.py` - Core fix
- `apply_team_beta_sidecar_fix_v1.3.py` - Patcher script

### Commit
`f391786` - fix(step5): honor subprocess-trained checkpoints when writing sidecar

---

## ML Target Update: holdout_quality (S111)

### Problem

holdout_hits as ML target produced R2=0.000155 on real CA Daily 3 data -- zero signal.

### Fix (S111)

holdout_quality (composite score) replaced holdout_hits.
CatBoost R2 immediately rose to +0.0046 on clean real data baseline.

### NN Y-Normalization Fix (S121)

train_single_trial.py line 499: added y_mean/y_std normalization with sidecar .json
for Step 6 inverse-transform. First positive NN R2: +0.020538.

> **Corrected 2026-11-11 (DESCRIPTION-ONLY, D14):** at `a15b524` the normalisation is at
> `train_single_trial.py:502-510`. Consequence for diagnostics: NN losses in the report are in
> z-scored units, tree losses in raw `holdout_quality` units, so loss values are not comparable
> across families. The S121 fix did not change the health-check grade: the two checks after it
> (2026-03-07T05:32Z and 2026-03-08T03:07Z, `diagnostics_outputs/history/`) are critical and
> carry "Early stop ratio 0.00" (B1).

### Feature Signal on Real Data (CatBoost)

| Feature Group | Importance | Notes |
|---------------|-----------|-------|
| mod-8 residue | ~32% | LCG low-bit algebraic structure |
| Prediction residuals | ~20% | pred_std, residual_abs_mean |
| mod-125 | <10% | |
| mod-1000 | <10% | |
| 24/62 base features | 0% | Battery Tier 1A added to address gap |

> **Note 2026-11-11 (DESCRIPTION-ONLY, D1; audit §7 item 1):** "62" was the base count before
> Battery Tier 1A (+23, `survivor_scorer.py:754-766`) and the S119 digit features (+4,
> `:618-621`); at `a15b524` Step 5 trains 89 of 91 columns. The importance shares above were
> measured on PWC-produced survivors before RANGE-MINER; they are history, not a current reading.
> Of the 89 columns, 36 carry one value across a single-generation survivor set (§15).

### Battery Tier 1A (S113) -- 23 columns added

F1 Spectral FFT (5): batt_fft_peak_mag, batt_fft_secondary_peak, batt_fft_spectral_conc,
  batt_fft_diff_peak, batt_fft_diff_conc
F5 Autocorrelation (12): batt_ac_lag_01..10, batt_ac_decay_rate, batt_ac_sig_lag_count
F7 Cumulative Sum (3): batt_cs_max_excursion, batt_cs_mean_excursion, batt_cs_zero_crossings
F6 Bit Frequency (3): batt_bf_hamming_mean, batt_bf_hamming_std, batt_bf_popcount_bias

### Z10xZ10xZ10 Digit Transition Features (S119) -- 4 columns

Full utilization requires Z10xZ10xZ10 kernel in sieve_gpu_worker.py -- TB proposal needed first.

> **Note 2026-11-11:** the four S119 columns (`hundreds_digit_agreement`, `tens_digit_agreement`,
> `ones_digit_agreement`, `expected_digit_match_count`) are computed in Step 3 from the seed's
> regenerated sequence (`survivor_scorer.py:618-621`) and are among the 53 per-seed columns (§15).

---

## 15. What the diagnostics actually see at a15b524 — RANGE-MINER, the 22-array NPZ, the window anchor

*(Added 2026-11-11. Sources: `docs/CHAPTER_14_AUDIT.md` §3–§5 and §7, anchors re-verified on
VM 101 this session.)*

**The v1.2.0 text above mentions none of RANGE-MINER, the 22-array NPZ contract, or the
window-anchor change.** None of them changed a Chapter 14 code path — `training_diagnostics.py`,
`training_health_check.py`, `per_survivor_attribution.py` and `diagnostics_llm_analyzer.py` read
whatever Step 5 trains on. What they changed is **what Step 5 trains on**, and therefore what every
importance, gradient and attribution in this chapter can mean.

### 15.1 What enters the model

Step 5 trains **89** columns: the 91 per-survivor features minus `score` and `confidence`
(`meta_prediction_optimizer_anti_overfit.py:1482`; names taken from the first record's sorted
keys, `:566-567`). Grouped by where each value comes from (audit §3):

| group | n | what it is | anchor |
|---|---|---|---|
| A. per-seed, from Step 3's regeneration of the seed's sequence (`skip=0`) compared with the training history | 53 | match/residual stats, CRT lanes, S119 digits, temporal stability, residue, Battery Tier 1A | `survivor_scorer.py:575`, `:592-690`, `:754-766` |
| B. **trial-level**, identical for every seed of a trial | 13 | `skip_min`, `skip_max`, `skip_range`, `forward_count`, `reverse_count`, `bidirectional_count`, `intersection_count`, `intersection_ratio`, `forward_only_count`, `reverse_only_count`, `survivor_overlap_ratio`, `bidirectional_selectivity`, `intersection_weight` | built once in `metadata_base`, `utils/canonical_records.py:214-243`; copied into every record `:246-257` |
| C. **run-level**, identical for every survivor | 17 | 14 `global_*`; `total_predictions`, `actual_mean`, `actual_std` | `full_scoring_worker.py:347-349`; `survivor_scorer.py:706`, `:716-717` |
| D. constant zero | 6 | `skip_mean`, `skip_std`, `skip_entropy`, `survivor_velocity`, `velocity_acceleration` (no producer); `best_offset` (hardcoded) | `survivor_scorer.py:784-791`; `:707` |

In a single-generation survivor set **36 of the 89 columns carry one value** (B + C + D); the NN
scaler maps each to exactly 0 after centring (`train_single_trial.py:488-490`).

### 15.2 What RANGE-MINER and the 22-array contract changed

- **The per-seed sieve evidence never reaches the model.** Of the canonical record's fields, only
  `seed`, `forward_match_rate`, `reverse_match_rate` and `score` are set per seed
  (`utils/canonical_records.py:248-253`), and none of the four is a model input: Step 3's merge
  list names 18 fields, all trial-level or placeholders (`survivor_scorer.py:774-779`). The NPZ
  arrays `forward_matches` / `reverse_matches` are therefore invisible to every diagnostic in this
  chapter. **SWITCHED-OFF-BY-RULING, S5:** "do NOT silently add them… Open a separately governed
  feature-schema decision" (`TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:167-171`, `:276`).
  Lifting it would let the model and every diagnostic see per-seed sieve match quality for the
  first time; it would need a feature-schema version, Step 3 merge-list changes in both paths,
  regenerated artifacts, retraining and Step 6 acceptance of the new schema hash.
- **The trial-level fields are what the old examples called per-seed drivers.** Under the canonical
  record `forward_count` is `len(fwd_map)` and `intersection_weight` is
  `len(both) / (len(fwd)+len(rev))` — one value per trial (`utils/canonical_records.py:234`,
  `:243`). §1.1, §2.2, §4.2 and §8.6 examples (D11) treat them as explaining one seed against
  another; they cannot.
- **`skip_entropy` is zero.** The 22 arrays carry no skip-shape statistic; the column is defaulted
  to 0.0 (`survivor_scorer.py:784-791`). Giving it values is **SWITCHED-OFF-BY-RULING, S6** ("They
  must not begin receiving nonzero values under the existing 89-feature schema",
  `…SYSTEM_MAP…:199-200`; "Expanding RANGE-MINER beyond 22 arrays — Not authorized", `:268`).
- **The empirical numbers in this chapter predate all three changes** — the December 2025 MSE
  table, the "95K survivors" scale, the S111 importance table — and were measured on
  PWC-produced survivors.

### 15.3 What the window-anchor change (Brief I) changed

Canonical array 4 keeps the wire name `offset` but now carries **the window anchor and never the
generator phase** (`utils/canonical_records.py:216-224`, comment and `"offset": ctx["window_anchor"]`;
commits `48a8705`, `69ca910`). `offset` is not a model input (not in the merge list), so no
diagnostic changes. A reader who meets **`best_offset`** in the feature list should know it is a
hardcoded zero (`survivor_scorer.py:707`) unrelated to array 4. Final acceptance of Brief I is
pending a completed production-shape run; `G-PROD-SHAPE` ran on 2026-08-22 and failed
(`docs/S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md:352`).

### 15.4 Consequence for reading any Chapter 14 output

No file among `training_diagnostics.py`, `per_survivor_attribution.py`, `training_health_check.py`
and `diagnostics_llm_analyzer.py` distinguishes groups B–D from group A (audit Q5: no namespace,
`global_` or constant-column test). An importance list, attribution share or top-3 feature set that
names a group B–D column is **contextual model dependence, not evidence distinguishing survivors**
(`…SYSTEM_MAP…:191-197`). The one policy bound that could surface groups C–D,
`unused_feature_pct`, is evaluated nowhere (B16).

---

## 16. Known gaps at a15b524

*(Added 2026-11-11.)* Each item carries exactly one label. "Governed" means already recorded in the
governance trail — listed as status, not as a new finding. Full evidence: `docs/CHAPTER_14_AUDIT.md`
§4–§6 and Addendum.

### 16.1 BROKEN (18)

| # | chapter says | code at `a15b524` | governance | a repair would have to |
|---|---|---|---|---|
| B1 | health check grades early-stop ratio from best/total rounds (§7.2) | reads `best_round` (`training_health_check.py:392`); producer writes `best_val_round` (`training_diagnostics.py:239`) → 0.00 → critical. **Host: 107/107 archived checks critical** | symptom only (`SESSION_CHANGELOG_20260215_S92.md:188`) | make the consumer read the key the producer writes, and prove a healthy report grades `ok` |
| B2 | gradient-spread cross-check (§7.2) | reads `model_specific.feature_gradient_spread` (`thc:373`); producer writes `gradient_health.gradient_spread_ratio` (`td:595-598`) | none found | read one agreed key where it is written |
| B3 | live introspection for all four families (§1.3, §3.2–3.4; checklist 3.3) | final tree models train inline with no hook (`mpo:1832`, `:2761`); flag consumed for NN only (`:2587`). Host: no xgb/lgb report exists | none (header patch "NN-first", a session note) | give the tree final-model path a diagnostics capture |
| B4 | XGB/LGB per-round loss and importance | only CatBoost rounds read (`tst:126-141`); `feature_importances_` (`:144`) absent on Booster objects [UNVERIFIED]; empty subclasses (`td:683-690`) | none found | capture from the objects actually built |
| B5 | importances keyed by feature name | `feature_{i}` (`td:481`), `f{i}` (`tst:147`); host CatBoost archive names `f4`, `f6` | none found | carry the sidecar's 89 names into the trial worker |
| B6 | health check evaluates the compare-models winner | wrong filename and key (`thc:160-166` vs `mpo:311`, `:325`); host has 60 `…_S88_*` files, no plain one | none found | one filename and key between writer and reader |
| B7 | LLM receives rounds, top features, NN health (§8.4) | prompt reads old-schema keys (`diagnostics_llm_analyzer.py:183-191`) | none found | build the prompt from schema-1.1.0 keys |
| B8 | SKIP_MODEL lasts 24 h, `model_skip_state.json` | `skip_duration_hours` read nowhere (`thc:72`); no writer for the state file; absent on host | none found | implement an expiry, or state that skips last until reset |
| B9 | incidents to `watcher_failures.jsonl`; diagnostics feed Strategy Advisor | four §7.5 functions undefined; Advisor reads `diagnostics_history/` (`parameter_advisor.py:210`); no incident log on host | partly (fact file §2.13) | define the bridge artifact with one writer and one reader |
| B10 | FIFO history pruning complete | pruner called only from unused `MultiModelDiagnostics.save` (`td:834`); live writer `thc:562` does not prune; 111 files on host | none found | call the pruner from the writer that runs |
| B11 | Step 5 writes `tier_comparison.json` (§9.4) | Step 5 never imports `per_survivor_attribution`; sole producer `c13:827-833`, unreachable | governed (`S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md:426`, `:614`) | part of P0-C |
| B12 | degrading trend recorded (§9.2) | `record_event` not a `LearningTelemetry` method (`selfplay_orchestrator.py:845`); swallowed | none found | add the event method or use an existing recorder |
| B13 | Chapter 13 RCA when Hit@20 drops (§9.3) | four blockers (§9.3 table, stops 2–5) | **governed** (`…SYSTEM_MAP…:89-101`, order `:255-256`) | P0-C in full |
| B14 | same path — the gate | trigger fields have no producer (`c13:546-557`) | governed as status (`…S95.md:178-180`) | emit what the gate tests, or test what is emitted |
| B15 | same path — LLM on regime shift | `StrategyAdvisor.request_diagnostics_analysis` undefined (`c13:842`) | none found | point the call at the function that exists (`diagnostics_llm_analyzer.py:308`) |
| B16 | `unused_feature_pct` bound | declared (`thc:67`, policy), evaluated nowhere (`thc:331-430`) | none found | compute and evaluate the metric |
| B17 | tasks 8.5, 8.6 validated (tracker) | synthetic tests assert on literals, call no production code (`test_phase_8_soak.py:137-196`) | none found | tests that exercise `InnerEpisodeTrainer` and `_check_episode_training_trend` |
| **B18** | NN per-feature gradients captured each detail round (§3.1) | `grad_input[0]` of the first layer is read (`td:429-431`) with a non-grad input (`tst:515`); **host: 0/75 real NN reports contain `feature_gradients`; spread always 1.0** (`td:563`). *New this pass — was audit UNVERIFIED #1* | none found; `S172_ATTRIBUTION…:426` describes it as present | capture input gradients by a path that produces them (the chapter's extra `requires_grad` pass is one), and show a real report carrying them |

### 16.2 SWITCHED-OFF-BY-RULING (10)

| # | held back | record | lifting it would enable |
|---|---|---|---|
| S1 | diagnostic/LLM retry parameters reaching Step 5 | step-scoped filter ruling, quoted `CLAUDE_CODE_INSTRUCTIONS_CHAIN_C_TRUTHFULNESS_HOTFIX.md:33-44`; `PIPELINE_BEHAVIOUR_MODEL.md:993-996` | a retry that differs from the failed run (also needs manifest names, CLI routes, B1 and B7) |
| S2 | real per-survivor attribution run | `…SYSTEM_MAP…:272`, `:275`; Gate A-NORM `:99` | Chapter 13 attributing real predictions per seed (needs all of P0-C plus B14, B15) |
| S3 | RCA classification raising triggers | `SESSION_CHANGELOG_20260214_S85.md:85`, `:95`; `…SYSTEM_MAP…:222-228` | a `regime_shift` retrain trigger (needs S2, trigger type, evidence conditions) |
| S4 | attribution-driven autonomous parameter changes | "Not authorized", `…SYSTEM_MAP…:277`; observe-only experiment after P0, `:278` | attribution steering search without a human step |
| S5 | `forward_matches`/`reverse_matches` as features | `…SYSTEM_MAP…:167-171`, `:276` | the model and diagnostics seeing per-seed sieve match quality (§15.2) |
| S6 | dead placeholders stay zero under the 89-schema | `…SYSTEM_MAP…:199-200`; RANGE-MINER beyond 22 arrays not authorized `:268` | per-seed skip shape in `skip_mean`/`skip_std`/`skip_entropy` (needs a carrier and schema bump) |
| S7 | no diagnostics on Optuna folds | "TB Trim #1", `SESSION_CHANGELOG_20260216_S94.md:59`; `mpo:1891-1892` | per-fold NN health during search (needs per-fold paths and working pruning, B10) |
| S8 | selfplay trend observe-only | `SESSION_CHANGELOG_20260213_S83.md:14`, `:46`; `selfplay_orchestrator.py:829` (session decision) | a degrading trend stopping or adjusting selfplay (needs diagnostics on, B12) |
| S9 | web dashboard and TensorBoard | deferred, `CHAPTER_14_IMPLEMENTATION_PROPOSAL_S69.md:374-382`; header patch `:63`; tracker `:74` | human-readable loss/importance views (needs B3–B5, B18) |
| S10 | `actual_mean`/`actual_std` exclusion, guardrails G1–G6 | request "Pending TB Approval", `S111_TEAM_BETA_BRIEFING.md:152-157`; no answering ruling found — under the 2026-11-11 owner rulings this is Michael's decision | the two history-constant columns leaving training and a preflight for near-constant columns (needs a schema-hash change and retrain) |

Per `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md` §0, every ruling above is a
recorded Team Beta review position that the owner may keep or lift. No recommendation is made.

### 16.3 DESCRIPTION-ONLY (18)

D1 62 features → 89 trained / 91 extracted · D2 hooks in `reinforcement_engine.py` → `train_single_trial.py:607-660` ·
D3 schema 1.0.0 → 1.1.0 (§3.5) · D4 health functions in `agents/watcher_agent.py` → `training_health_check.py` ·
D5 `DiagnosticsBundle` in `bundle_factory.py` → `diagnostics_llm_analyzer.py:112` · D6 overfit ratio train/val → val/train ·
D7 one `reinforcement_engine_config.json` block → code defaults; file absent on host · D8 header phase status ·
D9 §13.1 prerequisites vs §14 · D10 trend check in `chapter_13_orchestrator.py` → `selfplay_orchestrator.py:826` ·
D11 examples driven by trial-level / constant features · D12 "dead ReLU" measured pre-activation (host: 242/242 zero) ·
D13 selfplay function names · D14 "line 499" → `:502-510` · D15 `--enable-tensorboard` does not exist ·
D16 one threshold layer → two (`td:75-80` and policy) · D17 per-sample attribution "unique to NN" → all four families ·
D18 diagnostics "default OFF" → Step-5 manifest default `true` (`agent_manifests/reinforcement.json:179`).

---

## 17. Open items

*(Added 2026-11-11.)*

1. **[UNVERIFIED] B4 mechanism** — whether `lgb.train`/`xgb.train` Boosters expose
   `feature_importances_`. Needs an executed check (out of scope for a documentation pass).
2. **[UNVERIFIED] tracker soak results** (30/30 cycles, "real SHAP attribution"). Historical; the
   one on-disk report (`diagnostics_outputs/soak_report_real.json`, 2026-02-15) used a 62-feature
   model and a `ranked_predictions.json` that no longer exists.
3. **Which WATCHER action followed each of the last checks** (2026-03-07/08) — no WATCHER log was
   read; `[UNVERIFIED]`.
4. **Recorded attribution of the NN criticals.** `SESSION_CHANGELOG_20260225_S111.md:89-94` and
   `SESSION_CHANGELOG_20260307_S121.md:45` attribute them to un-normalised `y`; the archive shows
   B1 on every check before and after. For the owner to note; this chapter does not resolve it.
5. **S10** awaits an owner decision.
6. **Phases 4, 5, 9 and task 3.6** remain unbuilt or unrun — tracked in `docs/TODO_CONSOLIDATED.md`
   (Chapter 14 rows).
7. **Repairs for B1–B18** are separate briefs; this pass records them only.

---

## 18. Verification declaration (VIR-1…6)

**execution proof (VIR-1).** Every `file:line` added in this pass was opened on VM 101 at
`a15b524` in this session (Read, `sed -n`, `awk 'NR==n'`, `/bin/grep -n`), except where a line
is explicitly attributed to `docs/CHAPTER_14_AUDIT.md`, whose anchors were themselves re-opened
(Addendum A.2). Files opened are listed in the audit Addendum A.5. Host artifacts were opened with
`python3 -I` read-only one-liners; nothing was written to the host except this chapter and the
audit's appended addendum.

**clean control (VIR-2).** `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` was read as the model and not
edited. Design text verified **correct and unchanged** by this pass: §2.3 invariant, §7.1 decision,
§7.2 policy values, the existence of `training_diagnostics.py` / `per_survivor_attribution.py` /
grammar / schema, `capture_every_n` 5, the RETRY cap of 2, the skip threshold of 3, the
attribution backends (audit §1, 22 accurate claims).

**fault-injection control.** **Not applicable** to a documentation pass. No detector, gate or
harness was run that could pass vacuously; none was invented.

**completion sentinel.** §19.

**unavailable-observer behaviour.** Surfaces that could not be read are recorded as UNAVAILABLE,
never as clean: WATCHER run logs for the 2026-03 checks; library behaviour of xgboost/lightgbm
Boosters (not executed); ser8 archives; any rig. Absence claims in this chapter list their
surfaces: tracked tree, gitignored files (`.gitignore:41` `*.json`; `find` on the working tree and
`/`), git history (`git log -S`, `git log -- path`, `git show`), host state (`diagnostics_outputs/`,
`predictions/`, `configs/`, `agent_manifests/definitions.json`, `crontab -l`, `systemctl list-units`).

**audit claim scope.** What this chapter says against what the code at `a15b524` does. Whether the
design is right is not judged; no component is proposed for removal or simplification.

**governance trail searched.** `docs/TB_RULING_*` (30 files), `docs/PROPOSAL_*`, `docs/TEAM_ALPHA_*`
by `/bin/grep -l -i` for `training_diagnostics`, `training_health`, Chapter 14, skip registry,
`per_survivor_attribution`, `feature_gradients`; `TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md`;
`CLAUDE_CODE_INSTRUCTIONS_CHAIN_C_TRUTHFULNESS_HOTFIX.md`; `S111_TEAM_BETA_BRIEFING.md`;
`S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md`; `PROPOSAL_CATEGORY_B_NN_TRAINING_ENHANCEMENTS_v1_0.md`;
`PROPOSAL_EPISTEMIC_AUTONOMY_UNIFIED_v1_3.md`; session changelogs S83, S85, S90, S92–S95, S111, S121;
`CHAPTER_14_HEADER_PATCH.md`; `CHAPTER_14_IMPLEMENTATION_PROPOSAL_S69.md`;
`CHAPTER_13_IMPLEMENTATION_PROGRESS_v3_9.md`; `BACKLOG.md`; `TODO_CONSOLIDATED.md`.
**chapters searched.** This chapter (all headings; prose sections line by line; long sample-code
listings in §3.2–3.4, §5.2–6.5, §7.3–7.4, §8.1–8.6, §9.1–9.3, §10.2 by heading and targeted grep
and annotated at section level, not line by line); `CHAPTER_2_BIDIRECTIONAL_SIEVE.md` (model);
`PIPELINE_BEHAVIOUR_MODEL.md` §1.1.

---

## 19. Closure statement

### 19.1 Verified against

VM 101, `/home/michael/distributed_prng_analysis`, HEAD `a15b5247336255801de8cf67d52653701b1e8c80`,
2026-11-11, read-only.

### 19.2 What this pass did

- Added the TFM framing block, status header, numbering statement and "How this chapter was
  produced".
- Annotated §1–§14 and the appendices with dated corrections; marked §3.5, §5, §6, §8.1, §9.4,
  §12.3 and §13.1 **SUPERSEDED (2026-11-11)** with pointers, text retained.
- Added §15 (RANGE-MINER, 22-array contract, window anchor), §16 Known gaps (18 BROKEN,
  10 SWITCHED-OFF-BY-RULING, 18 DESCRIPTION-ONLY), §17 Open items, §18 VIR-1…6.
- Settled from VM 101: real producer reports carry **`best_val_round`**, never `best_round`;
  **`neural_net` is not currently skipped** (`consecutive_critical` 1 < 3).

### 19.3 What this chapter is NOT

Not a repair: no code, test, config or manifest was changed. Not a design review. Not a claim that
training diagnostics work — §16 says they mostly do not reach a consumer that uses them.

### 19.4 Length

The chapter started at 3,199 lines and is longer after this pass (3,9xx). **No heading was
removed and no text was deleted.** Two original lines were extended in place with their original
text kept as the prefix: the `**Version:** 1.2.0` header line (now also names the 1.3.0 overlay)
and the `**END OF CHAPTER 14**` marker (now says what follows it). Checked by diffing every
removed line of `git diff HEAD` against the new file.

### 19.5 Completion sentinel

```
CHAPTER 14 CORRECTION:  PASS
```

`PASS` is claimed for the declared scope only: every substantive claim the audit checked was
re-verified against live code at `a15b524`, labelled once, and reflected in the chapter with an
anchor; both §5 host questions are answered from host artifacts. It is **not** a claim that the
long sample-code listings were compared line by line (§18), and two `[UNVERIFIED]` items remain
(§17.1, §17.2).

**Files changed by this pass:** `docs/CHAPTER_14_TRAINING_DIAGNOSTICS.md`, and an appended
addendum in `docs/CHAPTER_14_AUDIT.md`.
