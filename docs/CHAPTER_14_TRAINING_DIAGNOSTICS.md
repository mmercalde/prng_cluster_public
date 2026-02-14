# Chapter 14: Training Diagnostics & Model Introspection

## PRNG Analysis Pipeline — Complete Operating Guide

**Version:** 1.2.0  
**Status:** ACTIVE — Phases 1, 3, 5, 6 Complete (S69-S73)  
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

### 1.3 Scope

Four capabilities, each applicable across all four model types:

| # | Capability | NN | XGBoost | LightGBM | CatBoost |
|---|-----------|-----|---------|----------|----------|
| 1 | Live Training Introspection | PyTorch hooks | eval_set callbacks | record_evaluation | get_evals_result |
| 2 | Per-Survivor Attribution | Input gradients | pred_contribs | pred_contrib | ShapValues |
| 3 | Web Dashboard Charts | Loss + dead neurons + gradients | Loss + gain importance | Loss + split/gain | Loss + PredictionValuesChange |
| 4 | TensorBoard | add_graph + add_histogram | add_scalars from evals | add_scalars from evals | add_scalars from evals |

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

### 7.2 New WATCHER Policy Entries

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

---

## 8. LLM Integration — DiagnosticsBundle & Grammar

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
```

---

**END OF CHAPTER 14**

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
