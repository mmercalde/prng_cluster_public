# PROPOSAL: Category B — Neural Net Training Enhancements
**Version:** 1.0.0
**Date:** 2026-02-15 (Session 91)
**Author:** Team Alpha (Claude)
**Reviewer:** Team Beta
**Status:** DRAFT — Awaiting Team Beta Architectural Review

---

## 1. Motivation

Step 5 `--compare-models` consistently shows neural_net as the worst performer:

| Model | Best R² (CV) | Duration | Notes |
|-------|-------------|----------|-------|
| neural_net | -1.743 | ~14 min | 97% of Step 5 runtime |
| lightgbm | -0.0044 | ~6 sec | |
| xgboost | -0.0059 | ~9 sec | |
| catboost | -0.000033 | ~14 sec | Winner |

**Root cause:** `SurvivorQualityNet` receives raw, unnormalized features from `survivors_with_scores.json`. Feature scales vary by orders of magnitude (probabilities 0-1 vs seed candidates in millions vs tiny p-values). The vanilla MLP architecture has no mechanism to handle this, causing:

1. **Gradient domination** — large-scale features overwhelm small-scale features in the linear layers
2. **Dead ReLU neurons** — gradient swings push neurons into permanently-zero output (diagnostics confirmed critical dead neuron percentages)
3. **Immediate overfitting** — ~50K parameters memorize hundreds of samples despite dropout

Chapter 14 Section 11 defines the path: Phase A (Diagnose) is complete. This proposal implements **Phase B (Fix informed by diagnosis)**.

---

## 2. Scope

Three new CLI flags for `train_single_trial.py`, consumed during neural_net training. These touch the Optuna search space because the WATCHER retry path and LLM advisory layer need to propose these values during parameter refinement.

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `--normalize-features` | bool | false | StandardScaler preprocessing before NN training |
| `--use-leaky-relu` | bool | false | Replace ReLU with LeakyReLU(0.01) in SurvivorQualityNet |
| `--dropout` | float | (none) | Override Optuna-suggested dropout value |

**Explicit non-scope:** These flags affect `neural_net` model type only. Tree models (lightgbm, xgboost, catboost) are unaffected. No changes to Steps 1-4 or Step 6.

---

## 3. Files Modified

### 3.1 `train_single_trial.py` — Argparse + Training Logic

**3.1.1 New argparse flags** (after existing `--enable-diagnostics`, ~line 666):

```python
# Category B: Neural net training enhancements (S91)
parser.add_argument('--normalize-features', action='store_true',
                    help='Apply StandardScaler to features before NN training')
parser.add_argument('--use-leaky-relu', action='store_true',
                    help='Use LeakyReLU(0.01) instead of ReLU in neural net')
parser.add_argument('--dropout', type=float, default=None,
                    help='Override dropout rate (takes precedence over Optuna suggestion)')
```

**3.1.2 Feature normalization** (in `train_neural_net()`, before tensor conversion, ~line 451):

```python
# Category B: Feature normalization
scaler = None
if normalize_features:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)  # transform only — no fit on validation
    if verbose:
        print(f"[NN] Features normalized: mean→0, std→1 ({X_train.shape[1]} features)", 
              file=sys.stderr)
```

The `scaler` object is saved alongside the model checkpoint so prediction (Step 6) can apply the same transform:

```python
# In model save block (~line 557):
torch.save({
    'state_dict': state_to_save,
    'feature_count': input_dim,
    'hidden_layers': hidden_layers,
    'dropout': dropout,
    'best_epoch': best_epoch,
    'scaler': scaler,              # NEW: None if normalization not used
    'use_leaky_relu': use_leaky_relu,  # NEW: architecture flag for reconstruction
}, save_path)
```

**3.1.3 Dropout override** (in `train_neural_net()`, ~line 460):

```python
# Category B: Dropout override (CLI takes precedence over Optuna/params)
dropout = dropout_override if dropout_override is not None else params.get('dropout', 0.3)
```

**3.1.4 LeakyReLU flag passthrough** (in `train_neural_net()`, ~line 466):

```python
model = SurvivorQualityNet(
    input_size=input_dim,
    hidden_layers=hidden_layers,
    dropout=dropout,
    use_leaky_relu=use_leaky_relu  # NEW
).to(device)
```

**3.1.5 Function signature change** — `train_neural_net()`:

```python
def train_neural_net(X_train, y_train, X_val, y_val, params, 
                     save_path=None, verbose=False, enable_diagnostics=False,
                     normalize_features=False,   # NEW
                     use_leaky_relu=False,        # NEW  
                     dropout_override=None):       # NEW
```

**3.1.6 main() wiring** — pass new args to `train_neural_net()`:

```python
if args.model_type == 'neural_net':
    result = train_neural_net(
        X_train, y_train, X_val, y_val, hyperparams,
        save_path=model_save_path, verbose=args.verbose,
        enable_diagnostics=args.enable_diagnostics,
        normalize_features=args.normalize_features,
        use_leaky_relu=args.use_leaky_relu,
        dropout_override=args.dropout
    )
```

### 3.2 `models/wrappers/neural_net_wrapper.py` — Architecture Change

**SurvivorQualityNet constructor** — add `use_leaky_relu` parameter:

```python
class SurvivorQualityNet(nn.Module):
    def __init__(self, input_size, hidden_layers=[256, 128, 64], 
                 dropout=0.3, use_leaky_relu=False):  # NEW param
        super().__init__()
        
        activation = nn.LeakyReLU(0.01) if use_leaky_relu else nn.ReLU()
        
        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                activation,
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
```

**Backward compatibility:** `use_leaky_relu=False` preserves existing ReLU behavior. No change to tree model wrappers.

### 3.3 `meta_prediction_optimizer_anti_overfit.py` — Subprocess CLI Construction

In `_s88_run_compare_models()`, where the subprocess command is built for each model type, add conditional flags for neural_net:

```python
cmd = [
    sys.executable, "meta_prediction_optimizer_anti_overfit.py",
    "--survivors", args_dict["survivors"],
    "--lottery-data", args_dict["lottery_data"],
    "--model-type", model_type,
    "--trials", str(trials),
]

# Category B: Pass NN enhancement flags if present
if model_type == "neural_net":
    if args_dict.get("normalize_features"):
        cmd.append("--normalize-features")
    if args_dict.get("use_leaky_relu"):
        cmd.append("--use-leaky-relu")
    if args_dict.get("dropout") is not None:
        cmd.extend(["--dropout", str(args_dict["dropout"])])
```

**Note:** This is the compare-models subprocess path. The single-model path (direct execution without `--compare-models`) already passes `--params` as JSON which can include these keys.

### 3.4 `meta_prediction_optimizer_anti_overfit.py` — Optuna Search Space

In `_sample_config()` (~line 286), add conditional activation search:

```python
config = {
    'hidden_layers': layers,
    'dropout': trial.suggest_float('dropout', 0.2, 0.5),
    'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
    'learning_rate': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
    'batch_size': trial.suggest_categorical('batch', [64, 128, 256]),
    'epochs': trial.suggest_int('epochs', 50, 150),
    'early_stopping_patience': trial.suggest_int('patience', 5, 15),
    'early_stopping_min_delta': trial.suggest_float('min_delta', 1e-4, 1e-2, log=True),
    'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw']),
    'loss': trial.suggest_categorical('loss', ['mse', 'huber']),
    'gradient_clip': trial.suggest_float('grad_clip', 0.5, 5.0),
    # Category B additions:
    'normalize_features': trial.suggest_categorical('normalize_features', [True, False]),
    'use_leaky_relu': trial.suggest_categorical('use_leaky_relu', [True, False]),
}
```

**Design decision:** `normalize_features` and `use_leaky_relu` are added to the Optuna search space so the optimizer can learn whether they help. On retry paths, the WATCHER can force them via CLI override regardless of what Optuna suggests.

### 3.5 `agent_manifests/reinforcement.json` — Parameter Bounds

Add to `parameter_bounds`:

```json
"normalize_features": {
    "type": "bool",
    "default": false
},
"use_leaky_relu": {
    "type": "bool",
    "default": false
},
"dropout_override": {
    "type": "float",
    "min": 0.0,
    "max": 0.8,
    "default": null
}
```

These bounds are enforced by the LLM advisory clamping in `_build_retry_params()` (lines 1542-1658 of `watcher_agent.py`).

### 3.6 `training_health_check.py` — Suggestion Mapping (Already Done)

`get_retry_params_suggestions()` (line 590) already emits `normalize_features: True` and `use_leaky_relu: True` based on diagnostics. No changes needed — the downstream consumers (`train_single_trial.py`) just couldn't act on them until now.

### 3.7 Step 6 Prediction Path — Scaler Loading

When Step 6 loads a neural_net checkpoint for prediction, it must apply the same scaler transform. The checkpoint now contains `scaler` (sklearn StandardScaler or None):

```python
# In prediction loading code:
checkpoint = torch.load(checkpoint_path)
scaler = checkpoint.get('scaler', None)
use_leaky_relu = checkpoint.get('use_leaky_relu', False)

# Reconstruct model with correct architecture
model = SurvivorQualityNet(
    input_size=checkpoint['feature_count'],
    hidden_layers=checkpoint['hidden_layers'],
    dropout=checkpoint.get('dropout', 0.3),
    use_leaky_relu=use_leaky_relu
)
model.load_state_dict(checkpoint['state_dict'])

# Apply scaler to prediction features
if scaler is not None:
    X_predict = scaler.transform(X_predict)
```

**File to modify:** This depends on which module loads the checkpoint for Step 6 prediction. Need to verify: is it `chapter_13_orchestrator.py`'s `_load_best_model_if_available()` or another loader?

---

## 4. Data Flow Diagram

```
WATCHER retry path:
  training_health_check.py
    → get_retry_params_suggestions()
      → {normalize_features: True, use_leaky_relu: True}
    → _build_retry_params() in watcher_agent.py
      → LLM proposes values → clamped by parameter_bounds
      → merged into step params
    → run_step(5, params={..., normalize_features: True, use_leaky_relu: True})
      → subprocess: train_single_trial.py --normalize-features --use-leaky-relu
        → StandardScaler applied → LeakyReLU architecture → train → save with scaler

Normal Optuna path:
  _sample_config() suggests normalize_features/use_leaky_relu per trial
    → passed to train_single_trial.py via --params JSON
    → same training logic applies
    → Optuna learns which combination works best
```

---

## 5. Existing Architecture Note: `batch_norm` Already Present

**Verified on live Zeus code (S91):** `SurvivorQualityNet` in `models/wrappers/neural_net_wrapper.py` already includes `nn.BatchNorm1d` in every hidden layer (always-on, line 65). The architecture is:

```
Linear → BatchNorm1d → ReLU → Dropout  (per hidden layer)
```

BatchNorm is **not** in the live Optuna search space (confirmed: `grep -n "batch_norm" meta_prediction_optimizer_anti_overfit.py` returns zero hits on production). This is correct — BatchNorm is always beneficial for NN and doesn't need to be a search variable.

**Implication for this proposal:** The NN already has intermediate activation normalization (BatchNorm). What's missing is **input feature normalization** (StandardScaler on raw features before they enter the network). These are complementary — BatchNorm normalizes between layers, StandardScaler normalizes before the first layer. Both are needed; only StandardScaler is missing.

---

## 6. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Scaler not saved → Step 6 predictions on wrong scale | Checkpoint explicitly stores scaler; Step 6 checks for it |
| LeakyReLU flag not stored → model reconstruction fails | Checkpoint stores `use_leaky_relu`; reconstruction reads it |
| Old checkpoints lack new fields | `checkpoint.get('scaler', None)` / `.get('use_leaky_relu', False)` — backward compatible |
| Dropout override conflicts with Optuna | CLI override takes explicit precedence; Optuna value used only when no override |
| StandardScaler fit on small dataset unreliable | With hundreds of samples, StandardScaler statistics are noisy but still dramatically better than raw scales spanning 6+ orders of magnitude |
| Tree models accidentally affected | All three flags are gated on `model_type == 'neural_net'` in both argparse consumption and subprocess construction |

---

## 7. Testing Plan

### 7.1 Unit Verification (pre-deployment)

1. `train_single_trial.py --model-type neural_net --normalize-features --data-path test.npz --trials 1` — verify scaler applied, saved in checkpoint
2. `train_single_trial.py --model-type neural_net --use-leaky-relu --data-path test.npz --trials 1` — verify LeakyReLU in model architecture
3. `train_single_trial.py --model-type neural_net --dropout 0.5 --data-path test.npz --trials 1` — verify override takes precedence over params
4. `train_single_trial.py --model-type lightgbm --normalize-features --data-path test.npz` — verify flag is ignored for tree models

### 7.2 Integration (WATCHER pipeline)

1. Run `--compare-models` with `normalize_features: true, use_leaky_relu: true` in params
2. Verify NN subprocess receives the flags
3. Compare NN R² against baseline (current -1.743)
4. Verify Step 6 loads checkpoint and applies scaler correctly

### 7.3 Retry Path

1. Force RETRY via synthetic training_health_check critical result
2. Verify `get_retry_params_suggestions()` emits normalize_features/use_leaky_relu
3. Verify `_build_retry_params()` includes them in retry params
4. Verify retry subprocess receives the flags

---

## 8. Implementation Order

1. `models/wrappers/neural_net_wrapper.py` — `use_leaky_relu` constructor param
2. `train_single_trial.py` — argparse + training logic + checkpoint fields
3. `meta_prediction_optimizer_anti_overfit.py` — subprocess CLI construction + Optuna search space
4. `agent_manifests/reinforcement.json` — parameter bounds
5. Step 6 prediction loader — scaler application
6. End-to-end test: `--compare-models` with new flags

---

## 9. Questions for Team Beta

1. **Optuna search space expansion concern:** Adding `normalize_features` and `use_leaky_relu` as categorical booleans doubles the effective search space. Should we instead **always enable** normalization (no Optuna choice) and only let Optuna search `use_leaky_relu`? Normalization is almost certainly always beneficial for NNs — the live architecture already has BatchNorm always-on (no toggle), suggesting this pattern.
2. **Step 6 checkpoint loader location:** Which module loads the NN checkpoint for prediction? Need to confirm the exact file and function for scaler wiring.
3. **GPU isolation invariant:** StandardScaler is sklearn (CPU-only). No CUDA/OpenCL implications. Confirmed compatible with the GPU isolation design from S72.
4. **NN runtime:** S91 test showed neural_net consuming 2+ hours for 20 Optuna trials (vs ~30 sec for all 3 tree models combined). Should the skip registry threshold (currently 3 consecutive criticals) be lowered to 1 for faster exclusion until Category B fixes are deployed?
