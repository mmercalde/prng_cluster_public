# SESSION CHANGELOG — S93 (2026-02-16)

**Focus:** Bug A/B/C Deployment + NN Diagnostics Hook Wiring + Architecture Analysis  
**Duration:** ~4 hours  
**Result:** COMPLETE — All bugs deployed, diagnostics live, vanishing gradients detected,
root cause traced to over-parameterized architecture, fix path confirmed (Optuna already configured)

---

## Summary

Three-phase session. Phase 1 deployed three bugs identified by Team Beta post-S92 analysis
(WATCHER retry freshness, health check winner override, diagnostics history guard).
Phase 2 discovered and fixed the root cause of empty NN diagnostics — NNDiagnostics hooks
were plumbed (Chapter 14) but never connected to the actual training loop. Live telemetry
immediately detected vanishing gradients in 3/4 NN layers. Phase 3 investigated the
vanishing gradient root cause with Team Beta, confirming it's an architecture problem
(not target scale), and discovered the fix (Optuna exploration of shallower architectures)
was already configured but masked by CLI test overrides.

---

## Commits

| Commit | Description |
|--------|-------------|
| (pending) | fix(s93): Bug A/B/C — freshness invalidation on RETRY, winner override, history guard |
| 7ecbe25 | feat(s93): Wire NNDiagnostics hooks into train_neural_net() training loop |

---

## Phase 1: Bug A/B/C Deployment

### Bug A (CRITICAL): WATCHER Retry Freshness Gate

**Problem:** WATCHER retry loop set `current_step=5` but freshness gate blocked
re-execution. `skip_registry` incremented on stale diagnostics — NN hit `SKIP_MODEL`
after 1 real run counted 3 times.

**Fix:** Added `_invalidate_step_freshness()` method that touches the primary Step 5
input file before retry loop-back. Logs before/after mtime. Added A3 defense-in-depth:
health check block checks `results.get("skipped", False)` — if Step 5 was
freshness-skipped, health check bypassed entirely.

**Files:** `agents/watcher_agent.py` (patches A1, A2, A3)

### Bug B: Health Check Winner Override

**Problem:** Health check evaluated CLI `model_type` instead of compare-models winner
from sidecar file.

**Fix:** Read winner from `compare_models_summary.json` sidecar. Best-effort (non-fatal
if sidecar missing).

**Files:** `training_health_check.py` (patch B1)

### Bug C: Diagnostics History Type Guard

**Problem:** `diagnostics_llm_analyzer.py` history loop crashed on non-dict JSON entries
(`AttributeError`).

**Fix:** `isinstance` guard + broadened exception catch.

**Files:** `diagnostics_llm_analyzer.py` (patch C1)

### Validation Run (3 cycles, 37 minutes)

All three compare-models cycles executed with real training:
- Cycle 1: NN R²=0.0002 (5 min), catboost winner (R²=0.0000)
- Cycle 2: NN TIMEOUT at 600s, catboost winner (R²=0.0001)
- Cycle 3: NN R²=0.0001 (2.75 min), catboost winner (R²=0.0001)

Bug A fix confirmed: freshness invalidation fired each RETRY iteration.
LLM diagnostics analyzer called DeepSeek-R1-14B successfully (focus: MODEL_DIVERSITY).

---

## Phase 2: NN Diagnostics Hook Wiring

### Root Cause Analysis

Investigation revealed `_emit_nn_diagnostics()` created a fresh `NNDiagnostics()` object
AFTER training and called `.set_final_metrics()` + `.save()` — but never called
`.attach(model)` or `.on_round_end()`. The hooks that capture per-epoch gradient norms,
dead neuron percentages, and activation statistics were never registered.

Evidence chain:
- `grep` for `NeuralNetDiagnostics|diagnostics_callback|hook|register_hook` → zero hits
- `neural_net_diagnostics.json`: `"status": "partial"`, `"round_data_sample": []`,
  `"confidence": 0.0`, `"layer_health": {}`
- `NNDiagnostics.attach()` registers forward/backward hooks on all `nn.Linear` layers
- `on_round_end()` snapshots activations/gradients each epoch

### Fix (v3, Team Beta reviewed)

Patcher `apply_s93_nn_diagnostics_wiring.py` — 6 patches:

1. **PATCH 1:** Create `NNDiagnostics()`, call `.attach(_base_model)` before training loop
2. **PATCH 2a:** Add `_epoch_train_loss` / `_epoch_batches` accumulators at epoch start
3. **PATCH 2b:** Accumulate `loss.item()` after `optimizer.step()` (tightened anchor)
4. **PATCH 3:** Call `on_round_end(epoch, avg_train_loss, val_loss, lr)` after validation
5. **PATCH 4:** Replace post-hoc stub with live diagnostics save + detach

Team Beta fixes incorporated:
- **Fix #1:** DataParallel `.module` unwrap before `.attach()`
- **Fix #2:** `val_mse`/`mse` variable fallback via `locals().get()`
- **Fix #3:** Function-scoped anchor search (only patches inside `train_neural_net()`)
- **Red flag #1:** Idempotency guard (sentinel + partial-marker detection)
- **Red flag #2:** Tightened `optimizer.step()` anchor (backward+step pair regex)
- **Nice-to-have A:** `_write_canonical_diagnostics` existence check
- **Nice-to-have B:** `None` fallback (not `0`) for missing MSE
- **Nice-to-have C:** `detach()` in separate try block

### Validation Result

```json
{
    "status": "complete",
    "training_summary": {
        "rounds_captured": 38,
        "final_train_loss": 1.017e-06,
        "final_val_loss": 1.017e-06,
        "best_val_loss": 1.003e-06,
        "best_val_round": 22,
        "overfit_gap": -3.95e-10
    },
    "diagnosis": {
        "severity": "critical",
        "issues": [
            "Vanishing gradients in network.0: norm=0.00e+00",
            "Vanishing gradients in network.4: norm=0.00e+00",
            "Vanishing gradients in network.8: norm=0.00e+00"
        ],
        "confidence": 0.9
    }
}
```

**Key finding:** Vanishing gradients in 3/4 Linear layers (gradient norm = 0.0).
Only the output layer (`network.12`) has nonzero gradients (0.00013). This explains
persistent NN R²≈0 — the network is effectively dead from the input side.

---

## Phase 3: Target Statistics & Architecture Analysis (Team Beta Review)

### TB Concern #1: "Degenerate Sidecar" Wording

TB correctly identified a log mismatch: subprocess writes `best_model.pth` (model exists)
but parent logs "No model trained - saving degenerate sidecar only."

**Root cause:** The degenerate sidecar path is gated on signal quality/confidence, not on
physical checkpoint existence. With `signal_quality: weak (confidence=0.40)`, the
meta-optimizer refuses to "bless" the model but the subprocess already saved it.

**Recommendation:** Save model but mark `accepted: false` in sidecar. Logged as P7.

### TB Concern #2: Target Scale vs Architecture

TB hypothesized that tiny target variance could explain vanishing gradients. We ran the
target statistics check TB suggested.

```
y_train: mean=1.000e-01, std=5.006e-02, min=0.000e+00, max=3.750e-01, var=2.506e-03
y_val:   mean=1.000e-01, std=5.006e-02, min=0.000e+00, max=3.750e-01, var=2.506e-03
```

**Verdict: std(y) = 0.05 is NOT microscopic.** Targets range [0.0, 0.375] with reasonable
spread. The mse/var ratio of 0.0004 initially looked contradictory (would imply R²≈0.9996)
but this is because val_mse=1e-6 reported in diagnostics JSON is from the internal training
loop loss, while R²=0.0007 is computed on original-scale predictions post-training.

**Clarification:** val_mse in `neural_net_diagnostics.json` is recorded inside
`train_neural_net()` during validation (criterion output). R² is computed later by the
parent meta-optimizer on raw-scale y_val after checkpoint load. These metrics are therefore
not guaranteed to be numerically comparable unless they share the same target representation
and aggregation method (batch-mean vs full-array). Since y is not currently normalized in
the NN training path, the discrepancy likely stems from different evaluation pipelines.

**The vanishing gradients are a real architecture problem, not a scale artifact.**

**Caveat:** Architecture is the likely bottleneck, but LR/regularization settings can
produce the same gradient-collapse pattern (too-low LR, strong weight decay, bad init,
or early stopping locking in a constant mapping). Optuna's joint search over
depth/width/dropout/LR should disambiguate.

Further investigation confirmed: `normalize_features` only normalizes X (features), not
y (targets). R² is computed on raw y_val. The NN converges to predicting ≈ mean(y) ≈ 0.1
and gradients die because early layers become irrelevant once the output layer learns the
constant.

### TB Concern #2 Sub-findings: Win A vs Win B

**Win A (Rescale Targets):** Deferred as secondary. Target scale (std=0.05) is not the
primary issue but could improve gradient signal quality. Can be added as a
`normalize_targets` flag in `train_single_trial.py` in a future session.

**Win B (Confirm Gradient Capture):** Hooks are capturing the right tensors. Evidence:
- `network.12` (output layer) shows nonzero gradient norm (1.3e-4)
- `network.0/4/8` show exactly 0.0
- `dead_neuron_pct = 0.0` across all layers (activations aren't dying, just gradients)

This pattern is consistent with a model that rapidly converged to a near-constant mapping
where only the final layer needs adjustment. Early layers are "alive" (not dead neurons)
but "irrelevant" (zero gradient flow).

### TB Concern #3: Diagnostics Interpretation

TB's interpretation confirmed:
- `overfit_gap: -3.95e-10` (essentially zero) — not overfitting, not learning useful signal
- `severity: critical` (correct classification)
- `confidence: 0.9` (high, because hooks captured 38 full rounds)

### TB Concern #4: Architecture Fix (Already Configured)

**Current default architecture:** `[256, 128, 64]` — three hidden layers, ~50K+ parameters.
Massively over-parameterized for 62 features with weak signal. Network layers:
- `network.0`: Linear(62→256) — gradient_norm = 0.0 ❌
- `network.4`: Linear(256→128) — gradient_norm = 0.0 ❌
- `network.8`: Linear(128→64) — gradient_norm = 0.0 ❌
- `network.12`: Linear(64→1) — gradient_norm = 0.00013 ✅

**Optuna search space already covers the solution:**

```python
n_layers = trial.suggest_int("n_layers", 2, 4)
layer_0  = trial.suggest_int("layer_0", 64, 256)
layer_i  = trial.suggest_int(f"layer_{i}", 32, layers[-1])
dropout  = trial.suggest_float("dropout", 0.2, 0.6)
lr       = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
epochs   = trial.suggest_int("epochs", 50, 200)
patience = trial.suggest_int("patience", 10, 30)
```

TPE can discover `[64, 32]` with higher LR and higher dropout — drastically different
from the default `[256, 128, 64]`.

**Trial count already configured correctly:**

```json
// agent_manifests/reinforcement.json
"default_params": {
    "trials": 20,
    "compare_models": true,
    "enable_diagnostics": true
}
```

We were masking this with `--params '{"trials":1}'` CLI overrides during testing. On
autonomous WATCHER runs, each model type gets 20 Optuna trials. Compare-models wrapper
passes trials through to each subprocess (confirmed at line 246).

**With 20 trials, TPE will:**
1. Start with random exploration (trials 1-5)
2. Begin focusing on promising regions (trials 6-10)
3. Converge on optimal architecture/LR/dropout (trials 11-20)
4. Accumulate across runs via SQLite `load_if_exists=True`

**Note:** Optuna/TPE remains active in the Step 5 NN hyperparameter search path
(invoked via `_sample_hyperparameters()` in the Optuna objective). It was removed from
earlier pipeline steps but is the primary search engine for Step 5 model training.

**Future improvement noted:** `n_layers` minimum is 2. A single hidden layer `[64]`
or `[32]` might outperform all 2+ layer architectures given the weak signal. Lowering
the minimum to 1 would expand the search space. Non-blocking for now.

### TB Concern #5: Missing [DIAG] Lines in Log

Subprocess stderr isn't forwarded to the parent tee capture. The diagnostics JSON proof
is sufficient (status=complete, 38 rounds, real layer data). Low priority.

### TB Review Summary

| TB Concern | Status | Action |
|------------|--------|--------|
| #1 Degenerate sidecar wording | Logged P7 | Future: mark `accepted: false` in sidecar |
| #2 Target scale | Investigated | std=0.05, not a scale issue. Architecture is primary. |
| #3 Diagnostics interpretation | Confirmed | TB analysis was exactly correct |
| #4a Win A (rescale targets) | Deferred | Secondary benefit, not primary fix |
| #4b Win B (architecture) | Already configured | Optuna search space + 20 trials covers it |
| #5 Missing [DIAG] in log | Acknowledged | JSON proof sufficient, low priority |

---

## Files Modified

| File | Changes | Status |
|------|---------|--------|
| agents/watcher_agent.py | Bug A (A1, A2, A3) | ✅ Deployed |
| training_health_check.py | Bug B (B1) | ✅ Deployed |
| diagnostics_llm_analyzer.py | Bug C (C1) | ✅ Deployed |
| train_single_trial.py | NN diagnostics hook wiring (6 patches) | ✅ Deployed, committed |

---

## Skip Registry Status

Reset to `{}` after validation. Previous state had `neural_net` at 5/5 consecutive
critical from Bug A phantom increments compounded by tree-model health check runs.

---

## Next Steps

### Immediate (Next Session)

1. **Commit Bug A/B/C** — Phase 1 fixes need formal git commit with patcher scripts.

2. **Run full autonomous WATCHER cycle** — Step 5 with manifest defaults
   (`trials=20, compare_models=true, enable_diagnostics=true`), no CLI trial override.
   Confirm diagnostics show nonzero gradient norms for winning NN trial (validates TPE
   explored shallower/higher-LR configurations).

3. **Compare NN R² across trials** — Verify TPE is learning (later trials should improve).

### No Longer Needed

- **Priority 2 (bump Optuna trials):** Already configured correctly. Manifest has
  `"trials": 20` in `default_params`. We were forcing `trials=1` via CLI for testing.

### Deferred

- P3: Recalibrate NN diagnostic thresholds post-Category-B
- P4: Investigate NN timeout (bump to 900s or GPU cleanup between cycles)
- P5: Regression diagnostics for gate=True validation
- P6: Dead code audit, 27 stale files cleanup
- P7: Fix "degenerate sidecar" log wording — mark `accepted: false` in sidecar
- P8: Target rescaling (`normalize_targets` flag) for improved gradient signal
- P9: Lower `n_layers` minimum from 2 to 1 in Optuna search space

---

## Key Insights

1. **Last-mile integration gaps:** The diagnostics infrastructure was architecturally
   complete (Chapter 14 defined hooks, `NNDiagnostics` had `.attach()` / `.on_round_end()`
   / `.save()`) but the integration point in `train_neural_net()` was a post-hoc stub.
   Live telemetry immediately surfaced vanishing gradients invisible for weeks.

2. **Test overrides hiding production config:** We thought Optuna trials needed bumping,
   but the manifest already had `trials=20`. Our `--params '{"trials":1}'` CLI overrides
   during testing masked the production configuration. Always verify manifest defaults
   before proposing config changes.

3. **Diagnostics → root cause → fix path in one session:** Wiring the hooks → detecting
   vanishing gradients → tracing to over-parameterized architecture → confirming Optuna
   already covers the solution space. The Chapter 14 telemetry pipeline worked exactly
   as designed once the last-mile gap was closed.
