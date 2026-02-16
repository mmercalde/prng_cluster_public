# SESSION CHANGELOG — S92 (2026-02-15)

**Focus:** Category B Phase 2.1 — Close Inline NN Trainer Gap  
**Duration:** ~3 hours  
**Result:** COMPLETE — NN subprocess routing deployed and validated through WATCHER

---

## Summary

Closed the architecture gap where single-shot NN training bypassed normalization/LeakyReLU
by routing it through train_single_trial.py subprocess (same path as compare-models).
Required three commits to resolve GPU isolation invariant violation and checkpoint naming bugs.

---

## Commits

| Commit | Description |
|--------|-------------|
| dd34310 | feat(cat-b): Phase 2.1 — route single-shot NN through train_single_trial.py subprocess |
| 3c8afca | fix(cat-b): Phase 2.1 hotfix — checkpoint rename + JSON metrics parsing |
| 3dac87d | fix(cat-b): Phase 2.1 GPU isolation — defer CUDA for NN subprocess routing (Team Beta Option B-lite) |

---

## What Changed

### 1. Subprocess Routing for Single-Shot NN (dd34310)

Added _export_split_npz() and _run_nn_via_subprocess() methods to
meta_prediction_optimizer_anti_overfit.py. When model_type == "neural_net",
_run_single_model() now routes through train_single_trial.py subprocess
instead of inline MultiModelTrainer._train_neural_net().

Team Beta Modifications Applied:
- Mod A: Export exact train/val split (no new split heuristics)
- Mod B: Atomic temp dir (outputs/tmp/) with cleanup on success, retain on fail
- Mod C: Fail hard by default; --allow-inline-nn-fallback escape hatch
- Mod D: Thread all Category B flags end-to-end

### 2. Checkpoint Rename + JSON Metrics (3c8afca)

Two bugs found during acceptance testing:
- train_single_trial.py saves as neural_net_trial-1.pth, patcher expected best_model.pth
- Metrics parsing expected key:value lines but subprocess outputs JSON on stdout

Fix: Rename after subprocess completion; parse JSON from stdout.

### 3. GPU Isolation Fix (3dac87d)

Root cause: Session 72 GPU Isolation Design Invariant states GPU-accelerated code
must NEVER run in the coordinating process when using subprocess isolation. The parent
process called initialize_cuda_early() in single-model mode, poisoning the NN
subprocess via CUDA context inheritance (1.6s crash, rc=1).

Fix (Team Beta Option B-lite): Extended CUDA deferral to cover NN subprocess routing:

    will_use_subprocess = args.compare_models or (
        args.model_type == "neural_net" and NN_SUBPROCESS_ROUTING_ENABLED
    )

Added NN_SUBPROCESS_ROUTING_ENABLED = True flag for future-proofing.
Added Option C defense-in-depth: env pass-through with logging (no invented device IDs).

---

## Validation Results

### WATCHER End-to-End Test (18:53 run)

- Mode: Single Model (neural_net) (Subprocess Isolation)
- GPU initialization DEFERRED to subprocess
- CUDA initialized: False
- NN subprocess: ~5 min runtime (previously 1.6s crash)
- Enriched checkpoint: normalize_features=True, use_leaky_relu=True, scaler arrays=(62,)
- All 4 model types trained in compare-models loop
- CatBoost won (R2=0.0002), pipeline completed successfully

### Enriched Checkpoint Verification

    normalize_features: True
    use_leaky_relu: True
    scaler_mean: shape=(62,)
    scaler_scale: shape=(62,)

---

## Category B Complete Status

| Phase | Description | Commit | Status |
|-------|-------------|--------|--------|
| 1 | train_single_trial.py normalization + LeakyReLU | 3c3f9ae | Done |
| 2 | Compare-models subprocess flag injection | fb8561c | Done |
| 3 | Step 6 scaler application + manifest | bc481f7 | Done |
| 2.1 | Single-shot NN subprocess routing | dd34310 | Done |
| 2.1a | Hotfix: checkpoint rename + JSON metrics | 3c8afca | Done |
| 2.1b | GPU isolation: CUDA deferral for NN subprocess | 3dac87d | Done |

---

## Known Issues Found (Not Blockers)

1. diagnostics_llm_analyzer.py line 255: 'str' has no attribute 'get' — blocks LLM retry refinement
2. WATCHER retry switches to catboost instead of retrying NN with modified params
3. NN diagnostic thresholds need recalibration post-Category-B normalization
4. Optuna trials=1 gives zero exploration budget for TPE sampler

---

## Next Session (S93) Priorities

### Priority 1: Increase NN Optuna Trials (HIGHEST IMPACT)

Bump from 1 to 10-20 in manifest default_params. This is the single highest-impact
change. TPE needs data to learn. With Category B normalization making the landscape
smooth, Optuna can actually explore learning rate, dropout, architecture, and weight
decay meaningfully. The Optuna DB accumulates across runs via deterministic study
names with load_if_exists=True, so trials 11-20 in the next run benefit from trials
1-10 in this run.

### Priority 2: Fix diagnostics_llm_analyzer.py Line 255

The 'str'.get() bug blocks smart LLM-guided retry refinement. History entries contain
strings where dicts are expected. Quick fix, big unlock — this restores the "smart"
half of the WATCHER retry loop where the LLM reads training_diagnostics.json and
proposes parameter adjustments via grammar-constrained output.

### Priority 3: Fix WATCHER Retry to Re-run NN

Stop switching to catboost on critical. Instead, retry NN with Optuna's next suggestion.
The whole retry-learn-retry loop only works if NN actually gets another shot. Currently:
- Attempt 1: NN trains, diagnostics flags critical
- WATCHER switches model_type to catboost (NN never retried)
- Retries 2-3 see fresh catboost artifacts, skip

After fix:
- Attempt 1: NN trains with params A, diagnostics flags critical
- Attempt 2: NN retries with modified params B (higher dropout, different LR)
- Attempt 3: NN retries with params C
- If still critical after 3: SKIP_MODEL fires, tree models take over

### Priority 4: Recalibrate NN Diagnostic Thresholds

"Early stop ratio 0.00" as automatic critical needs revisiting post-Category-B.
The normalization changes what "healthy" training looks like. With normalized inputs
and LeakyReLU, the gradient landscape is different — thresholds tuned for raw inputs
with ReLU may be too aggressive.

### Priority 5: Deferred Backlog

- Regression diagnostics: create synthetic data for gate=True validation (deferred since S86)
- Dead code audit: comment out old MultiModelTrainer inline NN path (never delete)
- 27 stale files cleanup from project knowledge (S85/S86 audit)
- Checkpoint rename -> explicit output path (Team Beta recommendation from S92)

---

## Files Modified

| File | Change |
|------|--------|
| meta_prediction_optimizer_anti_overfit.py | +2 methods, subprocess routing, GPU isolation, hotfixes |

## Patcher Files (cleanup candidates on Zeus)

| File | Can Remove |
|------|------------|
| apply_category_b_phase2_1_nn_subprocess.py | Yes |
| apply_phase2_1_hotfix.py | Yes |
| apply_phase2_1_gpu_isolation_fix_v3.py | Yes |

---

Session 92 — Team Alpha
