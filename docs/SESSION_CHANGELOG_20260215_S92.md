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

## Team Beta Post-Run Analysis (Critical Findings)

Team Beta review of the WATCHER end-to-end run revealed three distinct bugs that were
previously masquerading as a single "NN always fails" problem:

### Bug A: WATCHER Retry Without Re-Execution (CRITICAL)

After Step 5 succeeds and produces winner artifacts, WATCHER logs:
- "Training health check: model=neural_net severity=critical action=RETRY"
- Then immediately skips re-running Step 5 because output is "fresh"
- Then repeats the same health evaluation again
- Then hits 3 consecutive critical and SKIP_MODELs

The skip registry reached 3 even though NN only actually trained ONCE. The retry
semantics are applied, but Step 5 execution is bypassed by freshness gating, while
the skip counter increments anyway. This is a logic defect: you cannot both "RETRY
Step 5" and "skip Step 5 because outputs are fresh" and still count that as another
consecutive failure.

Fix options:
- Option 1 (best): When action=RETRY, bypass freshness checks for that step
  (delete/rename sentinel or pass --force-run)
- Option 2 (quick): If action=RETRY and Step 5 was skipped due to "fresh output,"
  do not increment consecutive critical (treat as "no new evidence")

### Bug B: Health Check Evaluates Requested Model, Not Winner

After compare-models selects catboost as winner and restores its artifacts, the health
check still evaluates model=neural_net and triggers retries. The health check reads the
CLI requested model_type rather than the actual winner model_type from the sidecar or
compare-models summary.

Fix: Health check should read the sidecar (or compare_models_summary) and evaluate the
artifact model_type, not the CLI requested model_type.

### Bug C: diagnostics_llm_analyzer.py Line 255 String-vs-Dict

build_diagnostics_prompt() expects dict entries in history but sometimes receives strings.
Causes: AttributeError: 'str' object has no attribute 'get'. Non-fatal by design (good)
but triggers repeated LLM server startups with zero value.

Fix: Coerce non-dict history entries or skip them with a warning.

### Impact

Bug A explains why NN has been hitting SKIP_MODEL so fast across multiple sessions —
it was never actually getting 3 chances, just 1 real run counted 3 times. This is the
primary reason the retry-learn-retry loop has been non-functional.

---

## Next Session (S93) Priorities

### Priority 1: Fix WATCHER Retry-Without-Rerun Bug (Bug A)

This is now the highest priority — without this fix, increasing Optuna trials is
pointless because retries never actually re-execute. The WATCHER must either force
Step 5 re-execution when action=RETRY, or stop counting stale results as new failures.

Team Beta requested three code snippets to start S93:
1. WATCHER Step 5 freshness skip logic ("Step 5: Fresh ... skipping")
2. Where consecutive critical is incremented / skip registry updated
3. build_diagnostics_prompt() history loop around line ~255

### Priority 2: Increase NN Optuna Trials

Bump from 1 to 10-20 in manifest default_params. TPE needs data to learn. With
Category B normalization making the landscape smooth, Optuna can explore learning
rate, dropout, architecture, and weight decay meaningfully. The Optuna DB accumulates
across runs via deterministic study names with load_if_exists=True.

This becomes effective AFTER Bug A is fixed (otherwise retries never re-execute).

### Priority 3: Fix Health Check Model Mismatch (Bug B)

Health check should evaluate the winner artifact model_type (from sidecar), not the
CLI requested model_type. After compare-models selects catboost, triggering NN retries
is wrong.

### Priority 4: Fix diagnostics_llm_analyzer.py (Bug C)

Harden build_diagnostics_prompt() history ingestion. Coerce string entries to dict
or skip with warning. Unblocks LLM-guided retry refinement.

### Priority 5: Recalibrate NN Diagnostic Thresholds

"Early stop ratio 0.00" as automatic critical needs revisiting post-Category-B.
The normalization changes what "healthy" training looks like.

### Priority 6: Deferred Backlog

- Regression diagnostics: create synthetic data for gate=True validation (deferred since S86)
- Dead code audit: comment out old MultiModelTrainer inline NN path (never delete)
- 27 stale files cleanup from project knowledge (S85/S86 audit)
- Checkpoint rename -> explicit output path (Team Beta recommendation from S92)

---

## Files Modified

| File | Change |
|------|--------|
| meta_prediction_optimizer_anti_overfit.py | +2 methods, subprocess routing, GPU isolation, hotfixes |

## Patcher Files (cleaned up)

All S92 patchers removed from Zeus after final commit:
- apply_category_b_phase2_1_nn_subprocess.py
- apply_phase2_1_hotfix.py
- apply_phase2_1_gpu_isolation_fix_v3.py

---

Session 92 — Team Alpha
