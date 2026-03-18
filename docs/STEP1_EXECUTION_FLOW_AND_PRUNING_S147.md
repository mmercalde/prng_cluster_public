# Step 1 Execution Flow & Pruning Logic
## Window Optimizer — Both Execution Paths
**Version:** S147  
**Date:** 2026-03-17

---

## Overview

Step 1 runs 4 sieve passes per Optuna trial. Two execution backends exist:

| Backend | When used | Key files |
|---------|-----------|-----------|
| **Original coordinator** | `--use-persistent-workers` NOT set | `coordinator.py` → `sieve_filter.py` |
| **PWC (persistent workers)** | `--use-persistent-workers` set | `persistent_worker_coordinator.py` → `sieve_gpu_worker.py` |

Both paths run the same 4 logical passes. Pruning implementation differs.

---

## The 4 Sieve Passes

| Pass | Type | Kernel | Seeds scanned | Speed |
|------|------|--------|---------------|-------|
| 1 | Forward constant-skip | `java_lcg` | Full range | Fast |
| 2 | Reverse constant-skip | `java_lcg_reverse` | Full range | Fast |
| 3 | Forward variable-skip (hybrid) | `java_lcg_hybrid` | Full range | Slow |
| 4 | Reverse variable-skip (hybrid) | `java_lcg_hybrid_reverse` | Full range | Slow |

Pass 3+4 are slow because: 5 strategies × full seed range = up to 5B evaluations per pass.

---

## Path A — Original Coordinator (pre-PWC)

```
WATCHER
  └─► window_optimizer.py (via subprocess)
        └─► window_optimizer_bayesian.py (Optuna TPE loop)
              └─► For each trial:
                    └─► window_optimizer_integration_final.py
                          └─► run_bidirectional_test()
                                ├─► coordinator.py → sieve_filter.py × 26 GPUs
                                │     [Pass 1: forward constant-skip]
                                │
                                ├─► GATE A1 ✅ (exists)
                                │     if forward_count == 0:
                                │       raise optuna.TrialPruned()
                                │       → Optuna ThresholdPruner skips trial
                                │
                                ├─► coordinator.py → sieve_filter.py × 26 GPUs
                                │     [Pass 2: reverse constant-skip]
                                │     → intersect → bidirectional_constant
                                │
                                ├─► if test_both_modes:
                                │     ├─► coordinator.py → sieve_filter.py × 26 GPUs
                                │     │     [Pass 3: forward hybrid × 5 strategies]
                                │     │
                                │     ├─► GATE A2 ✗ MISSING — confirmed bug (Q0)
                                │     │     (no check on hybrid forward survivors)
                                │     │
                                │     └─► coordinator.py → sieve_filter.py × 26 GPUs
                                │           [Pass 4: reverse hybrid × 5 strategies]
                                │           → intersect → bidirectional_variable
                                │
                                └─► Save survivors → NPZ accumulator
```

### Pruning in Path A

| Gate | Location | Status | Behavior |
|------|----------|--------|----------|
| A1 — constant-skip forward = 0 | `window_optimizer_integration_final.py` | ✅ Exists | Optuna `raise TrialPruned()` — skips entire trial |
| A2 — hybrid forward = 0 | Between Pass 3 and Pass 4 | ✗ Missing | Pass 4 runs unconditionally — **confirmed bug (Q0)** |

---

## Path B — PWC (Persistent Worker Coordinator)

```
WATCHER
  └─► sweep_run1.sh
        └─► window_optimizer.py (via subprocess)
              └─► window_optimizer_bayesian.py (Optuna TPE loop)
                    └─► For each trial:
                          └─► window_optimizer_integration_final.py
                                └─► run_trial_persistent()  [shim — PWC entry point]
                                      └─► PersistentWorkerCoordinator
                                            ├─► pwc.run_sieve_pass(java_lcg)
                                            │     [Pass 1: forward constant-skip]
                                            │     Zeus: execute_local_sieve_job()
                                            │     Rigs: _dispatch_to_worker() SSH
                                            │
                                            ├─► GATE B1 ✅ (exists)
                                            │     if not fwd_survivors:
                                            │       pwc.shutdown()
                                            │       return {"pruned": True}
                                            │
                                            ├─► pwc.run_sieve_pass(java_lcg_reverse)
                                            │     [Pass 2: reverse constant-skip]
                                            │     → intersect → bidirectional_constant
                                            │
                                            ├─► if test_both_modes:
                                            │     ├─► pwc.run_sieve_pass(java_lcg_hybrid)
                                            │     │     [Pass 3: forward hybrid × 5 strategies]
                                            │     │
                                            │     ├─► GATE B2 ✗ MISSING — confirmed bug (Q0)
                                            │     │     (no check on hybrid forward survivors)
                                            │     │
                                            │     └─► pwc.run_sieve_pass(java_lcg_hybrid_reverse)
                                            │           [Pass 4: reverse hybrid × 5 strategies]
                                            │           → intersect → bidirectional_variable
                                            │
                                            ├─► update_trial_stats() → dashboard
                                            ├─► pwc.shutdown()
                                            └─► return results → NPZ accumulator
```

### Pruning in Path B

| Gate | Location | Status | Behavior |
|------|----------|--------|----------|
| B1 — constant-skip forward = 0 | `persistent_worker_coordinator.py` | ✅ Exists | `return {"pruned": True}` — PWC shuts down, trial skipped |
| B2 — hybrid forward = 0 | Between Pass 3 and Pass 4 | ✗ Missing | Pass 4 runs unconditionally — **confirmed bug (Q0)** |

---

## Differences Between Paths

| Aspect | Path A (coordinator) | Path B (PWC) |
|--------|---------------------|--------------|
| Worker lifecycle | Spawned per job | Persistent — kept alive between chunks |
| Pruning mechanism | Optuna `raise TrialPruned()` | Manual `return {"pruned": True}` |
| Gate A1/B1 | `forward_count == 0` | `not fwd_survivors` |
| Gate A2/B2 | Missing | Missing |
| Dashboard updates | Via `coordinator._progress_writer` | Via `pwc._progress_writer` |
| Hybrid seed input | Full seed range | Full seed range |

---

## Confirmed Bug — Q0 (S147)

**Both paths** are missing a gate between Pass 3 and Pass 4.

If hybrid forward (Pass 3) finds zero survivors, the intersection `bidirectional_variable = fwd_h_map.keys() & rev_h_map.keys()` is guaranteed to be empty — making Pass 4 pure wasted compute.

### Proposed fix (both paths)

```python
fwd_h_survivors = fwd_h_result.get("survivors", [])
print(f"      Forward (variable): {len(fwd_h_survivors):,} survivors")

# Q0 fix — mirror of constant-skip gate B1
if not fwd_h_survivors:
    print("      Hybrid forward zero survivors — skipping hybrid reverse")
    # do NOT prune the trial — constant-skip results are still valid
    bidirectional_variable = set()
else:
    # Pass 4: hybrid reverse
    rev_h_result = pwc.run_sieve_pass(prng_hybrid_rev, ...)
    ...
```

**Critical:** This is a **skip**, not a trial prune. The constant-skip bidirectional result (719 survivors in S147 Trial 1) is preserved. Only Pass 4 is skipped.

**Team Beta ruling (Q0):** Option A — implement in both PWC and legacy coordinator paths.

---

## Pending Fixes (S147 TB Rulings)

| ID | Description | Target file(s) | Status |
|----|-------------|----------------|--------|
| Q0 | Add hybrid forward zero-survivor gate | `persistent_worker_coordinator.py`, `window_optimizer_integration_final.py` | Pending implementation |
| Q1 | Step 1 WATCHER timeout: `step_timeout_overrides: {1: 720}` | `agents/watcher_agent.py` or manifest | Pending |
| Q2 | Single strategy (balanced_hybrid) for full-range hybrid scan | `persistent_worker_coordinator.py`, manifest | Pending |

---

## Runtime Profile (S147 observed — 1B seeds, 26 GPUs)

| Pass | Estimated duration | Notes |
|------|--------------------|-------|
| Pass 1 (forward constant) | ~15 min | ~18,000 seeds/sec cluster-wide |
| Pass 2 (reverse constant) | ~15 min | Same throughput |
| Pass 3 (forward hybrid) | ~4-5 hrs | 5 strategies × 1B seeds |
| Pass 4 (reverse hybrid) | ~4-5 hrs | 5 strategies × 1B seeds |
| **Total per trial** | **~9-11 hrs** | With current 5-strategy config |
| **Total with Q2 (1 strategy)** | **~2.5-3 hrs** | ~5× faster hybrid passes |

WATCHER default timeout: 120 minutes — insufficient for full trial.  
TB-approved override: `step_timeout_overrides: {1: 720}` (12 hours).

---

*Generated S147 — 2026-03-17 — Team Alpha*
