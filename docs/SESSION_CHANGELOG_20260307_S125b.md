# Session Changelog — S125b
**Date:** 2026-03-07
**Session:** S125b (continuation of S125)
**Focus:** Fix cudaErrorDevicesUnavailable on Zeus RTX 3080 Ti GPUs

---

## Problem

Smoke test from S125 showed both Zeus RTX 3080 Ti GPUs (GPU0, GPU1) failing with:

```
cupy_backends.cuda.api.runtime.CUDARuntimeError: cudaErrorDevicesUnavailable:
CUDA-capable device(s) is/are busy or unavailable
sieve_filter.py line 136: with self.device:
```

All 3 retry rounds failed for both localhost GPUs. AMD rigs (ROCm) were unaffected.
Retry logic correctly migrated failed jobs to the RX 6600 rigs — which is why the
pipeline completed but without Zeus GPU contribution.

---

## Root Cause Diagnosis

### Investigation Path
- Confirmed no cupy imports anywhere in parent call chain (coordinator.py,
  window_optimizer.py, window_optimizer_integration_final.py, watcher_agent.py)
- Confirmed coordinator v1.6.1 explicitly removed all CuPy usage
- Confirmed `create_gpu_workers()` and `test_connectivity()` never touch CUDA
- Confirmed multiprocessing spawn inherits clean environment (no CUDA_VISIBLE_DEVICES)
- Confirmed `_partition_worker` imports only clean modules

### Root Cause: EXCLUSIVE_PROCESS Compute Mode
```
nvidia-smi --query-gpu=index,name,compute_mode --format=csv,noheader
0, NVIDIA GeForce RTX 3080 Ti, Exclusive_Process
1, NVIDIA GeForce RTX 3080 Ti, Exclusive_Process
```

Both Zeus RTX 3080 Ti GPUs were set to `Exclusive_Process` compute mode.
In this mode, **only one process can hold the CUDA context per GPU at a time.**

When the P0 partition worker process (spawned by S125 multiprocessing dispatcher)
ran `execute_distributed_analysis` → `execute_truly_parallel_dynamic` and dispatched
TWO sieve_filter.py subprocesses (one per GPU), the first subprocess to acquire each
GPU blocked the other. All retry rounds failed because the mode was persistent.

The "when one GPU fails the other picks it up" behavior was the coordinator's retry
logic migrating localhost jobs to the AMD rigs — the RTX cards were genuinely blocked.

### Why This Wasn't Seen Before S125
Prior to S125, `n_parallel=1` used the parent coordinator directly (no spawn).
The parent coordinator dispatched sieve subprocesses sequentially or with limited
concurrency. With `n_parallel=2` and multiprocessing.Process (S125), P0 dispatches
jobs to BOTH Zeus GPUs simultaneously — exposing the Exclusive_Process conflict.

---

## Fix Applied

### Immediate Fix (Zeus)
```bash
sudo nvidia-smi -i 0 -c DEFAULT
sudo nvidia-smi -i 1 -c DEFAULT
```

Output:
```
Set compute mode to DEFAULT for GPU 00000000:1A:00.0.
Set compute mode to DEFAULT for GPU 00000000:68:00.0.
0, NVIDIA GeForce RTX 3080 Ti, Default
1, NVIDIA GeForce RTX 3080 Ti, Default
```

### Permanent Fix (reboot persistence)
`/etc/rc.local` created on Zeus:
```bash
#!/bin/bash
nvidia-smi -i 0 -c DEFAULT
nvidia-smi -i 1 -c DEFAULT
exit 0
```
Permissions: `-rwxr-xr-x 1 root root` ✅

Zeus GPUs will always boot in Default compute mode from now on.

---

## Smoke Test Result

```
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 1 --end-step 1 \
  --params '{"trials": 1, "n_parallel": 2}' >> logs/smoke_s125b_defaultmode.log 2>&1
```

**Result: NO FAILURES** ✅

- `✅ Parallel | RTX 3080 Ti@localhost(gpu0)` — working
- `✅ Parallel | RTX 3080 Ti@localhost(gpu1)` — working
- All 26 GPUs operational including both RTX 3080 Ti cards
- n_parallel=2 multiprocessing dispatcher confirmed working end-to-end

---

## Files Modified

| File | Change | Location |
|------|--------|----------|
| `/etc/rc.local` | Created — sets DEFAULT compute mode on boot | Zeus system |

No Python code changes required. Root cause was infrastructure configuration.

---

## Git Commit

```bash
cd ~/distributed_prng_analysis
git add docs/SESSION_CHANGELOG_20260307_S125b.md
git commit -m "fix(s125b): Zeus GPU compute mode DEFAULT -- was Exclusive_Process

Root cause of cudaErrorDevicesUnavailable on RTX 3080 Ti GPU0/GPU1:
Both GPUs were in Exclusive_Process compute mode. With n_parallel=2
multiprocessing.Process dispatcher (S125), P0 dispatched sieve jobs
to both GPUs simultaneously -- Exclusive_Process blocked second process.

Fix: sudo nvidia-smi -i 0 -c DEFAULT && sudo nvidia-smi -i 1 -c DEFAULT
Permanent: /etc/rc.local sets DEFAULT on every boot.

Smoke test passed: all 26 GPUs operational, no localhost failures.
RTX 3080 Ti cards fully restored to n_parallel=2 workload."

git push origin main && git push public main
```

---

## Next Steps

With smoke test passing, proceed to production Optuna resume:

```bash
cd ~/distributed_prng_analysis
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 1 --end-step 1 \
  --params '{"trials": 200, "n_parallel": 2, "resume_study": true, \
             "study_name": "window_opt_1772507547", "enable_pruning": true}'
```

**Carry-forward items (unchanged from S125):**
- Wire `n_parallel` + `enable_pruning` into `agent_manifests/window_optimizer.json`
- Variable skip bidirectional count not wired into Optuna scoring
- sklearn warnings in Step 5
- Remove CSV writer from coordinator.py
- S110 root cleanup (884 files)
- S103 Part 2 — per-seed match rates

---

## Key Numbers (End of S125b)

| Metric | Value |
|--------|-------|
| Real draws | 18,068 |
| Bidirectional survivors (S120 baseline) | 85 (W8_O43) |
| Best NN R² | +0.020538 |
| Active Optuna study | `window_opt_1772507547.db` (21 trials) |
| Zeus GPU compute mode | DEFAULT ✅ |
| n_parallel=2 status | FULLY OPERATIONAL ✅ |

---

*Session S125b — Team Alpha*
*Root cause: infrastructure (compute mode), not code*
*All 26 GPUs now operational with n_parallel=2*
