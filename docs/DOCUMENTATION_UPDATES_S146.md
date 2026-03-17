# DOCUMENTATION UPDATES — S146
## Persistent Worker Coordinator Fixes & Web Dashboard

**Date:** March 16, 2026
**Session:** S146
**Applies to:** CHAPTER_1, CHAPTER_2, CHAPTER_9, CHAPTER_12, COMPLETE_OPERATING_GUIDE_v2_0

---

## CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md — ADDENDUM S146

### Append to "Persistent Worker Engine" section:

---

## Persistent Worker Engine — S146 Fixes & Invariants

### Architecture Invariants (CRITICAL — do not violate)

**Localhost semaphore (Bug 1, S146):**
Zeus local sieve dispatch MUST be gated by `_localhost_semaphore = threading.Semaphore(2)`.
Without this, concurrent Zeus local sieve_filter.py subprocesses saturate both CUDA GPUs
→ `cudaErrorDevicesUnavailable`. Matches `coordinator.py` `_localhost_semaphore` (line 269).

**Strategy dict format (Bug 10, S146):**
Strategies sent to workers MUST be full `StrategyConfig.to_dict()` dicts — all 6 fields required:
`name`, `max_consecutive_misses`, `skip_tolerance`, `enable_reseed_search`,
`skip_learning_rate`, `breakpoint_threshold`.
Sending only `max_consecutive_misses` + `skip_tolerance` causes `StrategyConfig.__init__()` crash
in `sieve_filter.py` line 485.

**Hybrid kernel signatures (Bug 4, S146):**
Forward and reverse hybrid use DIFFERENT kernel arg tails — never combine into one branch:

| Kernel | Tail args |
|--------|-----------|
| `java_lcg_hybrid_multi_strategy_sieve` (forward) | `float threshold, unsigned long long a, unsigned long long c` |
| `java_lcg_hybrid_reverse_sieve` (reverse) | `float threshold, int offset` (a,c hardcoded inside) |

**Hybrid threshold (Bug 5, S146):**
Hybrid passes MUST use `phase2_threshold` not `min_match_threshold`.
`hybrid_threshold = coerce_threshold(phase2_raw, threshold) if phase2_raw is not None else threshold`
This applies to BOTH the kernel launch AND the post-filter survivor loop.

**`log_gpu_result()` required (Bug 12, S146):**
PWC MUST call `self._progress_writer.log_gpu_result(hostname, gpu_id, gpu_type, seeds, elapsed)`
after every successful chunk dispatch. Without this the web dashboard shows 0 seeds/sec and
0 active workers even during a live run.

### Validated Operating Parameters (S146)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `worker_pool_size` | 4 | Validated stable; 8 causes memory pressure |
| `JOB_TIMEOUT_S` | 600 | 300 causes false timeouts on large chunks |
| `_localhost_semaphore` | Semaphore(2) | 2 = Zeus GPU count |
| Spawn stagger | 4.0s | ROCm HIP init stability |

### Validation Results (S146 — live hardware)

3 trials, 10M seeds, `--seed-start 500000000`, `--test-both-modes`, `--use-persistent-workers`:

| Pass | Result |
|------|--------|
| `java_lcg` forward constant | ✅ 43, 2 survivors |
| `java_lcg_reverse` constant | ✅ 40, 0 survivors |
| `java_lcg_hybrid` forward variable | ✅ 671, 201 survivors |
| `java_lcg_hybrid_reverse` variable | ✅ 62, 10 survivors |

Zero ❌ errors. Zero crashes. All 4 passes fully operational.

---

## CHAPTER_1_WINDOW_OPTIMIZER.md — ADDENDUM S146

### Append to "Execution Flags" or "CLI Reference" section:

---

## Persistent Worker Mode (S130/S146)

When `--use-persistent-workers` is passed, Step 1 sieve execution bypasses `coordinator.py`
and uses `persistent_worker_coordinator.py` (PWC) instead.

**Execution path:**
```
window_optimizer.py --use-persistent-workers
  → window_optimizer_integration_final.py
    → run_trial_persistent()
      → PersistentWorkerCoordinator.run_sieve_pass()
        Zeus:   _dispatch_local_sieve() → sieve_filter.py subprocess
        Remote: _dispatch_to_worker()  → sieve_gpu_worker.py --persistent
```

**Four sieve passes per trial (test-both-modes=True):**
1. Forward constant skip (`java_lcg`) → `sieve_gpu_worker.py`
2. Reverse constant skip (`java_lcg_reverse`) → `sieve_gpu_worker.py`
3. Forward hybrid variable skip (`java_lcg_hybrid`) → `sieve_gpu_worker.py`
4. Reverse hybrid variable skip (`java_lcg_hybrid_reverse`) → `sieve_gpu_worker.py`

**Key CLI flags:**
```
--use-persistent-workers    Enable PWC path (default: True as of S145)
--worker-pool-size 4        Workers per rig (validated: 4, not 8)
--seed-cap-nvidia 5000000   Zeus chunk size ceiling
--seed-cap-amd 2000000      Rig chunk size ceiling
```

---

## CHAPTER_2_BIDIRECTIONAL_SIEVE.md — ADDENDUM S146

### Append to architecture section:

---

## Persistent Worker Execution Path (S146)

When `use_persistent_workers=True`, sieve execution uses `sieve_gpu_worker.py` on remote AMD rigs
instead of SSH subprocess to `sieve_filter.py`. The worker runs as a persistent process,
accepting jobs via stdin JSON-lines IPC and returning results via stdout.

**Hybrid sieve kernel signatures (CRITICAL):**

Forward hybrid (`java_lcg_hybrid`):
```c
void java_lcg_hybrid_multi_strategy_sieve(
    ..., float threshold, unsigned long long a, unsigned long long c
)
```

Reverse hybrid (`java_lcg_hybrid_reverse`):
```c
void java_lcg_hybrid_reverse_sieve(
    ..., float threshold, int offset   // a,c hardcoded inside kernel
)
```

These signatures differ — the worker must dispatch them through separate code branches.

**Threshold semantics for hybrid:**
- Kernel threshold: `phase2_threshold` (NOT `min_match_threshold`)
- Post-filter threshold: same `phase2_threshold`
- `sieve_filter.py` behavior preserved for consistency

---

## CHAPTER_12_WATCHER_AGENT.md — ADDENDUM S146

### Update Step 1 execution path description:

---

## Step 1 Execution Path — Persistent Worker Mode (S146)

Step 1 (Window Optimizer) now defaults to persistent worker mode via manifest flag
`use_persistent_workers: true`. The execution path is:

```
WATCHER → watcher_agent.py → window_optimizer.py
  [--use-persistent-workers]
    → persistent_worker_coordinator.py (PWC)
      → sieve_gpu_worker.py (AMD rigs, persistent process)
      → sieve_filter.py subprocess (Zeus local, via _dispatch_local_sieve)
```

**Dashboard integration:** PWC writes to `/tmp/cluster_progress.json` via `ProgressWriter`:
- `register_node()` — on startup for each node
- `update_step()` — before each sieve pass (step name + total seeds)
- `log_gpu_result()` — after each successful chunk (per-node throughput)
- `update_progress()` — after each complete sieve pass
- `finish()` — on shutdown

**Web dashboard:** All routes now return 200. The `read_progress()` function in
`web_dashboard.py` guarantees `trial_stats` always has all 9 required keys with defaults,
preventing `Undefined.__format__` crashes on the overview/plots/stats routes.

---

## COMPLETE_OPERATING_GUIDE_v2_0.md — ADDENDUM S146

### Append to "Step 1 Operations" section:

---

## Step 1 Persistent Worker Operations (S146)

### Launch (via sweep script — preferred)
```bash
bash sweep_run1.sh
```

### Launch (direct test)
```bash
python3 window_optimizer.py \
    --lottery-file daily3.json \
    --strategy bayesian \
    --max-seeds 1073741824 \
    --prng-type java_lcg \
    --output optimal_window_config.json \
    --test-both-modes \
    --trials 50 \
    --use-persistent-workers \
    --worker-pool-size 4 \
    --seed-cap-nvidia 5000000 \
    --seed-cap-amd 2000000 \
    --enable-pruning \
    > logs/pwc_test.log 2>&1 &
```

### Monitor live
```bash
tail -f logs/pwc_test.log
# Web dashboard: http://45.32.131.224:5002
```

### Worker health check
```bash
# Check all workers alive (expect 12/12)
grep "worker alive\|Worker pool ready" logs/pwc_test.log | tail -15
```

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `cudaErrorDevicesUnavailable` | Missing localhost semaphore | Check PWC `_localhost_semaphore` |
| `UnboundLocalError: time` | Inline `import time` in retry block | Remove inline import |
| `StrategyConfig.__init__() missing args` | Strategy dict only has 2 fields | Use `s.to_dict()` for full dict |
| `Undefined.__format__` on dashboard | `trial_stats` missing from progress file | `read_progress()` now adds defaults |
| Hybrid chunks all ❌ | Wrong kernel arg tail for reverse | Check forward/reverse split in sieve_gpu_worker.py |
| Dashboard shows 0 seeds/sec | `log_gpu_result()` not called | PWC must call after each chunk |

---

**END OF S146 DOCUMENTATION UPDATES**
