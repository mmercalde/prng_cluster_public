
---

## 14. Inter-Chunk GPU Cleanup (Added 2026-01-26)

### Problem Identified

Step 1 forward sieves process seeds in chunks (~19K seeds/chunk). With large seed spaces (e.g., 500K seeds = 26 chunks), VRAM fragmentation accumulated without cleanup, causing intermittent GPU hangs:
```
Error: HW Exception by GPU node-11... reason: GPU Hang
```

### Root Cause

| Step | Chunks/Invocation | Cleanup Frequency | Result |
|------|-------------------|-------------------|--------|
| Step 1 | ~26 | Once at exit | **GPU hangs** |
| Step 2.5/3 | 1 | Every invocation | Stable |

### Fix Applied

Added inter-chunk cleanup to both forward sieve loops in `sieve_filter.py` (lines 230, 385):
```python
if chunk_start + chunk_size < seed_end:
    _best_effort_gpu_cleanup()
```

Also added `gc.collect()` to `_best_effort_gpu_cleanup()`.

### Validation

- 20/20 benchmark trials: 0 GPU hangs
- All 26 GPUs healthy post-run
- Performance overhead: <5%

---

## 15. Persistent Worker Execution Path (S146)

### Two Sieve Execution Backends

As of S146, Step 1 supports two sieve execution backends:

| Mode | Flag | Backend path |
|------|------|-------------|
| Default (legacy) | (none) | `coordinator.py` → `sieve_filter.py` |
| Persistent workers | `--use-persistent-workers` | `PWC` → `sieve_gpu_worker.py --persistent` |

The persistent worker path keeps sieve workers alive between trials, eliminating
SSH process spawn overhead on every chunk.

### Persistent Worker Call Chain

```
watcher_agent.py
  └─► window_optimizer_integration_final.py  (use_persistent_workers=True)
        └─► run_trial_persistent()   (persistent_worker_coordinator.py:669)
              └─► PersistentWorkerCoordinator
                    Zeus:    execute_local_sieve_job()  ──► sieve_filter.py
                    Remote:  _dispatch_to_worker()       ──► sieve_gpu_worker.py --persistent
```

### Hybrid Routing in sieve_gpu_worker.py

`sieve_gpu_worker.py` handles four sieve pass types:

| Pass type | Kernel family field | Arg tail |
|-----------|---------------------|----------|
| Constant skip forward | `prng_families` (base) | standard |
| Constant skip reverse | `prng_families` (reverse) | standard |
| Hybrid forward | `prng_families` (hybrid) | `threshold, a, c` |
| Hybrid reverse | `prng_families` (hybrid_reverse) | `threshold, offset` |

The hybrid forward and reverse branches are implemented as **separate elif blocks** —
they must not share kernel_args construction.

### S146 Validation

All 4 pass types validated on live hardware (3 trials, 10M seeds, Zeus + 3 rigs):
313 bidirectional survivors found (274 constant + 40 variable skip).
666 total in NPZ accumulator after S146 preprod run.

---

## 15. Persistent Worker Execution Path (S146)

### Two Sieve Execution Backends

As of S146, Step 1 supports two sieve execution backends:

| Mode | Flag | Backend path |
|------|------|-------------|
| Default (legacy) | (none) | `coordinator.py` → `sieve_filter.py` |
| Persistent workers | `--use-persistent-workers` | `PWC` → `sieve_gpu_worker.py --persistent` |

The persistent worker path keeps sieve workers alive between trials, eliminating
SSH process spawn overhead on every chunk.

### Persistent Worker Call Chain

```
watcher_agent.py
  └─► window_optimizer_integration_final.py  (use_persistent_workers=True)
        └─► run_trial_persistent()   (persistent_worker_coordinator.py:669)
              └─► PersistentWorkerCoordinator
                    Zeus:    execute_local_sieve_job()  ──► sieve_filter.py
                    Remote:  _dispatch_to_worker()       ──► sieve_gpu_worker.py --persistent
```

### Hybrid Routing in sieve_gpu_worker.py

`sieve_gpu_worker.py` handles four sieve pass types:

| Pass type | Kernel family field | Arg tail |
|-----------|---------------------|----------|
| Constant skip forward | `prng_families` (base) | standard |
| Constant skip reverse | `prng_families` (reverse) | standard |
| Hybrid forward | `prng_families` (hybrid) | `threshold, a, c` |
| Hybrid reverse | `prng_families` (hybrid_reverse) | `threshold, offset` |

The hybrid forward and reverse branches are implemented as **separate elif blocks** —
they must not share kernel_args construction.

### S146 Validation

All 4 pass types validated on live hardware (3 trials, 10M seeds, Zeus + 3 rigs):
313 bidirectional survivors found (274 constant + 40 variable skip).
666 total in NPZ accumulator after S146 preprod run.
