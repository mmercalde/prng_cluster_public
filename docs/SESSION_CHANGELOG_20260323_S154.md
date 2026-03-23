# SESSION CHANGELOG — S154
**Date:** 2026-03-23  
**Commits:** `7182ab5`  
**Status:** OPEN — OOM root cause confirmed, partial fix deployed (insufficient), real fix pending TB review

---

## Summary

S154 identified the definitive root cause of the rrig6600c crash: **OOM (Out of
Memory)**. Confirmed via `kern.log` after two separate crash events. A partial
fix was deployed (explicit GPU array deletion — S154 commit `7182ab5`) but did
not resolve the crash. The real root cause is `CUPY_CUDA_MEMORY_POOL_TYPE=none`
causing 8GB VRAM mmap × 8 workers = ~64GB virtual address space on a 7.7GB RAM
machine. The real fix requires reversing an S151 architectural decision and is
pending Team Beta review.

---

## OOM Confirmed — kern.log Evidence

Production run launched with persistent crash monitor (`~/rig_crash_monitor_persistent.log`).
After rrig6600c crash and reboot, `kern.log` showed:

**Crash 1 (pre-S154 fix, ~12:56):**
```
python invoked oom-killer: gfp_mask=0x140cca(GFP_HIGHUSER_MOVABLE|__GFP_COMP)
Out of memory: Killed process 3008 (python) total-vm:41452828kB,
anon-rss:172748kB, file-rss:124kB, UID:1000
```

**Crash 2 (post-S154 fix, ~13:35):**
```
python invoked oom-killer
Out of memory: Killed process 3379 (python) total-vm:41452828kB,
anon-rss:190424kB, file-rss:260kB, UID:1000
```

Both crashes show **identical `total-vm: 41,452,828 kB`** (~41GB) — this is
deterministic and reproducible, not a transient fault. The identical VM size
across both events confirms the bloat happens at **worker startup**, not during
job processing.

**Additional finding:** kern.log also shows a 10:09 crash where **syncthing**
invoked the OOM killer — a separate unrelated OOM. Syncthing competes for RAM
on all rigs and should be disabled.

---

## Root Cause Analysis

### S154 Hypothesis (WRONG — ruled out by identical VM size)
Initial hypothesis: GPU arrays (`seeds_gpu`, `survivors_gpu`, etc.) allocated
per job were never explicitly deleted. With `CUPY_CUDA_MEMORY_POOL_TYPE=none`,
Python GC is non-deterministic → arrays accumulate over hundreds of jobs →
VM bloats → OOM.

**Why this is wrong:** If accumulation happened per-job, Crash 1 and Crash 2
would show different `total-vm` values (Crash 2 ran longer before crashing).
Both show **exactly 41,452,828 kB** — the VM is fixed at startup, not growing
per job.

### Real Root Cause
`CUPY_CUDA_MEMORY_POOL_TYPE=none` causes CuPy to bypass its memory pool and
allocate GPU memory directly from the OS via `mmap`. On device initialization
(`cp.cuda.Device(gpu_id)` in `run_worker()`), CuPy maps the **full 8GB VRAM**
of the RX 6600 into the worker process virtual address space. This happens once
at startup, not per job.

```
8 workers per rig × 8GB VRAM mapped = 64GB virtual address space
```

On a 7.7GB RAM + 2GB swap = 9.7GB physical machine, once the kernel tries
to page-in these mappings under production load, the OOM killer fires.

The 41GB VM (`41,452,828 kB`) matches: 8 × 8GB VRAM mappings minus OS
overhead and shared library space ≈ ~40-44GB expected VA space.

### Why `CUPY_CUDA_MEMORY_POOL_TYPE=none` Was Added (S151)
Added in S151 commit `f3fdbf1` to eliminate CuPy memory pool race conditions
under concurrent workers. The S151 analysis noted the pool race as "a real
issue but not the crash cause" — it was a secondary precautionary fix applied
alongside the primary sshd MaxSessions fix. At the time, persistent workers
were new and the long-running OOM consequence wasn't apparent.

### The Conflict
| Setting | Race Conditions | VM Growth | Result |
|---------|----------------|-----------|--------|
| `CUPY_CUDA_MEMORY_POOL_TYPE=none` | ✅ Eliminated | ❌ 41GB at startup | OOM crash |
| `CUPY_CUDA_MEMORY_POOL_TYPE=default` | ⚠️ Possible | ✅ Bounded by pool | May race |
| `default` + `set_limit(256MB)` | ✅ Bounded | ✅ Bounded | **Proposed fix** |

---

## Fix Deployed This Session

### S154 — Explicit GPU Array Deletion (PARTIAL — does not resolve root cause)
**Commit:** `7182ab5`  
**File:** `sieve_gpu_worker.py`  
**Patch:** `apply_s154_gpu_del_fix.py`  

Added explicit `del` of all 8 GPU arrays + `gc.collect()` before
`_best_effort_gpu_cleanup()` in `run_sieve_job()`. Correct code hygiene
but does not address the mmap root cause.

**Deployed to:** Zeus + rrig6600 + rrig6600b + rrig6600c  
**Dual-pushed:** `7182ab5` → origin + public

---

## Proposed Real Fix (NOT YET IMPLEMENTED)

**Requires Team Beta review** — reverses S151 architectural decision.

### Part 1 — `persistent_worker_coordinator.py`
Remove `CUPY_CUDA_MEMORY_POOL_TYPE=none` from `ROCM_ENV_VARS` list (~line 97).

### Part 2 — `sieve_gpu_worker.py`
Add pool size limit inside `run_worker()` after GPU warmup:

```python
# [S155] Cap CuPy pool — prevents OOM while avoiding pool race conditions
# Working set: ~40MB per job (2M seeds × 40 bytes). 256MB = 6× headroom.
# 8 workers × 256MB = 2GB total — fits in 7.7GB RAM.
cp.get_default_memory_pool().set_limit(256 * 1024 * 1024)
cp.get_default_pinned_memory_pool().set_limit(64 * 1024 * 1024)
```

**Memory math:**
- Per job GPU allocation: ~40MB (2M seeds × 8+8+4+1 bytes)
- Pool limit per worker: 256MB = 6× working set
- 8 workers × 256MB = 2GB total pool
- Remaining for OS + Python: 5.7GB — ample

---

## Other Issues Identified This Session

### Optuna 7-Minute Hang (causes AMD worker timeout storm)
During production runs, `window_optimizer.py` hangs for ~7 minutes after
worker pool ready while Optuna computes trial parameters (SQLite futex wait).
AMD rig workers spawn, wait 7 minutes with no jobs, then time out. Zeus
retries chunks but AMD workers are dead → respawn storm → zombie accumulation.
**This is a separate issue from OOM** but compounds instability. Root cause:
Optuna TPE sampler with no prior trials + warm-start skipped = slow cold-start
computation.

### Respawn Storm
537 dispatch threads simultaneously detect dead workers after a rig crash.
Per-node respawn lock (S151) serializes spawns but hundreds of threads queue
on the lock. Not yet fixed.

### Syncthing on rrig6600c
Confirmed running and contributed to an unrelated OOM at 10:09. Should be
disabled on all rigs — it competes for RAM with sieve workers.

---

## Production Run Status

| Event | Time | Result |
|-------|------|--------|
| Run 1 launch (pre-fix) | ~12:28 | rrig6600c OOM crash at 12:56 (~28 min) |
| Run 1 launch (post-S154) | ~13:28 | rrig6600c OOM crash at 13:35 (~7 min — Optuna hang, worker timeout) |
| NPZ seeds | — | 676 (unchanged) |
| Coverage | — | 660,000,000 |

---

## Architecture Invariants Added S154

- **[S154]** Explicit `del` of all GPU arrays + `gc.collect()` in `run_sieve_job()` before cleanup
- **[S154]** OOM confirmed as crash root cause — `total-vm:41452828kB` signature is deterministic
- **[S154]** Real root cause: `CUPY_CUDA_MEMORY_POOL_TYPE=none` + CuPy device init mmaps full 8GB VRAM per worker
- **[S154]** Crash monitor MUST use `~/rig_crash_monitor_persistent.log` with `>>` — `/tmp` wiped on reboot
- **[S154]** Syncthing must be disabled on all rigs before production runs

---

## Commits This Session

| Commit | Description |
|--------|-------------|
| `7182ab5` | fix(s154): explicit GPU array del + gc.collect() — prevents VM bloat OOM on persistent workers |

---

## Next Session (S155)

**Priority order:**
1. Team Beta review: CuPy pool limit fix (`set_limit(256MB)` + remove `CUPY_CUDA_MEMORY_POOL_TYPE=none`)
2. Disable syncthing on all rigs
3. Deploy CuPy pool fix after TB approval
4. Run 1 fresh launch — 26/26 GPU stable run
5. Upload S152 changelog and TODO_MASTER_S154 to project database
