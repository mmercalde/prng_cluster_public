# PROPOSAL — CuPy Memory Pool Fix (S155)
**Submitted by:** Team Alpha  
**Date:** 2026-03-23  
**Session:** S154  
**Priority:** P0 — blocks all production runs  
**Requires TB ruling before implementation**

---

## Problem Statement

All three AMD rigs (rrig6600, rrig6600b, rrig6600c) crash with OOM during
production runs. The crash is deterministic and reproducible — confirmed across
3 separate OOM events on rrig6600c via `kern.log`:

```
Out of memory: Killed process XXXX (python) total-vm:41452828kB
```

**`total-vm: ~41GB`** on machines with **7.7GB RAM + 2GB swap = 9.7GB physical**.

The identical VM size across all three events confirms the bloat happens at
**worker startup**, not during job processing.

---

## Root Cause

`CUPY_CUDA_MEMORY_POOL_TYPE=none` (added S151, commit `f3fdbf1`) causes CuPy
to bypass its memory pool and allocate GPU memory directly from the OS via
`mmap`. On device initialization (`cp.cuda.Device(gpu_id)` in `run_worker()`),
CuPy maps the **full 8GB VRAM** of each RX 6600 into the worker process virtual
address space.

```
8 workers per rig × 8GB VRAM mapped = 64GB virtual address space
```

Under production load, the kernel attempts to back these VA mappings with
physical pages → OOM killer fires → worker killed → rig may hard-lock.

### Why `CUPY_CUDA_MEMORY_POOL_TYPE=none` Was Added (S151)
Added to eliminate CuPy memory pool **race conditions** under concurrent
workers. S151 analysis: *"CuPy memory pool race — Real issue but not the
crash cause."* It was a secondary precautionary fix alongside the primary
sshd MaxSessions fix. The long-running OOM consequence was not apparent at
the time because persistent workers were new.

### The Conflict
| Setting | Race Conditions | VM Growth | Outcome |
|---------|----------------|-----------|---------|
| `CUPY_CUDA_MEMORY_POOL_TYPE=none` | ✅ Eliminated | ❌ 41GB at startup | OOM — production blocked |
| `CUPY_CUDA_MEMORY_POOL_TYPE=default` (no limit) | ⚠️ Possible | ⚠️ Unbounded pool | May race or OOM |
| `default` + `set_limit(256MB)` per worker | ✅ Bounded | ✅ 2GB total | **Proposed fix** |

---

## Proposed Fix

### Part 1 — `persistent_worker_coordinator.py`
Remove `CUPY_CUDA_MEMORY_POOL_TYPE=none` from `ROCM_ENV_VARS` list (~line 97).

**Before:**
```python
ROCM_ENV_VARS = [
    ...
    "CUPY_CUDA_MEMORY_POOL_TYPE=none",
    ...
]
```

**After:**
```python
ROCM_ENV_VARS = [
    ...
    # [S155] Removed CUPY_CUDA_MEMORY_POOL_TYPE=none — caused 41GB VM mmap OOM.
    # Pool race condition now handled via set_limit() in sieve_gpu_worker.py.
    ...
]
```

### Part 2 — `sieve_gpu_worker.py`
Add pool size limit inside `run_worker()` after GPU warmup, before first job:

```python
# [S155] Cap CuPy memory pool — prevents OOM while bounding race conditions.
# CUPY_CUDA_MEMORY_POOL_TYPE=none caused 8GB VRAM mmap × 8 workers = 64GB VA → OOM.
# Pool limit: 256MB = 6× per-job working set (~40MB). 8 workers × 256MB = 2GB total.
cp.get_default_memory_pool().set_limit(256 * 1024 * 1024)
cp.get_default_pinned_memory_pool().set_limit(64 * 1024 * 1024)
```

---

## Memory Safety Analysis

### Per-job GPU allocation (2M seeds, non-hybrid):
| Array | Size |
|-------|------|
| `seeds_gpu` (uint64) | 16 MB |
| `survivors_gpu` (uint64) | 16 MB |
| `match_rates_gpu` (float32) | 8 MB |
| `best_skips_gpu` (uint8) | 2 MB |
| `survivor_count_gpu` | <1 MB |
| `residues_gpu` (window=10) | <1 MB |
| **Total** | **~42 MB** |

### Hybrid path extras:
| Array | Size |
|-------|------|
| `strategy_ids_gpu` (uint32, 2M) | 8 MB |
| `skip_sequences_gpu` (5000×10) | <1 MB |
| **Hybrid total** | **~50 MB** |

### Pool limit headroom:
- 256MB limit = **6× non-hybrid working set**, **5× hybrid working set**
- 8 workers × 256MB = **2GB total pool** across all workers on one rig
- Remaining RAM: 7.7GB - 2GB pool - ~1GB OS/Python = **~4.7GB free**

### Race condition mitigation:
With `set_limit(256MB)`, if two workers simultaneously request allocations
that would exceed the pool, CuPy raises `OutOfMemoryError` in the worker
rather than corrupting shared state. The worker catches this and returns an
error result — coordinator retries the chunk. This is safe and recoverable,
unlike the current OOM killer which kills the entire process.

---

## Questions for Team Beta

1. **Is `set_limit()` effective on ROCm/HIP backend?** CuPy's `set_limit()`
   is documented for CUDA. Does it correctly bound the HIP memory allocator
   on AMD GPUs, or does it only affect the CUDA path?

2. **Does removing `CUPY_CUDA_MEMORY_POOL_TYPE=none` reintroduce the S151
   race condition?** The original race was under concurrent workers sharing
   the same kernel cache — now addressed by per-worker `CUPY_CACHE_DIR`
   (S152). Is the pool race a separate concern from the cache race?

3. **Is 256MB pool limit appropriate?** Working set is ~42-50MB per job.
   256MB = 5-6× headroom. Should this be higher (512MB) for safety, or
   is 256MB sufficient?

4. **Should pool limit be configurable via env var?** Suggest
   `PRNG_CUPY_POOL_LIMIT_MB` defaulting to 256, to allow tuning without
   code changes.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `set_limit()` not effective on ROCm | Medium | High — OOM persists | TB confirms ROCm support |
| Pool race reintroduced | Low | Medium — worker crashes, coordinator retries | Per-worker CUPY_CACHE_DIR (S152) already in place |
| 256MB too small for some configs | Low | Low — worker OOM error, chunk retried | Raise to 512MB if needed |
| Fix breaks Step 4/5 workers | Low | Medium | `set_limit()` only in `sieve_gpu_worker.py` run_worker() |

---

## Files to Modify

| File | Change |
|------|--------|
| `persistent_worker_coordinator.py` | Remove `CUPY_CUDA_MEMORY_POOL_TYPE=none` from ROCM_ENV_VARS |
| `sieve_gpu_worker.py` | Add `set_limit(256MB)` in `run_worker()` after GPU warmup |

**No other files need changes.** Step 4/5 workers (`full_scoring_worker.py`,
`scorer_trial_worker.py`) set `CUPY_CUDA_MEMORY_POOL_TYPE=none` independently
via `os.environ.setdefault()` — they are unaffected by this change.

---

## Blocking Status

**Production is fully blocked until this fix is approved and deployed.**
All three AMD rigs OOM crash within 5-30 minutes of launch. The 26 GPU
cluster cannot complete a single trial of Run 1.
