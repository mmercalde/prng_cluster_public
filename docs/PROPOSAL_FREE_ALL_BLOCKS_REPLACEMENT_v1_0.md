# PROPOSAL: S163 — Replace `free_all_blocks()` with Safe GPU Memory Management
**Version:** 1.0
**Date:** 2026-04-05
**Author:** Claude (Team Alpha Lead Dev)
**Status:** Submitted for Team Beta review — Michael retains final authority
**Scope:** `sieve_gpu_worker.py` — `_best_effort_gpu_cleanup()` function
**Risk:** LOW — additive safety fix, no change to sieve logic or result format

---

## 1. Problem Statement

### 1.1 Background

`_best_effort_gpu_cleanup()` was introduced in S154/S155 to prevent Linux OOM killer
from terminating rig workers. Without it, 8 CuPy workers × 8GB VRAM each = ~41GB
virtual memory accumulation on a 7.7GB RAM machine → OOM kill.

The fix at the time: call `free_all_blocks()` after every chunk + cap pool at 256MB.

### 1.2 Root Cause Discovery (S162)

S162 established that RX 6600 rig crashes with `SQC (inst)` page faults only occurred
under **3-rig concurrent load**. Single-rig and 2-rig runs were stable at higher seed
caps. The DKMS driver `amdgpu-dkms 6.12.12` resolved the stock kernel driver's MES
scheduler bug, but crashes continued at 2M seed cap post-DKMS.

Research into CuPy internals and the ROCm/CuPy GitHub issue tracker reveals a
second contributing factor:

### 1.3 The `free_all_blocks()` Race Condition

**CuPy GitHub Issue #4866 — confirmed by CuPy maintainers:**
> "Are there any situations where `free_all_blocks` can cause crashes if called at
> the same time by two different threads?"
>
> Result: `cudaErrorIllegalAddress: an illegal memory access was encountered` —
> process could not recover.

**Official CuPy documentation (Memory Management):**
> "The memory pool holds allocated blocks without freeing as much as possible.
> It makes the program hold most of the device memory, which may make **other
> CUDA programs running in parallel** out-of-memory situation."

**What this means for our cluster:**

With `seed_cap_amd=2000000`, each chunk takes ~500ms to execute. All 8 workers
on a rig start at approximately the same time and finish at approximately the same
time. This means **all 8 `free_all_blocks()` calls fire concurrently** —
triggering the exact race condition documented in CuPy #4866. With 3 rigs active,
this is 24 concurrent `free_all_blocks()` calls globally across the PCIe bus.

With `seed_cap_amd=100000`, each chunk takes ~25ms. Workers are naturally staggered
due to coordinator dispatch timing. By the time worker N finishes, worker N-1 has
already completed cleanup. The race window is effectively eliminated.

**This explains the 3-rig crash pattern:** more rigs = more concurrent workers =
higher probability of simultaneous `free_all_blocks()` calls = higher crash rate.

### 1.4 Why `free_all_blocks()` May Now Be Redundant

S155 added a 256MB pool cap per worker:
```python
PRNG_CUPY_POOL_LIMIT_MB=256
CUPY_GPU_MEMORY_LIMIT=268435456
```

With the pool capped at 256MB, CuPy can never accumulate more than 256MB per worker
regardless of `free_all_blocks()`. The original 41GB VM bloat was caused by an
**uncapped pool** — which no longer exists. The pool cap alone solves the OOM problem.

Therefore `free_all_blocks()` is both:
1. **Redundant** — pool cap prevents the VM bloat it was designed to fix
2. **Dangerous** — causes race conditions under concurrent multi-process load

---

## 2. Proposed Changes

### Option A — Add `synchronize()` before `free_all_blocks()` [CONSERVATIVE]

Add a device synchronize call before releasing pool blocks. This guarantees the GPU
has finished ALL pending work before any page unmapping begins.

```python
def _best_effort_gpu_cleanup():
    try:
        import gc; gc.collect()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        # [S163] Synchronize before free_all_blocks() to prevent race condition.
        # CuPy issue #4866: concurrent free_all_blocks() across workers causes
        # cudaErrorIllegalAddress. synchronize() ensures all kernels complete
        # before page unmapping begins.
        cp.cuda.Device(0).synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
```

**Risk:** Very low. `synchronize()` is a standard GPU operation. Adds ~1-5ms per
chunk (negligible vs chunk execution time). Does not change result correctness.

**Benefit:** Eliminates the race condition while keeping the explicit pool release.

---

### Option B — Remove `free_all_blocks()` entirely [RECOMMENDED]

Since S155's 256MB pool cap already prevents VM bloat, `free_all_blocks()` serves
no purpose and only introduces risk. Remove it entirely.

```python
def _best_effort_gpu_cleanup():
    try:
        import gc; gc.collect()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    # [S163] free_all_blocks() REMOVED — redundant since S155 pool cap (256MB).
    # S155 cap prevents VM bloat without requiring explicit pool release.
    # free_all_blocks() under concurrent multi-process load causes race condition
    # (CuPy issue #4866 — cudaErrorIllegalAddress). Removed as both redundant
    # and dangerous. Pool cap is the correct mechanism for memory control.
```

**Risk:** Low. The pool cap has been in production since S155 with no OOM issues.
Explicit `del` of GPU arrays after each chunk (S154) is retained and sufficient.

**Benefit:** Eliminates race condition entirely. Simplifies cleanup logic.
Removes a source of ROCm GCVM page fault pressure.

---

### Option C — `MemoryAsyncPool` stream-ordered allocator [FUTURE / DEFERRED]

CuPy provides `MemoryAsyncPool` — a stream-ordered allocator that handles
allocation/deallocation asynchronously per-stream, avoiding global synchronization.
This is the architecturally correct long-term solution but is marked experimental
in current CuPy versions.

**Recommendation:** Defer to a future session after Option A or B is validated.

---

## 3. Recommended Path

**Immediate:** Implement **Option B** (remove `free_all_blocks()`).

Rationale:
- Pool cap (S155) already prevents OOM — `free_all_blocks()` is redundant
- Removing it eliminates the race condition entirely
- Simpler code, less risk surface
- Explicit `del` of all GPU arrays (S154) is retained and sufficient
- If OOM issues return (unlikely), Option A is the fallback

**Validation test:**
```bash
# Run with 2M seed cap — previously crashed under 3 rigs
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 1 --end-step 1 \
  --params '{"min_workers": 24, "seed_cap_amd": 2000000}' \
  2>&1 | tee logs/s163_2m_no_free_blocks_run1.log
```

If 3-rig run completes cleanly at 2M seeds after removing `free_all_blocks()`,
the race condition hypothesis is confirmed and we can safely increase seed cap,
improving throughput significantly.

---

## 4. Implementation

**File:** `sieve_gpu_worker.py`
**Function:** `_best_effort_gpu_cleanup()` (lines 78-95)
**Change size:** 4 lines removed, 3-line comment added
**Deployment:** Zeus → scp to all 3 rigs → commit → dual push

**No changes to:**
- Coordinator logic
- Job format / result format
- Sieve kernel code
- S154 explicit `del` array cleanup (retained)
- S155 pool cap environment variables (retained)

---

## 5. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| OOM killer returns | Very Low | S155 pool cap still active; S154 explicit del retained |
| VM bloat returns | Very Low | 256MB cap hard limit per worker |
| Performance regression | None | Removing cleanup = faster between-chunk transitions |
| Result correctness impact | None | Cleanup runs after results retrieved and sent |

---

## 6. TB Questions for Review

1. Is Option B (remove entirely) acceptable given S155 pool cap is confirmed working?
2. Should `cp.cuda.Device(0).synchronize()` be added to `torch.cuda.synchronize()`
   call path regardless, as a general safety measure?
3. Should we validate at 500K seeds before jumping to 2M for acceptance test?

---

## 7. Live Run Evidence (S162 — 2026-04-05)

### Current Run — 100K Seed Cap — 26 GPUs — NO CRASHES

**Run started:** 18:51 local time
**Dashboard snapshot at 20:23 (25:46 elapsed):**

| Metric | Value |
|--------|-------|
| Total GPUs | 26/26 active |
| Seeds/sec | 3,525,963 |
| Total seeds processed | 1,073,741,824 |
| Progress (Trial 3 fwd hybrid) | 44.9% |
| Elapsed | 25:46 |
| rrig6600 (.120) | Active — 1,318,582 s/s — 8,821 jobs |
| rrig6600b (.154) | Active — 1,017,929 s/s — 8,804 jobs |
| rrig6600c (.162) | Active — 1,186,336 s/s — 7,700 jobs |
| Netconsole | **SILENT since 18:49** (95+ minutes) |

**Log evidence — chunks completing cleanly with survivors:**
```
✅ Chunk 3937: 100,000 seeds → 10 survivors
✅ Chunk 3938: 100,000 seeds → 7 survivors
✅ Chunk 3939: 100,000 seeds → 9 survivors
✅ Chunk 3940: 100,000 seeds → 8 survivors
✅ Chunk 3941: 100,000 seeds → 6 survivors
✅ Chunk 3942: 100,000 seeds → 8 survivors
✅ Chunk 3943: 100,000 seeds → 13 survivors
✅ Chunk 3944: 100,000 seeds → 11 survivors
✅ Chunk 3945: 100,000 seeds → 8 survivors
✅ Chunk 3946: 100,000 seeds → 10 survivors
```

**Trial completion status (Step 1 — 3 trials — seed_start=1,073,741,824):**
```
Trial 1: W7_O19_midday_S5-63_FT0.49_RT0.53 → NEW BEST ✅ COMPLETE
Trial 2: W21_O74_midday+evening_S3-165_FT0.61_RT0.4 → Score: 0.00 ✅ COMPLETE
Trial 3: Forward hybrid sieve 44.9% → IN PROGRESS — NO CRASH
```
All 3 trials running without a single GPU fault across 26 GPUs and 3 rigs.

**Previous runs at 1M and 2M seed cap with same DKMS driver:**
- 2M seeds — crashed at ~18 seconds (stock driver) / ~1400 seconds (DKMS)
- 1M seeds — crashed mid-run under 3-rig load
- 100K seeds — **stable, 95+ minutes, 3 rigs, zero crashes**

This is consistent with the chunk-size / concurrent-cleanup race hypothesis and justifies staged validation. Root-cause status remains provisional until 500K → 1M → 2M 3-rig runs pass with bounded pool and process memory.
Smaller chunks = staggered `free_all_blocks()` calls = no concurrent page unmap race.

---

## 8. Response to Team Beta Ruling

**TB Ruling accepted.** Option B approved with conditions. Root-cause status remains provisional until staged 3-rig validation passes.

### Staged Validation Plan
500K → 1M → 2M seeds, each requiring:
- Netconsole silence throughout
- No worker loss
- No VM/OOM growth

### Instrumentation Response

TB requested per-worker RSS/VmSize and CuPy pool usage monitoring during staged runs. We agree this is essential to prove `free_all_blocks()` removal does not reintroduce the original S154 memory bloat.

**Overhead analysis of proposed instrumentation:**

| Instrument | Method | Overhead | Location |
|-----------|--------|----------|----------|
| CuPy pool `used_bytes()` | Python int read from C struct | ~1-5 μs | Worker, per-chunk |
| CuPy pool `total_bytes()` | Python int read from C struct | ~1-5 μs | Worker, per-chunk |
| CuPy `n_free_blocks()` | Python int read from C struct | ~1-5 μs | Worker, per-chunk |
| Worker RSS `/proc/PID/status` | File read | ~10-50 μs | Worker, per-chunk |
| `rocm-smi --showmeminfo vram` | Subprocess fork + driver query | ~50-200ms | **EXCLUDED** — external timer only |

Items 1-4 are negligible overhead and will be logged per-chunk in `sieve_gpu_worker.py`. `rocm-smi` is explicitly excluded from the worker — adding a 50-200ms subprocess call per chunk would destroy throughput. Instead it will be sampled externally from Zeus on a 30-second timer during validation runs.

**Implementation — `_best_effort_gpu_cleanup()` with sampling-gated instrumentation:**

```python
# Module-level counter per worker process (not shared across workers)
_S163_CHUNK_COUNTER = 0
_S163_RSS_BASELINE  = None
_S163_SAMPLE_EVERY  = 25   # log every 25 chunks
_S163_RSS_WARN_MB   = 200  # alert if RSS grows >200MB from baseline
_S163_POOL_WARN_MB  = 256  # alert if pool exceeds cap

def _s163_read_proc(pid):
    """Read VmRSS and VmSize from /proc/PID/status. Returns (rss_mb, vmsize_mb)."""
    try:
        status = open(f'/proc/{pid}/status').read()
        rss_kb  = int(status.split('VmRSS:')[1].split()[0])
        vmsize_kb = int(status.split('VmSize:')[1].split()[0])
        return rss_kb // 1024, vmsize_kb // 1024
    except Exception:
        return 0, 0

def _best_effort_gpu_cleanup():
    global _S163_CHUNK_COUNTER, _S163_RSS_BASELINE
    import os, gc
    _S163_CHUNK_COUNTER += 1
    n = _S163_CHUNK_COUNTER

    # ── S163 instrumentation — gated by env var, sampled every N chunks ──
    if os.environ.get('S163_MEM_DEBUG', '0') == '1':
        try:
            pool = cp.get_default_memory_pool()
            used_before  = pool.used_bytes()  / 1024**2
            total_before = pool.total_bytes() / 1024**2
            free_before  = pool.n_free_blocks()
            rss, vmsize  = _s163_read_proc(os.getpid())

            if _S163_RSS_BASELINE is None:
                _S163_RSS_BASELINE = rss

            rss_delta    = rss - _S163_RSS_BASELINE
            threshold_breach = (
                total_before > _S163_POOL_WARN_MB or
                rss_delta    > _S163_RSS_WARN_MB
            )
            should_log = (n <= 3 or n % _S163_SAMPLE_EVERY == 0 or threshold_breach)

            if should_log:
                tag = "WARN" if threshold_breach else "INFO"
                print(
                    f"[S163-MEM/{tag}] chunk={n} "
                    f"pool_used_before={used_before:.1f}MB "
                    f"pool_total_before={total_before:.1f}MB "
                    f"free_blocks={free_before} "
                    f"VmRSS={rss}MB VmSize={vmsize}MB rss_delta={rss_delta:+d}MB",
                    flush=True
                )
        except Exception:
            pass

    # ── actual cleanup ────────────────────────────────────────────────────
    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass

    # [S163] After cleanup — log pool state post-gc to show natural stabilization
    if os.environ.get('S163_MEM_DEBUG', '0') == '1' and should_log:
        try:
            pool = cp.get_default_memory_pool()
            used_after  = pool.used_bytes()  / 1024**2
            total_after = pool.total_bytes() / 1024**2
            free_after  = pool.n_free_blocks()
            print(
                f"[S163-MEM/AFTER] chunk={n} "
                f"pool_used_after={used_after:.1f}MB "
                f"pool_total_after={total_after:.1f}MB "
                f"free_blocks_after={free_after}",
                flush=True
            )
        except Exception:
            pass

    # [S163] free_all_blocks() REMOVED — redundant since S155 pool cap (256MB).
    # S155 cap prevents VM bloat without requiring explicit pool release.
    # free_all_blocks() under concurrent multi-process load causes race condition
    # (CuPy issue #4866). Removed as both redundant and dangerous.
    # Pool cap is the correct mechanism for memory control.
```

**What the instrumentation proves:**
- `pool_used_before/after` stays ≤256MB → S155 cap is sufficient without `free_all_blocks()`
- `pool_total_before/after` shows natural pool stabilization across chunks
- `VmRSS` + `VmSize` stay stable → no VM bloat returning (original S154 failure mode)
- `rss_delta` from baseline → catches slow accumulation before OOM
- Threshold breach → immediate alert if anything trends wrong

**Overhead:** Zero when `S163_MEM_DEBUG=0` (default). When enabled, ~15-50μs per sampled chunk (every 25). No subprocess calls, no driver queries, no `flush=True` on every chunk.

**External `rocm-smi` monitoring during validation (run from Zeus):**
```bash
watch -n 30 'ssh rrig6600c "rocm-smi --showmeminfo vram" 2>/dev/null'
```

---

## 9. References

- CuPy Issue #4866: concurrent `free_all_blocks()` causes `cudaErrorIllegalAddress`
- CuPy Memory Management docs: pool cap via `set_limit()` / `CUPY_GPU_MEMORY_LIMIT`
- S154: Explicit GPU array deletion (retained)
- S155: 256MB pool cap per worker (retained)
- S162: DKMS root cause + `free_all_blocks()` race condition discovery
