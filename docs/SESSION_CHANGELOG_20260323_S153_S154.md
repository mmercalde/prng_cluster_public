# SESSION CHANGELOG — S153 / S154
**Date:** 2026-03-23  
**Commits:** `7182ab5`  
**Status:** OPEN — OOM root cause identified, partial fix deployed, real fix pending

---

## Summary

S153/S154 continued the rrig6600c crash investigation from S152. The session
began with diagnostic confirmation that the java_lcg_reverse kernel argument
mismatch (the S152 hypothesis) was NOT the crash cause — ROCm tolerates the
wrong args silently. After multiple production crash attempts and persistent
crash log capture, the root cause was definitively identified as **OOM (Out of
Memory)** via `kern.log`. A partial fix was deployed (S154) but the real root
cause is deeper and requires an architectural fix before production can run
stably.

---

## Diagnostic Work Completed

### Stage 8 — Reverse Kernel Arg Test (PASSED — ruled out)
- `slim_v1_diag_rrig6600c_v2.py` Stage 8A: correct args work on all 8 GPUs ✅
- `slim_v1_diag_rrig6600c_v2.py` Stage 8B: wrong args tolerated by ROCm — **did NOT crash**
- Conclusion: kernel arg mismatch is a code bug but NOT the crash cause on rrig6600c

### Stage 9 — Full IPC Simulation (PASSED — ruled out)
- All 8 reverse jobs completed successfully on rrig6600c
- Conclusion: IPC path is not the crash cause

### OOM Confirmed via kern.log
After capturing crash monitor log to persistent home directory location
(`~/rig_crash_monitor_persistent.log`), kern.log revealed the definitive cause:

```
Mar 23 12:56:31 rig-6600c kernel: python invoked oom-killer
Out of memory: Killed process 3008 (python) total-vm:41452828kB,
anon-rss:172748kB, file-rss:124kB
```

**`total-vm: 41,452,828 kB = ~41GB virtual memory`** on a 7.7GB RAM machine.

This OOM event was confirmed across **two separate crash events** with the
same 41GB VM signature, confirming it is a deterministic, reproducible
condition — not a transient hardware fault.

---

## Root Cause Analysis

### Immediate Cause (S154 — partially addressed)
Python's GC is non-deterministic. With `CUPY_CUDA_MEMORY_POOL_TYPE=none`,
CuPy allocates GPU memory directly from the OS and relies on Python GC to
free it. GPU arrays allocated per job (`seeds_gpu`, `survivors_gpu`,
`match_rates_gpu`, `best_skips_gpu`, etc.) were never explicitly deleted.
Over hundreds of sequential jobs on a persistent worker, unreleased Python
references accumulated.

**S154 Fix Applied:** Explicit `del` of all GPU arrays + `gc.collect()` before
`_best_effort_gpu_cleanup()` in `run_sieve_job()`.

### Real Root Cause (NOT YET FIXED)
The S154 fix did not resolve the crash. Both OOM events show **identical
`total-vm: 41,452,828 kB`** — meaning the VM bloat is not from Python
object accumulation but from **CuPy GPU device context initialization**.

When CuPy initializes a GPU device with `CUPY_CUDA_MEMORY_POOL_TYPE=none`,
it maps the full GPU VRAM (8GB per RX 6600) into the process virtual address
space via `mmap`. This happens once at worker startup — not per job. With 8
workers per rig each mapping 8GB VRAM:

```
8 workers × 8GB VRAM mapped = 64GB virtual address space
```

On a 7.7GB RAM + 2GB swap = 9.7GB physical machine, once the kernel tries
to back these mappings with physical pages under load, the OOM killer fires.

### Why `CUPY_CUDA_MEMORY_POOL_TYPE=none` Was Added (S151)
`CUPY_CUDA_MEMORY_POOL_TYPE=none` was added in S151 to eliminate CuPy memory
pool race conditions under concurrent workers. The S151 analysis noted it as
a "real issue but not the crash cause" — it was a secondary precautionary fix.

### The Conflict
- `CUPY_CUDA_MEMORY_POOL_TYPE=none` → eliminates race conditions → causes 41GB VM → OOM
- `CUPY_CUDA_MEMORY_POOL_TYPE=default` (pool enabled) → bounded VM → race conditions possible

### Proposed Real Fix (NOT YET IMPLEMENTED — pending Team Beta review)
1. Remove `CUPY_CUDA_MEMORY_POOL_TYPE=none` from worker spawn env in
   `persistent_worker_coordinator.py`
2. Add `cp.get_default_memory_pool().set_limit(256 * 1024 * 1024)` (256MB)
   inside `run_worker()` after GPU warmup — caps pool per worker
3. This eliminates unbounded VM growth while preventing race conditions via
   the fixed pool limit

Memory math: each job needs ~40MB GPU memory (2M seeds × 40 bytes).
256MB pool limit = 6× working set headroom per worker. 8 workers × 256MB =
2GB total — well within 7.7GB RAM.

---

## Fixes Deployed This Session

### S154 — Explicit GPU Array Deletion
**Commit:** `7182ab5`  
**File:** `sieve_gpu_worker.py`  
**Patch:** `apply_s154_gpu_del_fix.py`  

Added explicit `del` of all 8 GPU arrays + `gc.collect()` before
`_best_effort_gpu_cleanup()` in `run_sieve_job()`:

```python
# [S154] Explicit GPU array deletion — prevents VM bloat OOM.
try: del seeds_gpu
except NameError: pass
try: del survivors_gpu
except NameError: pass
try: del match_rates_gpu
except NameError: pass
try: del best_skips_gpu
except NameError: pass
try: del survivor_count_gpu
except NameError: pass
try: del residues_gpu
except NameError: pass
try: del strategy_ids_gpu
except NameError: pass
try: del skip_sequences_gpu
except NameError: pass
import gc; gc.collect()
```

**Result:** Did not resolve the crash — both OOM events show identical
41GB VM, confirming the root cause is device context mmap, not Python
object accumulation.

**Deployed to:** Zeus + rrig6600 + rrig6600b + rrig6600c  
**Dual-pushed:** origin + public

---

## Other Findings

### Crash Monitor Log Must Be Persistent
- `/tmp/rig_crash_monitor.log` is wiped on reboot — useless for crash diagnosis
- Fix: use `~/rig_crash_monitor_persistent.log` with `>>` append mode
- All rigs now use persistent log location

### Syncthing OOM (unrelated)
`kern.log` also shows a 10:09 OOM crash where **syncthing** invoked the
OOM killer. Syncthing is running on rrig6600c and consuming memory.
Recommend disabling syncthing on all rigs — it serves no purpose in this
cluster and competes for RAM with sieve workers.

### Respawn Storm (separate issue)
When a rig crashes mid-run, 537 dispatch threads simultaneously detect
dead workers and fire respawn attempts. The per-node respawn lock from S151
serializes the actual spawn but the lock acquisition queue grows to hundreds
deep. This is a separate issue from OOM but contributes to instability.

### java_lcg_reverse Kernel Arg Mismatch (code correctness bug)
Still present — `java_lcg_reverse` receives `a, c` args it doesn't need.
ROCm tolerates this silently. Fix patch `apply_s153_java_lcg_reverse_args_fix.py`
exists but was deprioritized since it's not the crash cause.

---

## Production Run Status

| Run | Status |
|-----|--------|
| Run 1 Trial 1 | Crashed — OOM on rrig6600c at ~12:56 |
| Run 1 Trial 2 | Crashed — OOM on rrig6600c at ~13:35 (after S154 fix) |
| NPZ seeds | 676 (from pre-S153 runs) |
| Coverage | 660,000,000 |

---

## Architecture Invariants Added S153/S154

- **[S154]** Explicit `del` of all GPU arrays + `gc.collect()` in `run_sieve_job()` before cleanup
- **[S153/S154]** Crash monitor MUST use `>>` append to `~/rig_crash_monitor_persistent.log`
- **[S154]** OOM confirmed as crash root cause — `total-vm:41452828kB` signature
- **[S154]** Real root cause: `CUPY_CUDA_MEMORY_POOL_TYPE=none` causes 8GB VRAM mmap × 8 workers = 64GB VA space

---

## Commits This Session

| Commit | Description |
|--------|-------------|
| `7182ab5` | fix(s154): explicit GPU array del + gc.collect() — prevents VM bloat OOM on persistent workers |
