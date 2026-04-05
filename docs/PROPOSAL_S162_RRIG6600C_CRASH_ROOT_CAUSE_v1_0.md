# PROPOSAL: S162 — rrig6600c Crash Root Cause Analysis & Fix Options
**Version:** 1.0  
**Date:** 2026-04-04  
**Author:** Claude (Team Alpha Lead Dev)  
**Status:** Submitted for Team Beta review  
**Scope:** `persistent/pwc_worker_service.py`, `persistent/pwc_transport_tcp.py`, `sieve_filter.py`  
**Priority:** P0 — Blocks all 26-GPU production runs

---

## 1. Problem Statement

rrig6600c (192.168.3.162, 8× RX 6600, gfx1032) crashes with GPU page faults
every time all 3 AMD rigs run simultaneously under TCP-PWC. The crash has
persisted across SSH-PWC, ZMQ+SQLite, and now TCP-PWC transports — surviving
every architectural change made over the past month.

**Isolation test results (confirmed this session):**
| Config | Result |
|--------|--------|
| rrig6600c alone + Zeus (10 GPUs) | ✅ Full 3-trial run, 18m 52s, zero faults |
| rrig6600 alone + Zeus | ✅ Stable (historical) |
| rrig6600b alone + Zeus | ✅ Stable (historical) |
| rrig6600 + rrig6600c + Zeus (18 GPUs) | ✅ Clean (S158 historical) |
| All 3 rigs + Zeus (26 GPUs) | ❌ rrig6600c crashes every time |

The crash is **deterministically triggered by simultaneous 3-rig load** and
cannot be reproduced in any isolation configuration.

---

## 2. Crash Signature (Netconsole Evidence)

### 2.1 Previous crashes (pre-S162 cleanup patch)
```
GCVM_L2_PROTECTION_FAULT_STATUS: 0xFFFFFFFF
Faulty UTCL2 client ID: unknown (0x1ff)
WALKER_ERROR: 0x7 / PERMISSION_FAULTS: 0xf / MAPPING_ERROR: 0x1
Multiple GPUs, Multiple PIDs simultaneously
→ watchdog: hard LOCKUP on cpu 4
→ rig goes offline
```

### 2.2 This session (post-S162 cleanup patch deployed)
```
FIRST FAULT:
  GCVM_L2_PROTECTION_FAULT_STATUS: 0x00801231
  Faulty UTCL2 client ID: SQC (inst) (0x9)   ← SHADER INSTRUCTION CACHE
  WALKER_ERROR: 0x0 / PERMISSION_FAULTS: 0x3 / MAPPING_ERROR: 0x0

SECOND FAULT (same GPU, same PID, milliseconds later):
  GCVM_L2_PROTECTION_FAULT_STATUS: 0x00801031
  Faulty UTCL2 client ID: TCP (0x8)           ← TEXTURE CACHE

ESCALATION (100ms later):
  GCVM_L2_PROTECTION_FAULT_STATUS: 0xFFFFFFFF ← FULL CASCADE
  → qcm fence wait loop timeout expired
  → snd_hda_intel: Unable to change power state from D3hot to D0
  → rig offline
```

**Critical difference this session:** Only ONE GPU (`0000:16:00.0`), ONE PID
(`python pid 5141`). Previous crashes hit multiple GPUs and PIDs simultaneously.
The `SQC (inst)` client is the shader instruction cache — the GPU tried to
fetch kernel instructions from an address that was no longer mapped.

**Timing:** Crash at kernel uptime `3036s` (~50 minutes into session).
Previous crashes occurred at `~18 seconds` into the run. The cleanup patch
significantly delayed the crash onset, suggesting it partially addressed
memory accumulation but introduced a new failure mode.

---

## 3. Infrastructure Investigation (This Session)

### 3.1 Zeus Network Stack Analysis

**Active NIC:** `enp2s0` — Intel I210 Gigabit Network Connection (rev 03)  
**Speed:** 1000Mb/s Full Duplex  
**Architecture:** Single 1GbE NIC shared by ALL cluster traffic:
- 24 inbound result streams (AMD workers → Zeus)
- 24 outbound job dispatch streams (Zeus → AMD workers)
- Netconsole UDP streams (3 rigs → Zeus)
- SSH control connections
- Dashboard HTTP traffic

**TCP buffer configuration (pre-tuning, stock Ubuntu defaults):**
```
net.core.wmem_max    = 212,992 bytes (~208KB)
net.core.wmem_default = 212,992 bytes (~208KB)
net.ipv4.tcp_wmem    = 4096 / 16384 / 4,194,304
```

**With 24 worker connections:** `212KB / 24 = ~8.8KB effective send buffer per socket`  
Each job payload is ~3KB — functionally adequate but leaves no headroom for
burst congestion.

**TCP buffer configuration (post-tuning, applied this session):**
```
net.core.wmem_max    = 16,777,216 bytes (16MB)
net.core.wmem_default = 1,048,576 bytes (1MB)
net.core.rmem_max    = 16,777,216 bytes (16MB)
net.core.rmem_default = 1,048,576 bytes (1MB)
net.ipv4.tcp_wmem    = 4096 / 1,048,576 / 16,777,216
net.ipv4.tcp_rmem    = 4096 / 1,048,576 / 16,777,216
```
Persisted to `/etc/sysctl.conf`. Active immediately via `sysctl -w`.

### 3.2 Zeus CPU Thread Analysis

**CPU:** Intel i9-9920X — 12 cores / 24 threads (Hyper-Threading)  
**Thread allocation under 26-GPU load:**

| Role | Thread Count |
|------|-------------|
| TCP worker handler threads (1 per AMD GPU) | 24 |
| Zeus local GPU sieve workers | 2 |
| Accept thread | 1 |
| Lease/heartbeat thread | 1 |
| WATCHER / coordinator main thread | 1 |
| Python GIL arbitration overhead | ~2-3 |
| **Total** | **~31** |

Zeus is **oversubscribed** — 31 logical threads competing for 24 physical
threads. When all 24 worker handler threads simultaneously call `conn.send_obj()`
(job dispatch), all 24 threads compete for CPU time AND the single I210 NIC
send buffer simultaneously.

### 3.3 Dispatch Semaphore (Deployed This Session)

Added `threading.Semaphore(8)` around `conn.send_obj()` in
`persistent/pwc_transport_tcp.py` `_handle_client()`. Limits concurrent
outbound job payload sends to 8 at a time (one rig's worth).

**Assessment:** This is correct hygiene for a 1GbE NIC with 24 concurrent
senders. However, since TCP guarantees delivery integrity (no payload
corruption on delay), this fix alone does not explain the GPU crash mechanism.
It reduces NIC contention but cannot prevent a GPU page fault caused by
software-side memory mismanagement.

---

## 4. Cleanup Patch History & Causal Chain Hypothesis

### 4.1 S160 Root Cause Finding
Session S160 established that `_best_effort_gpu_cleanup()` was never called
between chunks in persistent workers. GPU memory accumulated across hundreds
of chunks until the ROCm driver hit an unrecoverable state.

**S160 fix:** Added `_best_effort_gpu_cleanup()` call in `zmq_sqlite_worker.py`
after each chunk result delivery. Validated: full 3-rig run completed with
zero GPU crashes (S160 session log).

### 4.2 S162 TCP Worker Deployment Gap
The S160 fix was never deployed to `persistent/pwc_worker_service.py` (the
TCP-PWC worker). Confirmed this session: all 3 rigs were missing the patch.

**S162 deployment:** `_best_effort_gpu_cleanup()` added to
`pwc_worker_service.py` between chunks. Deployed to all 3 rigs.

### 4.3 Current `_best_effort_gpu_cleanup()` Implementation
```python
def _best_effort_gpu_cleanup():
    # 1. Python GC
    gc.collect()
    # 2. PyTorch cache clear
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    # 3. CuPy memory pool release
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
```

### 4.4 New Crash Signature Analysis

After deploying the cleanup patch, the crash signature changed materially:
- **Crash delayed:** from ~18 seconds to ~50 minutes (3036s uptime)
- **Scope narrowed:** from multiple GPUs + PIDs to single GPU + PID
- **First fault client changed:** from `unknown (0x1ff)` to `SQC (inst) (0x9)`

The `SQC (inst)` client ID indicates the **shader instruction cache** —
specifically, the GPU's L1 instruction cache for fetching compiled kernel
binary code. This is architecturally distinct from data memory faults.

---

## 5. Team Alpha Hypothesis: free_all_blocks() → SQC Fault

**Hypothesis:** `cp.get_default_memory_pool().free_all_blocks()` on ROCm/HIP
(gfx1032) is aggressively releasing GPU virtual address mappings that include
active kernel instruction pages, causing the next chunk's kernel launch to
fault on instruction fetch.

**Supporting evidence:**
1. Crash onset changed from ~18s (no cleanup) to ~50min (with cleanup) —
   cleanup is working but introduces a new failure mode
2. SQC (inst) fault is specifically an instruction fetch failure, not data
3. Only rrig6600c crashes — rrig6600 and rrig6600b use identical code but
   don't crash. rrig6600c's ROCm driver may handle `free_all_blocks()` more
   aggressively

**Contradicting evidence:**
1. CuPy `RawKernel` compiled binaries are cached in Python-side dict
   (`self.compiled_kernels`) — not in the CuPy memory pool. `free_all_blocks()`
   should only release data arrays, not kernel code objects.
2. The SQC fault address `0x00007cbb41419000` is in high userspace — consistent
   with either a data buffer or a mapped code segment.
3. rrig6600 and rrig6600b have identical code + cleanup — if `free_all_blocks()`
   caused instruction unmapping, they should also crash.

**Confidence in hypothesis:** MEDIUM. The causal chain is plausible but not
proven. The timing correlation (cleanup deployed → SQC fault appears → delay
from 18s to 50min) is suggestive but not conclusive.

---

## 6. Internet Research Findings

### 6.1 SQC (inst) Fault — Known ROCm Pattern

ROCm GitHub Issue #5616 documents an identical fault signature on AMD GPU
with darktable (OpenCL workload):
```
GCVM_L2_PROTECTION_FAULT_STATUS: 0x008012B0
Faulty UTCL2 client ID: SQC (inst) (0x9)
WALKER_ERROR: 0x0 / PERMISSION_FAULTS: 0xb / MAPPING_ERROR: 0x0
→ MES failed to respond to msg=REMOVE_QUEUE
→ Failed to evict process queues
→ sq_intr errors → GPU reset
```
This is the same progression seen in our crash. The ROCm issue remains open
with no upstream fix. The pattern appears to be a ROCm driver bug in queue
eviction when instruction cache pages cannot be cleanly unmapped.

### 6.2 AMD_SERIALIZE_KERNEL Diagnostic Tool

ROCm HIP debugging documentation recommends:
```bash
AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3
```
This forces synchronous kernel execution, making the faulting kernel
identifiable in the call stack. **TB Action Item:** Should we add these
vars to `ROCM_ENV_VARS` for a diagnostic run on rrig6600c only?

### 6.3 CuPy free_all_blocks() Documentation

CuPy official documentation states `free_all_blocks()` releases all free
blocks in the memory pool back to the device. The memory pool manages
**data allocations** only — `cp.ndarray` objects, buffers, etc. Compiled
`RawKernel` objects are managed by a separate kernel cache (`cupy.cuda.compiler`
module), not the memory pool.

**Implication:** `free_all_blocks()` should NOT unmap kernel instructions.
The SQC fault may have a different root cause than the cleanup patch.

### 6.4 Alternative Hypothesis: GPU Memory Pressure

Under full 3-rig load, 24 AMD workers process chunks simultaneously. Each
RX 6600 has 8GB VRAM but each worker on rrig6600c competes with 7 other
workers on the same physical machine. If total GPU memory demand across 8
workers exceeds available VRAM, the ROCm driver may evict pages including
instruction cache pages to make room — causing the SQC fault.

This would explain why:
- rrig6600c crashes but rrig6600/b don't (rrig6600c gets jobs later, when
  other rigs' workers have already consumed their GPU memory — changing the
  timing of when rrig6600c's workers first allocate)
- The crash is delayed with cleanup patch (lower steady-state memory pressure)
- Single GPU/PID affected (whichever worker hits memory pressure first)

---

## 7. Changes Deployed This Session (Pre-TB)

| Change | File | Status |
|--------|------|--------|
| TCP dispatch semaphore (8 concurrent) | `persistent/pwc_transport_tcp.py` | ✅ Zeus only |
| Zeus TCP buffer tuning | `/etc/sysctl.conf` | ✅ Active + persisted |
| GPU cleanup in TCP worker | `persistent/pwc_worker_service.py` | ✅ All 3 rigs |

---

## 8. Fix Options — TB Ruling Requested

### Option A: Replace free_all_blocks() with gc-only cleanup (Team Alpha Recommendation)

**Change:** In `pwc_worker_service.py`, replace `_best_effort_gpu_cleanup()`
call with inline `gc.collect()` only:
```python
# Current (causes SQC fault hypothesis):
from sieve_filter import _best_effort_gpu_cleanup
_best_effort_gpu_cleanup()  # includes free_all_blocks()

# Proposed:
import gc
gc.collect()  # Python refs only — no CuPy pool operations
```

**Rationale:** If `free_all_blocks()` is the trigger, removing it eliminates
the SQC fault. The S154 fix (explicit `del` of GPU arrays in `run_sieve_job()`)
handles cleanup at the job level — `free_all_blocks()` between chunks is
additive, not essential.

**Risk:** If `free_all_blocks()` is NOT the cause, this change does nothing
and memory may accumulate more slowly (gc.collect() alone is less aggressive).
The underlying crash would eventually recur.

**Validation:** Full 26-GPU run. If rrig6600c survives all 3 trials, Option A
is confirmed.

---

### Option B: Add AMD_SERIALIZE_KERNEL=3 to rrig6600c ROCM_ENV_VARS (Diagnostic)

**Change:** Add `AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3` to rrig6600c
workers only (via per-node env override in coordinator).

**Rationale:** Forces synchronous kernel execution — GPU waits for each
kernel to complete before returning. If the fault is caused by a kernel
launching against freed memory, synchronous mode catches it at the launch
point rather than during async execution. This is a diagnostic, not a fix —
but it may prevent the crash by serializing execution and reducing GPU memory
pressure concurrency.

**Risk:** Significant throughput reduction on rrig6600c (~50-80% slower
per-GPU). Acceptable for diagnostic run only.

**Validation:** Run with AMD_SERIALIZE_KERNEL=3 on rrig6600c. If stable →
confirms async kernel overlap is the crash trigger. If still crashes →
rules out kernel timing as cause.

---

### Option C: Cap rrig6600c at 4 workers under 3-rig load (Workaround)

**Change:** Set `gpu_count=4` in `distributed_config.json` for rrig6600c
when all 3 rigs are active.

**Rationale:** This was the S155 stable operating point. 4 workers = half
the GPU memory pressure. The crash is likely caused by the combination of
8-worker memory pressure + 3-rig simultaneous load.

**Risk:** Permanently reduces cluster throughput from 26 GPUs to 22 GPUs
(~15% loss). Does not fix root cause.

**Validation:** Immediate — historically proven stable.

---

### Option D: Add synchronization barrier between rrig6600c worker starts (Architectural)

**Change:** In TCP startup sequence, stagger rrig6600c worker initialization
by introducing a 30-60 second delay after rrig6600 and rrig6600b workers
are fully online and have completed their first chunk. This gives rrig6600c
workers a "warm" environment with less simultaneous memory pressure.

**Rationale:** The crash timing (~50 minutes with cleanup, ~18 seconds without)
suggests it's triggered by peak concurrent GPU memory demand. Staggering
rrig6600c's entry into the workload reduces the peak.

**Risk:** Adds ~30-60 seconds to startup time per trial. May not fully
prevent crash if memory pressure builds up over multiple chunks.

---

## 9. Team Alpha Recommended Sequence

1. **Immediate:** TB ruling on Option A vs Option B
2. **If Option A approved:** Deploy gc-only cleanup, run full 26-GPU test
3. **If Option A fails:** Deploy Option B (AMD_SERIALIZE_KERNEL=3) as diagnostic
4. **If Option B confirms async timing:** Consider Option D (startup stagger)
5. **Fallback at any point:** Option C (4-worker cap) restores production capability

---

## 10. Open Questions for Team Beta

1. Does `cp.get_default_memory_pool().free_all_blocks()` on ROCm/HIP release
   any GPU virtual address mappings beyond data allocations? Specifically, can
   it affect compiled kernel instruction mappings?

2. Is the SQC (inst) fault consistent with GPU memory pressure (VRAM near
   capacity forcing driver page eviction) rather than software-triggered unmapping?

3. Given that rrig6600 and rrig6600b have identical cleanup code and do NOT
   crash, what is the architectural difference that makes rrig6600c specifically
   vulnerable? (Hardware is identical — same CPU, same GPU model, same ROCm version)

4. Is Option A safe with respect to memory accumulation? The S154 explicit
   `del` of GPU arrays in `run_sieve_job()` should handle cleanup at job
   level — is `free_all_blocks()` between chunks redundant or load-bearing?

5. Should `AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3` be added to
   rrig6600c's worker environment as a permanent diagnostic measure, or only
   for a targeted diagnostic run?

---

## 11. Files Requiring Changes

| File | Change | Node |
|------|--------|------|
| `persistent/pwc_worker_service.py` | Option A: gc-only cleanup | All 3 rigs |
| `persistent/pwc_transport_tcp.py` | Semaphore already deployed ✅ | Zeus |
| `/etc/sysctl.conf` | TCP buffers already deployed ✅ | Zeus |
| `distributed_config.json` | Option C only: gpu_count=4 for rrig6600c | Zeus |

---

*Proposal S162 — Team Alpha — 2026-04-04*
