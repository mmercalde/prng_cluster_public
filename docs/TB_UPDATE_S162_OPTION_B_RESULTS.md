# TB UPDATE — S162 Option B Diagnostic Results
**Date:** 2026-04-04  
**Author:** Claude (Team Alpha)  
**Ref:** PROPOSAL_S162_RRIG6600C_CRASH_ROOT_CAUSE_v1_0.md  
**Status:** Option B diagnostic complete — results submitted for TB ruling

---

## Option B Diagnostic Run Results

### Configuration
- `AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3` applied to rrig6600c workers ONLY
- All other settings unchanged from previous run
- All 3 rigs active, 26 GPUs total
- `_best_effort_gpu_cleanup()` (including `free_all_blocks()`) still active on all rigs

### Crash Timing Clarification
- **rrig6600c last rebooted:** 20:45:52 (kernel uptime resets to 0 on reboot)
- **Trial started:** ~21:52 (worker spawn sequence complete)
- **Crash occurred:** 21:53:25 (kernel uptime 4078s = rig uptime since reboot)
- **Trial runtime before crash:** approximately **1-2 minutes**

`AMD_SERIALIZE_KERNEL=3` did NOT delay the crash relative to trial start.
Previous runs crashed at ~18 seconds and ~50 minutes (trial runtime).
This run crashed at ~1-2 minutes. No meaningful improvement.

---

## New Netconsole Evidence

### Crash Sequence
```
21:53:25 — FIRST FAULT:
  GPU: 0000:16:00.0 (GPU6), PID 2986
  GCVM_L2_PROTECTION_FAULT_STATUS: 0x00801231
  Faulty UTCL2 client ID: SQC (inst) (0x9)  ← instruction cache
  WALKER_ERROR: 0x0 / MAPPING_ERROR: 0x0

21:53:25 — SIMULTANEOUS SECOND FAULT (48ms later):
  GPU: 0000:06:00.0 (GPU1), PID 2693
  GCVM_L2_PROTECTION_FAULT_STATUS: 0x80801231
  Faulty UTCL2 client ID: SQC (inst) (0x9)  ← SAME fault, different GPU

21:53:25 — CASCADE:
  Both GPUs → 0xFFFFFFFF
  WALKER_ERROR: 0x7 / PERMISSION_FAULTS: 0xf / MAPPING_ERROR: 0x1

21:53:34 — QCM TIMEOUTS:
  0000:16:00.0: qcm fence wait loop timeout expired
  0000:06:00.0: qcm fence wait loop timeout expired
  0000:03:00.0: Trap debug id already reserved

21:53:44 — FULL COLLAPSE:
  0000:19:00.0: qcm fence wait loop timeout expired
  0000:19:00.0: Failed to evict process queues
  0000:06:00.0: Failed to restore queues
  0000:16:00.0: Failed to restore queues
  amdgpu: Failed to suspend process 0x8010 through 0x8017  (8 processes = all 8 workers)

21:53:55 — CPU SOFT LOCKUP:
  watchdog: BUG: soft lockup - CPU#3 stuck for 23s! [python:2997]
```

---

## Key Findings From Option B

### Finding 1: AMD_SERIALIZE_KERNEL=3 does NOT prevent the crash
Serial kernel execution (no async overlap) made no difference to crash timing
from trial start. Option B is **ruled out as a fix**. Async kernel overlap
is **not the root cause**.

### Finding 2: SQC (inst) fault persists under serial execution
The first fault is STILL `SQC (inst)` even when `AMD_SERIALIZE_KERNEL=3`
forces synchronous kernel execution. This means:
- The SQC fault is NOT caused by async kernel overlap
- The fault occurs even when only one kernel runs at a time per GPU
- The instruction cache miss happens at a deterministic point regardless of serialization

### Finding 3: Two GPUs fault simultaneously with identical signature
`0000:16:00.0` (GPU6, PID 2986) and `0000:06:00.0` (GPU1, PID 2693) both
show `SQC (inst)` at nearly the same wall-clock time (48ms apart). Two
separate worker processes on two separate GPUs hit the same instruction
cache fault simultaneously. This is a system-level trigger, not a
single-worker event.

### Finding 4: CPU soft lockup — new evidence
```
watchdog: BUG: soft lockup - CPU#3 stuck for 23s! [python:2997]
```
A CPU core (not GPU) is locked for 23 seconds in a Python process. This
indicates the ROCm/KFD kernel driver is deadlocking a CPU core during
the GPU fault handling sequence. This is a kernel-level event, not
application-level.

### Finding 5: All 8 workers fail to suspend simultaneously
```
amdgpu: Failed to suspend process 0x8010 through 0x8017
```
All 8 rrig6600c GPU worker processes fail to suspend at the same instant.
The KFD driver cannot cleanly shut down any of the 8 workers once the
cascade begins — consistent with a driver-level resource exhaustion under
full concurrent load.

---

## Crash Progression Summary Across All Runs

| Run | Config | First fault | Trial runtime | Fault client |
|-----|--------|-------------|---------------|--------------|
| Pre-cleanup | No cleanup | Multi-GPU/PID | ~18 seconds | unknown (0x1ff) |
| Post-cleanup | `free_all_blocks()` active | Single GPU/PID | ~50 minutes | SQC (inst) first |
| Option B | `AMD_SERIALIZE_KERNEL=3` | Two GPUs/PIDs | ~1-2 minutes | SQC (inst) first |

**Pattern:** The SQC (inst) fault appears consistently as the triggering
event once cleanup is active. AMD_SERIALIZE_KERNEL neither prevents it
nor changes its character — only the cleanup patch changes the crash mode.

---

## Team Alpha Analysis

### What we can rule out
1. ❌ Async kernel overlap — eliminated by Option B (serial execution same crash)
2. ❌ TCP send buffer contention — eliminated by semaphore + buffer tuning (no change)
3. ❌ Zeus NIC bandwidth — TCP tuning applied, no effect on crash
4. ❌ rrig6600c hardware defect — stable in isolation and 2-rig configs

### What the evidence points toward
The CPU soft lockup + simultaneous 8-worker KFD suspend failure + identical
SQC fault on two separate GPUs simultaneously strongly suggests:

**The ROCm KFD (Kernel Fusion Driver) hits a resource limit or race condition
under full 3-rig concurrent load that causes GPU virtual memory management
to fail.** The SQC (inst) fault is the first visible symptom — the GPU
tries to fetch kernel instructions from a VA that the KFD has invalidated
or reclaimed under memory pressure from 24 concurrent workers.

This is consistent with TB's earlier framing: "a 3-rig concurrency-triggered
system bug whose first visible casualty is rrig6600c."

### What we cannot determine without deeper tooling
- Whether the KFD hits a queue/process limit (8 pasids × 3 rigs = 24 concurrent HW queues)
- Whether VRAM pressure from 24 concurrent 2M-seed buffers triggers GPU VA reclaim
- Whether a ROCm driver bug in multi-node peer queue management is involved

---

## Questions for Team Beta

1. **KFD hardware queue limits:** The RX 6600 (gfx1032) reports
   `Max Queue Number: 128` per device. With 8 workers per rig × 3 rigs = 24
   concurrent processes each using multiple queues, is it possible the system
   is hitting a KFD global queue resource limit across all GPUs on the rig?

2. **`Failed to suspend process` cascade:** All 8 workers fail simultaneously.
   Is this consistent with a KFD internal lock contention (e.g., global mutex
   held during GPU reset blocking all other processes)?

3. **SQC (inst) + serial execution:** If `AMD_SERIALIZE_KERNEL=3` does not
   prevent SQC instruction cache faults, what ROCm-level mechanism could
   cause instruction cache invalidation outside of kernel launch events?

4. **Option A revisited:** Given that Option B failed, should Option A
   (`gc.collect()` only, removing `free_all_blocks()`) be tested on rrig6600c
   only — not all 3 rigs — to determine if the cleanup patch is contributing
   to KFD memory pressure?

5. **Option C implementation:** If TB approves Option C as production
   workaround, should rrig6600c be capped at 4 workers permanently, or
   should we attempt 6 workers as an intermediate test?

---

## Recommended Next Steps (Team Alpha Proposal)

**Immediate production capability:** Implement Option C — cap rrig6600c
at 4 workers. This gives 22 GPUs (Zeus 2 + rrig6600 8 + rrig6600b 8 +
rrig6600c 4) and is historically proven stable (S155).

**Continued diagnosis (parallel):** Test Option A on rrig6600c ONLY with
4-worker cap active. If `free_all_blocks()` is contributing to KFD memory
pressure, removing it on rrig6600c may allow 8 workers to be stable.

**Longer term:** Research whether ROCm 6.4.3 has known KFD multi-process
concurrent queue management bugs on gfx1032 under full cluster load.

---

*TB Update S162 — Option B Results — Team Alpha — 2026-04-04*
