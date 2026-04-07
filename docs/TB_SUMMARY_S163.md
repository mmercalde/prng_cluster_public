# Team Beta Summary — S163
**Date:** 2026-04-06
**Author:** Claude (Team Alpha Lead Dev)
**Session:** S163
**HEAD:** `68ed075`

---

## Session Overview

S163 targeted three items: NPZ bug fix, `free_all_blocks()` removal from `sieve_filter.py`,
and staged validation at 500K → 1M → 2M seed caps. All three were partially addressed
with significant findings requiring TB review.

---

## Item 1 — NPZ Bug Fix ✅ CLOSED

**Fix:** Removed duplicate `import numpy as np` at line 62 inside `convert_survivors_to_binary.py`.
Python was treating `np` as a local variable, causing `UnboundLocalError` when the module-level
`np.array()` call was reached before the local import.

**Status:** Deployed, verified. NPZ conversion confirmed OK — 887 survivors, 64.9x compression.
Low-variance warning noted but confirmed expected behavior for window_size=6 (discrete match
rate values, not degenerate data). Closed.

---

## Item 2 — `free_all_blocks()` Removal ⚠️ PARTIALLY VALIDATED

### What Was Done
- `_best_effort_gpu_cleanup()` in `sieve_filter.py` patched per TB-approved proposal
  (PROPOSAL_FREE_ALL_BLOCKS_REPLACEMENT_v1_0.md)
- `free_all_blocks()` removed from both default and pinned memory pools
- Sampling-gated instrumentation added behind `S163_MEM_DEBUG=1` env var
- Deployed to Zeus + all 3 rigs. Committed `68ed075`.

**Note:** Initial patch was incorrectly applied to `sieve_gpu_worker.py` (SSH-PWC path,
not used in TCP-PWC runs). Corrected and applied to `sieve_filter.py` (confirmed correct
file via grep of `pwc_worker_service.py` import chain). `sieve_gpu_worker.py` reverted
to backup.

### Run Results — 500K Seed Cap, 3 Trials

**Trial 1:**
- 3,140 bidirectional survivors (vs 887 in S162 at 100K seed cap)
- 21:08 elapsed
- Netconsole: SILENT throughout
- AMD rigs: stable, no faults
- Aggregate throughput: ~20M sps (vs ~1.9M sps at 100K — 10x improvement)

**Trial 1 completed successfully. AMD rig stability confirmed with free_all_blocks() removed.**

**Trial 2/3 status:** Run hung after Trial 1 completion. `window_optimizer.py` showed
139% CPU / 2GB RAM / 48+ minutes with no output files written. Killed.

### Zeus `cudaErrorDevicesUnavailable` — New Pattern

**Critical finding:** Zeus local GPU path failed on EVERY local chunk systematically
(chunks 43, 87, 131, 175... — every 44 chunks, in pairs). This is a regression from
S162 behavior (250 failed chunks total) — S163 shows near-100% Zeus local chunk failure.

**Persistence mode confirmed already enabled** — `nvidia-smi -pm 1` returned
"Persistence mode is already Enabled." P8 idle is not the cause.

**S163_MEM_DEBUG=1 did not propagate to worker processes** — grep of log returned
empty. Env var set on SSH session is not being inherited by coordinator-spawned
subprocesses. Instrumentation produced zero output.

---

## TB Questions for Review

### Q1 — Zeus `cudaErrorDevicesUnavailable` Root Cause

The systematic Zeus local chunk failure (every chunk, in pairs) is new this session.
Three candidates:

**Candidate A — S163 sieve_filter.py restructure:**
The new `_best_effort_gpu_cleanup()` now touches CuPy pool objects (`pool.used_bytes()`,
`pool.total_bytes()`, `pool.n_free_blocks()`) on every chunk even when `S163_MEM_DEBUG=0`,
due to module-level global counter increment (`_S163_CHUNK_COUNTER += 1`). This runs
inside Zeus's local sieve subprocess. Could this CuPy pool access be contending with
the active CUDA context?

**Candidate B — Semaphore(2) simultaneous dispatch:**
`_localhost_semaphore = threading.Semaphore(2)` dispatches both Zeus GPU jobs
simultaneously. Both try to initialize CuPy context at the same moment. The pair
pattern (chunks always fail together) is consistent with this. This was present in
S162 but only caused 250/10,738 failures — why is it now causing near-100% failure?

**Candidate C — 500K chunk size interaction:**
At 100K seeds/chunk (~25ms), Zeus jobs complete quickly and stagger naturally.
At 500K seeds/chunk (~125ms), both semaphore slots complete at nearly the same time,
maximizing simultaneous context re-initialization on the next dispatch. Could larger
chunks be causing more precise synchronization of the two Zeus workers?

**TB ruling requested:** Which candidate(s) should be investigated first? Proposed
test: re-run at 500K without `S163_MEM_DEBUG=1` to isolate whether the cleanup
restructure is involved.

### Q2 — S163_MEM_DEBUG Propagation

The `S163_MEM_DEBUG=1` env var set in the SSH launch command did not reach worker
subprocesses. The coordinator launches Zeus local subprocesses via `subprocess.Popen`
and remote workers via TCP. Neither path inherits the SSH session environment.

**TB ruling requested:** What is the correct mechanism to propagate `S163_MEM_DEBUG`
to worker subprocesses? Options:
- A: Add `S163_MEM_DEBUG` to the env dict passed to `subprocess.Popen` in
  `persistent_worker_coordinator.py` Zeus local path
- B: Add `S163_MEM_DEBUG` to the ROCm env vars block pushed to remote TCP workers
- C: Read from a config file rather than env var

### Q3 — Post-Trial-1 Hang

`window_optimizer.py` ran 48+ minutes after Trial 1 with no output. No output files
written. No daemon_state.json. The `cudaErrorDevicesUnavailable` errors were all logged
at `18:30:45` simultaneously — a batch dump at trial completion, not during execution.

**TB ruling requested:** Is the hang in the inter-trial coordinator cleanup (TCP worker
teardown between trials) or in the NPZ accumulator merge? Should we add a timeout to
the inter-trial transition?

---

## Recommended Next Steps (pending TB ruling)

1. Re-run at 500K seeds **without** `S163_MEM_DEBUG=1` — isolate whether cleanup
   restructure causes Zeus failures
2. If Zeus failures persist → investigate Candidate B (semaphore) or C (chunk size)
3. Fix `S163_MEM_DEBUG` propagation per TB ruling on Q2
4. Investigate inter-trial hang — add timeout or diagnostic logging to transition
5. If 500K run completes clean → step up to 1M for next staged validation

---

## Commits This Session

| Hash | Description |
|------|-------------|
| `378fc40` | docs(s163): updated TODO master |
| `03d9013` | fix(s163): remove free_all_blocks() from sieve_gpu_worker.py (WRONG FILE — reverted) |
| `68ed075` | fix(s163): remove free_all_blocks() from correct file sieve_filter.py, revert sieve_gpu_worker.py |

---

## Current Infrastructure State

| Component | State |
|-----------|-------|
| Zeus HEAD | `68ed075` |
| sieve_filter.py | S163 patch — free_all_blocks() removed, instrumentation added |
| sieve_gpu_worker.py | Reverted to pre-S163 backup |
| All 3 rigs | amdgpu-dkms 6.12.12 pinned ✅ |
| bidirectional_survivors.json | S162 victory run — 887 seeds (not updated this session) |
| optimal_window_config.json | S162 victory run (not updated this session) |
| bidirectional_survivors_binary.npz | S163 — 887 seeds, NPZ bug fixed ✅ |
