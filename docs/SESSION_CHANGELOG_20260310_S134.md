# SESSION CHANGELOG — S134
**Date:** 2026-03-10
**Session:** S134
**Author:** Team Alpha (Claude)
**Status:** CLOSED — persistent worker engine redesigned as standalone module

---

## Summary

S134 reverted coordinator.py persistent worker integration to pre-S130 baseline and implemented a clean standalone `persistent_worker_coordinator.py`. This eliminated the coordinator.py regression risk and gave the persistent worker path its own lifecycle management, worker pool, and chunk dispatch logic. Dry-run harness (25 tests) validated. Deployed to Zeus and all 3 rigs. Two runtime errors discovered during live test — carried forward to S135.

---

## Architecture Decision: Standalone Persistent Worker Coordinator

### Problem with S130/S133 approach
Persistent worker logic was embedded inside `coordinator.py`, sharing state with the standard SSH path. Every fix risked regressing the default (non-persistent) path. Gate 4 (additive routing) was repeatedly violated as semaphore blocks, result handling, and chunk sizing all had cross-path entanglement.

### Decision
Revert `coordinator.py` to pre-S130 (`0996582`) — zero persistent worker references. Implement all persistent worker logic in a new standalone file: `persistent_worker_coordinator.py`.

**Routing:**
- No flag (default): `coordinator.py` → `execute_truly_parallel_dynamic` → `sieve_filter.py` subprocess (unchanged)
- `--use-persistent-workers`: `persistent_worker_coordinator.py` → `sieve_gpu_worker.py` IPC (new path)
- Gate in `window_optimizer_integration_final.py`: `getattr(coordinator, 'use_persistent_workers', False)`

---

## Files Created / Modified

### New: `persistent_worker_coordinator.py`
Full standalone implementation:
- `WorkerHandle` dataclass with SSH Popen, GPU index, node IP, alive flag
- `PersistentWorkerCoordinator` class: `start_pool()`, `shutdown_pool()`, `run_sieve_pass()`, `_dispatch_to_worker()`, `_dispatch_local_sieve()`, `run_trial_persistent()`
- Worker spawn with stagger (4s between workers per rig) for ROCm stability
- Heartbeat drain loop on startup (reads until `"status"` + `"ready"` in line)
- Zeus localhost always uses local sieve path (`_dispatch_local_sieve`)
- Remote rigs use persistent SSH `sieve_gpu_worker.py --persistent`
- 4 sieve passes: forward, reverse, hybrid-forward, hybrid-reverse
- Results written to `results/window_opt_{pass}_{window}_{offset}_t{trial}.json`

### Modified: `window_optimizer_integration_final.py`
- Added `run_trial_persistent()` integration function
- `getattr(coordinator, 'use_persistent_workers', False)` gate
- `TestResult` pruned return (Bug 6 fix from S135)
- `dataset_path` propagation (Bug 1 fix from S135)

### Modified: `coordinator.py`
- Reverted to commit `0996582` (pre-S130)
- Seed caps (5M/2M) preserved from S131
- Zero persistent worker references

### New: `test_persistent_worker_harness.py`
25-test dry-run harness:
- T01–T10: WorkerHandle construction, pool size, node loading
- T11–T15: Local sieve path, result format parsing
- T12–T20: Chunk sizing (14 chunks for 5M seeds / 13 workers)
- T21–T25: Result JSON format, NPZ conversion compatibility

**Result: 25/25 PASS** before live deploy.

---

## ROCm Stability Guards

- Worker spawn stagger: 4s between each worker on same rig
- `HSA_OVERRIDE_GFX_VERSION=10.3.0` in all SSH worker launch commands
- `source ~/rocm_env/bin/activate` before each worker
- Max 4 persistent workers per rig (not 8) — memory pressure validated at 4-way

---

## Live Test — Runtime Errors (Carried to S135)

Two runtime errors discovered during first live pipeline test:

**Error 1:** `FileNotFoundError: ''` — `dataset_path` not passed to workers
**Error 2:** `TypeError: unhashable type: 'dict'` — worker result nested under `"result"` key, not unwrapped

Both diagnosed, fixes designed. Full Bug 1–9 resolution completed in S135.

---

## Deploy Commands (Historical Reference)

```bash
# Zeus
scp ~/Downloads/persistent_worker_coordinator.py rzeus:~/distributed_prng_analysis/
scp ~/Downloads/window_optimizer_integration_final.py rzeus:~/distributed_prng_analysis/
scp ~/Downloads/coordinator.py rzeus:~/distributed_prng_analysis/

# All 3 rigs
for IP in 192.168.3.120 192.168.3.154 192.168.3.162; do
  scp ~/Downloads/sieve_gpu_worker.py michael@$IP:~/rocm_env/sieve_gpu_worker.py
done
```

---

## Carry-Forward to S135

1. Bug 1: `dataset_path` never passed to workers
2. Bug 2: Worker result nested under `"result"` key
3. Binary pipe mode for large JSON (Bug 3)
4. Local sieve format mismatch (Bug 4)
5. Empty pipe response detection (Bug 5)
6. TestResult pruned constructor (Bug 6)
7. skip_range string format in NPZ conversion (Bug 7)
8. SSH banner consuming heartbeat (Bug 8)
9. Per-worker dispatch lock for concurrent chunk collision (Bug 9)

---

*Session S134 — 2026-03-10 — Team Alpha*
*Architecture redesign: persistent_worker_coordinator.py standalone. 25/25 harness. Live test errors → S135.*
