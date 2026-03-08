# Session Changelog — S130
**Date:** 2026-03-08
**Commit:** `89c7597`
**Tag:** `s130-applied`
**Branch:** main (direct commit, both remotes)
**Status:** CLOSED — merged, soak test passed (Gate 1 partially validated, see caveat)

---

## Summary

S130 implemented persistent GPU sieve worker support in the TFM pipeline. Instead of spawning a new subprocess and initializing ROCm for every job, workers now boot once and process multiple jobs via stdin/stdout JSON-lines IPC. The feature is disabled by default (`use_persistent_workers=False`) and must be explicitly enabled via CLI or WATCHER manifest.

**Measured throughput gain: +150% (2.5x) — 832,300 sps → 2,082,140 sps aggregate.**

---

## Files Changed

### `coordinator.py`
- `__init__` signature: added `use_persistent_workers=False`, `worker_pool_size=8`
- `__init__` body: stores flags, initializes `_persistent_worker_registry = {}`
- `argparse`: added `--use-persistent-workers` (action=store_true) and `--worker-pool-size` (int, default=8)
- `main()` coordinator instantiation: wires both new args through
- New method: `_get_or_spawn_worker(worker)` — returns live Popen handle for (hostname, gpu_id), spawning via SSH if needed. Waits for `{"status": "ready"}` signal before returning.
- New method: `_shutdown_all_persistent_workers()` — sends shutdown command to all live workers, reaps processes. Hooked into existing `finally` cleanup block.
- New method: `execute_persistent_worker_job(job, worker)` — sends job via stdin JSON-line, reads result from stdout JSON-line. Three fallback paths to `execute_remote_job`: spawn failure, broken pipe, worker death mid-job.
- `execute_gpu_job()`: additive gate added **above** semaphore acquisition block. Activates only when `use_persistent_workers=True AND job.search_type='residue_sieve' AND not localhost`. Existing local and remote paths unchanged.
- `finally` cleanup: calls `_shutdown_all_persistent_workers()` when `use_persistent_workers=True`

### `window_optimizer.py`
- `run_bayesian_optimization()` signature: added `use_persistent_workers=False`, `worker_pool_size=8`
- `MultiGPUCoordinator` instantiation inside `run_bayesian_optimization`: passes both new params
- `argparse`: added `--use-persistent-workers` and `--worker-pool-size`
- `__main__` bayesian call: wires both args through via `getattr`

### `agent_manifests/window_optimizer.json`
- Version bumped: `1.3.0` → `1.4.0`
- `default_params`: added `use_persistent_workers: false`, `worker_pool_size: 8`
- `actions[0].args_map`: added `use-persistent-workers`, `worker-pool-size`
- `parameter_bounds`: added full entries for both params with descriptions and effect notes

---

## IPC Protocol (coordinator ↔ sieve_gpu_worker.py)

Protocol verified against actual `sieve_gpu_worker.py` source:

| Direction | Message | Field match |
|-----------|---------|-------------|
| Worker → Coordinator (startup) | `{"status": "ready", "gpu_id": N, "device": "..."}` | ✅ |
| Coordinator → Worker (job) | `{"command": "sieve", "job": {...}}` | ✅ |
| Worker → Coordinator (success) | `{"status": "ok", "job_id": "...", "elapsed_s": N, "result": {...}}` | ✅ |
| Worker → Coordinator (error) | `{"status": "error", "job_id": "...", "error": "...", "traceback": "..."}` | ✅ |
| Coordinator → Worker (shutdown) | `{"command": "shutdown"}` | ✅ |

---

## Four Hard Gates — Status

**Gate 1 — Fault tolerance: PARTIALLY VALIDATED**
- ✅ No job loss under normal persistent worker operation (exit=0, hard failures=0 across all soak jobs)
- ✅ Three fallback paths to `execute_remote_job` present in code: spawn failure, broken pipe, worker death mid-job
- ⚠️ Live fallback-on-worker-death not exercised — induced `kill -9` in Phase 2 did not land. Workers spawn as SSH subprocesses owned by Zeus; `pgrep` on rrig6600 cannot see them. Kill method that reaches the actual process not yet identified.
- ⚠️ Live fault tolerance path remains unexercised until a successful induced-kill test is completed.

**Gate 2 — Manifest/WATCHER wiring: ✅ SATISFIED**
- `use_persistent_workers` and `worker_pool_size` present in `default_params`, `actions.args_map`, and `parameter_bounds`
- Closes S97 class of silent parameter drop

**Gate 3 — GPU-clean invariant (S72): ✅ SATISFIED**
- `coordinator.py` contains zero CuPy/HIP imports
- All new methods manage subprocess handles and pipes only

**Gate 4 — Additive routing: ✅ SATISFIED**
- Persistent path is a branch above semaphore acquisition in `execute_gpu_job()`
- `execute_local_job()` and `execute_remote_job()` structurally unchanged
- Feature is off by default

---

## Soak Test Results — S130

**Script:** `soak_s130.sh` | **Date:** 2026-03-08 15:23 PDT

| Phase | Description | Result |
|-------|-------------|--------|
| Phase 1 | Persistent path engagement (10 jobs × 20M seeds) | ✅ 10/10 — 48 workers confirmed each run, ~75.9s avg |
| Phase 2 | Induced worker failure — job loss check | ✅ exit=0, hard failures=0 — kill did not land (see Gate 1 caveat) |
| Phase 3 | SSH fallback path verification (5 jobs × 20M seeds) | ✅ 5/5 — no S130 lines, ~69.4s avg |
| Phase 4 | Throughput vs baseline (200M seeds, persistent) | ✅ 2,082,140 sps — +150.1% vs 832,300 sps baseline |

**Total: 17/17 checks passed.**

### Throughput Analysis

The 2.5x aggregate gain comes from **cluster utilization improvement**, not per-job speedup. Per-job wall time with persistent workers (~76s) is slightly higher than SSH path (~69s) for the same 20M seed job. The gain is that persistent workers bypass SSH semaphore acquisition and connection setup latency between jobs, keeping all 24 AMD GPUs saturated with minimal dead time. The coordinator can dispatch the next job to a persistent worker as soon as the previous one completes, vs SSH path which must re-establish connection and re-initialize ROCm each time.

---

## Carry-Forward Items

- **Gate 1 open item**: identify correct kill method for SSH-spawned worker subprocesses and run a successful induced-kill soak test
- Seed cap patch to `window_optimizer_integration_final.py` (4 sites) — still not implemented
- `apply_caps.py` with final measured values — still not run
- S128 and S129B changelogs — written this session, need to be committed
- RETRY param-threading — next session priority
