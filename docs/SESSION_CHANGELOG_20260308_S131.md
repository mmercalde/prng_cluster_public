# Session Changelog — S131
**Date:** 2026-03-08
**Commits:** `1d85da4`, `d168f83`, `50a4146`
**Branch:** main (direct commits, both remotes)
**Status:** CLOSED

---

## Summary

S131 closed three carry-forward items from S129B/S130: Gate 1 fault tolerance
validation, seed cap propagation, and the apply_caps.py maintenance tool.
Item 4 (RETRY param-threading) was confirmed already complete since S82 —
stale carry-forward note cleared.

---

## Item 1 — Gate 1 Close: Fallback Logging + Empty-Readline Dead-Pipe Fix

**Commit:** `1d85da4`

### Problem
S130 Gate 1 was partially validated. The induced `kill -9` test in S130 Phase 2
failed to land on a worker process — `pgrep -f 'sieve_gpu_worker.*rrig6600'`
matched the outer bash shell instead of the actual SSH transport processes,
because the processes use IPs (`192.168.3.120`, `.154`, `.162`), not hostnames.

### Root Cause Analysis (Three Layers)
1. **Wrong grep pattern** — hostname-based grep never matched IP-based transports
2. **Wrong kill timing** — 25s sleep meant the worker had already completed its
   first ~12s job and was idle; killing an idle worker triggers no fallback
3. **Empty-readline race** — even when the kill landed mid-job, `readline()`
   returns `""` on a closed pipe. The old code treated `""` as "no data yet"
   and slept, never re-checking `proc.poll()` until job timeout

### Fixes Applied (`coordinator.py`)

**Fix 1 — All 4 fallback log lines enriched:**
Every fallback path now emits a structured log line with `worker`, `job_id`,
`reason` (spawn_failure / broken_pipe / worker_dead / read_error), and
`recovery=execute_remote_job`. Previously these logs were missing `job_id`
and used inconsistent formats.

**Fix 2 — Empty-readline dead-pipe detection:**
Added `proc.poll()` check immediately inside the `if not line:` branch. When
`kill -9` closes the pipe, `readline()` returns `""` → `poll()` now fires
immediately → fallback triggers within one 20ms sleep cycle.

### Gate 1 Phase 2 Test Result
- Correct grep: `pgrep -f 'ssh.*192\.168\.3\.[0-9]+.*sieve_gpu_worker'`
- Correct timing: 15s sleep (workers ready at ~4s, first job done at ~16s)
- Kill landed on PID 23259 (`192.168.3.120` gpu0 transport)
- Result:
  ```
  [S130][FALLBACK] worker=('192.168.3.120', 0) job_id=sieve_002
  reason=worker_dead recovery=execute_remote_job
  ```
- Exit code: 0, no job loss, no duplication

**Gate 1 CLOSED. ✅**

---

## Item 2 — Seed Cap Patch: Constructor Defaults + Explicit Wiring + Manifest

**Commit:** `d168f83`

### Problem
S129B-A patched the CLI `argparse` defaults in `coordinator.py` (40k→5M/2M)
but left the constructor defaults at `40000`/`19000`. Any code that instantiates
`MultiGPUCoordinator` directly (without CLI) silently got stale caps.

`window_optimizer_integration_final.py` had 2 such instantiation sites (not 4
as noted in S129B carry-forward — confirmed by live grep). Neither passed
`seed_cap_*` explicitly.

`agent_manifests/window_optimizer.json` had no `seed_cap_nvidia`/`seed_cap_amd`
entries, so WATCHER could never pass these values through.

### Fixes Applied

**`coordinator.py` line 233:**
```python
# Before
seed_cap_nvidia: int = 40000, seed_cap_amd: int = 19000, seed_cap_default: int = 19000
# After
seed_cap_nvidia: int = 5000000, seed_cap_amd: int = 2000000, seed_cap_default: int = 2000000
```

**`window_optimizer_integration_final.py` (2 sites):**
Both `_MCC(...)` and `_WMCC(...)` instantiations now explicitly pass:
```python
seed_cap_nvidia=5_000_000,
seed_cap_amd=2_000_000,
```

**`agent_manifests/window_optimizer.json` v1.4.0 → v1.5.0:**
- `actions[0].args_map`: added `seed-cap-nvidia`, `seed-cap-amd`
- `default_params`: added `seed_cap_nvidia: 5000000`, `seed_cap_amd: 2000000`
- `parameter_bounds`: added full entries for both with descriptions and effect notes

---

## Item 3 — apply_caps.py: Deploy as Maintenance Tool

**Commit:** `50a4146`

`apply_caps.py` was a carry-forward item from S128/S129B — a re-runnable tool
for applying Phase A/B throughput probe results to `coordinator.py` and
`gpu_optimizer.py`. The current values are already correct from S129B-A, so
the script was not run. Deployed to Zeus as a maintenance tool for future
re-probing sessions.

The script takes `--rtx-phaseA-sps`, `--amd-phaseA-sps`, `--rtx-phaseB-ceiling`,
`--amd-phaseB-ceiling`, `--safety-factor` and computes/applies `seed_cap_nvidia`,
`seed_cap_amd`, `scaling_factor` automatically with backups.

---

## Item 4 — RETRY Param-Threading: Already Complete

Carry-forward note was stale. RETRY param-threading was fully implemented in
S76 (`_handle_training_health`, `_build_retry_params`, `_get_max_training_retries`)
and proven end-to-end in S82 (all 11 assertions passed). No work required.

---

## Carry-Forward to S132

- S110 root cleanup (884 files)
- sklearn warnings in Step 5
- Remove CSV writer from coordinator.py
- Regression diagnostic gate: set `gate=True`
- S103 Part 2: per-seed match rates
- Phase 9B.3: deferred selfplay component
- Variable skip bidirectional count not wired into Optuna scoring
- Node failure resilience: single rig dropout can crash Optuna study

---

## Commit Log

| Hash | Description |
|------|-------------|
| `1d85da4` | S131: Gate 1 close — fallback logging + empty-readline dead-pipe fix |
| `d168f83` | S131: seed cap patch — constructor defaults, explicit wiring, manifest |
| `50a4146` | S131: add apply_caps.py maintenance tool for future GPU throughput re-probing |

---

*Session S131 — 2026-03-08 — Team Alpha*
*All three S129B/S130 carry-forward items closed. Both remotes at `50a4146`.*
