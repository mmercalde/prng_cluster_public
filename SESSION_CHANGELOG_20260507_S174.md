# SESSION CHANGELOG — 2026-05-07 (S174)

**Branch:** `main` (origin + public)
**HEAD before session:** `7313a43` (S170/S171/S173 v2 baseline)
**HEAD after session:** *uncommitted patches on disk; awaiting TB final sign-off + commit*

---

## Session arc

1. Continued from S173 v2 evening handoff: TB Step 1 forced FT=0.73 reproduction
2. Found stale S173 contamination, executed full cleanup
3. Ran S174 baseline (clean cluster, gate dispatched at ready=2)
4. TB diagnosed ready-gate wiring bug from baseline tarball
5. Wrote and applied TB-approved coordinator-side hard ready-gate patch
6. Validated patch with positive-case gate test
7. D1 (forced FT=0.73 50k 425M) ready, pending TB sign-off

---

## TB Step 1 forced FT=0.73 — three runs (post-S173-handoff)

**Run 1** (2026-05-04): rrig6600b operator-rebooted mid-trial → CONTAMINATED, archived as `CONTAMINATED_run1_*`. Excluded from analysis.

**Run 2** (2026-05-05 17:43): All 8500 chunks completed. **Then** 4 minutes after trial finalization, GCVM_L2 fault on rrig6600c gpu0+gpu6 simultaneously, followed by SMU cascade on rrig6600 gpu0 (signature matches 2026-05-03 manifests #1 and #2). Crash phase = post-completion teardown, NOT chunk_active. Tarballed: `s173_tb_step1_run2_crash_20260505_182552.tar.gz`.

**Run 3** (2026-05-05 18:31, post-reboot of all 3 rigs): Clean completion in 1:55, no faults, no manifests. Tarballed: `s173_tb_step1_run3_CLEAN_20260505_185619.tar.gz`.

Operator concern: TB analysis was conflating data from multiple historical sessions (surface area too large to isolate run #2's signature). Decision: discard all S173 instrumentation outputs and treat next run as the first valid post-reset data point.

---

## Cleanup (TB-approved reset)

**Deleted (S174 instrumentation outputs only):**
- `~/crash_dumps/2026*` on ser8 (~2400 dirs, daemon manifests back to 2026-04-17)
- `/tmp/crash_daemon_*.log` on ser8
- All `s173_tb_step1_*.tar.gz` from `~/Downloads/` and `/tmp/`
- Zeus `logs/s173_*.log s173_*.jsonl CONTAMINATED_run1_* netconsole_all_rigs.log pwc_startup_diag_simple.jsonl`
- Each rig `/tmp/prng_active_worker_gpu*.json* /tmp/prng_gpu_bus_map*.json`
- crash_forensic_daemon.py process (PID 30759) killed

**Preserved:**
- All source code, daily3.json, optuna_studies/*.db, `bidirectional_survivors_all.npz` (S145-R1 accumulator at 20,949 seeds), Python venv
- Per-worker `/tmp/pwc_tcp_worker_*.log` (PWC operational, predates S173)
- Worker heartbeat `.events.jsonl` (predates S173)
- Historical run logs (s112_*, s120_*, s162_*, etc.) for forensic continuity

**Post-cleanup:** rrig6600 SSH host key updated post-reboot; all 3 rigs healthy at 8/8/8 GPUs.

---

## S174 baseline run

**Goal:** post-reset instrumentation/healthy-cluster baseline (NOT crash repro). TB conditional approval with 5 conditions.

**RUN_ID:** `s174_baseline_pool8_25k_20260506_174650`
**Config:** pool=8, chunk=25k, max_seeds=213M, mode=open Optuna, --min-workers 24
**Organic Optuna config:** W27_O52_evening_S1-132_FT0.74_RT0.31
**Outcome:** 8520/8520 chunks completed, 0 faults, 0 manifests, 8/8/8 GPUs after run, all rigs reachable

**Critical finding from log inspection** (and TB tarball analysis):
- `[PWC-TCP] all 26/26 workers online — proceeding to init` at 17:47:16.317
- `[PWC-TCP] 2/26 workers ready — dispatching` at 17:47:16.819 ← **0.5 sec after init broadcast**
- `[java_lcg] 213,000,000 seeds → 8520 chunks across 3 workers`

PWC dispatched with only 2 workers ready despite `--min-workers 24` set. Remaining 24 workers came ready within ~1 sec and picked up jobs. PWC ledger confirmed all 26 eventually participated (24 AMD: 8118 chunks, 2 Zeus: 402 chunks).

Per TB Section 5: ready < 24 at dispatch makes this run INVALID as controlled baseline. Per operational reality: cluster healthy, instrumentation working.

**Launcher hygiene issue:** outer log truncated at `--- ready-worker gate (waiting up to 240s) ---`. No bundle, no observation, no summary written by launcher. Bash control issue with `set -euo pipefail` over SSH `nohup`.

**Manual bundle assembly** via `manual_bundle_s174_baseline.sh`:
- 500 KB tarball: `s174_baseline_pool8_25k_20260506_174650_bundle.tar.gz`
- Per-rig rocm-smi/ps/active_worker JSON/gpu_bus_map (current state, not at-fault — but no fault occurred)
- Run log, netconsole, PWC startup diag, S173 ledger, optimal_window_config.json
- Summary populated (chunks_completed=17040 was 2× from grep counting both [PWC] INFO and PersistentWorkerCoordinator INFO lines per chunk; true count is 8520)

Tarball delivered to TB.

---

## Ready-gate root cause (TB-confirmed via tarball)

```
window_optimizer.py line 617:
    coordinator.pwc_min_workers = pwc_min_workers   # set on MultiGPUCoordinator

persistent_worker_coordinator.py line 1537 (run_trial_persistent shim):
    pwc = PersistentWorkerCoordinator(
        config_file=..., worker_pool_size=..., seed_cap_*=...,
        pwc_transport=..., pwc_host=..., pwc_port=..., node_allowlist=...,
        # NO min_workers PASSED
    )
    # PWC.__init__ default: self.min_workers = min_workers (param default = 1)

persistent_worker_coordinator.py line 801:
    if count >= self.min_workers:   # self.min_workers = 1 (NOT 24)
        return count
```

`coordinator.pwc_min_workers = 24` was being set on `MultiGPUCoordinator`, never propagated to the `PersistentWorkerCoordinator` instance constructed inside `run_trial_persistent()`. PWC defaulted to `min_workers=1`, so `2 >= 1` was true and dispatch fired at the millisecond `_tcp_wait_ready` polled `ready_count() == 2`.

Even on timeout, `_tcp_wait_ready` returned `count` (whatever it was) and the caller continued — no abort path existed.

---

## S174 ready-gate hard fix (TB-approved architectural change)

**Patch script:** `apply_s174_ready_gate_fix.py` (idempotent, AST-verified, per-file backup)

**Changes:**
- `persistent_worker_coordinator.py`:
  - `run_trial_persistent()` signature gains `min_workers: int = 1` param
  - PWC ctor inside `run_trial_persistent()` receives `min_workers=min_workers`
  - `_tcp_wait_ready()` rewritten:
    - Success path: emits `[PWC-TCP] READY GATE PASSED: N/M ready (min_workers=K) — dispatch allowed`, returns count
    - Timeout path: emits `[PWC-TCP] READY GATE FAILED: N/M ready < min_workers=K — aborting before dispatch`, calls `self.shutdown()` (TB hardening 1) inside try/except, then `raise RuntimeError(...)`
  - Dispatch site at `run_sieve_pass()`: replaces `if _ready == 0: return error` with `if _ready < self.min_workers: log + raise RuntimeError`. Adds `dispatch confirmed: N ready workers (min_workers=K)` log on success.

- `window_optimizer_integration_final.py:352`:
  - `run_trial_persistent()` call gains `min_workers=getattr(coordinator, 'pwc_min_workers', 1)` — wires the value MultiGPUCoordinator sets to the PWC instance.

Backups: `persistent_worker_coordinator.py.bak.s174_1778207008`, `window_optimizer_integration_final.py.bak.s174_1778207008`.

Verify mode confirms all 6 markers present. py_compile clean on both files. Single production caller of `run_trial_persistent` is the patched call site (test harnesses default to min_workers=1 preserving their existing behavior).

---

## Gate validation run (positive case)

**RUN_ID:** `s174_gate_validation_20260507_192521`
**Config:** pool=8, chunk=25k, max_seeds=5M, --min-workers 24, trials=1
**Timeline:**
- 19:25:22.828 — coordinator init
- 19:25:48.560 — `READY GATE PASSED: 26/26 ready (min_workers=24) — dispatch allowed`
- 19:25:48.564 — `dispatch confirmed: 26 ready workers (min_workers=24)`
- 19:25:49.213 — first chunk completed (+653 ms after gate)
- 19:25:50.696 — last chunk completed (~1.5 sec compute phase)
- Bayesian optimization complete

**TB acceptance criteria — all PASS:**
1. ✅ READY GATE PASSED with ready ≥ 24 (got 26/26)
2. ✅ First job_assign after gate (gate 19:25:48.560 → first chunk 19:25:49.213)
3. ✅ No legacy `N ready worker(s) — dispatching` line (zero grep matches)
4. ✅ Defense-in-depth `dispatch confirmed` line emitted

**Cluster post-run:** 8/8/8 GPUs all rigs, no faults, no netconsole entries.

**Negative test (--min-workers 27):** NOT YET RUN. Cost up to 180s timeout. TB will decide whether required before D1.

---

## Launcher hygiene (recurring)

Both `launch_s174_baseline_pool8_25k.sh` and `launch_s174_gate_validation.sh` truncate their post-run summary section due to bash `set -euo pipefail` interactions during/after Python foreground execution over SSH. Forensic data lives in the run log either way; the patched coordinator now enforces the gate contract regardless of launcher behavior.

Outstanding question to TB: rewrite launchers in Python before D1, or accept current bash behavior since safety is in code?

---

## Files delivered (uncommitted on Zeus)

- `apply_s174_ready_gate_fix.py` (idempotent patch, applied — backups present)
- `launch_s174_gate_validation.sh` (5M seed positive-case validator)
- `launch_s174_baseline_pool8_25k.sh` (S174 baseline launcher with TB conditions; truncates summary)
- `manual_bundle_s174_baseline.sh` (post-hoc forensic bundle assembler)

**Modified files (uncommitted):**
- `persistent_worker_coordinator.py` (S174 hard gate)
- `window_optimizer_integration_final.py` (min_workers wiring)

---

## Open / pending

- **TB sign-off** on gate validation as patch acceptance, OR require negative test before D1
- **TB sign-off** on D1 launcher reusing patched coordinator path
- **TB call** on launcher rewrite (bash → Python) vs. accept current
- **Operator action** when above resolved: review `git diff`, commit S174 patches with message documenting TB approval lineage, dual-push to `origin` + `public`, remove `*.bak.s174_*` from working tree (gitignore or rm)
- **Negative test run** (--min-workers 27) if TB requires before D1
- **D1 run** (S174_D1_FT073_50K_425M, forced W21/O66/skip10-209/FT0.73/RT0.31, pool=8/50k/425M)

---

## Standing rules honored

- ✅ All code delivered to `/mnt/user-data/outputs/`, downloaded to ser8, scp'd to Zeus
- ✅ No git commits from Claude sandbox
- ✅ TB approval obtained before architectural change (ready-gate patch)
- ✅ Source code preserved through cleanup (only logs/results deleted)
- ✅ Idempotent patch with AST verify and per-file backup
- ⚠️ Web dashboard + monitor_all reminder: should be enforced before every trial launch (gate validation run skipped them — operator flagged correctly)
