# SESSION CHANGELOG — S163-KARG Forensic Upgrade
## Date: 2026-04-18
## Session Focus: Crash Analysis + Forensic Tooling Upgrade (TB-Approved)

---

## 1. WHAT WAS ANALYZED

### 1.1 S163-KARG Run Results
- Run: `s163_karg_json_100k_5trials_2044.log`, launched 20:44, seed_cap_amd=100K, 5 trials
- Patches active: [S163-KARG] cp.int32() typing fix + JSON guard (100K limit)
- Outcome: Step 1 completed (score: 1.0000). Two GPU crashes occurred before completion.

### 1.2 Secondary Crash — rrig6600b GPU1 (Trial 2, Silent)
- Time: 20:56:11
- Trial: **Trial 2** — W17_O25_midday+evening_S3-174 (window_size=17, skip_range=(3,174))
- Worker died silently mid-job sieve_1447 with no GCVM kernel fault in netconsole
- Autonomous recovery: reconnected at 21:06:20, resumed at 21:06:42
- Root cause: unknown — pre-daemon launch, no kernel evidence captured

### 1.3 Primary Crash — rrig6600c (Trial 3, Fatal)
- Time: 21:16:45–21:16:56
- Trial: **Trial 3** — W10_O61_midday_S1-243 (window_size=10, skip_range=(1,243))
- Required manual reboot. rrig6600 and rrig6600b remained healthy throughout.

**Fault sequence (from netconsole_since_launch.log):**
- Stage 1 (21:16:45.860): Real, readable fault codes — NOT 0xFFFFFFFF.
  - GPU 0000:16:00.0, STATUS=0x00801231, SQC(inst), pid 145822, vmid:8
    WALKER_ERROR=0x0 (page table walk succeeded — page EXISTED)
    PERMISSION_FAULTS=0x3 (R+X denied)
  - GPU 0000:06:00.0, STATUS=0x80081071, TCP, pid 145352, vmid:8
    WALKER_ERROR=0x0, PERMISSION_FAULTS=0x7 (R+W+X denied)
- Stage 2 (21:16:46.002): Cascade to 0xFFFFFFFF on GPU 0000:19:00.0, pid 145917, vmid:8
- Stage 3 (21:16:54): qcm fence wait loop timeout on 16:00.0 and 06:00.0
- Terminal (21:16:56.176): device lost from bus! on both 06:00.0 and 16:00.0

**Key findings:**
- Crash occurred MID-COMPUTE during Trial 3 forward sieve, NOT inter-trial idle
- All 3 faulting GPUs share vmid:8 — shared virtual address space context
- Stage 1 faults show pages existed but had permissions stripped mid-kernel execution
- Same address 0x0000790de5615000 faults repeatedly — kernel retry loop on dead mapping
- rocm_smi at crash time: ALL GPUs Performance Level=high, SCLK 2630-2740MHz
- Power management hypothesis FORMALLY RULED OUT

**Root cause assessment (TB-reviewed):**
- GPU virtual memory mapping invalidated while kernels mid-execution across multiple devices
- Leading hypothesis: CuPy memory lifecycle interference in shared HIP context
- Classification: "narrowed" not "identified" per TB — allocator/pool involvement is a
  serious lead but not yet demonstrated
- KARG patch: "appears distinct from prior KARG crash" — not yet formally closed per TB

**Optuna Trial Map (confirmed from pipeline log):**
| Trial | Config | window/skip | Incident |
|-------|--------|-------------|----------|
| 1 | W25_O9_evening_S8-89 | W25/S8-89 | None |
| 2 | W17_O25_midday+evening_S3-174 | W17/S3-174 | rrig6600b GPU1 silent crash |
| 3 | W10_O61_midday_S1-243 | W10/S1-243 | rrig6600c FATAL |
| 4 | W23_O6_evening_S6-137 | W23/S6-137 | None |
| 5 | W29_O42_evening_S9-85 | W29/S9-85 | None |

**Open unknowns (TB-confirmed):**
1. Worker process isolation model on rrig6600c — shared HIP context vs separate OS processes
2. Root cause of rrig6600b GPU1 silent crash at 20:56 (no kernel evidence)
3. Whether vmid:8 shared across devices is ROCm driver VMID reuse or true shared context
4. CuPy pool configuration and whether it acts as a cross-device invalidator

---

## 2. WHAT WAS BUILT

### 2.1 Team Beta Incident Report (docx)
- File: `TB_Incident_Report_rrig6600c_S163KARG.docx`
- 12 sections: timeline, fault evidence, GPU state, root cause, ruled-out hypotheses,
  open unknowns, diagnostic steps, evidence artefacts, sign-off
- Corrected mid-session: Trial 1 → Trial 3 after pipeline log cross-reference
- TB verdict: "Good incident report. Strong timeline. Strong localization.
  Useful hypotheses. Root cause is narrowed, not identified."

### 2.2 crash_forensic_daemon.py — v2 (TB-Approved)
**New capture functions added (zero removals from existing behavior):**

- `capture_pipeline_tail()` — greps watcher/pipeline log for trial/Optuna/phase
  markers → `zeus/pipeline_tail.log`. Immediately answers trial+phase at capture
  time. Gap that caused post-session archaeology this session.

- `write_capture_context()` — structured `zeus/capture_context.json` with triggers,
  worker counts (with reachability separated cleanly), pipeline log path, launch time.
  Unreachable rigs shown as `{count_last_observed, reachable: false}`.

- `pull_worker_heartbeats()` — recursive SCP of
  `~/worker_log_snapshots/worker_heartbeats/` from each rig →
  `<rig>/worker_heartbeats/`. Writes MISSING_HEARTBEATS.txt if absent.

- `capture_process_topology()` — `ps -eo pid,ppid,pgid,sid,etime,cmd` + `pstree -ap`
  → `<rig>/ps_full.log` and `<rig>/pstree.log`. Directly resolves Unknown 1
  (process isolation/shared parent question).

- `capture_worker_proc_details()` — reads heartbeat JSONs, extracts PIDs, queries
  `/proc/<pid>/status`, cmdline, exe, GPU env vars → `<rig>/worker_proc_details.json`.
  Marks `missing_at_capture: true` for PIDs already gone.

**Other daemon fixes:**
- `COORDINATOR_PROCESS_PATTERN` constant added (configurable) — now includes
  `persistent_worker_coordinator` alongside watcher_agent and window_optimizer.
  Prevents false coordinator_dead triggers.
- Default `--log-pattern` changed from `s163_karg_100k_5trials` to `s163_karg`
  (broader, matches any future S163-KARG run).

### 2.3 pwc_worker_service.py — v2 with TB-spec heartbeat instrumentation (TB-Approved)
**New constants:**
- `HEARTBEAT_SCHEMA_VERSION = "1.0.0"`
- `HEARTBEAT_DIR = "~/worker_log_snapshots/worker_heartbeats"`
- `_STATES` frozenset of 10 canonical state names

**New module-level helpers:**
- `_utc_now_iso()` — UTC ISO timestamp with milliseconds
- `_safe_cupy_pool_stats()` — CuPy default pool used/total bytes, fully defensive
  (pinned pool fields are None — no fake zeros)
- `_atomic_write_json()` — tmp+fsync+replace, never partial writes
- `_append_jsonl()` — append-only compact JSON lines
- `_ensure_heartbeat_dir()` — creates directory at startup

**New instance fields:**
- `_last_job_start_ts`, `_last_kernel_launch_ts`, `_last_kernel_return_ts`,
  `_last_result_send_ts`, `_last_done_job_id` — persistent across state transitions
  so the latest JSON always carries full job-boundary history

**`_emit_heartbeat()` instance method:**
- Emits at all 10 canonical states: connected, init_start, init_done, idle,
  job_start, pre_kernel, post_kernel, result_sent, exception, shutdown
- idle throttled to 2s minimum to avoid excessive writes
- `phase` strictly from job payload — never falls back to search_type
- Persistent timestamps updated on owning transition, serialized on every emit
- `_current_job_id` cleared AFTER result_sent heartbeat (TB blocker fix)
- Wrapped in try/except at every level — instrumentation CANNOT kill the worker
- Warns at DEBUG on non-canonical state names

**Emit call sites added:**
- `run_forever()`: connected, shutdown (normal + exception paths)
- `_wait_for_init()`: init_start
- `_import_sieve()`: init_done, exception on import failure
- `_main_loop()`: idle (throttled), job_start, result_sent, shutdown
- `_execute_job()`: pre_kernel, post_kernel, exception

---

## 3. REVIEW HISTORY

| Round | Reviewer | Verdict |
|-------|----------|---------|
| v1 | TB | "Mergeable after correction pass" — 4 required fixes, 2 improvements |
| v2 | TB | "Approved in principle — one small blocker" (_current_job_id ordering) |
| v3 | TB | **"Approved for deployment on next staged validation run"** |

---

## 4. WHAT IS NOT YET COMMITTED

Per hard rule: Team Alpha NEVER commits. Michael commits and dual-pushes.

Files ready for deployment:
- `~/Downloads/crash_forensic_daemon.py` → `~/crash_forensic_daemon.py` (ser8 only)
- `~/Downloads/pwc_worker_service.py` → Zeus + all 3 rigs

**Suggested commit message:**
```
[S163-KARG-HB] TB-approved forensic upgrade: crash daemon + worker heartbeats

crash_forensic_daemon.py:
- Add pipeline_tail.log capture (trial/phase context at crash time)
- Add capture_context.json (structured trigger+worker state)
- Add worker heartbeat pull from each rig
- Add process topology capture (ps + pstree, resolves VMID unknown)
- Add worker /proc details per PID from heartbeats
- Expand COORDINATOR_PROCESS_PATTERN to include persistent_worker_coordinator
- Broaden default --log-pattern

pwc_worker_service.py:
- TB-spec worker heartbeat JSON/JSONL at all 10 lifecycle states
- Persistent last_*_ts fields survive across state transitions
- Atomic writes, append-only JSONL, fully defensive (never kills worker)
- phase field strictly from payload, no search_type fallback
- _current_job_id cleared AFTER result_sent heartbeat
- Canonical _STATES enforcement at DEBUG level
```

**Dual-push required:**
```bash
git add persistent/pwc_worker_service.py
git commit -m "[S163-KARG-HB] TB-approved forensic upgrade: crash daemon + worker heartbeats"
git push origin main && git push public main
```

Note: `crash_forensic_daemon.py` lives on ser8, not in the repo. No git action needed for it.

---

## 5. DEPLOYMENT COMMANDS (run on ser8)

```bash
# 1. Deploy daemon (ser8 local)
cp ~/Downloads/crash_forensic_daemon.py ~/crash_forensic_daemon.py

# 2. Deploy worker service to Zeus
scp ~/Downloads/pwc_worker_service.py \
    rzeus:~/distributed_prng_analysis/persistent/pwc_worker_service.py

# 3. Deploy worker service to all 3 rigs + create heartbeat directory
for rig in rrig6600 rrig6600b rrig6600c; do
  scp ~/Downloads/pwc_worker_service.py \
      $rig:~/distributed_prng_analysis/persistent/pwc_worker_service.py
  ssh $rig "mkdir -p ~/worker_log_snapshots/worker_heartbeats"
done

# 4. Verify heartbeat dirs exist
for rig in rrig6600 rrig6600b rrig6600c; do
  ssh $rig "ls ~/worker_log_snapshots/worker_heartbeats/ && echo $rig OK"
done
```

---

## 6. NEXT SESSION PRIORITIES (before next 100K × 3 trial run)

1. **Deploy** crash_forensic_daemon.py + pwc_worker_service.py per section 5
2. **Commit + dual-push** pwc_worker_service.py per section 4
3. **Reboot all 3 rigs** with snd-power.conf + amdgpu.conf confirmed active
   (update-initramfs -u required — see S163 FINAL pending fix in memory)
4. **Fix Zeus rc.local** — nvidia-smi -c DEFAULT must survive reboot
5. **Commit KARG + JSON guard patches** — still uncommitted working-tree
   modifications on Zeus (committed tonight or next session)
6. **Run 100K × 3 trials** with full monitoring stack:
   - Terminal 1: `python3 ~/crash_forensic_daemon.py --log-pattern s163_karg`
   - Terminal 2: `python3 ~/trial_progress_monitor.py`
   - Terminal 3: `bash ~/worker_log_snapshot.sh start`
7. **If 3 trials clean**: advance to 300K × 3 trials
8. **Investigate rrig6600b GPU1 silent crash** (sieve_1447 at 20:56):
   `grep -n "sieve_1447\|ERROR\|Exception\|Traceback" \
   ~/crash_dumps/20260417_211658_.../rrig6600b/snapshots/pwc_tcp_worker_192_168_3_154_gpu1.log | tail -20`

---

## 7. ACCEPTANCE CRITERIA FOR NEXT RUN

On first crash (or manual trigger test):
- `zeus/pipeline_tail.log` present and shows trial number
- `zeus/capture_context.json` present with worker_counts_at_capture
- `<rig>/worker_heartbeats/` populated with JSON + JSONL files
- `<rig>/ps_full.log` and `<rig>/pstree.log` present
- `<rig>/worker_proc_details.json` present with PID/PPID topology
- On crashed rig: all above attempted, UNREACHABLE.txt written if rig gone

On any worker heartbeat JSON:
- `state` is one of the 10 canonical states
- `last_job_start_ts`, `last_kernel_launch_ts` etc. persist after job completion
- `phase` is "forward", "reverse", "bidirectional", or "unknown" — never "residue_sieve"
- `cupy_pool.pinned_used_bytes` is null, not zero
- `result_sent` event JSONL line contains correct job_id

---

*Team Alpha (Claude) — Session 2026-04-18*
*TB sign-off: Approved for deployment on next staged validation run*
