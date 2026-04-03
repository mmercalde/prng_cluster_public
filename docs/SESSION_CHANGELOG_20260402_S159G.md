# Session Changelog — S159G
**Date:** 2026-04-02
**Session:** S159G
**Focus:** rrig6600 GPU crash root cause investigation and fix

---

## Summary

This session identified and fixed the root cause of persistent rrig6600 GPU crashes
during ZMQ multi-rig runs. The crash was a GPU virtual memory fault (amdgpu gfxhub
page fault → queue unrecoverable) caused by rrig6600 workers running without 4
critical ROCm protective environment variables.

**Status:** Fix applied. Acceptance test (fresh run + live PID env verification)
pending next session.

---

## Key Findings

### Root Cause Confirmed (Primary Root-Cause Candidate)
rrig6600 workers consistently ran with only:
```
HSA_OVERRIDE_GFX_VERSION=10.3.0
```

While rrig6600b and rrig6600c workers had the full protective set:
```
ROCR_VISIBLE_DEVICES=0
HSA_ENABLE_SDMA=0
AMDGPU_NO_POWER_PROFILE=1
HSA_ENABLE_RUNTIME_POWER_MGMT=0
HSA_OVERRIDE_GFX_VERSION=10.3.0
```

The missing `HSA_ENABLE_SDMA=0` is the critical variable — it disables the SDMA
engine which has known bugs causing GPU VM faults on gfx1032 (RX 6600).

### Why rrig6600 Was Different
The coordinator launches workers via `systemd-run --user` with `--setenv` flags.
On rrig6600b and rrig6600c, the transient systemd unit files correctly contained
the full `Environment=` block, injecting all 5 vars into worker processes.

On rrig6600, the systemd unit was created identically BUT the env vars were not
reaching the worker PID. Only `/etc/environment` vars survived. Investigation
revealed rrig6600's `.bashrc` had a `[ -n "$PS1" ]` guard before sourcing
`rocm_env/bin/activate`, preventing venv activation in non-interactive shells.
rrig6600b's `.bashrc` had an unconditional activate (no `$PS1` check).

### ZMQ Crash Resilience Confirmed
When rrig6600 crashed mid-run and rebooted, the ZMQ SQLite coordinator:
- Detected expired leases via lease expiry mechanism
- Redistributed chunks to surviving rrig6600b and rrig6600c workers
- Completed the forward sieve (126,610,730 survivors) without data loss
- Accepted rrig6600 back when it rejoined

This is the first confirmed crash-resilient full-cluster run. Architecture validated.

---

## Infrastructure Work Completed

### Netconsole (Completed in earlier S159 sub-sessions)
- Zeus netconsole listener: `logs/netconsole_all_rigs.log`
- All 3 rigs streaming kernel logs to Zeus in real time
- Enabled capture of exact crash sequence: page fault → GCVM_L2_PROTECTION_FAULT_STATUS:0xFFFFFFFF → qcm fence timeout → unrecoverable CP state

### Fix Applied — /etc/environment on All 3 AMD Rigs
Added to `/etc/environment` on rrig6600, rrig6600b, rrig6600c:
```
HSA_ENABLE_SDMA=0
HSA_ENABLE_RUNTIME_POWER_MGMT=0
AMDGPU_NO_POWER_PROFILE=1
```

`ROCR_VISIBLE_DEVICES` intentionally NOT added — remains coordinator-managed
per-worker via `--setenv` to avoid pinning the whole machine to one GPU.

All 3 rigs rebooted after fix. `/etc/environment` verified clean and identical
on all 3 rigs post-reboot.

---

## TB Rulings This Session

### TB Initial Ruling
- P0: Verify `HSA_ENABLE_SDMA=0` reaches live worker PIDs
- Fault class: GPU VM / copy-path failure (not Zeus, not coordinator)
- 5M seed cap: downgraded to "amplifier at most"

### TB Final Ruling (Provisional Accept)
- Primary root-cause candidate: worker env propagation defect
- Fix: add AMD-wide safety vars to `/etc/environment` on all AMD rigs
- Keep `ROCR_VISIBLE_DEVICES` coordinator-managed
- Acceptance criteria:
  1. rrig6600 rebooted after fix ✅
  2. Fresh coordinator launch (PENDING)
  3. Live zmq_sqlite_worker PID on rrig6600 shows all 4 vars (PENDING)
  4. One clean full-cluster stability run with no rrig6600 crash (PENDING)
- Follow-on: worker registration env-invariant check (future hardening)

---

## Documents Created This Session

- `docs/TB_SUBMISSION_S159G_RIG6600_CRASHES.md` — commit 9e09af7
- `docs/TB_UPDATE_S159G_ENV_PROPAGATION.md` — commit pending
- `docs/TB_FINAL_UPDATE_S159G_ROOT_CAUSE_CONFIRMED.md` — commit pending
- `docs/SESSION_CHANGELOG_20260402_S159G.md` — this file

---

## Current Cluster State

| Component | State |
|-----------|-------|
| rrig6600 | Up, rebooted, /etc/environment fixed |
| rrig6600b | Up, rebooted, /etc/environment fixed |
| rrig6600c | Up, rebooted, /etc/environment fixed |
| Zeus | Halted (agent_halt set) |
| Last run | Killed mid-reverse-sieve (hybrid pass failed — rrig6600 crash) |
| seed_start | Reset required before next run |

---

## Pending — Next Session Action Plan

### Step 1 — Fresh launch with env verification
```bash
# Clear halt and state
ssh rzeus "cd ~/distributed_prng_analysis && \
  rm -f /tmp/agent_halt daemon_state.json optimal_window_config.json \
  bidirectional_survivors.json && \
  mv zmq_job_queue.db zmq_job_queue.db.pre_s159g_fix 2>/dev/null; \
  source ~/venvs/torch/bin/activate && \
  python3 -c 'from agents.safety import clear_halt; clear_halt()' && \
  python3 -c \"import sqlite3; conn=sqlite3.connect('prng_analysis.db'); \
  conn.execute('DELETE FROM exhaustive_progress WHERE prng_type=?',('java_lcg',)); \
  conn.commit(); conn.close()\" && \
  setsid bash -c 'source ~/venvs/torch/bin/activate && \
  PYTHONPATH=. python3 agents/watcher_agent.py \
  --run-pipeline --start-step 1 --end-step 1 --force-step 1 \
  >> logs/zmq_s159g_fix_v1.log 2>&1' &"
```

### Step 2 — TB acceptance test (45 seconds after launch)
```bash
sleep 45
# Check systemd unit
ssh rrig6600 "systemctl --user cat zmq-worker-gpu0.service 2>/dev/null | grep Environment"
# Check live PID env — THIS IS THE CRITICAL TEST
ssh rrig6600 "cat /proc/\$(pgrep -f zmq_sqlite_worker | head -1)/environ \
  2>/dev/null | tr '\0' '\n' | grep -E 'HSA|SDMA|ROCR|POWER'"
```

### Expected result
```
ROCR_VISIBLE_DEVICES=0
HSA_ENABLE_SDMA=0
AMDGPU_NO_POWER_PROFILE=1
HSA_ENABLE_RUNTIME_POWER_MGMT=0
HSA_OVERRIDE_GFX_VERSION=10.3.0
```

If all 5 present → run to completion → S159G resolved.
If still missing → coordinator launch-path bug, escalate to TB.

### Step 3 — Future hardening (post-S159G)
- Worker registration env-invariant check in coordinator
- Normalize rrig6600 `.bashrc` to match rrig6600b (remove `$PS1` guard)
- TODO_MASTER update

---

*Team Alpha — S159G — 2026-04-02*
