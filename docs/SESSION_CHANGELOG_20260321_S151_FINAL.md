# SESSION CHANGELOG — S151
**Date:** 2026-03-21  
**Commits:** `38393b5` → `cc113b5` → `f3fdbf1` → `e4905c2` → `12dbeaf` → `fd38d17`  
**Status:** CLOSED — root cause found, partial Run 1 complete

---

## Summary

S151 was an extended debugging session that began with Run 1 launch and
descended into a full day of rrig6600c crash investigation. After exhausting
numerous theories (hardware, kernel, slim_v1, CPU, CuPy memory pool, PCIe),
the root cause was identified as **`MaxSessions 5` in rrig6600c's sshd_config**
— a security hardening setting that limited SSH sessions to 5, causing workers
5-7 to be rejected by sshd. Combined with `ClientAliveInterval 300`, this
created a sustained crash loop whenever the rig operated with 8 workers.

Fix applied, documented, and committed. Trial 1 of Run 1 completed with 4,473
bidirectional survivors (W5_O77 config). Run 1 study active.

---

## Root Cause: rrig6600c sshd_config

**Full analysis:** `docs/ROOT_CAUSE_ANALYSIS_RRIG6600C_S151.md`

**Fix applied to `/etc/ssh/sshd_config` on rrig6600c:**
```
MaxSessions 5    →  MaxSessions 50
ClientAliveInterval 300  →  ClientAliveInterval 0
ClientAliveCountMax 2    →  ClientAliveCountMax 3
MaxStartups (missing)    →  MaxStartups 50:30:100
```
Backup: `/etc/ssh/sshd_config.bak_s151`
Applied with: `sudo systemctl reload sshd`

---

## Fixes Deployed

### Fix 1 — CUPY_CUDA_MEMORY_POOL_TYPE=none
**Commit:** `f3fdbf1`  
**File:** `persistent_worker_coordinator.py` (ROCM_ENV_VARS)  
Added `CUPY_CUDA_MEMORY_POOL_TYPE=none` — disables CuPy memory pool on
workers. Eliminates pool management race conditions under concurrent workers.
Already used in Chapter 4 full scoring worker.

### Fix 2 — Per-node respawn lock + stagger
**Commit:** `12dbeaf`  
**Patch:** `apply_s151_respawn_stagger.py`  
**File:** `persistent_worker_coordinator.py`  
Added `_node_respawn_locks` dict. `_ensure_worker_alive()` now acquires
per-node lock and sleeps `ROCM_SPAWN_STAGGER_S` (4s) before each respawn.
Converts parallel respawn hammer to serialized stagger.

### Fix 3 — rrig6600c sshd_config (PRIMARY FIX)
**Not a code change** — SSH daemon config on rrig6600c.  
See root cause analysis document.

### Fix 4 — rrig6600c gpu_count restored to 8
**Commit:** `fd38d17`  
`distributed_config.json` — rrig6600c gpu_count restored from 4 to 8
after sshd fix confirmed.

---

## slim_v1 Status

slim_v1 was reverted to pre-S150 worker on all rigs during S151 debugging.
**slim_v1 was NOT the root cause** — confirmed by root cause analysis.
slim_v1 should be re-enabled in S152 after verifying 8-worker stability
on rrig6600c with new sshd config.

Current state:
- Zeus coordinator: has slim_v1 parser (backward compatible)
- All 3 rigs: running pre-slim_v1 `sieve_gpu_worker.py` (backup)

---

## Diagnostic Tools Built

### Crash Monitor (`rig_crash_monitor.sh`)
Deployed to all 3 rigs. Logs per-second: worker count, CPU load, RAM usage,
VRAM usage, SSH connection count. Essential for future diagnosis.

**Deploy:**
```bash
scp ~/Downloads/rig_crash_monitor.sh rrig6600:~/
scp ~/Downloads/rig_crash_monitor.sh rrig6600b:~/
scp ~/Downloads/rig_crash_monitor.sh rrig6600c:~/
ssh rrig6600 "chmod +x ~/rig_crash_monitor.sh && nohup ~/rig_crash_monitor.sh > /tmp/rig_crash_monitor.log 2>&1 &"
ssh rrig6600b "chmod +x ~/rig_crash_monitor.sh && nohup ~/rig_crash_monitor.sh > /tmp/rig_crash_monitor.log 2>&1 &"
ssh rrig6600c "chmod +x ~/rig_crash_monitor.sh && nohup ~/rig_crash_monitor.sh > /tmp/rig_crash_monitor.log 2>&1 &"
```

**Read:**
```bash
ssh rrig6600c "tail -20 /tmp/rig_crash_monitor.log"
```

### Stress Test Harness (`test_rrig6600c_stress.py`)
Replicates PWC spawn + dispatch pattern. Useful for validating worker
count limits before production runs.

---

## Run 1 Status

| Item | Value |
|---|---|
| Study | `window_opt_1774137256` (fresh — old studies archived) |
| Seed range | 0 → 1,073,741,824 |
| Trials completed | 1 |
| Best config | W5_O77 — 4,473 bidirectional |
| NPZ accumulator | 666 seeds (checkpoint not fired — process died) |
| Workers at end | 20/20 alive (rrig6600c at 4 workers) |

**Note:** Trial 1 survivors (4,473 bidirectional) are in Optuna DB but
NOT in NPZ — process killed before checkpoint fired. Survivors lost.
Fresh start required for S152.

---

## Infrastructure Changes

### rrig6600c sshd_config
- `MaxSessions`: 5 → 50
- `ClientAliveInterval`: 300 → 0
- `ClientAliveCountMax`: 2 → 3
- `MaxStartups`: added `50:30:100`

### Crash monitors deployed
- All 3 rigs: `~/rig_crash_monitor.sh` → `/tmp/rig_crash_monitor.log`

---

## Commits This Session

| Commit | Description |
|--------|-------------|
| `38393b5` | config(s151): limit rrig6600c to 6 workers |
| `cc113b5` | config(s151): revert rrig6600c to 8 workers |
| `f3fdbf1` | fix(s151): CUPY_CUDA_MEMORY_POOL_TYPE=none to worker env |
| `e4905c2` | config(s151): set rrig6600c gpu_count=4 |
| `12dbeaf` | fix(s151): per-node respawn lock + stagger |
| `fd38d17` | fix(s151): root cause rrig6600c — MaxSessions 5, restored gpu_count=8 |

---

## Architecture Invariants Added S151

- **[S151]** `CUPY_CUDA_MEMORY_POOL_TYPE=none` in all worker spawn envs
- **[S151]** Per-node respawn lock — serializes respawns with 4s stagger
- **[S151]** rrig6600c sshd MaxSessions 50 (was 5)
- **[S151]** Crash monitor deployed to all rigs
- **[S151]** sshd config parity required across all rigs (add to preflight)
