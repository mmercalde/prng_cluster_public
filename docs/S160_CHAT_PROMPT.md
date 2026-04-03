# S160 Chat Prompt
**Date:** 2026-04-03 (next session)
**Continues from:** S159G — 2026-04-02

---

## Context

You are Team Alpha (Lead Dev/Implementation) on a distributed PRNG analysis system.
26 GPUs across 4 nodes: Zeus (2×RTX 3080Ti, CUDA) + rrig6600/rrig6600b/rrig6600c
(8×RX 6600 each, ROCm). ZMQ+SQLite coordinator (S158D) is the active transport.

## What Happened Last Session (S159G)

We identified and applied the fix for persistent rrig6600 GPU crashes.

**Root cause (primary candidate):** rrig6600 workers were running without 4 critical
ROCm protective env vars that rrig6600b and rrig6600c had. Specifically missing:
- `HSA_ENABLE_SDMA=0` — disables SDMA engine with known GPU VM fault bugs
- `HSA_ENABLE_RUNTIME_POWER_MGMT=0`
- `AMDGPU_NO_POWER_PROFILE=1`
- `ROCR_VISIBLE_DEVICES=0` (per-worker, coordinator-managed)

**Fix applied:** Added to `/etc/environment` on ALL 3 AMD rigs:
```
HSA_ENABLE_SDMA=0
HSA_ENABLE_RUNTIME_POWER_MGMT=0
AMDGPU_NO_POWER_PROFILE=1
```
All 3 rigs rebooted. `/etc/environment` verified clean and identical post-reboot.

**Also confirmed:** ZMQ SQLite lease recovery works — when rrig6600 crashed mid-run,
surviving rigs completed the forward sieve without data loss. Architecture is
crash-resilient.

## Current Cluster State

- Zeus: halted (`/tmp/agent_halt` set), needs fresh launch
- All 3 rigs: up, rebooted, fix applied
- Last run: killed mid-reverse-sieve
- DB: needs `exhaustive_progress` reset for `java_lcg` before fresh run
- seed_start=0 reset required

## TB Acceptance Criteria (Must Pass Before S160 Is Done)

1. Fresh coordinator launch ← do this first
2. Live `zmq_sqlite_worker` PID on rrig6600 shows ALL of:
   - `HSA_ENABLE_SDMA=0`
   - `HSA_ENABLE_RUNTIME_POWER_MGMT=0`
   - `AMDGPU_NO_POWER_PROFILE=1`
   - `HSA_OVERRIDE_GFX_VERSION=10.3.0`
   - `ROCR_VISIBLE_DEVICES=0`
3. One clean full-cluster stability run — rrig6600 must NOT crash

## First Action This Session

Fresh launch + immediate env verification:

```bash
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

Then 45 seconds later — TB acceptance test:
```bash
ssh rrig6600 "systemctl --user cat zmq-worker-gpu0.service 2>/dev/null | grep Environment"
ssh rrig6600 "cat /proc/\$(pgrep -f zmq_sqlite_worker | head -1)/environ \
  2>/dev/null | tr '\0' '\n' | grep -E 'HSA|SDMA|ROCR|POWER'"
```

Expected: all 5 vars present. If yes — let run complete. If no — escalate to TB.

## If S159G Acceptance Test Passes

Post-run tasks:
1. Commit SESSION_CHANGELOG_20260402_S159G.md and TB docs
2. Normalize rrig6600 `.bashrc` — remove `[ -n "$PS1" ]` guard before activate
3. Add worker env-invariant check to coordinator (TB-recommended hardening)
4. Update TODO_MASTER

## Key Files
- Coordinator: `zmq_sqlite_coordinator.py` (`--use-zmq-sqlite` flag)
- Worker: `zmq_sqlite_worker.py`
- Log: `logs/zmq_s159g_fix_v1.log`
- Netconsole: `logs/netconsole_all_rigs.log`
- TB docs: `docs/TB_*_S159G_*.md`

## Architecture Invariants
- seed_cap_amd: 2,000,000
- Zeus semaphore: 2
- ZMQ ports: job=5557, result=5558
- Git: dual-push `git push origin main && git push public main`
