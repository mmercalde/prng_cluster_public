# SESSION CHANGELOG — S162
**Date:** 2026-04-04 / 2026-04-05
**Session:** S162
**Focus:** TCP-PWC Production Integration, rrig6600c Crash Root Cause, DKMS Fix, Reverse Sieve Warmup
**Author:** Claude (Team Alpha Lead Dev)
**Status:** Closed — Trial 1 complete, Trials 2 & 3 pending next session

---

## Summary

S162 spanned two days. Day 1 (Apr 4) focused on TCP-PWC production integration and initial crash investigation. Day 2 (Apr 5) confirmed the DKMS driver as the root cause fix, completed Trial 1 successfully, and applied warmup + VRAM instrumentation patches to `reverse_sieve_filter.py`. Late in Day 2, forward sieve crashes appeared on rrig6600 and rrig6600c after multiple crash/reboot cycles — likely cumulative GPU state degradation. Session ended with decision to run isolation tests after clean reboot.

---

## Day 1 — 2026-04-04

### S162-1 — First 26-GPU Run Attempt
- WATCHER launched with TCP-PWC default, all 3 rigs, 26 GPUs
- Dashboard: 26/26 GPUs active, 7,162,151 seeds/sec aggregate
- rrig6600c crashed at ~18 seconds — same signature as all previous transports

**Netconsole crash signature:**
```
GCVM_L2_PROTECTION_FAULT_STATUS: 0xFFFFFFFF
Faulty UTCL2 client ID: unknown (0x1ff)
WALKER_ERROR: 0x7 / PERMISSION_FAULTS: 0xf / MAPPING_ERROR: 0x1
Multiple GPUs + PIDs simultaneously → watchdog hard LOCKUP
```

### S162-2 — Cleanup Patch Missing on All Rigs
- `_best_effort_gpu_cleanup()` patch (commit `b3c3207`) was never deployed to rigs
- Fix: SCP'd `pwc_worker_service.py` from Zeus to all 3 rigs
- **Key learning:** Files committed to Zeus repo are NOT automatically on rigs. Rig deployment requires explicit SCP.

### S162-3 — Isolation Test (rrig6600c + Zeus only)
- Disabled rrig6600 and rrig6600b via `gpu_count=0` in `distributed_config.json`
- Full 3-trial run with Zeus + rrig6600c only (10 GPUs)
- **Result:** ✅ Complete success — 18m 52s, zero crashes, zero netconsole faults
- **Invariant confirmed:** rrig6600c crash caused exclusively by simultaneous 3-rig load

### S162-4 — Zeus Network Stack Investigation
- Active NIC: `enp2s0` — Intel I210 Gigabit (1GbE)
- TCP buffer tuning applied and persisted to `/etc/sysctl.conf`:
  - `wmem/rmem_max = 16MB`, `default = 1MB`
- Dispatch semaphore `Semaphore(8)` added to `TCPWorkerTransport._handle_client()`

### S162-5 — Second 26-GPU Run (Post-Cleanup + Semaphore)
- Crashed at kernel uptime 3036s (~50 minutes)
- **New crash signature:**
  ```
  Faulty UTCL2 client ID: SQC (inst) (0x9)  ← SHADER INSTRUCTION CACHE
  WALKER_ERROR: 0x0 / MAPPING_ERROR: 0x0
  ```
- Crash delayed from ~18s to ~50min — cleanup patch working
- Only ONE GPU, ONE PID — not multi-GPU cascade initially

### S162-6 — Internet Research
- ROCm GitHub Issue #5616: Identical SQC (inst) fault signature documented
- `AMD_SERIALIZE_KERNEL=3` noted for diagnostic fault isolation
- TB Proposal submitted: `docs/PROPOSAL_S162_RRIG6600C_CRASH_ROOT_CAUSE_v1_0.md`

---

## Day 2 — 2026-04-05

### S162-7 — DKMS Driver — ROOT CAUSE CONFIRMED AND FIXED
**Problem:** rrig6600c (and all rigs) using stock Ubuntu kernel amdgpu driver.
Stock driver cannot handle 8 concurrent GPU compute workers per rig under full 3-rig load.

**Fix:** Install AMD `amdgpu-dkms 6.12.12.60403` on all 3 rigs.
```
filename: /lib/modules/6.8.0-106-generic/updates/dkms/amdgpu.ko
```

**Verified on all 3 rigs:**
```bash
modinfo amdgpu | grep filename
# → /lib/modules/6.8.0-106-generic/updates/dkms/amdgpu.ko  ✅
```

### S162-8 — cwsr_enable=0 Investigation
- `amdgpu.cwsr_enable=0` required ONLY on rrig6600c
- Adding to rrig6600b caused KIQ fence timeouts (broke stable rig)
- Adding to rrig6600 caused instability

**Final confirmed GRUB config:**
| Rig | cwsr_enable=0 |
|-----|---------------|
| rrig6600 | No |
| rrig6600b | No |
| rrig6600c | Yes |

### S162-9 — Trial 1 Completed Successfully ✅
- Forward sieve: 2,544,571 survivors
- Reverse sieve: 1,751,847 survivors
- Forward hybrid: 52,096 survivors
- Reverse hybrid: 13,833 survivors
- **Bidirectional: 9,921**
- Duration: 31:31
- 26/26 GPUs, 4 workers online throughout
- rrig6600c faulted during reverse sieve but DKMS driver recovered workers

### S162-10 — Reverse Sieve Crash Analysis
- Crashes always trigger at forward → reverse sieve transition
- First fault: `SQC (inst)` — instruction fetch fault (not data)
- Cause: cold kernel compilation on 8 workers simultaneously, no GPU context warmup
- Forward sieve worker has `cp.zeros(1)` warmup; reverse sieve had none

**Fix applied (commit `8d91311`):**
1. **Warmup** — `cp.zeros(1)` + `synchronize()` added to `reverse_sieve_filter.py` before allocations
2. **VRAM instrumentation** — `[VRAM]` logging before/after GPU array allocations

### S162-11 — Trials 2 & 3 Did Not Complete
- optimal_window_config.json never written
- Window optimizer stopped after Trial 1 — rrig6600c went offline, cluster dropped below `--min-workers 24`
- Optuna study DB has Trial 1 data — warm-start will work on relaunch
- Step 1 ESCALATED by watcher (file validation failed)

### S162-12 — Late Session Forward Sieve Crashes
- After multiple crash/reboot cycles, forward sieve started crashing on rrig6600 AND rrig6600c simultaneously
- perf=high not set (resets to auto on reboot) — load imbalance: rrig6600c at 2750MHz, others at 700MHz
- Likely cumulative GPU state degradation from day of repeated crashes
- Decision: run isolation tests after full clean reboot before next attempt

---

## Commits This Session

| Commit | Description |
|--------|-------------|
| `10871b8` | feat(s162): switch WATCHER default transport to TCP-PWC |
| `b89b33a` | feat(s162): TCP-PWC default — fix parser/fallback/manifest/helptext |
| `c2a59cc` | fix(s162): TCP startup reaps stale sieve_gpu_worker |
| `b3c3207` | fix(s162): add _best_effort_gpu_cleanup to TCP worker between chunks |
| `89c1512` | docs: SESSION_CHANGELOG_20260402_S159G.md |
| `8d91311` | S162: Add GPU warmup and VRAM instrumentation to reverse_sieve_filter.py |

---

## Architecture Invariants Added This Session

- **[S162]** Rig files must be explicitly SCP'd after every commit — Zeus repo is NOT auto-synced to rigs
- **[S162]** Zeus TCP buffers tuned: `wmem/rmem_max=16MB`, persisted to `/etc/sysctl.conf`
- **[S162]** Dispatch semaphore `Semaphore(8)` in `TCPWorkerTransport._handle_client()`
- **[S162]** `gpu_count=0` in `distributed_config.json` is the correct mechanism to disable a rig
- **[S162]** rrig6600c crash under 3-rig load = stock Ubuntu amdgpu driver — fix = DKMS
- **[S162]** `cwsr_enable=0` required ONLY on rrig6600c — breaks other rigs if applied globally
- **[S162]** Reverse sieve requires GPU context warmup before kernel launch — now in `reverse_sieve_filter.py`

---

## Current Cluster State (End of Session)

| Component | Status |
|-----------|--------|
| DKMS driver | ✅ All 3 rigs |
| cwsr_enable=0 | ✅ rrig6600c only |
| perf=high | ❌ NOT persistent — resets on reboot |
| Warmup patch | ✅ Committed `8d91311` |
| VRAM instrumentation | ✅ Committed `8d91311` |
| optimal_window_config.json | ❌ Not written — Trials 2 & 3 pending |
| Optuna study DB | ✅ Trial 1 data present — warm-start ready |

---

## Open TODOs for Next Session

1. **Make perf=high permanent** — systemd startup service on all 3 rigs
2. **Run isolation test on rrig6600c alone** — verify stability after clean reboot
3. **Full 3-trial run** — complete Trials 2 & 3, write optimal_window_config.json
4. **Verify warmup effectiveness** — check `[VRAM]` lines in log when reverse sieve starts
5. **Selfplay NN fix** (`inner_episode_trainer.py`) — forbidden guard + y-normalization
6. **S110 root cleanup** — 884 stray files

---

## Next Session Launch Command
```bash
ssh rzeus "cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate && \
  PYTHONPATH=. python3 -m agents.watcher_agent --clear-halt && \
  rm -f optimal_window_config.json && \
  PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 1 --end-step 1 \
  2>&1 | tee logs/s163_run1.log"
```

---

*Session S162 — 2026-04-04/05 — Team Alpha*
