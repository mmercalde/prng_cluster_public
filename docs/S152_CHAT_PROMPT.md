# S152 Chat Prompt — PRNG Distributed Analysis System

## Session Priority

**P0 — Start here:**
1. Deploy crash monitors on all rigs before any launch
2. Re-enable slim_v1 on all rigs (confirmed NOT root cause of crashes)
3. Verify rrig6600c stable with 8 workers (sshd MaxSessions fixed)
4. Fresh Run 1 launch — confirm 26/26 GPUs, watch through hybrid passes

---

## Root Cause Resolved This Session (S151)

**rrig6600c crashed repeatedly due to `MaxSessions 5` in sshd_config.**

With 8 workers needing 8 SSH connections, sshd rejected workers 5-7.
Combined with `ClientAliveInterval 300`, this caused sustained crash loops.

**Fix applied:**
- rrig6600c `/etc/ssh/sshd_config`: MaxSessions 50, ClientAliveInterval 0,
  ClientAliveCountMax 3, MaxStartups 50:30:100
- Backup: `/etc/ssh/sshd_config.bak_s151`

**Full analysis:** `docs/ROOT_CAUSE_ANALYSIS_RRIG6600C_S151.md`

---

## Current System State (end of S151)

**Last commit: `fd38d17` on both remotes.**

### Infrastructure state
| Item | State |
|---|---|
| rrig6600c sshd | ✅ Fixed — MaxSessions 50 |
| rrig6600c gpu_count | ✅ 8 (restored) |
| slim_v1 on rigs | ⚠️ REVERTED — re-enable S152 P0 |
| Crash monitors | ✅ Deployed — `~/rig_crash_monitor.sh` on all rigs |
| CUPY_CUDA_MEMORY_POOL_TYPE=none | ✅ In worker env |
| Respawn lock+stagger | ✅ Committed `12dbeaf` |

### Run 1 status
| Item | Value |
|---|---|
| Study | `window_opt_1774137256` |
| Seed range | 0 → 1,073,741,824 |
| Trials completed | 1 |
| Best config | W5_O77 — 4,473 bidirectional |
| NPZ accumulator | 666 seeds (survivors lost — process killed) |
| Status | **NEEDS FRESH START** |

---

## S152 P0 Sequence

```bash
# Step 1: Deploy crash monitors
ssh rrig6600 "nohup ~/rig_crash_monitor.sh > /tmp/rig_crash_monitor.log 2>&1 &"
ssh rrig6600b "nohup ~/rig_crash_monitor.sh > /tmp/rig_crash_monitor.log 2>&1 &"
ssh rrig6600c "nohup ~/rig_crash_monitor.sh > /tmp/rig_crash_monitor.log 2>&1 &"

# Step 2: Re-enable slim_v1 on rigs
scp rzeus:~/distributed_prng_analysis/sieve_gpu_worker.py ~/Downloads/
scp ~/Downloads/sieve_gpu_worker.py rrig6600:~/distributed_prng_analysis/
scp ~/Downloads/sieve_gpu_worker.py rrig6600b:~/distributed_prng_analysis/
scp ~/Downloads/sieve_gpu_worker.py rrig6600c:~/distributed_prng_analysis/

# Step 3: Fresh Run 1
ssh rzeus "cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate && \
  rm -f optimal_window_config.json /tmp/agent_halt && \
  PYTHONPATH=. python3 agents/watcher_agent.py --clear-halt && \
  bash sweep_run1.sh"

# Step 4: Verify 26/26 GPUs alive
ssh rzeus "grep 'Worker pool ready' ~/distributed_prng_analysis/logs/sweep_run1_production.log | tail -3"

# Step 5: Watch monitor during hybrid passes
ssh rrig6600c "tail -f /tmp/rig_crash_monitor.log"
```

---

## Key Commands

### Monitor sweep
```bash
ssh rzeus "tail -f ~/distributed_prng_analysis/logs/sweep_run1_production.log | \
  grep --line-buffered -E 'SAVE|NEW BEST|Running.*sieve|Total bidirectional|Worker pool ready|quarantine'"
```

### Check NPZ
```bash
ssh rzeus "python3 -c \"import numpy as np; d=np.load('distributed_prng_analysis/bidirectional_survivors_all.npz'); print('NPZ seeds:', len(d[list(d.keys())[0]]))\""
```

### Kill all
```bash
ssh rzeus "pkill -f window_optimizer; pkill -f watcher_agent"
ssh rrig6600 "pkill -f sieve_gpu_worker 2>/dev/null"
ssh rrig6600b "pkill -f sieve_gpu_worker 2>/dev/null"
ssh rrig6600c "pkill -f sieve_gpu_worker 2>/dev/null"
```

### Check rig monitors
```bash
ssh rrig6600 "tail -3 /tmp/rig_crash_monitor.log"
ssh rrig6600b "tail -3 /tmp/rig_crash_monitor.log"
ssh rrig6600c "tail -3 /tmp/rig_crash_monitor.log"
```

---

## Infrastructure

**Zeus:** `rzeus`, `~/distributed_prng_analysis/`, `~/venvs/torch/bin/activate`  
**Rigs:** `rrig6600` (192.168.3.120), `rrig6600b` (192.168.3.154), `rrig6600c` (192.168.3.162)  
**Dashboard:** `45.32.131.224:5002`  
**Git:** dual-push `origin` + `public` always  
**Rig deploys:** No git on rigs — use scp from Zeus

---

## P1 Backlog

- Write survivors to disk incrementally (not just in memory)
- Investigate worker session drops after extended runs (S138 open item)
- `--force-step` flag for WATCHER
- sweep_run2.sh warm-start
- Circuit breaker v3 (TB-approved, pending implementation)
- Telegram alert on worker quarantine
- Add sshd MaxSessions check to PWC preflight
- S110 root cleanup, sklearn warnings, Chapter 13 wire-up, selfplay NN fix
