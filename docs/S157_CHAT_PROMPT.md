# S157 Chat Prompt
**Date:** 2026-03-27  
**Last commit:** (see below after final push)  
**Cluster state:** 21/21 GPUs running, 3-trial sweep in progress

---

## Current State

### Cluster
- Zeus: 2× RTX 3080Ti — Active
- rrig6600 (192.168.3.120): 8× RX 6600 — Active, 8 workers
- rrig6600b (192.168.3.154): 8× RX 6600 — Active, 8 workers
- rrig6600c (192.168.3.162): 3× RX 6600 — Active, **3 workers MAX**
- Web dashboard: 45.32.131.224:5002

### S156 Completed Work
- PWC lifecycle bug identified and bandaid deployed (v1 → v2)
- BIOS reflash on rrig6600c — eliminated hard kernel crash
- rrig6600c stable at 3 workers, crashes at 4+
- S155 + S156 changelogs written and committed
- Run 1 complete (seeds 0→1,073,741,824), coverage advancing to Run 2 range

### Active Run
- 3-trial sweep running on 21 GPUs
- seed_start=2,147,483,648 (Run 2 range)
- Monitor: `python3 ~/remote_crash_monitor.py`
- Log: `ssh rzeus "tail -f ~/distributed_prng_analysis/logs/sweep_run1_production.log"`

---

## P0 — S157 Primary Focus

### rrig6600c 4th-Worker Cliff Investigation
Team Beta proposal `PROPOSAL_S157_rrig6600c_4worker_cliff.md` APPROVED.

**Observed behavior:**
```
3 workers → py_procs=5, PageTables=~13,000 kB → STABLE
4 workers → py_procs=6, PageTables=~9,260 kB  → HARD CRASH
```

**Key insight from TB:** PageTables at crash (9,260 kB) is LOWER than stable (13,000 kB). This points to concurrent ROCm init timing, not memory pressure.

**Test plan (strict order):**

**Phase 1 — Serialized startup (highest priority)**
Modify `PWC.startup()` to spawn workers with 12-15s stagger + warmup job between each.
If stable at 4 workers → root cause is concurrent ROCm init timing → fix is larger stagger.

**Phase 2 — Idle-only persistent workers**
Spawn 4 workers, do no work for 2-3 minutes.
If crash → hardware/platform issue. If stable → workload-triggered.

**Phase 3 — GPU set permutation**
Test sets: {0,1,2,3}, {0,1,2,4}, {0,1,3,4}
If crash only with specific GPU → faulty riser/slot.

**Phase 4 — Hardware swap**
Swap suspected GPU/riser with known-good rig.

**Phase 5 — Per-worker memory verification**
Capture VSZ/RSS at 3-worker stable vs 4th worker spawn.

---

## P1 — After rrig6600c Resolved

1. Phase B architectural fix — session-scoped PWC (PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md)
2. S110 root cleanup (884 files in project root)
3. sklearn warnings in Step 5
4. Selfplay NN fix (inner_episode_trainer.py forbidden guard)

---

## Launch Command (working, tested)

```bash
ssh rzeus "cd ~/distributed_prng_analysis && \
  rm -f /tmp/agent_halt daemon_state.json && \
  echo '{\"window_size\": 768, \"offset\": 0, \"skip_min\": 1, \"skip_max\": 147, \"sessions\": [\"midday\", \"evening\"], \"forward_threshold\": 0.55, \"reverse_threshold\": 0.61}' > optimal_window_config.json && \
  source ~/venvs/torch/bin/activate && \
  setsid bash -c 'PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 1 --end-step 1 --force-step 1 >> logs/sweep_run1_production.log 2>&1' &"
```

## Kill Command
```bash
ssh rzeus "touch /tmp/agent_halt && pkill -9 -f 'watcher_agent|window_optimizer|web_dashboard' 2>/dev/null"
ssh rrig6600  "pkill -9 -f sieve_gpu_worker 2>/dev/null"
ssh rrig6600b "pkill -9 -f sieve_gpu_worker 2>/dev/null"
ssh rrig6600c "pkill -9 -f sieve_gpu_worker 2>/dev/null"
```

---

## Key Architecture Invariants (S156)

- rrig6600c MAX 3 workers until 4-worker cliff resolved
- v2 bandaid active — targeted `--persistent` scoped cleanup, SIGTERM first
- BIOS reflash changed failure mode: hard crash → soft CuPy OOM
- Launch requires: `rm -f daemon_state.json` + valid `optimal_window_config.json` placeholder
- PWC v1 broad pkill REMOVED — use v2 only
- dual-push always: `git push origin main && git push public main`
