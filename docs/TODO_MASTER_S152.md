# TODO MASTER — S152
**Generated:** 2026-03-21 (end of S151)  
**Last commit:** `fd38d17`

---

## 🔴 P0 — Start Here

1. **Deploy crash monitors before any launch**
   ```bash
   ssh rrig6600 "nohup ~/rig_crash_monitor.sh > /tmp/rig_crash_monitor.log 2>&1 &"
   ssh rrig6600b "nohup ~/rig_crash_monitor.sh > /tmp/rig_crash_monitor.log 2>&1 &"
   ssh rrig6600c "nohup ~/rig_crash_monitor.sh > /tmp/rig_crash_monitor.log 2>&1 &"
   ```

2. **Verify rrig6600c sshd fix holds with 8 workers**
   - Launch sweep_run1.sh fresh
   - Confirm 26/26 GPUs active at first trial
   - Watch monitor through hybrid passes (Pass 3/4)

3. **Re-enable slim_v1 on all rigs** (confirmed NOT the root cause)
   ```bash
   scp rzeus:~/distributed_prng_analysis/sieve_gpu_worker.py ~/Downloads/
   scp ~/Downloads/sieve_gpu_worker.py rrig6600:~/distributed_prng_analysis/
   scp ~/Downloads/sieve_gpu_worker.py rrig6600b:~/distributed_prng_analysis/
   scp ~/Downloads/sieve_gpu_worker.py rrig6600c:~/distributed_prng_analysis/
   ```
   Note: Zeus coordinator already has slim_v1 parser. Only worker files need updating.

4. **Fresh Run 1 launch** (previous Run 1 study has only 1 trial, survivors lost)
   ```bash
   ssh rzeus "cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate && \
     rm -f optimal_window_config.json /tmp/agent_halt && \
     PYTHONPATH=. python3 agents/watcher_agent.py --clear-halt && \
     bash sweep_run1.sh"
   ```

---

## 🔴 P1 — Critical Path

5. **Write survivors to disk as found** (not just in memory accumulator)
   Current: survivors held in memory, written only at trial end or checkpoint
   Risk: process hang/kill loses entire trial's survivors
   Fix: write incremental NPZ after every N chunks (e.g. every 50 chunks)

6. **Investigate worker session drops after extended runs**
   Documented in S138: workers drop after hours of operation
   rrig6600b workers died at ~22:09 after ~2 hours
   Need keepalive or session TTL refresh mechanism

7. **`--force-step` flag for WATCHER**
   Freshness check blocks every resume — workaround is `rm -f optimal_window_config.json`
   Add `--force-step N` to bypass freshness for step N

8. **Investigate why rrig6600c had MaxSessions 5**
   Was this intentional security hardening for a specific purpose?
   Check git history, setup logs, operating manual

---

## 🟡 P2 — After Run 1 Stable

9. **sweep_run2.sh** — warm-start from Run 1 best params via `enqueue_trial()`

10. **Add sshd config check to PWC preflight**
    Verify `MaxSessions >= worker_pool_size + 2` on all remote nodes

11. **Add sshd config to REMOTE_NODE_SETUP_CHECKLIST.md**

12. **Circuit breaker v3** — TB-approved design pending implementation
    Requires: `_get_available_workers()` include dead-but-recoverable handles

13. **Telegram alert on worker quarantine / rig reboot detection**

---

## 🟢 Backlog (unchanged from S150)

14. S110 root cleanup (884 files in project root)
15. sklearn warnings in Step 5
16. Remove dead CSV writer from `coordinator.py`
17. Regression diagnostics gate → set to True
18. S103 Part 2
19. Chapter 13 wire-up: `dispatch_selfplay()`, `dispatch_learning_loop()`
20. Selfplay NN two-part fix
21. Walk-forward simulation
22. Remove legacy dict-list coordinator parser (after slim_v1 stable)

---

## Current State

| Item | Value |
|---|---|
| Last commit | `fd38d17` |
| Run 1 study | `window_opt_1774137256` |
| Trials done | 1 (survivors lost — fresh start needed) |
| NPZ | 666 seeds |
| rrig6600c sshd | Fixed — MaxSessions 50 |
| rrig6600c gpu_count | 8 (restored) |
| slim_v1 on rigs | REVERTED (re-enable S152 P0) |
| Crash monitors | Deployed on all 3 rigs |
