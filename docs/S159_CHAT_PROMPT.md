# S159 Chat Prompt — PRNG Distributed Analysis System

## Session Priority

**P0 — ZMQ 2-rig validation in progress:**
- rrig6600 + Zeus running clean via ZMQ+SQLite+systemd-run
- Current run: `logs/zmq_systemd_v1.log`
- Monitor: `http://45.32.131.224:5002`
- Tail log: `ssh rzeus "tail -f ~/distributed_prng_analysis/logs/zmq_systemd_v1.log"`

**P1 — After current run completes:**
1. Add rrig6600c for 2-rig full validation (enable linger first)
2. Add rrig6600b for 3-rig validation
3. Increase seed_cap_amd from 2M → 10M for better GPU utilization
4. Write S159 session changelog and commit

---

## What Was Completed This Session (S158/S158D)

### Root Cause — PWC Multi-Rig Crashes
- PWC launches 537 threads simultaneously → GIL thrashing → SSH pipe timeouts
- S158B: ThreadPoolExecutor bounded dispatch — first clean 2-rig run (PWC)
- S158C: Inter-node stagger — REVERTED (caused spawn crashes)

### S158D — ZMQ+SQLite Coordinator (NEW ARCHITECTURE)
- Zeus runs ZMQ PUSH/PULL sockets (ports 5557/5558)
- SQLite job queue with lease expiry, worker identity, idempotent results
- Workers launched ONCE, run independently via TCP — no persistent SSH pipes

### S158D-B — systemd-run --user Worker Launcher
- Replaced nohup SSH launch with systemd-run --user transient services
- Workers survive SSH session teardown (systemd owns process, not bash)
- loginctl enable-linger required on each rig (already done on rrig6600)

### S158D-E — Zeus CUDA Mask Hardening
- CUDA_VISIBLE_DEVICES set at Popen time per Zeus worker
- CUPY_CACHE_DIR isolated per Zeus GPU
- os.environ.setdefault in worker (launcher mask wins)

### Current Commit State
```
2d5db89 fix(s158d-b): systemd-run --user launcher + Zeus CUDA isolation
e454fbc fix(s158d-b+e): systemd-run launcher + CUDA mask hardening
074ab62 fix(s158d): bind ZMQ sockets BEFORE launching workers
14e7406 fix(s158d): remove proc.wait() — SSH fire-and-forget
cd74f61 fix(s158d): newline join for background worker launches
31d0a8a fix(s158d): v3 worker — correct sieve result parsing + defensive logging
a30f5dc config(s158d): enable ZMQ by default, disable PWC
```

---

## Cluster State

**Topology:**
- Zeus: `~/distributed_prng_analysis/`, venv `~/venvs/torch/bin/activate`
- rrig6600 (192.168.3.120): linger enabled ✅
- rrig6600b (192.168.3.154): linger NOT YET enabled
- rrig6600c (192.168.3.162): linger NOT YET enabled

**ZMQ Ports:** job=5557, result=5558
**Zeus IP on LAN:** 192.168.3.127
**Worker identity format:** `hostname:gpuN`

**Current run:** zmq_systemd_v1 — Zeus + rrig6600, ZMQ+systemd-run
**seed_cap_amd:** 2,000,000 (increase to 10M after 3-rig stable)

---

## Key Architecture Facts

**Why ZMQ works:**
- SSH used ONCE per rig at startup to launch systemd services
- All compute traffic is TCP directly to Zeus ZMQ sockets
- No persistent SSH connections during compute

**Why GPU utilization is low (~0% between chunks):**
- Chunk size 2M seeds → GPU burst ~50-100ms
- ZMQ round-trip ~600ms between chunks
- Fix: increase seed_cap_amd to 10-20M

**systemd service management on rigs:**
```bash
# Check worker status
ssh rrig6600 "systemctl --user status zmq-worker-gpu0 --no-pager | head -5"
# Stop all workers
ssh rrig6600 "for i in {0..7}; do systemctl --user stop zmq-worker-gpu\$i 2>/dev/null; done"
```

---

## Launch Sequence (for next session)

**Kill everything:**
```bash
ssh rzeus "touch /tmp/agent_halt && pkill -9 -f 'watcher_agent|window_optimizer|zmq_sqlite_worker' 2>/dev/null"
ssh rrig6600  "for i in {0..7}; do systemctl --user stop zmq-worker-gpu\$i 2>/dev/null; done"
ssh rrig6600b "for i in {0..7}; do systemctl --user stop zmq-worker-gpu\$i 2>/dev/null; done"
ssh rrig6600c "for i in {0..7}; do systemctl --user stop zmq-worker-gpu\$i 2>/dev/null; done"
```

**Enable linger on remaining rigs (interactive SSH required for sudo):**
```bash
ssh rrig6600b  # then: sudo loginctl enable-linger michael
ssh rrig6600c  # then: sudo loginctl enable-linger michael
```

**Reset coverage:**
```bash
ssh rzeus "cd ~/distributed_prng_analysis && python3 -c \"
import sqlite3
conn = sqlite3.connect('prng_analysis.db')
conn.execute('DELETE FROM exhaustive_progress WHERE prng_type=?', ('java_lcg',))
conn.commit(); conn.close(); print('reset')
\""
```

**Launch:**
```bash
ssh rzeus "cd ~/distributed_prng_analysis && \
  rm -f /tmp/agent_halt daemon_state.json optimal_window_config.json && \
  source ~/venvs/torch/bin/activate && \
  setsid bash -c 'PYTHONPATH=. python3 agents/watcher_agent.py \
  --run-pipeline --start-step 1 --end-step 1 --force-step 1 \
  >> logs/zmq_3rig_v1.log 2>&1' &"
```

---

## Active TODOs (priority order)

1. Complete current 2-rig ZMQ validation run
2. Enable linger on rrig6600b and rrig6600c
3. 3-rig validation run (all 26 GPUs)
4. Increase seed_cap_amd 2M → 10M after 3-rig stable
5. Write S158/S159 session changelogs
6. S110 root cleanup (884 files) — deferred
7. sklearn warnings in Step 5 — deferred

---

## Git Workflow Reminder
- Dual-push always: `git push origin main && git push public main`
- origin = `git@github.com:mmercalde/prng_cluster_project.git`
- public = `git@github.com:mmercalde/prng_cluster_public.git`
- Clone: `git clone https://github.com/mmercalde/prng_cluster_public.git`
