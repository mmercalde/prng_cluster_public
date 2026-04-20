# MASTER TODO LIST — S163-KARG
**Compiled:** 2026-03-02 (S114) | **Updated:** 2026-04-19 (S163-KARG)
**Status:** S163-KARG complete. DPM pin deployed. Skip-dedup fix live. Production run active.

---

## ✅ COMPLETED THIS SESSION (S163-KARG)

| Item | Session |
|------|---------|
| Root cause S163 GPU crashes: missing `amdgpu.conf` (gfxoff=0) + `snd-power.conf` on rrig6600/rrig6600c | S163 |
| `amdgpu-dpm-pin.service` deployed to all 3 rigs — pins all 24 GPUs to `manual` DPM at boot | S163-KARG |
| `gpu-enum-heal.service` removed (stale 12-GPU config) | S163-KARG |
| `rocm-perf-auto` cron removed (was overriding DPM pin) | S163-KARG |
| rrig6600b GRUB recovered via ser8 USB adapter + chroot | S163-KARG |
| **[S163-KARG-DEDUP]** Skip fwd/rev dedup when summary-only — eliminates 2.5h post-trial CPU burn | S163-KARG |
| **[S163-KARG]** Vectorized `deduplicate_survivors()` via numpy lexsort | S163-KARG |
| **[S163-KARG]** Vectorized NPZ accumulator merge via searchsorted | S163-KARG |
| **[S163-KARG-NPZ]** Schema backfill for missing fields in older prior NPZ schemas | S163-KARG |
| **[S163-KARG-PWC]** `node_allowlist` propagated into `PersistentWorkerCoordinator._load_config()` | S163-KARG |
| **[S163-KARG-FIX1]** n_parallel flag inheritance — 8-hop propagation chain | S163-KARG |
| **[S163-KARG-PORT]** Pre-fork `fuser -k` TCP port cleanup added to NP2 path | S163-KARG |
| **[S163-KARG-KILL]** Pre-fork `pkill -9 pwc_worker_service` on all rigs added to NP2 path | S163-KARG |
| `n_parallel` reverted to 1 — n_parallel=2 concept scrapped | S163-KARG |
| `reset_seed_coverage.py` utility script deployed to Zeus | S163-KARG |
| Harness A (NPZ dedup/merge): 22/22 PASS on Zeus | S163-KARG |
| Harness B (n_parallel structural): 9/9 PASS on Zeus | S163-KARG |
| GDM disabled + multi-user.target permanent on all 3 rigs | S163-KARG |
| Zeus nvidia-compute-mode systemd service installed | S163-KARG |

---

## 🔴 P1 — HIGH PRIORITY (Next Session)

### Run Step 2
- [ ] **Run Step 2 — Scorer Meta-Optimizer** with accumulated survivors from `bidirectional_survivors_all.npz`
  Current accumulator: 20,912 seeds (as of last S163 run)
  ```bash
  ssh rzeus "cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate && \
    PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 2 --end-step 2 \
    2>&1 | tee logs/step2_run1.log"
  ```

### Fix crash_forensic_daemon.py — 3 bugs
- [ ] **Startup false-DOWN bug** — crash monitor shows DOWN incorrectly on launch before any crash events. Fix initial state detection logic.
- [ ] **Log-caching bug** — daemon caches pipeline_log path at startup, never refreshes. New runs in same session point to old log. Fix: re-resolve newest matching log on each capture cycle.
- [ ] **dmesg capture empty** — `dmesg_tail.log` consistently empty. Fix: replace with `sudo journalctl -k --since <launch_time>` pulled from all reachable rigs immediately at capture time.

### Web dashboard auto-start
- [ ] Dashboard must auto-start when watcher_agent launches — currently requires manual restart every run.
- [ ] Add to `monitor_all.sh` or wire into watcher startup.

---

## 🟠 P2 — MEDIUM PRIORITY

### Zeus RTX 3080Ti Performance — TB Proposal Required
- [ ] **Zeus TCP worker path** — remove `_is_localhost` bypass in `persistent_worker_coordinator.py`
  so Zeus 3080 Tis use the same TCP persistent worker path as AMD rigs, not the slow local sieve path.
  - Current: Zeus dispatches via `execute_local_sieve_job()` subprocess — ~10K s/s
  - Expected after fix: Zeus via TCP workers — ~100K+ s/s (matching AMD rigs)
  - Requires launching `pwc_worker_service` on Zeus itself connecting to `localhost:5600`
  - **TB proposal required before implementing**

### TCP-PWC Job Pre-fetch — TB Proposal Required
- [ ] **Implement job pre-fetch in `pwc_worker_service.py`** — worker requests next chunk from Zeus
  before finishing current one, so GPU never idles between chunks.
  - Discussed S159F (implemented for ZMQ), never ported to TCP-PWC
  - ~10 line change to worker main loop
  - Deploy to all 4 nodes after Zeus TCP worker path is fixed
  - **TB proposal required before implementing**

### S163 — Remove `free_all_blocks()` Race Condition
- [ ] **Implement S163 proposal** — remove `free_all_blocks()` from `_best_effort_gpu_cleanup()`
  TB proposal: approved. Staged validation: 500K → 1M → 2M seeds.
  Enables raising `seed_cap_amd` beyond 100K if 2M passes clean.

### run_status.json / Flask status endpoint
- [ ] Add `run_status.json` auto-push to public repo after each trial (enables Claude autonomous polling)
- [ ] OR host lightweight Flask status endpoint on Zeus (separate port from web_dashboard)

### Selfplay NN fix
- [ ] `inner_episode_trainer.py` still has hardcoded forbidden guard blocking NN in selfplay.
  Fix: remove forbidden check + add y-normalization to selfplay path (matching S121 fix, commit `6e5f76c`).

---

## 🟡 P3 — LOWER PRIORITY

### S110 Root Cleanup
- [ ] 884 files need organization in project root

### sklearn Warnings Step 5
- [ ] Fix sklearn warnings in Step 5 anti-overfit training

### Remove Dead CSV Writer
- [ ] Remove dead CSV writer from `coordinator.py`

### Regression Diagnostic
- [ ] `gate=True` regression diagnostic

### S103 Part 2 / Phase 9B.3
- [ ] Deferred

### ZMQ Job Pre-fetch
- [ ] Implement job pre-fetch in `zmq_sqlite_worker.py` — worker pulls 2 jobs at startup
  ~10 line change. Deploy to all 4 nodes.

### Chapter 13 — Autonomy Wire-up
- [ ] Wire `dispatch_selfplay()` into WATCHER post-Step-6
- [ ] Wire `dispatch_learning_loop()` into WATCHER

### PWC Transport Adapter v1
- [ ] TB-approved longer horizon item. `--pwc-transport ssh|tcp` flag.

---

## 🔵 INFRASTRUCTURE STATE (End of S163-KARG)

| Component | State |
|-----------|-------|
| Zeus HEAD | `c7f00a6` |
| All 3 rigs | `amdgpu-dkms 6.12.12` pinned ✅ |
| DPM pin service | `manual` on all 24 AMD GPUs at boot ✅ |
| Zeus nvidia persistence | Enabled, P2 state ✅ |
| TCP-PWC transport | Active, 26/26 GPUs ✅ |
| seed_cap_amd | 100K (stable) |
| n_parallel | 1 (n_parallel=2 scrapped) |
| Skip-dedup fix | Live — post-trial ~seconds not hours ✅ |
| Vectorized NPZ merge | Live ✅ |
| reset_seed_coverage.py | Deployed to Zeus ✅ |
| bidirectional_survivors_all.npz | 20,912 seeds accumulated |

---

*Updated: S163-KARG — 2026-04-19 — Team Alpha*
