# S149 Chat Prompt — PRNG Distributed Analysis System

## Session Priority

**P1 — Start here:**
1. Deploy S149-A ROCR isolation fix + single-rig smoke test
2. If smoke passes: preprod soak at worker_pool_size=8
3. If soak passes: commit + relaunch production sweep with 8 workers/rig
4. Monitor Run 1 to completion (if still running)

---

## Current System State (end of S148)

**Both remotes synced at `0550842`.**

### Run 1 status
- Launched S148: W12 / T0.30 / seed_start=610M / 1.07B seed range
- All 26 GPUs active, WATCHER infinite timeout (Q1)
- AMD rigs contributing ~3% throughput (ROCR bug — fix ready)
- Zeus carrying ~90% of sweep on 2× RTX 3080 Ti
- Dashboard: `http://45.32.131.224:5002`
- Resume if crashed: `bash sweep_run1.sh --resume`

### S149-A ROCR isolation fix — TB APPROVED, NOT YET DEPLOYED

**Root cause identified S148:** `_spawn_worker()` sets `HIP_VISIBLE_DEVICES={gpu_id}`
and `CUDA_VISIBLE_DEVICES={gpu_id}` but NOT `ROCR_VISIBLE_DEVICES={gpu_id}`.
`sieve_gpu_worker.py` explicitly documents `Device(0)` assumes ROCR isolation is in
place — but the spawner never provided it. `HIP_VISIBLE_DEVICES` is unreliable for
GPUs 4–7 on ROCm, so workers above GPU3 collide → crash → `worker_pool_size=4` ceiling.

**TB ruling:** Approve one-line fix. Keep stagger 4.0s. Validate via single-rig
smoke + preprod before production. Pipelining deferred to S149-B.

**Fix:** Add `f"ROCR_VISIBLE_DEVICES={gpu_id}"` to `_spawn_worker()` in
`persistent_worker_coordinator.py`.

**Patch scripts ready (download to Zeus):**
- `apply_s149a_rocr_isolation.py` — one-line patch, exit-code correct
- `verify_s149a_rocr_fix.py` — 19/19 post-fix harness

**Harness results (S148):**
- Pre-fix detection harness: 15/15 (confirms bug exists)
- Post-fix verify harness: 19/19 (confirms fix correct, blast radius zero)
- TB live repo verification: concurs — fix correct, minimal, right

---

## S149-A Deploy Sequence

```bash
# 1. SCP patch scripts
scp ~/Downloads/apply_s149a_rocr_isolation.py rzeus:~/distributed_prng_analysis/
scp ~/Downloads/verify_s149a_rocr_fix.py rzeus:~/distributed_prng_analysis/

# 2. Apply and verify source
python3 apply_s149a_rocr_isolation.py --dry-run   # no backup created
python3 apply_s149a_rocr_isolation.py             # exit 0
python3 verify_s149a_rocr_fix.py                  # 19/19
```

### Single-rig smoke test (rrig6600 only, worker_pool_size=8)

TB acceptance criteria:
- All 8 workers heartbeat within WORKER_HEARTBEAT_TIMEOUT_S
- No worker exits or quarantine during startup
- rocm-smi --showuse shows GPU% > 0 on all 8 GPUs during job execution
- Jobs complete and return valid survivor counts
- No HSA init errors in log

```bash
ssh rrig6600 "rocm-smi --showuse 2>/dev/null"   # before: confirm all 0%
# run smoke with worker_pool_size=8 on rrig6600 only
ssh rrig6600 "rocm-smi --showuse 2>/dev/null"   # after: expect all >0%
```

### Short soak (all rigs, worker_pool_size=8)

```bash
bash sweep_preprod.sh   # 50M seeds, 5 trials
# Monitor: expect ~6M s/s per rig vs ~1,050 s/s before fix
```

### Commit after soak passes

```bash
git add persistent_worker_coordinator.py \
      apply_s149a_rocr_isolation.py \
      verify_s149a_rocr_fix.py \
      test_s149_rocr_isolation_harness.py \
      docs/TB_RULING_REQUEST_ROCR_ISOLATION_S149.md \
      docs/SESSION_CHANGELOG_S149.md \
      docs/TODO_MASTER_S149.md \
      docs/S150_CHAT_PROMPT.md
git commit -m "fix(s149a): ROCR_VISIBLE_DEVICES isolation — enable 8 workers/rig"
git push origin main && git push public main
```

---

## Infrastructure

**Zeus:** `rzeus`, `~/distributed_prng_analysis/`, `~/venvs/torch/bin/activate`
**Rigs:** `rrig6600` (192.168.3.120), `rrig6600b` (192.168.3.154), `rrig6600c` (192.168.3.162)
**Dashboard:** `45.32.131.224:5002`

### Kill all workers
```bash
ssh rzeus "pkill -f 'watcher_agent.py'; pkill -f 'window_optimizer.py'"
ssh rrig6600 "pkill -f sieve_gpu_worker 2>/dev/null"
ssh rrig6600b "pkill -f sieve_gpu_worker 2>/dev/null"
ssh rrig6600c "pkill -f sieve_gpu_worker 2>/dev/null"
```

### After Run 1 completes — commit NPZ
```bash
git add -f bidirectional_survivors_all.npz bidirectional_survivors_binary.npz
git commit -m "data(s149-r1): sweep run 1 complete — W12/T0.30 empirical governance"
git push origin main && git push public main
```

---

## P1 TODO (S149 order)

1. Deploy S149-A ROCR fix → single-rig smoke → preprod soak → commit
2. Monitor / triage Run 1 (if still running)
3. After Run 1 completes: commit NPZ, record survivor count
4. Q0 zero-survivor branch live verification (TB P2 item from S147)
5. Chapter 13 wire-up — dispatch_selfplay() + dispatch_learning_loop()

## S149-B (after ROCR validated)
- IPC pipelining / pre-fetch (TB: next optimization after ROCR, not distant backlog)
- Reduce spawn stagger 4.0s → 1.0s (only after 8-worker stability proven)

## Backlog
- phase2_threshold calibration (hybrid passes, currently 0.50 — synthetic-era)
- Selfplay NN two-part fix (inner_episode_trainer.py)
- S110 root cleanup (884 files)
- sklearn warnings in Step 5
- Remove CSV writer from coordinator.py
- Regression diagnostics gate → True
- S103 Part 2
- Phase 9B.3 (deferred)
- sweep_run2.sh, sweep_run3.sh, sweep_run4.sh

---

## Key S148 findings for reference

### Threshold calibration (deployed)
W12 + T0.30 = ~5 false fwd survivors/trial vs ~1.36M at old W8/T0.25.
First production sweep under empirical threshold governance.

### ROCR diagnosis (fix ready, not yet deployed)
- worker_pool_size=4 ceiling was not a stability limit — it was a GPU isolation bug
- Fix is one line in _spawn_worker() — TB approved, live-repo verified
- Expected outcome: rigs go from ~1,050 s/s to ~6M s/s each

### IPC starvation (deferred to S149-B)
Even with 8 workers, each worker dispatches one job at a time. GPU utilization
per worker ~1% due to SSH round-trip latency. ROCR fix multiplies active GPUs 8×
first — measure real baseline before attacking pipelining.
