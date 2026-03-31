# MASTER TODO LIST — S159
**Compiled:** 2026-03-02 (S114) | **Updated:** 2026-03-30 (S159)
**Status:** S159 — rrig6600 ZMQ isolation validation in progress. S159 slim
persistence patch (TB-approved) ready to deploy. 3-rig validation pending.

---

## 🔴 P1 — HIGH PRIORITY (Active / Next Session)

### ZMQ Validation Ladder (current focus)
- [ ] **Confirm rrig6600 2-rig run result** — check `logs/zmq_systemd_v1.log`
  tail for clean completion or error signature before advancing:
  ```bash
  ssh rzeus "tail -80 ~/distributed_prng_analysis/logs/zmq_systemd_v1.log"
  ```
- [ ] **Deploy S159 slim persistence patch** — files ready for download:
  - `apply_s159_slim_result_persistence.py` (patch script, for audit)
  - `zmq_sqlite_coordinator.py` (patched file, deploy directly)
  ```bash
  scp ~/Downloads/zmq_sqlite_coordinator.py rzeus:~/distributed_prng_analysis/
  ```
  TB ruling: Approved. Backward + forward compliant. Primary bottleneck addressed.

- [ ] **VACUUM or rotate zmq_job_queue.db after S159 deploy** — existing DB has
  old blob rows; SQLite does not shrink automatically on DELETE. Either:
  ```bash
  # Option A — vacuum in place (safe, slow on 7.5GB)
  ssh rzeus "cd ~/distributed_prng_analysis && sqlite3 zmq_job_queue.db 'VACUUM;'"
  # Option B — rotate (faster, clean slate)
  ssh rzeus "cd ~/distributed_prng_analysis && mv zmq_job_queue.db zmq_job_queue.db.pre_s159 && echo rotated"
  ```
  Rotation preferred — old DB is corrupt/bloated and the current run will have
  already completed or been restarted.

- [ ] **Enable linger on rrig6600b and rrig6600c** (requires interactive SSH):
  ```bash
  ssh rrig6600b   # then: sudo loginctl enable-linger michael
  ssh rrig6600c   # then: sudo loginctl enable-linger michael
  ```
  rrig6600: already enabled ✅

- [ ] **3-rig validation run** (Zeus + rrig6600 + rrig6600c) — after S159 patch
  deployed and linger enabled on rrig6600c:
  ```bash
  # Reset coverage first
  ssh rzeus "cd ~/distributed_prng_analysis && python3 -c \"
  import sqlite3
  conn = sqlite3.connect('prng_analysis.db')
  conn.execute('DELETE FROM exhaustive_progress WHERE prng_type=?', ('java_lcg',))
  conn.commit(); conn.close(); print('reset')
  \""
  # Kill everything
  ssh rzeus "touch /tmp/agent_halt && pkill -9 -f 'watcher_agent|window_optimizer|zmq_sqlite_worker' 2>/dev/null"
  ssh rrig6600  "for i in {0..7}; do systemctl --user stop zmq-worker-gpu\$i 2>/dev/null; done"
  ssh rrig6600c "for i in {0..7}; do systemctl --user stop zmq-worker-gpu\$i 2>/dev/null; done"
  # Launch
  ssh rzeus "cd ~/distributed_prng_analysis && \
    rm -f /tmp/agent_halt daemon_state.json optimal_window_config.json && \
    source ~/venvs/torch/bin/activate && \
    setsid bash -c 'PYTHONPATH=. python3 agents/watcher_agent.py \
    --run-pipeline --start-step 1 --end-step 1 --force-step 1 \
    >> logs/zmq_3rig_v1.log 2>&1' &"
  ```

- [ ] **4-node full run (all 26 GPUs)** — after 3-rig validates clean:
  Add rrig6600b (enable linger first). Log to `zmq_4node_v1.log`.

- [ ] **Increase seed_cap_amd 2M → 10M** — after 3-rig stable. Low GPU
  utilization (~0% between chunks) is by design at 2M (GPU burst ~50-100ms
  vs ZMQ round-trip ~600ms). 10M chunks will saturate GPUs.
  Edit `distributed_config.json` or pass `--seed-cap-amd 10000000`.

- [ ] **Set manifest default_params.use_zmq_sqlite** — currently `true`
  (ZMQ active default) but `param_docs.use_zmq_sqlite.default` says `false`.
  Inconsistency flagged S159. Decision deferred until after 3-rig soak.
  TB recommendation: keep `false` as default until slim persistence path is
  fully soaked; flip to `true` after successful 4-node run.
  File: `agent_manifests/window_optimizer.json`
  Fields to sync:
  - `default_params.use_zmq_sqlite` (currently `true`)
  - `param_docs.use_zmq_sqlite.default` (currently `false`)

### S159 Follow-up Hardening (TB caveats — not blockers)
- [ ] **Harden cleanup: don't delete .npz files if final output write failed**
  Current code calls `self.db.cleanup(run_id)` even if the `json.dump()` to
  `output_file` raises. This deletes chunk payload .npz files, making postmortem
  recovery impossible. Fix: move `cleanup()` call inside a `try/finally` only
  after confirming output file written successfully, or skip cleanup on write
  failure and log a recovery path.
  File: `zmq_sqlite_coordinator.py` → `run_sieve_pass()` finalization block.

- [ ] **Confirm skip_sequences / strategy_ids empty-list behavior is acceptable**
  S159 patch drops these from aggregated ZMQ output (they were never consumed
  by any pipeline step, but TB flagged as semantic caveat). Verify no diagnostic
  or debug tooling expects populated skip_sequences / strategy_ids in the
  ZMQ result path. If any tooling found: add pass-through in get_results().

### S159 Changelog
- [ ] **Write SESSION_CHANGELOG_20260330_S159.md** and dual-push.

---

## 🟠 P2 — MEDIUM PRIORITY (Next 3-6 Sessions)

### Production Sweep
- [ ] **Resume sweep Run 1** — after ZMQ 4-node soak validated. Clear halt,
  delete stale output files, relaunch with seed_start=0 reset.
  Study: `window_opt_1773792529`

### Chapter 13 — Autonomy Wire-up
- [ ] **Wire `dispatch_selfplay()` into WATCHER** post-Step-6
- [ ] **Wire `dispatch_learning_loop()` into WATCHER**
- [ ] **Wire Chapter 13 orchestrator into WATCHER daemon**
- [ ] **Integration test: WATCHER → Chapter 13 → Selfplay full loop**

### Selfplay NN Fix
- [ ] **Remove forbidden guard + add y-normalization to `inner_episode_trainer.py`**
  selfplay path. (S121 fix applied to `train_single_trial.py` but not selfplay.)

---

## 🟡 P3 — DEFERRED

- [ ] **S110 root cleanup** — 884 stray files in project root. Low urgency.
- [ ] **sklearn warnings in Step 5** — harmless deprecation warnings.
- [ ] **Remove CSV writer from coordinator.py** — dead weight.
- [ ] **Regression diagnostic gate=True**
- [ ] **S103 Part 2**
- [ ] **Phase 9B.3**
- [ ] **k_folds runtime clamp** — `val_fold_size < 3000` edge case. TB review needed.
- [ ] **Upload stale files to Claude Project** — `agents/watcher_agent.py`,
  `persistent_worker_coordinator.py`, `window_optimizer_integration_final.py`,
  `hybrid_strategy.py`, updated chapter docs.

---

## Architecture Invariants (never break)
- Zeus localhost semaphore = 2
- `bidirectional_survivors_binary.npz` always git-tracked; commit after every Step 1 run
- `watcher_policies.json` version-controlled
- Dual-push every commit: `git push origin main && git push public main`
- `use-zmq-sqlite` default in manifest = conservative (false) until 4-node soak
- ZMQ validation ladder: 1-rig ✅ → 2-rig (validating) → 3-rig → 4-node
- TB approval required before architectural changes
