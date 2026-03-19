# MASTER TODO LIST — S147
**Compiled:** 2026-03-02 (S114) | **Updated:** 2026-03-18 (S147)
**Status:** Q0/Q1/Q2 patches deployed and verified. Ready to resume production sweep Run 1.

---

## 🔴 P1 — HIGH PRIORITY (Next 1-3 Sessions)

### Production Sweep — Resume Run 1
- [ ] **Resume sweep Run 1** — Q0/Q1/Q2 patches applied. Clear halt, delete stale
  output files, relaunch:
  ```bash
  from agents.safety import clear_halt; clear_halt()
  rm -f optimal_window_config.json bidirectional_survivors.json
  bash sweep_run1.sh
  ```
  Study: `window_opt_1773792529` (Trial 1 incomplete — will restart from seed_start=610M)
  Expected per-trial time: ~2.5-3 hours with Q2 single strategy (was ~9-11 hours)
  Step 1 now has infinite timeout (Q1 fix)

- [ ] **Q0 zero-survivor branch live proof** — Monkeypatch `PersistentWorkerCoordinator` inside the module so `run_trial_persistent()` receives mocked results forcing `java_lcg_hybrid` to return `[]`. Exercises the actual skip-Pass-4 branch live without GPUs. TB recommendation.
- [ ] **Monitor hybrid forward prune rate** — With Q0 gate now active, log how often
  hybrid forward returns zero survivors. This data will inform whether further
  strategy reduction is needed for future runs.

- [ ] **After Run 1 completes** — commit NPZ accumulator:
  ```bash
  git add -f bidirectional_survivors_all.npz bidirectional_survivors_binary.npz
  git commit -m "data(S147): sweep run 1 complete"
  git push origin main && git push public main
  ```

### Chapter 13 — Autonomy Wire-up (Critical Path)
- [ ] **Wire `dispatch_selfplay()` into WATCHER** post-Step-6
- [ ] **Wire `dispatch_learning_loop()` into WATCHER**
- [ ] **Wire Chapter 13 orchestrator into WATCHER daemon**
- [ ] **Integration test: WATCHER → Chapter 13 → Selfplay full loop**

---

## 🟠 P2 — MEDIUM PRIORITY (Next 3-6 Sessions)

### Sweep Runs 2-4
- [ ] **sweep_run2.sh** — seeds 1,073,741,824 → 2,147,483,648
- [ ] **sweep_run3.sh** — seeds 2,147,483,648 → 3,221,225,472
- [ ] **sweep_run4.sh** — seeds 3,221,225,472 → 4,294,967,295

### Hybrid Scan Optimization (post Run 1 data)
- [ ] **Measure Q0 gate prune rate** from Run 1 logs — how often does hybrid
  forward = 0? This determines if further optimization is needed.
- [ ] **Consider Q2 refinement phase** — after constant-skip sweep identifies
  best windows, run all-5-strategy hybrid on top windows only (TB Q4 partial)

### Project Knowledge Updates Needed
- [ ] **Upload to Claude Project** (stale files):
  - `agents/watcher_agent.py` (3,084 lines — project has stale 1,863-line root copy)
  - `persistent_worker_coordinator.py` (not in project)
  - `window_optimizer_integration_final.py` (not in project)
  - `hybrid_strategy.py` (not in project)
  - Updated 5 chapter docs from S147

### Documentation
- [ ] **Update `THRESHOLD_GOVERNANCE.md`** — synthetic-era artifact
- [ ] **Update smoke test script `s145r1_smoke_tests.sh`** — checks for stale
  `bidirectional_survivors_all.json` (replaced by `.npz`)

### Neural Net & Selfplay
- [ ] **Remove NN forbidden guard in `inner_episode_trainer.py`** (lines 497-502)
- [ ] **Add y-normalization to selfplay inner trainer**
- [ ] **Chapter 14 Phase 8: Selfplay + Ch13 wiring**

### Optuna
- [ ] **Wire variable skip bidirectional count into Optuna scoring**
- [ ] **Node failure resilience**

---

## 🟢 P3 — LOW PRIORITY / DEFERRED

- [ ] **S110 root cleanup** — 884 files in project root
- [ ] **sklearn warnings fix in Step 5**
- [ ] **Remove CSV writer from coordinator.py**
- [ ] **Regression diagnostic gate** — set `gate=True`
- [ ] **S103 Part 2**
- [ ] **Phase 9B.3** (deferred)
- [ ] **Fix `soak_s130.sh`** — calls coordinator.py directly, cannot test PWC
- [ ] **TRSE Rules B and C** — revisit after sweep results
- [ ] **Bundle Factory Track 2** — fill 3 stub retrieval functions
- [ ] **`--force-step N` flag for WATCHER**
- [ ] **Persistent worker session drops on AMD rigs** — keepalive/TTL fix

---

## Architecture Invariants (S147 additions)

- **Q0/Q2 patches are interdependent** — Q0A references `_hybrid_strategies`
  introduced by Q2. Always deploy together. `apply_s147_q0_q1_q2.py` enforces this.
- **Step 1 timeout is now infinite** — `step_timeout_overrides={0:1, 1:0, 5:360}`.
  `0 * 60 = 0 → S145 guard fires → float('inf')`.
- **Hybrid scan uses balanced_hybrid strategy only** (Q2) for full-range scan.
  All-5-strategy hybrid reserved for refinement phase only (TB ruling).
- **Q0 is skip not prune** — when hybrid forward = 0, Pass 4 is skipped but
  constant-skip results are preserved. Trial completes normally.
- **run_trial_persistent is a standalone function** — uses `pwc.logger`, not
  `self.logger`. This must be respected in any future patches to that function.

---

*Updated S147 — 2026-03-18 — Team Alpha*
