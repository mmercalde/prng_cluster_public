# MASTER TODO LIST — S148
**Compiled:** 2026-03-02 (S114) | **Updated:** 2026-03-19 (S148)
**Sources:** TODO_MASTER_S145.md + S147 + S148 session history
**Status:** Threshold calibration complete. Production sweep Run 1 ready to relaunch.
Q0/Q1/Q2 patches deployed and verified (S147). Threshold defaults empirically grounded (S148).

---

## 🔴 P1 — HIGH PRIORITY (Next 1-3 Sessions)

### Production Sweep — Ready to Relaunch

- [ ] **Decide: raise window to 12 before Run 1?**  
  Calibration finding: window=12 + threshold=0.30 = near-zero noise (1 false survivor per 200k).  
  Current manifest: window_size=8.  
  If yes → update `agent_manifests/window_optimizer.json` default `window_size` to 12 and `distributed_config.json` window bounds default before launching.  
  If no → proceed with window=8 + threshold=0.30 (still 10x improvement over old defaults).

- [ ] **Launch production sweep Run 1** — 1,073,741,824 seeds, 50 trials, all 26 GPUs.
  All S147 patches (Q0/Q1/Q2) + S148 threshold calibration applied.
  ```bash
  bash sweep_run1.sh
  tail -f logs/sweep_run1_production.log
  ```

- [ ] **After Run 1 completes** — commit NPZ accumulator:
  ```bash
  git add -f bidirectional_survivors_all.npz bidirectional_survivors_binary.npz
  git commit -m "data(s148-r1): sweep run 1 — threshold=0.30 empirical calibration"
  git push origin main && git push public main
  ```

### Q0 Live Branch Verification (TB P2 item from S147)

- [ ] **Verify Q0 zero-survivor branch live** — In S147, hybrid forward found 4 survivors
  so the gate didn't fire. TB requested a monkeypatched live run forcing `java_lcg_hybrid`
  to return `[]` to exercise the actual skip-Pass-4 branch.
  Use `verify_q0_hybrid_gate.py` with mock patch — no GPUs needed.

### Chapter 13 — Autonomy Wire-up (CRITICAL PATH)

- [ ] **Wire `dispatch_selfplay()` into WATCHER** — Stub exists, not triggered post-Step-6.
- [ ] **Wire `dispatch_learning_loop()` into WATCHER** — Same gap.
- [ ] **Wire Chapter 13 orchestrator into WATCHER daemon**
- [ ] **Integration test: WATCHER → Chapter 13 → Selfplay full loop**

---

## 🟠 P2 — MEDIUM PRIORITY (Next 3-6 Sessions)

### Threshold Calibration — Remaining Items

- [ ] **Phase2_threshold calibration** — `phase2_threshold=0.50` (hybrid passes) not
  tested in S148 calibration. Remains synthetic-era default. Schedule for S149+.
- [ ] **Multi-PRNG threshold calibration** — S148 calibrated `java_lcg` only. Other PRNG
  families may need separate calibration if used in production.
- [ ] **Real draw validation** — Repeat calibration experiment with actual CA Daily 3 draws
  (not synthetic) to verify zero-noise threshold holds under real-world noise.

### Neural Net & Training

- [ ] **sklearn warnings fix**
- [ ] **XGBoost device mismatch warning fix**
- [ ] **Selfplay NN two-part fix:**
  - Remove NN forbidden guard in `inner_episode_trainer.py` (lines 497–502)
  - Add y-normalization to selfplay path (port from `train_single_trial.py` S121 fix)

### TRSE Integration

- [ ] **TRSE Rules B and C — revisit after sweep results**
- [ ] **Per-segment pipeline runs**

### Bundle Factory

- [ ] **Fill 3 stub retrieval functions in `bundle_factory.py`**

### Optuna / Window Optimizer

- [ ] **Wire variable skip bidirectional count into Optuna scoring**
- [ ] **Node failure resilience**

### Model Persistence

- [ ] **Feature names into `best_model.meta.json` at training time**

---

## 🟢 P3 — LOW PRIORITY / DEFERRED

### Code Cleanup

- [ ] **S110 root cleanup** — 884 files in project root
- [ ] **Remove CSV writer from coordinator.py**
- [ ] **Regression diagnostic gate** — Set `gate=True`
- [ ] **Fix soak_s130.sh** — calls coordinator.py directly, cannot test PWC
- [ ] **Update smoke test script `s145r1_smoke_tests.sh`** — still checks for old `.json` survivor file

### Documentation

- [ ] **Update `THRESHOLD_GOVERNANCE.md` survivor band** — 1K–10K is synthetic-era.
  Steps 2–6 handle millions of survivors. Document needs updating. (Discovery S145)
- [ ] **Upload updated docs to Claude Project**
- [ ] **sweep_run2.sh, sweep_run3.sh, sweep_run4.sh** — build progressive sweep scripts

### Research / Experimental

- [ ] **S103 Part 2: per-seed match rates**
- [ ] **Phase 9B.3: deferred selfplay component**
- [ ] **Walk-forward simulation** — architecturally feasible via date-sliced history files
- [ ] **Linear complexity Tier 1B**
- [ ] **Binary matrix rank feature**
- [ ] **PA Pick 3 clean sweep** — `pa_pick3_scraper.py` Rev 1.1 ready

### Post-Sweep Analysis (After Run 1 Completes)

- [ ] **Yield decay analysis** — survivors/seeds per session
- [ ] **Seed distribution analysis** — clustering in specific ranges?
- [ ] **Survivor quality vs seed range** — avg score across sessions
- [ ] **Practical sufficiency threshold** — when to stop expanding frontier

---

## ✅ RECENTLY COMPLETED (Reference)

| Item | Session | Commit |
|------|---------|--------|
| **S148 threshold calibration — 5 files patched** | **S148** | **pending** |
| **Manifest informational fields updated** | **S148** | **pending** |
| **THRESHOLD_GOVERNANCE.md + baselines/ created** | **S148** | **pending** |
| Q0 hybrid forward gate (skip-not-prune) | S147 | `667ece5` |
| Q1 Step 1 infinite timeout (WATCHER) | S147 | `667ece5` |
| Q2 Single balanced_hybrid strategy for full-range scan | S147 | `667ece5` |
| Q0/Q1/Q2 live verified 19/19 on Zeus | S147 | `bf2549b` |
| Dashboard trial_stats persistence fix | S147 | `60393a0` |
| Dashboard GPU display → seeds/GPU metric | S147 | `60393a0` |
| 5 chapter docs updated (S146 PWC invariants) | S147 | `c797045` |
| STEP1_EXECUTION_FLOW_AND_PRUNING_S147.md created | S147 | `0e01dcb` |
| sweep_preprod.sh validated end-to-end | S146 | `8ac4047` |
| PWC all 4 sieve passes operational | S146 | `ec3cd1f` |
| 313 bidirectional survivors in S146 | S146 | — |
| 666 total in NPZ accumulator | S146 | — |
| Empirical threshold calibration experiments run | S148 | `e051ee2` |

---

## 🏗️ ARCHITECTURE INVARIANTS (Never Change Without Team Beta)

- Step order: 1→2→3→4→5→6 (static)
- Feature schema: hash-validated Step 5→6
- Authority separation: Chapter 13 decides, WATCHER executes
- GPU isolation: Parent never initializes CUDA before NN subprocess spawn
- Manifest param governance: Every new CLI param in `default_params`
- Never restore from backup — always fix forward
- Comment out dead code, never delete
- Always fetch live file content via SSH before patching
- Zeus GPU compute mode: DEFAULT — enforced via `/etc/rc.local` (S125b)
- GPU seed caps: Phase B concurrent ceiling × 0.85
- Persistent workers: off by default — opt-in only
- TRSE skip_on_fail: Step 0 failures must never halt pipeline
- NPZ survivors: `bidirectional_survivors_binary.npz` must always remain git-tracked
- NPZ accumulator: `bidirectional_survivors_all.npz` must always remain git-tracked
- Warm-start: driven from `trial_history_context` only — no dataset-specific hardcodes
- Pipeline launch: always use `nohup`, never tmux
- Pruning: `enable_pruning` gates BOTH primary block AND ThresholdPruner (S145)
- Manifest JSON: never add `//` comments — JSON does not support comments
- **[S148] Threshold invariant: baseline ∈ [search_min, search_max] must be preserved**
- **[S148] Threshold floor ≥ 0.30 (window=8) / ≥ 0.20 (window=12) — empirically grounded**
- **[S148] Threshold ceiling 0.75 — known signal safe to this value**

---

*Generated S101 — 2026-02-20 | Updated S148 — 2026-03-19*
