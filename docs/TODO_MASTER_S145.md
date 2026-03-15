# MASTER TODO LIST — S145
**Compiled:** 2026-03-02 (S114) | **Updated:** 2026-03-15 (S145)
**Sources:** TODO_MASTER_S144.md + S145 session history
**Status:** S145-R1 progressive sweep framework validated. NPZ accumulator
operational. Ready for production sweep Run 1.

---

## 🔴 P1 — HIGH PRIORITY (Next 1-3 Sessions)

### Production Sweep — Ready to Launch
- [ ] **Recalculate production sweep timing** — `test_both_modes=true` runs
  4 sieves per trial (fwd constant, fwd variable, rev constant, rev variable),
  not 2. At 1.07B seeds: ~68 min/trial not ~34 min/trial. Update timeout and
  scheduling before launch.
- [ ] **Update smoke test script `s145r1_smoke_tests.sh`** — Checks 1-3 look
  for `bidirectional_survivors_all.json` (replaced by `.npz`). Update before
  next smoke test run.
- [ ] **Launch production sweep Run 1** — Cold start, 1,073,741,824 seeds,
  50 trials, all 26 GPUs, pruning enabled, `test_both_modes=true`.
  ```bash
  nohup bash -c 'PYTHONPATH=. python3 agents/watcher_agent.py \
  --run-pipeline --start-step 0 --end-step 1 \
  > logs/sweep_run1_production.log 2>&1' &
  ```
  After Run 1: note study name from log for Runs 2-4 resume.
- [ ] **After Run 1 completes** — commit NPZ accumulator to both remotes:
  ```bash
  git add -f bidirectional_survivors_all.npz bidirectional_survivors_binary.npz
  git commit -m "data(s145-r1): sweep run 1 — seeds 0→1,073,741,824"
  git push origin main && git push public main
  ```

### Chapter 13 — Autonomy Wire-up (NEXT PRIORITY)
- [ ] **Wire `dispatch_selfplay()` into WATCHER** — Stub exists, not triggered
  post-Step-6. ~180 lines.
- [ ] **Wire `dispatch_learning_loop()` into WATCHER** — Same gap.
- [ ] **Wire Chapter 13 orchestrator into WATCHER daemon**
- [ ] **Integration test: WATCHER → Chapter 13 → Selfplay full loop**

### Chapter 14 — Diagnostics
- [ ] **Chapter 14 Phase 8: Selfplay + Ch13 Wiring**
- [ ] **Chapter 14 Phase 9: First Diagnostic Investigation**

### Live Autonomous Operation
- [ ] **Activate `draw_ingestion_daemon.py` with real draw data**
- [ ] **Wire `daily3_scraper.py` into WATCHER scheduler**
- [ ] **24-hour synthetic soak test**

### Blocking Bugs
- [ ] **WATCHER Step 1 timeout** — 900 min set, verify sufficient for 50
  trials × ~68 min = ~57 hours. May need further increase.
- [ ] **Persistent worker session drops on AMD rigs**
- [ ] **WATCHER validation threshold** — `>=100` causes false ESCALATE
- [ ] **`--force-step N` flag for WATCHER**

---

## 🟠 P2 — MEDIUM PRIORITY (Next 3-6 Sessions)

### Documentation
- [ ] **Update `THRESHOLD_GOVERNANCE.md`** — 1K–10K survivor band is
  synthetic-era artifact. Steps 2–6 handle millions of survivors (Step 2
  samples 50K, Step 3 chunks in batches of 1K). Document needs updating
  to reflect real-data scale. (Discovery S145)

### Neural Net & Training
- [ ] **sklearn warnings fix**
- [ ] **XGBoost device mismatch warning fix**
- [ ] **NN Y-label normalization** — `train_single_trial.py` line 499
- [ ] **Phase 3B: Tree parallel workers** — Team Beta review required

### TRSE Integration
- [ ] **TRSE Rules B and C — revisit after sweep results**
- [ ] **Per-segment pipeline runs**

### Bundle Factory Track 2
- [ ] **Fill 3 stub retrieval functions in `bundle_factory.py`**

### Optuna / Window Optimizer
- [ ] **Wire variable skip bidirectional count into Optuna scoring**
- [ ] **Node failure resilience**
- [ ] **k_folds runtime clamp** — Team Beta review required

### Model Persistence
- [ ] **Feature names into `best_model.meta.json` at training time**

---

## 🟢 P3 — LOW PRIORITY / DEFERRED

### Code Cleanup
- [ ] **S110 root cleanup** — 884 files in project root
- [ ] **Dead code audit: MultiModelTrainer inline path**
- [ ] **Remove CSV writer from coordinator.py**
- [ ] **Regression diagnostic gate** — Set `gate=True`
- [ ] **Remove 27 stale project files from Claude Project**

### Documentation Sync
- [ ] **Write chapter docs patch script**
- [ ] **Upload updated docs to Claude Project**

### Research / Experimental
- [ ] **S103 Part 2: per-seed match rates**
- [ ] **Phase 9B.3: deferred selfplay component**
- [ ] **Phase 3A vmap k_folds scaling**
- [ ] **Web dashboard refactor**
- [ ] **PA Pick 3 follow-up experiment** — Clean sweep on PA data, no
  warm-start, session-split midday vs evening independently.
  Scraper: `pa_pick3_scraper.py` Rev 1.1.
  Requires warm-start fix ✅ (completed S144).

### Post-Sweep Analysis (After Production Sweep Completes)
- [ ] **Yield decay analysis** — Plot survivors/seeds_searched per session.
  If yield drops toward zero in sessions 3-4, supports hypothesis that
  signal is concentrated in lower seed IDs.
- [ ] **Seed distribution analysis** — Plot survivor seed values. Clustering
  in specific ranges = evidence of constrained initialization source.
- [ ] **Survivor quality vs seed range** — Compare avg per-seed `score`
  across sessions.
- [ ] **Practical sufficiency threshold** — Yield in sessions 3-4 < 5% of
  session 1 AND quality not materially better → supports sufficiency claim.
  Until then: progressive frontier expansion only.

### Operational
- [ ] **Results/ FIFO cleanup automation** — WATCHER post-run hook
- [ ] **`--force-step` flag for WATCHER**

---

## ✅ RECENTLY COMPLETED (Reference)

| Item | Session | Commit |
|------|---------|--------|
| **S145-R1 NPZ accumulator validated (smoke test passed)** | **S145** | **`ad5ab8d`** |
| **S145-R1 progressive sweep framework deployed** | **S145** | **`3940517`** |
| **Primary prune block gated on enable_pruning** | **S145** | **`ad5ab8d`** |
| **enable_pruning scoped into run_bidirectional_test()** | **S145** | **`ad5ab8d`** |
| **best_result None + KeyError guards (all-pruned case)** | **S145** | **`ad5ab8d`** |
| **NPZ→NPZ merge replaces JSON accumulator** | **S145** | **`ad5ab8d`** |
| **WATCHER fresh-study invariant conditionalized on study_name** | **S145** | **`3940517`** |
| **Manifest: max_seeds=1.07B, window_trials=50, timeout=900** | **S145** | **`3940517`** |
| Remove W8_O43 warm-start hardcode | S144 | `58aedb6` |
| PA Pick 3 scraper Rev 1.1 (Wild Ball handling) | S143 | — |
| PA sieve validation harness (CPU, 3-tier) | S143 | — |
| S142-C: NP2 canonical backfill | S142 | `51aed27` |
| TRSE Rule A applied in n_parallel partition worker | S139B | `25cc2de` |
| Window size max 500→50 (3 files) | S139 | `7d035c6` |
| NPZ pipe deadlock fix | S138 | `3624e3c` |
| 167-trial Optuna run — best=1,384,186 W2_O14_evening | S138 | — |
| Persistent workers (+150% throughput, 2,082,140 sps) | S130 | ✅ |
| Soak test 17/17 pass | S130 | ✅ |
| First clean real-data Steps 1–6 run | S120 | ✅ |
| 85 bidirectional survivors (W8_O43, real data) | S120 | ✅ |
| All 26 GPUs operational | S125b | ✅ |

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
- NPZ survivors: `bidirectional_survivors_binary.npz` must always remain
  git-tracked — explicit commit after every Step 1 run
- NPZ accumulator: `bidirectional_survivors_all.npz` must always remain
  git-tracked — explicit commit after every Step 1 run (S145)
- Warm-start: driven from `trial_history_context` only — no dataset-specific
  hardcodes in optimizer code (enforced S144)
- Pipeline launch: always use `nohup`, never tmux
- Pruning: `enable_pruning` gates BOTH primary block in
  `window_optimizer_integration_final.py` AND ThresholdPruner in
  `window_optimizer_bayesian.py` (enforced S145)
- Manifest JSON: never add `//` comments — JSON does not support comments

---

*Generated S101 — 2026-02-20 | Updated S145 — 2026-03-15*
