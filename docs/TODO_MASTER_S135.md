# MASTER TODO LIST — S135
**Compiled:** 2026-03-02 (S114) | **Updated:** 2026-03-10 (S135)
**Sources:** TODO_MASTER_S132.md + S132–S135 session history
**Status:** Persistent worker engine fully operational. 14/14 chunks clean. Both remotes at `6ac3312`.

---

## 🔴 P1 — HIGH PRIORITY (Next 1-3 Sessions)

### Run More Trials — Critical Path to NN
- [ ] **Run 200-trial Step 1 to grow survivor pool** — 85 survivors insufficient for NN. Need 500+. Resume `window_opt_1772507547.db` (21 trials) with 200 total trials. Use `--resume-study --study-name window_opt_1772507547 --use-persistent-workers`. Persistent workers now 14/14 clean.
- [ ] **k_folds runtime clamp** — When `val_fold_size < 3000`, clamp to `max(3, n_train // 3000)`. Needs Team Beta review. (S101 deferred)

### Chapter 13 + Selfplay Wire-up (Key Autonomy Gap)
- [ ] **Wire `dispatch_selfplay()` into WATCHER** — Stub exists in `watcher_agent.py`, not triggered post-Step-6. ~180 lines.
- [ ] **Wire `dispatch_learning_loop()` into WATCHER** — Same gap.
- [ ] **Wire Chapter 13 orchestrator into WATCHER daemon** — `chapter_13_orchestrator.py` exists but WATCHER cannot dispatch it post-Step-6.
- [ ] **Integration test: WATCHER → Chapter 13 → Selfplay full loop** — Selfplay A1–A5 validated in isolation; end-to-end with WATCHER trigger never verified.

### Optuna / Window Optimizer
- [ ] **Node failure resilience** — Single rig dropout can crash entire Optuna study. Need graceful degradation when a rig goes offline mid-study.

---

## 🟠 P2 — MEDIUM PRIORITY (Next 3-6 Sessions)

### Neural Net & Training
- [ ] **sklearn warnings fix** — `UserWarning: X does not have valid feature names, but LGBMRegressor was fitted with feature names`. Add `warnings.filterwarnings` suppression in `meta_prediction_optimizer_anti_overfit.py`. (First seen S109, recurring)
- [ ] **XGBoost device mismatch warning fix** — `Falling back to prediction using DMatrix due to mismatched devices`. Add `predict()` method to `XGBoostWrapper` pre-converting numpy to DMatrix. `fix_xgboost_device.sh` written S97 — verify status on Zeus.
- [ ] **NN Y-label normalization** — `train_single_trial.py` line 499. Add `y_mean/y_std` normalization, save sidecar for Step 6 inverse-transform. Agreed S112 to fix after run.
- [ ] **Phase 3B: Tree parallel workers** — 2 tree trials simultaneously with pinned `CUDA_VISIBLE_DEVICES`. Trees = 93% of wall clock (~24 min). Target: ~12 min. Requires Team Beta review.

### Persistent Worker Enhancements
- [ ] **Z10×Z10×Z10 kernel missing from `sieve_gpu_worker.py`** — Digit feature scoring unavailable via persistent worker path. TB proposal needed before implementing. (S133 discovery)
- [ ] **rrig6600c throughput deficit** — i5-8400T CPU ~50% throughput vs other rigs. Consider excluding from Step 1 partition or weighting down.

### TRSE Integration
- [ ] **TRSE v1 production wire-up** — `trse_step0.py` exists, smoke test passed (S123). Full 0→6 smoke test verified (S123) but production run with real data not yet done.
- [ ] **Per-segment pipeline runs** — Run Steps 1–6 separately per TRSE regime segment. Depends on TRSE production wire-up.

### Live Autonomous Operation
- [ ] **Activate `draw_ingestion_daemon.py` with real draw data** — Daemon coded (22KB), tested with synthetic draws, never activated with live data. Ignition switch for 24/7 autonomous operation.
- [ ] **Wire `daily3_scraper.py` into WATCHER scheduler** — WATCHER needs a scheduler thread invoking it every N minutes → triggering Chapter 13 on new draw detection.
- [ ] **24-hour synthetic soak test** — Run `synthetic_draw_injector.py` feeding draws every 60s with full Chapter 13 loop.

### Bundle Factory Track 2
- [ ] **Fill 3 stub retrieval functions in `bundle_factory.py`** — `_retrieve_recent_outcomes()`, `_retrieve_trend_summary()`, `_retrieve_open_incidents()`. Currently return empty lists.

### Chapter 14 Remaining
- [ ] **Chapter 14 Phase 8: Selfplay + Ch13 Wiring** — Episode diagnostics, trend detection, root cause analysis. Phase 8A done S83, Phase 8B blocked on per_survivor_attribution.
- [ ] **Chapter 14 Phase 9: First Diagnostic Investigation** — Real `--compare-models --enable-diagnostics` run on real data.

### Model Persistence
- [ ] **`--save-all-models` flag** — Save all 4 Step 5 model types for post-hoc AI analysis. Currently only winner saved.
- [ ] **Feature names into `best_model.meta.json` at training time** — Proper fix vs current fallback extraction from trained LightGBM model. (S86 partial fix in place)

---

## 🟢 P3 — LOW PRIORITY / DEFERRED

### Code Cleanup
- [ ] **S110 root cleanup** — 884 files need cleanup. 58 stray docs already moved (S109).
- [ ] **Dead code audit: MultiModelTrainer inline path** — Comment out. Unreachable in subprocess mode.
- [ ] **Remove CSV writer from coordinator.py** — Dead weight.
- [ ] **Regression diagnostic gate** — Set `gate=True`.
- [ ] **Remove 27 stale project files from Claude Project** — Identified S85/S86.

### Documentation Sync
- [ ] **Write chapter docs patch script** — Update ALL stale chapter docs simultaneously. Stale: Ch12 (missing dispatch module, bundle factory, lifecycle, new CLI args), Ch13 (Phase 7 complete, selfplay wiring gap), Ch14 (Phases 1–4 complete), COMPLETE_OPERATING_GUIDE_v2_0 (missing S85+), README (wrong GPU counts). Three locations: Zeus `docs/`, ser8 `~/Downloads/`, Claude Project.
- [ ] **Upload updated docs to Claude Project** — After patch script runs, upload manually.

### Research / Experimental
- [ ] **S103 Part 2: per-seed match rates** — Continuation work.
- [ ] **Phase 9B.3: deferred selfplay component** — After 9B.2 validation.
- [ ] **Phase 3A vmap k_folds scaling** — Increase `--k-folds` to 10/20/30. Deferred S99.
- [ ] **Web dashboard refactor** — Chapter 14 visualization. Long-term.

### Operational
- [ ] **Always launch pipeline in tmux** — `tmux new-session -s pipeline` before WATCHER.
- [ ] **Results/ FIFO cleanup automation** — WATCHER post-run hook.

---

## ✅ RECENTLY COMPLETED (Reference)

| Item | Session | Status |
|------|---------|--------|
| Persistent worker engine — 9 bugs fixed, 14/14 chunks clean | S135 | ✅ `b5b6532` |
| Per-worker dispatch lock (chunk collision fix) | S135 | ✅ |
| SSH binary pipe + banner drain | S135 | ✅ |
| NPZ skip_range string parsing | S135 | ✅ |
| Persistent worker engine redesign (standalone module) | S134 | ✅ |
| rig-6600c crash fix — hybrid kernel signature | S133-B | ✅ |
| Semaphore regression fix | S133-A | ✅ |
| TODO audit — 5 stale items cleared | S132 | ✅ `5525f35` |
| Variable skip bidi count wired into Optuna scoring | S124 | ✅ `c17eaa5` |
| sklearn KFold n_splits guard | S124 | ✅ `c17eaa5` |
| WATCHER validation threshold ≥100 → ≥50 | S122 | ✅ `1498e3f` |
| Gate 1 close — fallback logging + dead-pipe fix | S131 | ✅ `1d85da4` |
| Seed cap patch — constructor defaults, wiring, manifest v1.5.0 | S131 | ✅ `d168f83` |
| apply_caps.py deployed to Zeus | S131 | ✅ `50a4146` |
| Persistent workers (+150% throughput, 2,082,140 sps) | S130 | ✅ |
| Soak test 17/17 pass | S130 | ✅ |
| Manifest bug fix (study_name, enable_pruning, n_parallel) | S127 | ✅ |
| Optuna study cleanup (1 active study) | S127 | ✅ |
| TPESampler multivariate=True | S119 | ✅ |
| Z10×Z10×Z10 digit features (4 new) | S119 | ✅ |
| Dataset split (daily3_midday + daily3_evening) | S119 | ✅ |
| Pruning confirmed firing (42% prune rate) | S120 | ✅ |
| Bayesian window optimizer resume mechanism (7 bugs fixed) | S115-S116 | ✅ |
| First clean real-data Steps 1–6 run | S120 | ✅ |
| 85 bidirectional survivors (W8_O43, real data) | S120 | ✅ |
| All 26 GPUs operational | S125b | ✅ |
| n_parallel=2 + Zeus GPU compute mode fix (DEFAULT) | S125/S125b | ✅ |
| Battery Tier 1A (23 features: FFT, autocorr, cumsum, bitfreq) | S113 | ✅ |
| NPZ v3.1 (24 metadata fields, 7 intersection fields) | S113 | ✅ |
| Real CA Daily 3 data (18,068 draws) | S112 | ✅ |
| Chapter 14 Phases 1–7b (RETRY loop proven) | S69-S82 | ✅ |
| Strategy Advisor fully deployed | S66/S68/S75 | ✅ |
| Chapter 13 full implementation | S57-S83 | ✅ |
| Selfplay A1–A5 validated | S116 | ✅ |
| TRSE Step 0 smoke test (0→6 verified) | S123 | ✅ |

---

## 🏗️ ARCHITECTURE INVARIANTS (Never Change Without Team Beta)

- Step order: 1→2→3→4→5→6 (static)
- Feature schema: hash-validated Step 5→6
- Authority separation: Chapter 13 decides, WATCHER executes, components cannot self-promote
- GPU isolation: Parent never initializes CUDA before NN subprocess spawn
- Manifest param governance: Every new CLI param MUST be in manifest `default_params` or WATCHER silently drops it
- Never restore from backup — always fix forward
- Comment out dead code, never delete
- Always fetch live file content via SSH before patching
- Zeus GPU compute mode: DEFAULT (never Exclusive_Process) — enforced via `/etc/rc.local` (S125b)
- GPU seed caps: Phase B concurrent ceiling × 0.85 — never increase without re-running Phase B probe
- Persistent workers: off by default (`use_persistent_workers=false`) — opt-in only
- Persistent worker path: standalone `persistent_worker_coordinator.py` — never embed in `coordinator.py`

---

*Generated S101 — 2026-02-20 | Updated S135 — 2026-03-10*
