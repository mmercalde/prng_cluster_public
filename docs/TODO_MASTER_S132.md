# MASTER TODO LIST — S132
**Compiled:** 2026-03-02 (S114) | **Updated:** 2026-03-08 (S132 — live code verified)
**Sources:** Fresh clone audit of prng_cluster_public @ `5525f35` + project changelogs
**Status:** Persistent workers live (+150% throughput). All 26 GPUs operational. Both remotes at `5525f35`.

---

## 🔴 P1 — HIGH PRIORITY (Next 1-3 Sessions)

### Run More Trials — Critical Path to NN
- [ ] **Run 200-trial Step 1 to grow survivor pool** — 85 survivors insufficient for NN. Need 500+. Resume `window_opt_1772507547.db` (24 complete, 26 pruned). Use `--resume-study --study-name window_opt_1772507547 --trials 200`. Persistent workers active — 2.5× baseline throughput.

### Chapter 13 + Selfplay Wire-up (Key Autonomy Gap)
- [ ] **Wire `dispatch_selfplay()` post-Step-6 in `run_pipeline()`** — CLI path exists (lines 2974-2977) but `run_pipeline()` ends at line 2105 with no selfplay trigger. ~180 lines to implement. Verified absent from run_pipeline.
- [ ] **Wire `dispatch_learning_loop()` post-Step-6 in `run_pipeline()`** — Same gap. CLI path exists but never triggered from pipeline.
- [ ] **Wire Chapter 13 orchestrator into WATCHER daemon** — `chapter_13_orchestrator.py` exists, not dispatched post-Step-6.
- [ ] **Integration test: WATCHER → Chapter 13 → Selfplay full loop** — Selfplay A1–A5 validated in isolation; end-to-end with WATCHER trigger never verified.

### Node Failure Resilience
- [ ] **Optuna rig dropout resilience** — No `StorageInternalError`/`OperationalError` retry logic found in `window_optimizer_bayesian.py`. Single rig dropout can crash entire Optuna study. Needs graceful degradation.

---

## 🟠 P2 — MEDIUM PRIORITY (Next 3-6 Sessions)

### Neural Net & Training
- [ ] **sklearn warnings fix** — `UserWarning: X does not have valid feature names, but LGBMRegressor was fitted with feature names`. No `filterwarnings` suppression in live `meta_prediction_optimizer_anti_overfit.py`. Noisy, masks real issues.
- [ ] **`--save-all-models` flag** — Not in live `meta_prediction_optimizer_anti_overfit.py` (only in `.bak` files). Save all 4 Step 5 model types for post-hoc AI analysis.
- [ ] **k_folds runtime clamp (deeper fix)** — S124 `n_splits <= n_samples` guard confirmed present. The S101 deeper clamp `max(3, n_train // 3000)` for `val_fold_size < 3000` is a separate P2 improvement. Needs Team Beta review.
- [ ] **XGBoost device mismatch warning** — No fix in live code. `fix_xgboost_device.sh` written S97, status on Zeus unconfirmed. Add `predict()` to `XGBoostWrapper` pre-converting numpy to DMatrix.

### TRSE / Per-Segment
- [ ] **Per-segment pipeline runs** — Run Steps 1–6 per TRSE regime segment. TRSE Step 0 wired (confirmed). Per-segment dispatch logic not implemented.

### Live Autonomous Operation
- [ ] **Activate `draw_ingestion_daemon.py` with real draw data** — File exists (21KB), never activated with live data. Ignition switch for 24/7 autonomous operation.
- [ ] **Wire `daily3_scraper.py` into WATCHER scheduler** — No reference in `watcher_agent.py`. Needs scheduler thread + Chapter 13 trigger on new draw detection.
- [ ] **24-hour synthetic soak test** — `synthetic_draw_injector.py` → draws every 60s, full Chapter 13 loop. Proof before going live.

### Bundle Factory Track 2
- [ ] **Fill 3 stub retrieval functions in `bundle_factory.py`** — All 3 confirmed stubs returning empty: `_retrieve_recent_outcomes()`, `_retrieve_trend_summary()`, `_retrieve_open_incidents()`. Reads Chapter 13 summaries, run_history.jsonl, watcher_failures.jsonl.

### Chapter 14 Remaining Phases (verified against live code)
- [ ] **Phase 4: Dashboard visualization** — `chart_loss_curves()`, `chart_feature_importance()`, `chart_survivor_attribution()`, `chart_nn_health()`, `chart_diagnosis_panel()` — 0 hits in `web_visualizer.py`/`web_dashboard.py`. Not implemented.
- [ ] **Phase 5: TensorBoard** — `SummaryWriter`, `add_scalars`, `add_histogram`, `add_graph` — 0 hits in live code. Optional but designed. `--enable-tensorboard` flag not wired.
- [ ] **Phase 8: Selfplay + Ch13 diagnostics wiring** — `post_draw_root_cause_analysis()`, episode diagnostics — 0 hits in live code. `per_survivor_attribution.py` exists (17KB, PyTorch dynamic graph + grad_x_input + CatBoost SHAP confirmed) but not wired into selfplay loop or Chapter 13.
- [ ] **Phase 9: First Diagnostic Investigation** — Real `--compare-models --enable-diagnostics` run on real data. Requires Phase 8 first.

**Note — `diagnostics_analysis.gbnf` missing from repo** — Phase 7 LLM integration calls `request_llm_diagnostics_analysis()` (wired in `watcher_agent.py` lines 1688/1702) but the grammar file is absent. Needs investigation — may be on Zeus only or may need to be created.

### Phase 3B
- [ ] **Phase 3B: Tree parallel workers** — 2 tree trials simultaneously with pinned `CUDA_VISIBLE_DEVICES`. Trees = 93% of wall clock (~24 min). Target: ~12 min. Requires Team Beta review. No implementation in live code.

---

## 🟢 P3 — LOW PRIORITY / DEFERRED

### Code Cleanup
- [ ] **S110 root cleanup** — 680 total files in root (verified). Still significant clutter.
- [ ] **MultiModelTrainer inline path dead code** — `NN_SUBPROCESS_ROUTING_ENABLED = True` flag exists. Inline path still present but unreachable in subprocess mode. Comment out.
- [ ] **Regression diagnostic gate** — Set `gate=True`. Confirmed gate=True absent in live code — currently always False.
- [ ] **Remove 27 stale project files from Claude Project** — Identified S85/S86.

### Research / Experimental
- [ ] **S103 Part 2: per-seed match rates** — Continuation work.
- [ ] **Phase 9B.3: deferred selfplay component** — After 9B.2 validation.
- [ ] **Phase 3A vmap k_folds scaling** — Increase `--k-folds` to 10/20/30. Deferred S99.
- [ ] **Web dashboard refactor** — Chapter 14 visualization. Long-term.
- [ ] **Feature names into `best_model.meta.json` at training time** — `train_single_trial.py` has no feature_names write. Current fallback from trained LightGBM is fragile. S86 partial fix in place.

### Operational
- [ ] **Always launch pipeline in tmux** — Prevents SSH drop loss.
- [ ] **Results/ FIFO cleanup automation** — WATCHER post-run hook.

---

## ✅ VERIFIED COMPLETE (Live Code Confirmed @ 5525f35)

| Item | File:Line | Session | Evidence |
|------|-----------|---------|----------|
| WATCHER threshold >=50 (was >=100) | `watcher_agent.py:169` | S122/S123 | `"bidirectional_survivors*.json": 50` |
| Variable skip bidi count wired into Optuna | `window_optimizer_integration_final.py` | S124 | 4 hits: `_variable_bidi_count`, `_total_bidi` |
| sklearn KFold n_splits guard (n_splits <= n_samples) | `meta_prediction_optimizer_anti_overfit.py:2026` | S124 | `[S124] Guard: clamp n_splits` |
| NN Y-label normalization | `train_single_trial.py:497` | S121 | `[S121] Y normalization` |
| TRSE Step 0 wired in WATCHER | `watcher_agent.py:387` | S121/S122 | `0: "trse_step0.py"` |
| Manifest v1.5.0 — seed caps, enable_pruning, n_parallel | `agent_manifests/window_optimizer.json` | S127/S131 | version=1.5.0 |
| CSV writer removed from coordinator.py | `coordinator.py` | — | 0 grep hits confirmed |
| Gate 1 fault tolerance (dead-pipe fix) | `coordinator.py` | S131 | `[S130][FALLBACK]` paths |
| Seed cap constructor defaults (5M/2M) | `coordinator.py:233` | S131 | confirmed |
| Seed cap explicit wiring in integration file | `window_optimizer_integration_final.py` | S131 | 2 sites patched |
| Persistent workers (+150% throughput) | `coordinator.py` | S130 | `use_persistent_workers` |
| Zeus GPU compute mode DEFAULT | `/etc/rc.local` on Zeus | S125b | `nvidia-smi -c 0` on boot |
| n_parallel=2 multiprocessing.Process dispatcher | `window_optimizer_integration_final.py` | S125 | process partitioning |
| RETRY param-threading (training health) | `agents/watcher_agent.py` | S76/S82 | 11 assertions passed |
| apply_caps.py maintenance tool deployed | `apply_caps.py` | S131 | commit `50a4146` |
| TPESampler multivariate=True | `window_optimizer_bayesian.py` | S119 | confirmed |
| Z10×Z10×Z10 digit features (4 new) | `survivor_scorer.py` | S119 | confirmed |
| Battery Tier 1A (23 features) | `survivor_scorer.py` | S113 | confirmed |
| NPZ v3.1 (24 metadata fields) | `convert_survivors_to_binary.py` | S113 | confirmed |
| Bayesian window optimizer resume (7 bugs) | `window_optimizer_bayesian.py` | S115/S116 | confirmed |
| Strategy Advisor fully deployed | `parameter_advisor.py` etc. | S66/S68/S75 | confirmed |
| Chapter 13 full implementation | `chapter_13_orchestrator.py` | S57-S83 | confirmed |
| Ch14 Phase 1: Core diagnostics (NNDiagnostics, TreeDiagnostics) | `training_diagnostics.py` | S69 | 41KB, confirmed |
| Ch14 Phase 2: Per-survivor attribution (grad_x_input, CatBoost SHAP) | `per_survivor_attribution.py` | S70+ | 17KB, confirmed |
| Ch14 Phase 3: Engine wiring (--enable-diagnostics, subprocess hooks) | `train_single_trial.py`, `meta_prediction_optimizer_anti_overfit.py` | S70/S73 | confirmed |
| Ch14 Phase 6: WATCHER integration (check_training_health, skip registry) | `agents/watcher_agent.py:104` | S76/S82 | confirmed |
| Ch14 Phase 7: LLM diagnostics integration (request_llm_diagnostics_analysis) | `agents/watcher_agent.py:1688` | S81 | confirmed wired |
| Selfplay A1–A5 validated (isolation) | `selfplay_orchestrator.py` | S116 | confirmed |
| Real CA Daily 3 data (18,068 draws) | `daily3.json` on Zeus | S112 | confirmed |

---

## 🏗️ ARCHITECTURE INVARIANTS (Never Change Without Team Beta)

- Step order: 1→2→3→4→5→6 (static)
- Feature schema: hash-validated Step 5→6
- Authority separation: Chapter 13 decides, WATCHER executes, components cannot self-promote
- GPU isolation: Parent never initializes CUDA before NN subprocess spawn
- Manifest param governance: Every new CLI param MUST be in manifest `default_params` AND `actions[0].args_map` or WATCHER silently drops it
- Never restore from backup — always fix forward
- Comment out dead code, never delete
- Always fetch live file content via SSH before patching
- Zeus GPU compute mode: DEFAULT (never Exclusive_Process) — enforced via `/etc/rc.local`
- GPU seed caps: Phase B concurrent ceiling × 0.85 — never increase without re-running Phase B probe
- Persistent workers: off by default (`use_persistent_workers=false`) — opt-in only

---

*Generated S101 — 2026-02-20 | Live-code verified S132 — 2026-03-08*
*Verification: fresh git clone @ 5525f35, grep/AST audit of every open item*
