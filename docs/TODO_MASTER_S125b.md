# MASTER TODO LIST — S125b
**Compiled:** 2026-03-02 (S114) | **Updated:** 2026-03-07 (S125b)
**Sources:** TODO_MASTER_S120.md + S121–S125b changelogs
**Status:** n_parallel=2 fully operational. All 26 GPUs confirmed working. Ready for 200-trial production run.

---

## 🔴 P1 — HIGH PRIORITY (Next 1-3 Sessions)

### Immediate Next Action
- [ ] **200-trial production Optuna resume** — Resume `window_opt_1772507547.db` (21 trials) to 200 total trials with `n_parallel=2 --enable-pruning`. Need 500+ survivors for NN. Command:
  ```bash
  PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 1 --end-step 1 \
    --params '{"trials": 200, "n_parallel": 2, "resume_study": true, \
               "study_name": "window_opt_1772507547", "enable_pruning": true}'
  ```

### Step 1/2 Throughput Investigation (Next Focus After 200-Trial Run)
- [ ] **Re-probe AMD seed cap** — `seed_cap_amd` is hardcoded at 19,000 seeds/job (set conservatively during ROCm stability work S96B). RX 6600 cards finish in ~8.5s and sit idle. Each card can likely handle much larger chunks. Re-run capacity probe to find true ceiling. Per-rig throughput at current cap: ~18,000 seeds/sec (8 GPUs × ~2,200/sec). Potential: significantly higher with larger job sizes.
- [ ] **Re-probe RTX seed cap** — `seed_cap_nvidia` set at 2,800,000 capacity but jobs are capped at 19,000 seeds same as AMD. RTX 3080 Ti cards are massively underutilized. Probe true throughput ceiling with larger chunk sizes.
- [ ] **Benchmark full 26-GPU throughput at higher caps** — After individual GPU probes, measure end-to-end seeds/sec across all 26 GPUs with optimized caps. Current ~54,000 seeds/sec (24 AMD GPUs) should be significantly improvable.
- [ ] **Add throughput metrics to Optuna trial scoring** — Seeds/sec per trial currently not tracked in Optuna. Add as metadata for regime analysis.

### Uncommitted Fixes / Manifest Gaps
- [ ] **Wire `n_parallel` + `enable_pruning` into manifest** — `agent_manifests/window_optimizer.json` `default_params` missing these. Patched on Zeus (S120) but NOT committed to either repo. Causes WATCHER to silently drop these params if passed via pipeline.
- [ ] **WATCHER validation threshold fix** — `>=100` survivor threshold causes false ESCALATE on Steps 1 and 3 with real data. Lower to `>=50` or make configurable. Affects `watcher_agent.py`. (S120)

### Chapter 13 + Selfplay Wire-up (Key Autonomy Gap)
- [ ] **Wire `dispatch_selfplay()` into WATCHER** — Stub exists in `watcher_agent.py`, not triggered post-Step-6. ~180 lines.
- [ ] **Wire `dispatch_learning_loop()` into WATCHER** — Same gap.
- [ ] **Wire Chapter 13 orchestrator into WATCHER daemon** — `chapter_13_orchestrator.py` exists but WATCHER cannot dispatch it post-Step-6.
- [ ] **Integration test: WATCHER → Chapter 13 → Selfplay full loop** — Selfplay A1–A5 validated in isolation; end-to-end with WATCHER trigger never verified.

### Documentation Sync (Three Locations Must Stay In Sync)
- [ ] **Write chapter docs patch script** — Python patch script to update ALL stale chapter docs simultaneously on Zeus AND ser8. Stale chapters confirmed: Ch12 (missing dispatch module, bundle factory, lifecycle, new CLI args), Ch13 (Phase 7 complete status, selfplay wiring gap), Ch14 (Phases 1–4 complete, current phase), COMPLETE_OPERATING_GUIDE_v2_0 (missing S85+), README (6+ weeks stale, wrong GPU counts).
- [ ] **Upload updated docs to Claude Project** — After patch script runs on Zeus+ser8, upload revised files here manually to keep AI context current.

---

## 🟠 P2 — MEDIUM PRIORITY (Next 3-6 Sessions)

### Neural Net & Training Warnings
- [ ] **sklearn warnings fix** — `UserWarning: X does not have valid feature names, but LGBMRegressor was fitted with feature names` — add `warnings.filterwarnings` suppression in `meta_prediction_optimizer_anti_overfit.py`. (S109, recurring S112+S120)
- [ ] **XGBoost device mismatch warning fix** — `Falling back to prediction using DMatrix due to mismatched devices` — `fix_xgboost_device.sh` written S97, status on Zeus unconfirmed — verify and apply.
- [ ] **NN Y-label normalization** — `train_single_trial.py` line 499 — add `y_mean/y_std` normalization, save in sidecar for Step 6 inverse-transform. Unlocks NN convergence on narrow target range.
- [ ] **k_folds runtime clamp** — When `val_fold_size < 3000`, clamp to `max(3, n_train // 3000)`. Needs Team Beta review. (S101 deferred)

### GPU Utilization
- [ ] **Phase 3B: Tree parallel workers** — 2 tree trials simultaneously with pinned `CUDA_VISIBLE_DEVICES`. Trees = 93% of wall clock. Target: ~12 min. Requires Team Beta review.
- [ ] **Phase 3A vmap k_folds scaling** — Increase `--k-folds` to 10/20/30. Deferred S99.
- [ ] **Benchmark Phase 3A on real data** — S99 certified on test/noise. Needs real survivors.

### Live Autonomous Operation ("Flip The Switch")
- [ ] **Activate `draw_ingestion_daemon.py` with real draw data** — Daemon coded, tested synthetic, never activated live.
- [ ] **Wire scraper (`daily3_scraper.py`) into WATCHER scheduler** — Trigger every N minutes → Chapter 13 on new draw detection.
- [ ] **24-hour synthetic soak test** — `synthetic_draw_injector.py` feeding draws every 60s. Proof before going live.
- [ ] **Phase 9B.3: Selfplay policy feedback into Step 1 params** — `learned_policy_candidate.json` ACCEPT does not yet modify Step 1 search parameters.

### Bundle Factory Track 2
- [ ] **Fill 3 stub retrieval functions in bundle_factory.py** — `_retrieve_recent_outcomes()`, `_retrieve_trend_summary()`, `_retrieve_open_incidents()`. Return empty lists.

### Chapter 14 Remaining Phases
- [ ] **Chapter 14 Phase 5: FIFO history pruning** — Unbounded growth of `diagnostics_history/`. Keep last N.
- [ ] **Chapter 14 Phase 6: WATCHER integration** — Wire Ch14 diagnostics into WATCHER decision loop.
- [ ] **Chapter 14 Phase 7: LLM integration** — Feed training diagnostics to LLM advisor.
- [ ] **Chapter 14 Phase 8B: Selfplay wiring** — Phase 8A done S83. Phase 8B blocked on `per_survivor_attribution`.
- [ ] **Chapter 14 Phase 9: First real diagnostic investigation** — `--compare-models --enable-diagnostics` with 500+ survivors.

### Model Persistence
- [ ] **`--save-all-models` flag** — Save all 4 Step 5 model types. Only winner currently saved.
- [ ] **Feature names into best_model.meta.json at training time** — S86 partial fix in place. Full fix needed.

### Monitoring / Alerting
- [ ] **Telegram notification hardening** — Boot notifications only confirmed on rrig6600b (S63). rrig6600 and rrig6600c scripts need deployment. 12-GPU check in rrig6600b script also needs updating to 8.
- [ ] **Node watchdog / auto-restart layer** — WATCHER retries 3x on rig dropout but cannot reboot rigs. Watchdog for D-state zombies + SSH reboot for unattended 24/7.

---

## 🔬 POSSIBILITY ADDONS — PRNG Detection Research

> Context: 85 real survivors, W8_O43 confirmed optimal. Regime boundaries at draw
> counts 3 and 8 suggest dual RNG systems / pre-test draws / session resets.

### Battery Tier 1B
- [ ] **Berlekamp-Massey linear complexity** — `batt_lc_complexity`. Pure LCG→complexity ~32; XOR-shift mixed→64-128.
- [ ] **XOR-shift lag autocorrelation** — Extend to lags 12, 25, 27. `batt_ac_lag_12/25/27`.
- [ ] **Bit avalanche coefficient** — `batt_bf_avalanche`. LCG: ~1 bit change. XOR-shift: ~50%.
- [ ] **Bit plane runs test** — `batt_bp_plane_0_runs`, `batt_bp_plane_1_runs`.
- [ ] **Spectral lattice analysis** — Plot (d[i], d[i+1]) pairs. LCG shows Marsaglia lattice. XOR destroys it.
- [ ] **Real survivor battery profile comparison** — Compare `batt_fft_spectral_conc` / `batt_ac_decay_rate` between real 85 survivors and synthetic.

### PRNG Registry
- [x] **XOR-shift variants already in registry** — `xorshift32/64/128` implemented. ✅
- [ ] **Run Step 1 with xorshift variants** — If linear complexity confirms XOR mixing, `--prng-type xorshift64`. Zero code changes.
- [ ] **WATCHER PRNG autonomy policy** — Policy to switch `prng_type` autonomously.
- [ ] **`java_lcg_xor` hybrid PRNG** — LCG state update + XOR-shift output transformation.

### Per-Session Analysis
- [ ] **Per-segment pipeline runs** — `daily3_midday.json` (8,515) and `daily3_evening.json` (9,553) exist. Run separately and compare survivor profiles.
- [ ] **Digit triplet frequency drift probe** — Z10×Z10×Z10 features deployed (S119). Analyze drift across sessions as regime diagnostic.

---

## 🟡 P3 — BACKLOG (Deferred, No Hard Timeline)

### Deferred Work
- [ ] **S103 Part 2: Per-seed match rates continuation** — Work remaining from S103.
- [ ] **Feed results/ trial history to Strategy Advisor** — Window optimization history as advisor bundle context.
- [ ] **Phase 9B.3: Automatic policy proposal heuristics** — Deferred until 9B.2 validated.

### Infrastructure
- [ ] **S110 root cleanup** — 884 files need cleanup (58 stray docs moved S109).
- [ ] **FIFO pruning for results/ directory** — 93MB+. Keep last N runs.
- [ ] **_record_training_incident() in S76 retry path** — Audit trail for retries.
- [ ] **Web dashboard refactor** — Chapter 14 visualization. Long-term.

### Code Cleanup
- [ ] **Dead code audit: MultiModelTrainer inline path** — Comment out. Unreachable in subprocess mode.
- [ ] **Remove CSV writer from coordinator.py** — Dead weight.
- [ ] **Remove 27 stale project files from Claude Project** — Identified S85/S86.
- [ ] **Regression diagnostic gate** — Set `gate=True`.
- [ ] **Variable skip bidirectional count** — Not wired into Optuna scoring (`TestResult` only returns `bidirectional_count=len(bidirectional_constant)`). (S116 carry-forward)

### Operational
- [ ] **Always launch pipeline in tmux** — `tmux new-session -s pipeline` before WATCHER. Prevents SSH drop loss.
- [ ] **Results/ FIFO cleanup automation** — WATCHER post-run hook.

---

## ✅ RECENTLY COMPLETED (Reference)

| Item | Session | Status |
|------|---------|--------|
| Zeus GPU compute mode → DEFAULT (rc.local permanent) | S125b | ✅ |
| n_parallel=2 fully operational — all 26 GPUs confirmed | S125b | ✅ |
| cudaErrorDevicesUnavailable root cause diagnosed + fixed | S125b | ✅ |
| n_parallel=2 CUDA collision fix (multiprocessing.Process) | S125 | ✅ |
| coordinator=self dead routing var fixed (Bug A) | S125 | ✅ |
| Optuna n_jobs=1 always (Bug B) | S125 | ✅ |
| TRSE Step 0 integration | S121-S123 | ✅ |
| Bayesian window optimizer resume mechanism verified | S115-S116 | ✅ |
| S116 7-bug manifest fix | S116 | ✅ |
| First clean real-data Steps 1–6 run | S120 | ✅ |
| 85 bidirectional survivors (W8_O43, real data) | S120 | ✅ |
| Pruning confirmed firing (42% prune rate) | S120 | ✅ |
| Battery Tier 1A (23 features: FFT, autocorr, cumsum, bitfreq) | S113 | ✅ |
| NPZ v3.1 (24 metadata fields, 7 intersection fields) | S113 | ✅ |
| Real CA Daily 3 data (18,068 draws) | S112 | ✅ |
| Chapter 13 full implementation | S57-S83 | ✅ |
| Phase 7 WATCHER integration | S57-S59 | ✅ |
| Chapter 14 Phases 1-4 | S69-S82 | ✅ |
| Strategy Advisor fully deployed | S66/S68/S75 | ✅ |
| Selfplay A1–A5 validated | S116 | ✅ |
| Multivariate TPE on TPESampler | S119 | ✅ |
| Z10×Z10×Z10 digit features (4 new) | S119 | ✅ |

---

## 🏗️ ARCHITECTURE INVARIANTS (Never Change Without Team Beta)

- Step order: 0→1→2→3→4→5→6 (static)
- Feature schema: hash-validated Step 5→6
- Authority separation: Chapter 13 decides, WATCHER executes, components cannot self-promote
- GPU isolation: Parent never initializes CUDA before NN subprocess spawn
- Zeus GPU compute mode: DEFAULT (never Exclusive_Process) — enforced via /etc/rc.local
- Manifest param governance: Every new CLI param MUST be in manifest `default_params` or WATCHER silently drops it
- Never restore from backup — always fix forward
- Comment out dead code, never delete
- Always fetch live file content via SSH before patching

---

*Generated S101 — 2026-02-20 | Updated S114 — 2026-03-02 | Updated S120 — 2026-03-07 | Updated S125b — 2026-03-07*
