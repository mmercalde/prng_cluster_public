# MASTER TODO LIST — S127
**Compiled:** 2026-03-02 (S114) | **Updated:** 2026-03-07 (S127)
**Sources:** TODO_MASTER_S126.md + S127 session history
**Status:** First clean real-data Steps 1–6 complete. 85 survivors, CatBoost winner. All 26 GPUs operational (S125b). Manifest bug fixed (S127). Optuna studies cleaned to 1 active (S127).

---

## 🔴 P1 — HIGH PRIORITY (Next 1-3 Sessions)

### Active Bugs / Uncommitted Fixes
- [ ] **WATCHER validation threshold fix** — `>=100` survivor threshold causes false ESCALATE on Steps 1 and 3 with real data. Lower to `>=50` or make configurable. Affects `watcher_agent.py`. (S120)
- [ ] **Commit manifest fix to both repos** — `agent_manifests/window_optimizer.json` re-patched S127 (3 bugs fixed: missing `enable_pruning`/`n_parallel` in `default_params`; `resume-study`/`study-name` missing from `actions[0].args_map`; `study_name` defaulting to `""`). File delivered, NOT YET COMMITTED.

### Step 1 n_parallel Concurrent GPU Fix
- [ ] **Fix Zeus double-dispatch (n_parallel partition design)** — `n_parallel=2` uses Optuna `n_jobs=2` ThreadPoolExecutor; both threads hit Zeus simultaneously → CUDA context collision. Fix: two separate Python processes each owning one partition exclusively. Process A owns Zeus+rrig6600+rrig6600b, Process B owns rrig6600c. Both share same SQLite study with `timeout=20s`. (S120 root cause confirmed)

### More Survivors
- [ ] **Run 200-trial Step 1 to grow survivor pool** — 85 survivors insufficient for NN. Need 500+. Resume `window_opt_1772507547.db` with 200 total trials.
- [ ] **k_folds runtime clamp** — When `val_fold_size < 3000`, clamp to `max(3, n_train // 3000)`. Needs Team Beta review. (S101 deferred)

### Chapter 13 + Selfplay Wire-up (Key Autonomy Gap)
- [ ] **Wire `dispatch_selfplay()` into WATCHER** — Stub exists in `watcher_agent.py`, not triggered post-Step-6. ~180 lines.
- [ ] **Wire `dispatch_learning_loop()` into WATCHER** — Same gap.
- [ ] **Wire Chapter 13 orchestrator into WATCHER daemon** — `chapter_13_orchestrator.py` exists but WATCHER cannot dispatch it post-Step-6.
- [ ] **Integration test: WATCHER → Chapter 13 → Selfplay full loop** — Selfplay A1–A5 validated in isolation; end-to-end with WATCHER trigger never verified.

### Documentation Sync (Three Locations Must Stay In Sync)
- [ ] **Write chapter docs patch script** — Python patch script (same pattern as prior session scripts) to update ALL stale chapter docs simultaneously on Zeus AND ser8. Stale chapters confirmed: Ch12 (missing dispatch module, bundle factory, lifecycle, new CLI args), Ch13 (Phase 7 complete status, selfplay wiring gap), Ch14 (Phases 1–4 complete, current phase), COMPLETE_OPERATING_GUIDE_v2_0 (missing S85+), README (6+ weeks stale, wrong GPU counts). Three locations: Zeus `~/distributed_prng_analysis/docs/`, ser8 `~/Downloads/CONCISE_OPERATING_GUIDE_v1.0/`, Claude Project (manual upload after script runs).
- [ ] **Upload updated docs to Claude Project** — After patch script runs on Zeus+ser8, upload revised files here manually to keep AI context current.

### GPU Throughput Investigation (Next Priority — S128)

> **Context:** S125b smoke test revealed major underutilization. `seed_cap_amd=19,000` and
> `seed_cap_nvidia=40,000` were set conservatively during ROCm instability (S96B) and
> CUDA Exclusive_Process fix (S125b). Measured throughput: RTX ~33,000 seeds/sec,
> RX 6600 ~11,500 seeds/sec. `gpu_optimizer.py` RX 6600 profile is 2.3x understated
> (5,000 listed vs 11,500 measured). Estimated gain: **~6x cluster throughput** from
> config constants only — zero code refactoring. Full plan: `docs/GPU_THROUGHPUT_INVESTIGATION_PLAN_v1_0.md`
> **Decision S127:** Run throughput probes BEFORE 200-trial run (not after).

- [ ] **Phase A — Single card isolated ceiling (RTX)** — Step-ladder probe: 100k → 500k → 1M → 2M → 5M seeds, one card, no concurrent workers. Record seeds/sec and peak VRAM at each step. Stop at OOM or >20% throughput drop. Monitor: `watch -n 1 "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader"` (S126)
- [ ] **Phase A — Single card isolated ceiling (RX 6600)** — Same step-ladder: 100k → 250k → 500k → 1M → 2M seeds, one card isolated via `ROCR_VISIBLE_DEVICES=0`. Monitor: `rocm-smi --showmeminfo vram`. ROCm documented limit: "do not exceed 2 simultaneous workloads" — single-card ceiling will differ from full-rig ceiling. (S126)
- [ ] **Phase B — Full concurrent rig ceiling (RTX)** — Both Zeus RTX cards simultaneously at: 100k → 300k → 750k → Phase_A×0.5 → Phase_A×0.75. Stop at any OOM or HIP error. This is the real production number. (S126)
- [ ] **Phase B — Full concurrent rig ceiling (AMD)** — All 8 RX 6600 workers per rig simultaneously at same step-ladder as RTX Phase B. Key metric: gap between Phase A and Phase B = the "ROCm multi-worker tax". Expected: Phase B ceiling = 50–70% of Phase A. (S126)
- [ ] **Phase C — Stability test** — 50 consecutive jobs at chosen caps (Phase B ceiling × 0.85) across all 26 GPUs via normal WATCHER pipeline. Gate before updating production config. (S126)
- [ ] **Update `coordinator.py` line 233** — Set `seed_cap_nvidia`, `seed_cap_amd`, `seed_cap_default` to Phase B ceiling × 0.85. Two-line edit only. (S126)
- [ ] **Update `gpu_optimizer.py` lines 17/18/35/36** — Set `seeds_per_second` to Phase A measured values. Recalculate `scaling_factor` as `RTX_sps / RX6600_sps` (expected ~2.9, currently wrong at 6.0). (S126)
- [ ] **Commit both files to both repos** — Standard dual-push. Add measured caps to Architecture Invariants below. (S126)

### Tree Model Bottleneck (Phase 3B)
- [ ] **Phase 3B: Tree parallel workers** — 2 tree trials simultaneously with pinned `CUDA_VISIBLE_DEVICES`. Trees = 93% of wall clock. Target: ~12 min. Requires Team Beta review.

---

## 🟠 P2 — MEDIUM PRIORITY (Next 3-6 Sessions)

### Neural Net & Training Warnings
- [ ] **sklearn warnings fix** — `UserWarning: X does not have valid feature names, but LGBMRegressor was fitted with feature names` — add `warnings.filterwarnings` suppression in `meta_prediction_optimizer_anti_overfit.py`. Noisy and masks real issues. First seen S109, confirmed recurring S112+S120.
- [ ] **XGBoost device mismatch warning fix** — `Falling back to prediction using DMatrix due to mismatched devices` — add `predict()` method to `XGBoostWrapper` that pre-converts numpy arrays to `DMatrix`. Script `fix_xgboost_device.sh` was written in S97 but status on Zeus is unconfirmed — verify and apply.
- [ ] **NN Y-label normalization** — `train_single_trial.py` line 499 — add `y_mean/y_std` normalization, save in sidecar for Step 6 inverse-transform. Agreed in S112 to fix after run. Unlocks NN convergence on narrow target range.
- [ ] **NN needs 500+ survivors** — Category B confirmed working (S120). CV R²=-198 to -7763 on 54 samples is purely data volume. Fix upstream via more Step 1 trials. Do NOT add auto-skip gate — selfplay + learning loop will resolve.

### GPU Utilization
- [ ] **Phase 3A vmap k_folds scaling** — Increase `--k-folds` to 10/20/30. Deferred S99.
- [ ] **Benchmark Phase 3A on real data** — S99 certified on test/noise. Needs real survivors.

### Live Autonomous Operation ("Flip The Switch")
- [ ] **Activate `draw_ingestion_daemon.py` with real draw data** — Daemon is coded (22KB), tested with synthetic draws, but never activated with live lottery data. This is the ignition switch for 24/7 autonomous operation.
- [ ] **Wire scraper (`daily3_scraper.py`) into WATCHER scheduler** — Scraper fetches new draws and exists at `~/daily3_scraper.py`. WATCHER needs a scheduler thread invoking it every N minutes → triggering Chapter 13 on new draw detection. Scraper failure handling: retry with backoff, escalate after 10 consecutive failures via Telegram.
- [ ] **24-hour synthetic soak test** — Run `synthetic_draw_injector.py` feeding draws every 60s with full Chapter 13 loop processing. Proof of sustained autonomous operation before going live.
- [ ] **Phase 9B.3: Selfplay policy feedback into Step 1 params** — Selfplay produces `learned_policy_candidate.json`, LLM accepts/rejects, but ACCEPT does not yet modify Step 1 search parameters. This is the difference between "automated" and "learns/improves over time".

### Bundle Factory Track 2
- [ ] **Fill 3 stub retrieval functions in bundle_factory.py** — `_retrieve_recent_outcomes()`, `_retrieve_trend_summary()`, `_retrieve_open_incidents()`. Return empty lists. Reads Chapter 13 summaries, `run_history.jsonl`, `watcher_failures.jsonl`.

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
- [ ] **Telegram notification hardening** — Boot notifications only confirmed working on rrig6600b (S63 session). rrig6600 and rrig6600c scripts need deployment. 12-GPU check in rrig6600b script also needs updating to 8.
- [ ] **Node watchdog / auto-restart layer** — WATCHER retries 3x on rig dropout but cannot reboot rigs. A watchdog layer that detects D-state zombies (ROCm HIP initialization storm) and triggers SSH reboot is needed for unattended 24/7 operation.

---

## 🔬 POSSIBILITY ADDONS — PRNG Detection Research

> Context: 85 real survivors, W8_O43 confirmed optimal. Regime boundaries at draw
> counts 3 and 8 suggest dual RNG systems / pre-test draws / session resets.

### Battery Tier 1B
- [ ] **Berlekamp-Massey linear complexity** — `batt_lc_complexity`. Pure LCG→complexity ~32; XOR-shift mixed→64-128.
- [ ] **XOR-shift lag autocorrelation** — Extend to lags 12, 25, 27 (canonical XOR-shift amounts). `batt_ac_lag_12/25/27`.
- [ ] **Bit avalanche coefficient** — `batt_bf_avalanche`. LCG: ~1 bit change. XOR-shift: ~50%.
- [ ] **Bit plane runs test** — `batt_bp_plane_0_runs`, `batt_bp_plane_1_runs`.
- [ ] **Spectral lattice analysis** — Plot (d[i], d[i+1]) pairs. LCG shows Marsaglia lattice. XOR destroys it.
- [ ] **Real survivor battery profile comparison** — Compare `batt_fft_spectral_conc` / `batt_ac_decay_rate` between real 85 survivors and synthetic.

### PRNG Registry
- [x] **XOR-shift variants already in registry** — `xorshift32/64/128` implemented. ✅
- [ ] **Run Step 1 with xorshift variants** — If linear complexity confirms XOR mixing, `--prng-type xorshift64`. Zero code changes.
- [ ] **WATCHER PRNG autonomy policy** — Policy to switch `prng_type` autonomously. Infrastructure complete, purely policy layer.
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

### Operational
- [ ] **Always launch pipeline in tmux** — `tmux new-session -s pipeline` before WATCHER. Prevents SSH drop loss.
- [ ] **Results/ FIFO cleanup automation** — WATCHER post-run hook.

---

## ✅ RECENTLY COMPLETED (Reference)

| Item | Session | Status |
|------|---------|--------|
| Manifest 3-bug fix (enable_pruning/n_parallel/study_name flow) | S127 | ✅ |
| Optuna studies cleaned — 13 deleted, 1 active remains | S127 | ✅ |
| Zeus GPU compute mode fix (DEFAULT, was Exclusive_Process) | S125b | ✅ |
| n_parallel=2 multiprocessing.Process dispatcher | S125 | ✅ |
| All 26 GPUs operational (smoke test) | S125b | ✅ |
| Multivariate TPE on TPESampler | S119 | ✅ |
| Z10×Z10×Z10 digit features (4 new) | S119 | ✅ |
| Dataset split (daily3_midday + daily3_evening) | S119 | ✅ |
| Gap 4b + Gap 5 pruning wire-up | S119 | ✅ |
| Pruning confirmed firing (42% prune rate) | S120 | ✅ |
| Bayesian window optimizer resume mechanism | S115-S116 | ✅ |
| S116 7-bug manifest fix | S116 | ✅ |
| First clean real-data Steps 1–6 run | S120 | ✅ |
| 85 bidirectional survivors (W8_O43, real data) | S120 | ✅ |
| NPZ S120 committed both repos (1496dad) | S120 | ✅ |
| Category B (normalize+leaky) confirmed deployed | S120 | ✅ |
| NN data volume root cause confirmed | S120 | ✅ |
| n_parallel Zeus double-dispatch root cause diagnosed | S120 | ✅ |
| Battery Tier 1A (23 features: FFT, autocorr, cumsum, bitfreq) | S113 | ✅ |
| NPZ v3.1 (24 metadata fields, 7 intersection fields) | S113 | ✅ |
| Real CA Daily 3 data (18,068 draws) | S112 | ✅ |
| Chapter 13 full implementation | S57-S83 | ✅ |
| Phase 7 WATCHER integration | S57-S59 | ✅ |
| Chapter 14 Phases 1-4 | S69-S82 | ✅ |
| Strategy Advisor fully deployed | S66/S68/S75 | ✅ |
| Selfplay A1–A5 validated | S116 | ✅ |
| draw_ingestion_daemon.py coded | S57-S59 | ✅ |
| synthetic_draw_injector.py coded + tested | S57-S79 | ✅ |
| daily3_scraper.py v1.5 (18,068 draws scraped) | S112 | ✅ |
| Telegram boot notifications (rrig6600b) | S63 | ✅ |

---

## 🏗️ ARCHITECTURE INVARIANTS (Never Change Without Team Beta)

- Step order: 1→2→3→4→5→6 (static)
- Feature schema: hash-validated Step 5→6
- Authority separation: Chapter 13 decides, WATCHER executes, components cannot self-promote
- GPU isolation: Parent never initializes CUDA before NN subprocess spawn
- Manifest param governance: Every new CLI param MUST be in manifest `default_params` or WATCHER silently drops it
- Manifest param governance: Every new CLI param MUST be in manifest `default_params` AND `actions[0].args_map` or WATCHER silently drops it (confirmed S127)
- Never restore from backup — always fix forward
- Comment out dead code, never delete
- Always fetch live file content via SSH before patching
- Zeus GPU compute mode: DEFAULT (never Exclusive_Process) — enforced via `/etc/rc.local` (S125b)
- GPU seed caps: set to Phase B concurrent ceiling × 0.85 — never increase without re-running Phase B probe

---

*Generated S101 — 2026-02-20 | Updated S114 — 2026-03-02 | Updated S120 — 2026-03-07 | Updated S126 — 2026-03-07 | Updated S127 — 2026-03-07*
