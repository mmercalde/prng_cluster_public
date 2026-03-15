# MASTER TODO LIST — S143
**Compiled:** 2026-03-02 (S114) | **Updated:** 2026-03-14 (S143)
**Sources:** TODO_MASTER_S139.md + S140–S143 session history
**Status:** PA experiment complete. Warm-start hardcode identified as architectural violation. System reverted to CA. Ready for clean CA Step 1 run.

---

## 🔴 P1 — HIGH PRIORITY (Next 1-3 Sessions)

### Active Run — Monitor & Validate
- [ ] **200-trial fresh Step 1 study in progress** — Cold start, 50M seeds, TRSE Rule A active (window cap=32), n_parallel=2, all 26 GPUs. Monitor for completion and NPZ output. Verify temp file merge works end-to-end.
- [ ] **Update distributed_config.json search_bounds** — After 200-trial run completes, use empirical results to tighten offset, skip, threshold bounds. Do NOT pre-empt with guesses.
- [ ] **Verify NPZ accumulator fix end-to-end** — Confirm `bidirectional_survivors_binary.npz` contains correct survivor counts from both partitions after run completes.

### Chapter 13 — Autonomy Wire-up (NEXT PRIORITY)
- [ ] **Wire `dispatch_selfplay()` into WATCHER** — Stub exists in `watcher_agent.py`, not triggered post-Step-6. ~180 lines.
- [ ] **Wire `dispatch_learning_loop()` into WATCHER** — Same gap.
- [ ] **Wire Chapter 13 orchestrator into WATCHER daemon** — `chapter_13_orchestrator.py` exists but WATCHER cannot dispatch it post-Step-6.
- [ ] **Integration test: WATCHER → Chapter 13 → Selfplay full loop** — Selfplay A1–A5 validated in isolation; end-to-end with WATCHER trigger never verified.

### Chapter 14 — Diagnostics & Selfplay (NEXT PRIORITY)
- [ ] **Chapter 14 Phase 8: Selfplay + Ch13 Wiring** — Episode diagnostics, trend detection, root cause analysis. Phase 8A done S83, Phase 8B blocked on per_survivor_attribution.
- [ ] **Chapter 14 Phase 9: First Diagnostic Investigation** — Real `--compare-models --enable-diagnostics` run on real data.

### Live Autonomous Operation (Depends on Ch13/14)
- [ ] **Activate `draw_ingestion_daemon.py` with real draw data** — Daemon coded, tested with synthetic draws, never activated with live data. Ignition switch for 24/7 autonomous operation.
- [ ] **Wire `daily3_scraper.py` into WATCHER scheduler** — Scraper exists. WATCHER needs scheduler thread invoking it every N minutes → triggering Chapter 13 on new draw detection.
- [ ] **24-hour synthetic soak test** — Run `synthetic_draw_injector.py` feeding draws every 60s with full Chapter 13 loop. Proof of sustained autonomous operation before going live.

### Warm-Start Hardcode Fix (S143) — BLOCKS MULTI-STATE ANALYSIS
- [ ] **Remove hardcoded W8_O43 warm-start from `window_optimizer_bayesian.py`** ~line 547.
  Current code enqueues `{'window_size':8,'offset':43,'skip_min':5,'skip_max':56,
  'forward_threshold':0.49,'reverse_threshold':0.49}` as Trial 0 on every fresh study.
  **Problems:** (1) CA-specific empirical values hardcoded in general-purpose optimizer.
  (2) Cross-contaminates any non-CA dataset — confirmed to suppress PA signal in S143.
  (3) Anchors TPE to potentially stale regime. (4) Redundant — `trial_history_context`
  already provides smarter warm-starts. (5) Violates no-hardcoding invariant.
  **Fix:** Remove enqueue block. Drive warm-start entirely from `trial_history_context`
  when present — if all 6 params available, enqueue; otherwise Optuna explores freely.
  CA manifest supplies W8_O43 explicitly. Any other dataset gets no warm-start.
  TRSE wire-up completes the loop with regime-aware candidates from recent draws.

### Blocking Bugs (Fix as encountered)
- [ ] **WATCHER Step 1 timeout** — 480 min hardcoded will kill long runs. Make configurable or remove for autonomous operation.
- [ ] **Persistent worker session drops on AMD rigs** — keepalive/TTL fix needed for 24/7 operation.
- [ ] **WATCHER validation threshold fix** — `>=100` survivor threshold causes false ESCALATE. Lower to `>=50` or make configurable. *Verify if S122/S123 already covers this.*
- [ ] **`--force-step N` flag for WATCHER** — bypass freshness gate without manually deleting output files.

---

## 🟠 P2 — MEDIUM PRIORITY (Next 3-6 Sessions)

### Neural Net & Training
- [ ] **sklearn warnings fix** — `UserWarning: X does not have valid feature names, but LGBMRegressor was fitted with feature names`. Add `warnings.filterwarnings` suppression in `meta_prediction_optimizer_anti_overfit.py`. (First seen S109, recurring)
- [ ] **XGBoost device mismatch warning fix** — `Falling back to prediction using DMatrix due to mismatched devices`. Add `predict()` method to `XGBoostWrapper` pre-converting numpy to DMatrix.
- [ ] **NN Y-label normalization** — `train_single_trial.py` line 499. Add `y_mean/y_std` normalization, save sidecar for Step 6 inverse-transform. Agreed S112 to fix after run.
- [ ] **Phase 3B: Tree parallel workers** — 2 tree trials simultaneously with pinned `CUDA_VISIBLE_DEVICES`. Trees = 93% of wall clock (~24 min). Target: ~12 min. Requires Team Beta review.

### TRSE Integration
- [ ] **TRSE Rules B and C — revisit after 200-trial results** — Rules B (skip) and C (offset) are logged only, disabled per TB S121 shuffle test. Empirical data from 167-trial run showed strong skip/offset clustering — may justify promoting to applied after 200-trial confirms. Requires Team Beta review.
- [ ] **Per-segment pipeline runs** — Run Steps 1–6 separately per TRSE regime segment. Depends on TRSE production wire-up.

### Bundle Factory Track 2
- [ ] **Fill 3 stub retrieval functions in `bundle_factory.py`** — `_retrieve_recent_outcomes()`, `_retrieve_trend_summary()`, `_retrieve_open_incidents()`. Currently return empty lists.

### Optuna / Window Optimizer
- [ ] **Wire variable skip bidirectional count into Optuna scoring** — `TestResult` only returns `bidirectional_count=len(bidirectional_constant)`. Variable skip count not reflected in Optuna objective. (S115 carry-forward)
- [ ] **Node failure resilience** — Single rig dropout can crash entire Optuna study. Need graceful degradation when a rig goes offline mid-study.
- [ ] **k_folds runtime clamp** — When `val_fold_size < 3000`, clamp to `max(3, n_train // 3000)`. Needs Team Beta review. (S101 deferred)

### Model Persistence
- [ ] **Feature names into `best_model.meta.json` at training time** — Proper fix vs current fallback extraction from trained LightGBM model. (S86 partial fix in place)

---

## 🟢 P3 — LOW PRIORITY / DEFERRED

### Code Cleanup
- [ ] **S110 root cleanup** — 884 files need cleanup.
- [ ] **Dead code audit: MultiModelTrainer inline path** — Comment out. Unreachable in subprocess mode.
- [ ] **Remove CSV writer from coordinator.py** — Dead weight.
- [ ] **Regression diagnostic gate** — Set `gate=True`.
- [ ] **Remove 27 stale project files from Claude Project** — Identified S85/S86.

### Documentation Sync
- [ ] **Write chapter docs patch script** — Update ALL stale chapter docs simultaneously.
- [ ] **Upload updated docs to Claude Project** — After patch script runs.

### Research / Experimental
- [ ] **S103 Part 2: per-seed match rates** — Continuation work.
- [ ] **Phase 9B.3: deferred selfplay component** — After 9B.2 validation.
- [ ] **Phase 3A vmap k_folds scaling** — Increase `--k-folds` to 10/20/30. Deferred S99.
- [ ] **Web dashboard refactor** — Chapter 14 visualization. Long-term.
- [ ] **PA Pick 3 follow-up experiment (S143, deferred)** — Clean 200-trial Step 1 run
  on PA data with no warm-start (requires warm-start fix first). Run session-split:
  `pa_pick3_midday.json` and `pa_pick3_evening.json` independently via `dataset_split.py`.
  Hypothesis: midday (confirmed RNG) produces massive survivors; evening (physical balls)
  produces near-zero. If both produce strong signal — deeper investigation warranted.
  Scraper: `pa_pick3_scraper.py` Rev 1.1. Data: `pa_pick3.json` (18,003 draws, 2000–2026).
  S143 finding: 389,041 bidirectional survivors across 15 biased trials (W8 warm-start
  suppressed signal). Best clean trial: W3_O59_evening 220,168 survivors.

### Operational
- [ ] **Always launch pipeline in tmux** — `tmux new-session -s pipeline` before WATCHER.
- [ ] **Results/ FIFO cleanup automation** — WATCHER post-run hook.
- [ ] **`--force-step` flag for WATCHER** — Override freshness gate for specific steps without deleting output files manually.

---

## ✅ RECENTLY COMPLETED (Reference)

| Item | Session | Commit |
|------|---------|--------|
| PA Pick 3 scraper Rev 1.1 (Wild Ball handling) | S143 | — |
| PA sieve validation harness (CPU, 3-tier) | S143 | — |
| PA 15-trial Step 1 experiment — 389,041 bidir survivors | S143 | — |
| Warm-start hardcode identified as arch violation | S143 | — |
| System reverted to CA data | S143 | — |
| S142-C: NP2 canonical backfill, _worker_obj writes removed | S142 | `51aed27` |
| S142-B: NP2 terminal flag + trial history backfill | S142 | `a2210ad` |
| S142-TB2: WAL mode on prng_analysis.db | S142 | `b5ea9f7` |
| TRSE Rule A applied in n_parallel partition worker | S139B | `25cc2de` |
| Window size max 500→50 (3 files) | S139 | `7d035c6` |
| Trial ceiling fix — subtract existing complete trials | S138B | `4849bff` |
| NPZ pipe deadlock fix — temp file instead of Queue | S138 | `3624e3c` |
| NPZ pipe deadlock smoke test — all 5 checks passed | S138 | — |
| 167-trial Optuna run — best=1,384,186 W2_O14_evening | S138 | — |
| Gate 1 close — fallback logging + dead-pipe fix | S131 | `1d85da4` |
| Seed cap patch — constructor defaults, wiring, manifest v1.5.0 | S131 | `d168f83` |
| Persistent workers (+150% throughput, 2,082,140 sps) | S130 | ✅ |
| Soak test 17/17 pass | S130 | ✅ |
| Manifest bug fix (study_name, enable_pruning, n_parallel) | S127 | ✅ |
| TPESampler multivariate=True | S119 | ✅ |
| First clean real-data Steps 1–6 run | S120 | ✅ |
| 85 bidirectional survivors (W8_O43, real data) | S120 | ✅ |
| All 26 GPUs operational | S125b | ✅ |
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
- TRSE skip_on_fail: Step 0 failures must never halt pipeline
- NPZ survivors: `bidirectional_survivors_binary.npz` must always remain git-tracked

---

*Generated S101 — 2026-02-20 | Updated S143 — 2026-03-14*
