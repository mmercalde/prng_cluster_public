# MASTER TODO LIST — S114
**Compiled:** 2026-03-02
**Sources:** Project files, session changelogs S4–S113, chat history
**Status:** Real data pipeline run in progress — Steps 1-2 pending clean 10M seed run

---

## 🔴 P1 — HIGH PRIORITY (Next 1-3 Sessions)

### Pipeline & Data
- [ ] **Verify S101 full run results** — First clean Step 1→6 run on real data since S97 symlink fix. Check survivor count, Step 5 R² on real data, prediction pool quality.
- [ ] **k_folds runtime clamp (S101 deferred)** — When `val_fold_size < 3000`, clamp to `max(3, n_train // 3000)`. Remove reliance on CLI default. Needs Team Beta review.
- [ ] **n_jobs parallelism investigation** — Optuna `n_jobs` currently hardcoded to 2 (one per GPU). Explore whether increasing helps tree model throughput.
- [ ] **Multi-trial dispatch** — Currently one trial per GPU dispatch. Deferred from S100/S101.

### Tree Model Bottleneck (Phase 3B)
- [ ] **Phase 3B: Tree parallel workers** — Run 2 tree trials simultaneously with pinned `CUDA_VISIBLE_DEVICES`. Trees are 93% of current wall clock (~24 min). Target: ~12 min. HIGH priority. Requires Team Beta review before implementation.

---

## 🟠 P2 — MEDIUM PRIORITY (Next 3-6 Sessions)

### GPU Utilization
- [ ] **Phase 3A vmap k_folds scaling** — Increase `--k-folds` from current value to 10/20/30 to measure vmap N throughput on dual RTX 3080 Ti. Deferred from S99.
- [ ] **Benchmark Phase 3A on real data** — S99 certified Phase 3A but testing was on test/noise data. Needs real survivors_with_scores.json to measure true speedup.

### Bundle Factory Track 2
- [ ] **Fill 3 stub retrieval functions in bundle_factory.py** — `_retrieve_recent_outcomes()`, `_retrieve_trend_summary()`, `_retrieve_open_incidents()`. Currently return empty lists. Track 2 implementation reads Chapter 13 summaries, run_history.jsonl, watcher_failures.jsonl. Zero dispatch rework needed — same API.

### Chapter 14 Remaining Phases
- [ ] **Chapter 14 Phase 5: FIFO history pruning** — Prevent unbounded growth of diagnostics_history/. Keep last N entries, prune oldest.
- [ ] **Chapter 14 Phase 6: WATCHER integration** — Wire Chapter 14 diagnostics into WATCHER decision loop.
- [ ] **Chapter 14 Phase 7: LLM integration** — Feed training diagnostics to LLM advisor.
- [ ] **Chapter 14 Phase 8: Selfplay wiring** — Episode diagnostics + trend detection (Phase 8A done S83, Phase 8B blocked on per_survivor_attribution).
- [ ] **Chapter 14 Phase 9: First diagnostic investigation** — Real `--compare-models --enable-diagnostics` run on real data.

### Model Persistence
- [ ] **`--save-all-models` flag** — Save all 4 Step 5 model types for post-hoc AI analysis. Currently only winner is saved. Needed for Strategy Advisor deeper analysis.
- [ ] **Feature names into best_model.meta.json at training time** — Proper fix vs current fallback extraction from trained LightGBM model. (S86 partial fix in place — full fix needed.)

---

## 🔬 POSSIBILITY ADDONS — PRNG Detection Research (Investigate After Real Data Run)

> **Context:** S113 mathematical analysis showed 53 bidirectional survivors out of
> 500M total seeds tested (1 in 9.4M rate — comparable to Powerball odds). The
> sparsity and cluster size validates the bidirectional sieve. The following are
> research directions to detect whether the lottery applies additional bit
> manipulation (XOR-shift, post-processing) on top of the base Java LCG.

### XOR-Shift & Bit Manipulation Detection

- [ ] **Berlekamp-Massey linear complexity test** — Convert draw history to bit
  sequence, compute shortest LFSR that generates it. Pure LCG → complexity ~32.
  XOR-shift mixed → complexity ~64-128. Combined generator → very high complexity.
  Add as `batt_lc_complexity` feature. *(Battery Tier 1B candidate)*

- [ ] **XOR-shift lag autocorrelation extensions** — Extend existing
  `batt_ac_lag_01..10` to cover lags 12, 25, 27 specifically. These are the
  canonical XOR-shift amounts. Anomalous spikes at these lags in real survivors
  vs synthetic would confirm XOR mixing. Add as `batt_ac_lag_12`,
  `batt_ac_lag_25`, `batt_ac_lag_27`. *(Battery Tier 1B candidate)*

- [ ] **Bit avalanche coefficient** — Flip one input bit, measure how many output
  bits change. Pure LCG: ~1 bit (low avalanche). XOR-shift: ~50% bits (high
  avalanche by design). Add as `batt_bf_avalanche`. *(Battery Tier 1B candidate)*

- [ ] **Bit plane runs test** — Extract individual bit planes (bit 0, bit 1, etc.)
  from all draws. Test each plane independently for randomness. XOR-shift mixes
  bit planes — pure LCG has structured low-order bits. Add as
  `batt_bp_plane_0_runs`, `batt_bp_plane_1_runs`. *(Battery Tier 1B candidate)*

- [ ] **Spectral lattice analysis** — Plot consecutive draw pairs (d[i], d[i+1])
  in 2D. Pure LCG shows visible lattice structure (Marsaglia theorem). XOR mixing
  destroys the lattice. Visual + quantitative test. Compare real 53 survivors vs
  synthetic survivors.

- [ ] **Real survivor battery profile analysis** — After first clean real-data
  Step 3 run, compare `batt_fft_spectral_conc` and `batt_ac_decay_rate` between
  real 53 survivors and synthetic survivors. Fundamental difference in profiles
  would indicate XOR mixing is present without writing any new code.

### PRNG Registry Extensions

- [x] **XOR-shift variants already in registry** — `xorshift32`, `xorshift64`,
  `xorshift128` (plus hybrid/reverse variants) are fully implemented with GPU
  kernels in `prng_registry.py`. NO new implementation needed. ✅

- [ ] **Run Step 1 with xorshift variants** — If linear complexity test confirms
  XOR mixing is present, simply pass `--prng-type xorshift64` (or xorshift32,
  xorshift128) to Step 1. Compare survivor counts vs `java_lcg`. Zero code
  changes required — parameter change only.

- [ ] **WATCHER PRNG autonomy policy** — Define policy for WATCHER to
  autonomously switch `prng_type` between runs. Required additions: (1) add
  `prng_type` to WATCHER GBNF grammar as valid action, (2) PRNG performance
  tracking sidecar, (3) Strategy Advisor PRNG context, (4) escalation policy
  e.g. "3 consecutive 0-survivor runs → try next PRNG family". Infrastructure
  already complete — purely a policy/decision layer addition.

- [ ] **Java LCG + XOR post-processing hybrid** — Implement `java_lcg_xor` in
  registry: standard LCG state update + XOR-shift output transformation. Test
  whether this PRNG type finds more survivors than pure `java_lcg`.

---

## 🟡 P3 — BACKLOG (Deferred, No Hard Timeline)

### Strategy Advisor Enhancement
- [ ] **Feed results/ trial history to Strategy Advisor** — Window optimization trial history (per-trial survivor counts, parameter combinations, forward/reverse outcomes) as additional advisor bundle context. Gives LLM richer signal for regime shift detection and search strategy recommendations. (Added this session S101.)

### Phase 9B
- [ ] **Phase 9B.3: Automatic policy proposal heuristics** — Deferred until 9B.2 validated. Auto-generate heuristic policy proposals without LLM.

### Infrastructure
- [ ] **FIFO pruning for results/ directory** — Keep last N pipeline runs of results/, prune oldest. Currently 93MB and growing with every run. Add to pipeline cleanup step or WATCHER post-run hook.
- [ ] **_record_training_incident() in S76 retry path** — Audit trail for retries. Was in removed dead code. Low priority but clean architecture gap.
- [ ] **Web dashboard refactor** — Chapter 14 visualization. Long-term future item.

### Code Cleanup
- [ ] **Dead code audit: MultiModelTrainer inline path** — Comment out (never delete per policy). Confirmed unreachable in subprocess mode but still present.
- [ ] **Remove 27 stale project files from Claude project** — Identified S85/S86 documentation audit. Stale files confuse AI context.

### Operational
- [ ] **Always launch pipeline in tmux** — Standard procedure going forward. Prevents loss of run if SSH drops. Command: `tmux new-session -s pipeline` before WATCHER launch. (Noted this session S101.)
- [ ] **Results/ FIFO cleanup automation** — Script or WATCHER post-run hook to prune old coordinator result files.

---

## ✅ RECENTLY COMPLETED (Reference)

| Item | Session | Status |
|------|---------|--------|
| Battery Tier 1A deployed (23 features: FFT, autocorr, cumsum, bitfreq) | S113 | ✅ COMPLETE |
| NPZ v3.1 format (24 metadata fields, 7 intersection fields) | S113 | ✅ COMPLETE |
| NPZ mandatory commit protocol established | S113 | ✅ COMPLETE |
| Real CA Daily 3 data (18,068 draws) — synthetic data retired | S112 | ✅ COMPLETE |
| 53 bidirectional survivors found W8_O43_S5-56 | S112 | ✅ COMPLETE |
| k_folds manifest implementation (default=20) | S100 | ✅ COMPLETE |
| enable_vmap rename (was batch_size_nn) | S99 | ✅ COMPLETE |
| WATCHER health retry stale diagnostics fix | S99 | ✅ COMPLETE |
| Phase 3A vmap batching certified | S99 | ✅ COMPLETE |
| S97 symlink fix (survivors_with_scores.json) | S97 | ✅ COMPLETE |
| persistent_workers manifest param-routing fix | S97 | ✅ COMPLETE |
| Phase 2.2 Optuna NN subprocess routing | S94 | ✅ COMPLETE |
| Category B Phase 2.1 single-shot NN subprocess | S92-S93 | ✅ COMPLETE |
| Strategy Advisor fully deployed + verified | S66/S68/S75 | ✅ COMPLETE |
| Optuna v3.4 restoration | S89 | ✅ COMPLETE |
| Soak Tests A, B, C | S86-S87 | ✅ COMPLETE |
| Chapter 13 full implementation | S57-S83 | ✅ COMPLETE |
| Phase 7 WATCHER integration | S57-S59 | ✅ COMPLETE |
| Chapter 14 Phases 1-4 | S69-S82 | ✅ COMPLETE |
| S96B persistent GPU workers | S96B | ✅ COMPLETE |
| NPZ v3.0 format | S80 | ✅ COMPLETE |
| Multi-model compare-models | S72 | ✅ COMPLETE |

---

## 🏗️ ARCHITECTURE INVARIANTS (Never Change Without Team Beta)

- Step order: 1→2→3→4→5→6 (static)
- Feature schema: 62 features, lexicographic ordering, hash-validated Step 5→6
- Authority separation: Chapter 13 decides, WATCHER executes, components cannot self-promote
- GPU isolation: Parent never initializes CUDA before NN subprocess spawn
- Manifest param governance: Every new CLI param MUST be declared in manifest default_params or WATCHER silently drops it
- Never restore from backup — always fix forward
- Comment out dead code, never delete

---

*Generated S101 — 2026-02-20 | Updated S114 — 2026-03-02*
*Added: PRNG Detection Research (XOR-shift, Berlekamp-Massey, bit avalanche)*
