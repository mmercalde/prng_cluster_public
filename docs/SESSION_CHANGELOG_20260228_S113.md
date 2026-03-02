# SESSION CHANGELOG: 2026-02-28 Session 113

**Date:** February 28, 2026
**Session:** 113
**Focus:** Battery Tier 1A deployment, NPZ data integrity discovery, pipeline re-run prep

---

## Work Completed

### 1. Battery Tier 1A — Deployed ✅

- `apply_s113_battery_tier1a.py` patcher created and run on Zeus
- `survivor_scorer.py` patched: 577 → 782 lines (+205)
- All 4 patches applied cleanly, smoke test passed (23 columns confirmed)
- Deployed to all 4 nodes: Zeus, rig-6600 (120), rig-6600b (154), rig-6600c (162)
- Committed: `a513dbb` — feat(S113): Battery Tier 1A — 23 statistical features

**Battery columns added:**
| Group | Columns | Count |
|-------|---------|-------|
| F1 Spectral FFT | batt_fft_peak_mag, batt_fft_secondary_peak, batt_fft_spectral_conc, batt_fft_diff_peak, batt_fft_diff_conc | 5 |
| F5 Autocorrelation | batt_ac_lag_01..10, batt_ac_decay_rate, batt_ac_sig_lag_count | 12 |
| F7 Cumulative Sum | batt_cs_max_excursion, batt_cs_mean_excursion, batt_cs_zero_crossings | 3 |
| F6 Bit Frequency | batt_bf_hamming_mean, batt_bf_hamming_std, batt_bf_popcount_bias | 3 |

### 2. train_history.json / holdout_history.json Created ✅

- Generated from daily3.json (18,068 draws), 80/20 chronological split
- train_history.json: 14,454 draws
- holdout_history.json: 3,614 draws
- Both gitignored (by design) — live on Zeus only

### 3. full_scoring.json manifest updated ✅

- min_scored_survivors: 100 → 10 (both occurrences)
- Committed: `293e860`

### 4. Step 3 Run — INVALID (see issues below)

- Step 3 ran successfully: 50/50 jobs, 87 features per survivor
- BUT: ran against synthetic NPZ (49,882 seeds, not 53 real survivors)
- Results discarded — survivors_with_scores.json is synthetic garbage

---

## Critical Issues Discovered

### Issue 1: NPZ Contains Synthetic Data 🔴

**Root cause:** `bidirectional_survivors_binary.npz` on Zeus dates to Feb 23
and contains 49,882 synthetic survivors (seeds 0-49,999, window_size 2-6).

The 53 real survivors from S112 were never committed to git. S112 produced
`bidirectional_survivors.json` but it was lost — gitignored and not on Zeus.

**Evidence:**
```
Survivor count: 49,882
Seeds: 0, 1, 2, 3, 4 ... 49,995-49,999  ← sequential = synthetic
Window sizes: {2: 48792, 4: 212, 3: 876}  ← NOT W8_O43
```

**Fix required:** Re-run Steps 1-2 against daily3.json to regenerate real survivors.
**Immediately after Step 1:** Force-commit the NPZ before anything else.

### Issue 2: NPZ Never Auto-Committed After Step 1 🔴

`bidirectional_survivors_binary.npz` IS tracked in git (committed historically),
but `*.json` is gitignored. After Step 1 runs and writes a fresh NPZ, it must
be manually force-committed or the synthetic version remains in git as a trap.

**Mandatory protocol going forward:**
```bash
# Immediately after Step 1 completes:
git add -f bidirectional_survivors_binary.npz
git commit -m "data(SXX): Real survivor NPZ — W8_O43 XX survivors"
git push
```

### Issue 3: Step 3 Run Was on Synthetic Data 🔴

The Step 3 run this session (27 minutes, 49,882 survivors scored, 87 features)
was entirely on synthetic data. All results discarded. survivors_with_scores.json
is invalid and must be regenerated after Steps 1-2 re-run.

---

## Lessons Learned

### Deploy Sequence Error

Correct sequence for patching:
1. Copy patcher to Zeus via scp from ser8
2. Run patcher on Zeus (creates backup THEN patches)
3. Only after patcher confirms success, deploy to rigs

This session: deploy commands were given before patcher ran, causing files to
be deployed without backup. Git saved us — original was recoverable via
`git checkout survivor_scorer.py`.

### rig-6600c Omission

Deploy commands must always include all 4 nodes:
- Zeus (localhost)
- 192.168.3.120 (rig-6600)
- 192.168.3.154 (rig-6600b)
- 192.168.3.162 (rig-6600c) ← was missing from patcher output, must always be added

---

## Git Commits This Session

| Hash | Description |
|------|-------------|
| `a513dbb` | feat(S113): Battery Tier 1A — 23 statistical features |
| `293e860` | config(S113): Lower min_scored_survivors threshold to 10 |

---

## Open Issues (Updated Master List)

### 🔴 CRITICAL — Blocks Real Data Pipeline

| # | Item | Status |
|---|------|--------|
| 1 | Re-run Steps 1-2 against daily3.json | NEXT ACTION |
| 2 | Force-commit NPZ immediately after Step 1 | MANDATORY PROTOCOL |
| 3 | Re-run Step 3 against real 53 survivors | After Steps 1-2 |
| 4 | Re-run Steps 4-6 on real data | After Step 3 |

### 🔴 HIGH — Known Fixes Ready

| # | Item | Notes |
|---|------|-------|
| 5 | NN y-normalization fix (train_single_trial.py line 499) | 30 min fix, known |
| 6 | TB guardrails G1-G6 | Approved, pending implementation |

### 🟡 MEDIUM — After Real Data Pipeline

| # | Item | Notes |
|---|------|-------|
| 7 | Phase 3B: Tree parallel workers (2 trees/GPU) | HIGH priority after real data — trees are 93% of Step 5 wall clock. Team Beta approval required before implementation. Proposal: PROPOSAL_PHASE3_CONCURRENT_TRIAL_BATCHING_v1_0.md |
| 8 | Phase 3A: NN concurrent batching (CUDA streams) | After Phase 3B validated. Pack 3-5 SurvivorQualityNet per GPU |
| 9 | Benchmark Phase 3 on real data only | Cannot benchmark until real survivors in pipeline |
| 10 | GlobalStateTracker regime enhancement | After battery features on real data |
| 11 | Chapter 13 activation on real data | After 1B survivors available + battery features |
| 12 | Step 2 scorer re-optimization with more survivors | After 1B run |
| 13 | Feature importance in sidecar (G5) | Pending |
| 14 | sklearn warnings Step 5 | Since S109 |
| 15 | Battery Tier 1B (Runs Analysis + Linear Complexity, 6 cols) | After Tier 1A validated on real data |

### 🟢 LOW — Deferred

| # | Item | Notes |
|---|------|-------|
| 16 | S110 root cleanup (884 files) | Deferred |
| 17 | Remove dead CSV writer coordinator.py | Deferred |
| 18 | S103 Part 2 | Deferred |
| 19 | Phase 9B.3 heuristics | Deferred |
| 20 | GPU4/5 failure pattern on rigs | Fails then retries on Zeus successfully — not blocking |

---

## Next Session Start

```bash
# Step 1 — Regenerate real survivors
PYTHONPATH=. python3 agents/watcher_agent.py \
    --run-pipeline --start-step 1 --end-step 1 \
    --params '{"lottery_file": "daily3.json"}'

# IMMEDIATELY after Step 1 completes — commit NPZ before Step 2
git add -f bidirectional_survivors_binary.npz
git commit -m "data(S114): Real survivor NPZ — W8_O43 real CA Daily 3 survivors"
git push

# Step 2
PYTHONPATH=. python3 agents/watcher_agent.py \
    --run-pipeline --start-step 2 --end-step 2 \
    --params '{"lottery_file": "daily3.json"}'

# Step 3 — real survivors this time
PYTHONPATH=. python3 agents/watcher_agent.py \
    --run-pipeline --start-step 3 --end-step 3 \
    --params '{"lottery_file": "daily3.json"}'
```

---

*Session 113 — Team Alpha*
*Battery Tier 1A deployed. NPZ synthetic data trap identified and documented.*
*Real data pipeline re-run required before any results are valid.*
