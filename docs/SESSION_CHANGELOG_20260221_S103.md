# SESSION CHANGELOG — February 20-21, 2026 (S103)

**Focus:** Step 2 Architectural Deep Dive + Step 1 Root Cause Fix  
**Outcome:** Step 1 per-seed match rate bug found, fixed, tested, and deployed. Step 2 draw-history dependency confirmed as architectural failure. Composite objective designed. Team Beta proposal ready. Step 2 v4.0 blocked pending TB ruling on tautology risk.

---

## Summary

Session began as a Step 2 investigation. Deep historical search confirmed `scorer_trial_worker.py` has been evaluating survivors against draw history — something Step 2 was never designed to do. Recovered S101 changelog from Zeus confirming Spearman replaced neg-MSE but draw-history dependency was never removed. Exhaustive analysis of seven objective candidates selected a composite of Weighted Separation Index (70%) + IQR Ratio (30%).

During NPZ field inspection, a deeper root cause was discovered upstream in Step 1: `window_optimizer_integration_final.py` was discarding per-seed `match_rate` computed by the GPU sieve kernel and replacing it with trial-level aggregate counts. This meant all quality fields in the NPZ were identical for every seed from the same trial — zero signal for ML ranking. This was traced, confirmed, patched, tested, and deployed on Feb 21.

Step 2 v4.0 implementation remains blocked: further analysis revealed the proposed WSI objective is also tautological (same draw-match measurement as the sieve). Team Beta ruling required before proceeding.

---

## Part 1 — Step 2 Root Cause Analysis (Feb 20)

### Key Finding

Step 2's objective function has had two distinct but compounding problems:

**Problem 1 — Wrong metric** (partially addressed in S101):
neg-MSE was replaced with Spearman. Spearman is better than MSE, but still misaligned with Step 2's purpose — it requires two independent arrays with different ground truth to be meaningful, and in our context there is only one ground truth: the NPZ sieve data.

**Problem 2 — Wrong input** (never addressed until S103):
Both neg-MSE and Spearman were fed scores derived from literal draw matching (`prng_sequence == draw_value`). This is architecturally wrong from first principles. Step 2 was never designed to compare survivors against draws. The bidirectional sieve (Step 1) already did that. Re-doing it in Step 2 is redundant, and with synthetic data where the true seed may be outside the searched range, it always returns 0.0.

**The correct framing of Step 2:**

> Step 2 is a meta-optimizer. Its job is to find which values of `residue_mod_1/2/3`, `max_offset`, and `temporal_window_size` produce a scoring function that **most powerfully separates high-quality survivors from low-quality ones** — using the bidirectional sieve's own evidence (the NPZ fields) as the reference signal. No draw history. No literal matching. The sieve is the ground truth.

---

### S101 Context (Recovered from Zeus This Session)

S101 fixed two real bugs in v3.4 → v3.5:

1. **neg-MSE → Spearman** — MSE collapsed on low-variance distributions. Spearman is rank-based and more robust. Correct improvement.
2. **`random.seed(42)` → per-trial unique seed** — All 100 trials were evaluating the same 450 survivors. Fixed to `random.seed(params['optuna_trial_number'])`. **This fix is correct and is preserved in v4.0.**

**What S101 did NOT fix:** The draw-history dependency. Spearman was still being fed scores computed from `prng_sequence == draw_value`. The degenerate guard fired before Spearman ever ran (`std < 1e-12` because all scores were 0.0). S101 correctly diagnosed this as a separate problem and handed it to S102/S103.

**S103 conclusion on Spearman:** Even with the draw-history dependency removed, Spearman remains the wrong algorithm for Step 2's purpose.

---

### Architectural Analysis

#### The Redundancy Problem

| What | Current Step 2 | Chapter 13 |
|------|----------------|------------|
| Compare survivors to draws | ✅ (literal equality — always 0.0) | ✅ (post-draw diagnostics, hit rate) |
| Score survivors by draw match rate | ✅ (Optuna objective — broken) | ✅ (holdout_hits label refresh via Full Scoring re-run) |
| Produce ranked survivor weights | ✅ (broken — all zero) | ✅ (survivor performance → reinforce/decay) |

Step 2 has been duplicating Chapter 13's function — badly — while leaving its own actual function (parameter optimization) undone.

#### Clean Architectural Separation (Restored)

```
Step 1: Window Optimizer
        → Find optimal SIEVE PARAMETERS (window, skip, thresholds)
        → Output: bidirectional_survivors.json, optimal_window_config.json

Step 2: Scorer Meta-Optimizer
        → Find optimal SCORING PARAMETERS using SIEVE QUALITY as ground truth
        → Output: optimal_scorer_config.json
        → Ground truth: NPZ intrinsic fields (forward_matches, reverse_matches)
        → NO draw history. NO literal matching.

Step 3: Full Scoring
        → Apply optimal scorer config to ALL survivors (50 + 14 features per seed)
        → Output: survivors_with_scores.json

Step 4: ML Meta-Optimizer
        → Optimize ML architecture and hyperparameters via Bayesian search

Step 5: Anti-Overfit Training
        → Train final model with overfitting prevention

Step 6: Prediction Generator
        → Generate prediction pools using trained model

Ch. 13  → Live draw outcome feedback → label refresh → retrain trigger
```

---

### Algorithm Analysis — Objective Function Candidates

Seven candidates evaluated against alignment, Optuna landscape quality, and robustness:

| Candidate | Verdict |
|-----------|---------|
| neg-MSE | ❌ Retired S101 — collapses on low-variance distributions |
| Spearman | ❌ Misaligned — requires two independent ground truths; we have one |
| CV (std/mean) | ✅ Good but superseded by IQR Ratio |
| Shannon Entropy | ⚠️ Rejected — rewards noise and randomness |
| Gini Coefficient | ✅ Good but outlier-sensitive; superseded by IQR Ratio |
| IQR Ratio | ✅ Selected — secondary (30%), robust spread measure |
| Weighted Separation Index | ✅ Selected — primary (70%), uses sieve as ground truth |

**Selected composite:** `0.7 × WSI + 0.3 × IQR_Ratio`

---

### Newly Discovered: NN Params Polluting Step 2 Search Space

Live code inspection of `generate_scorer_jobs.py` confirmed the search space includes `hidden_layers`, `dropout`, `learning_rate`, `batch_size` — Step 5 (Anti-Overfit Training) parameters that have no effect in Step 2 and are polluting `optimal_scorer_config.json`.

Corrected v4.0 search space: `residue_mod_1/2/3`, `max_offset`, `temporal_window_size` only.

---

## Part 2 — Step 1 Root Cause & Fix (Feb 21)

### Discovery

During NPZ field inspection, a root cause bug was found upstream of Step 2. The NPZ on Zeus showed:

- Only 28 unique values across 37,846 survivors in `forward_matches` / `reverse_matches`
- `intersection_ratio = 1.0` for all survivors (degenerate)
- `bidirectional_selectivity = 1.0` for all survivors (degenerate)
- `score = 17126` for every seed from trial 40 (trial's total survivor count)

**Root cause traced through three files:**

**`sieve_filter.py`** — GPU kernel correctly computes per-seed `match_rate` (0.0–1.0) for every survivor. The data exists and is returned.

**`window_optimizer_integration_final.py`** — `extract_survivors_from_result()` discarded `match_rate` entirely, returning only seed integers. The accumulator then stamped trial-level aggregate counts onto every survivor:

```python
# BUG (v2.0): trial-level count stamped on every seed from same trial
'score': len(bidirectional_constant),   # same value for all 17,126 seeds
```

**`convert_survivors_to_binary.py`** — Mapped `forward_matches` ← `forward_count` and `reverse_matches` ← `reverse_count` (trial aggregates, not per-seed rates).

### Why This Matters

Per the mathematical whitepaper (Section 7), bidirectional survivors form a **manifold of near-consistent seeds** — `S_near = { s : d(s, s*) ≤ ε }`. ML's job is to rank within that manifold. The ranking signal is how strongly each seed passed the sieve — its individual `match_rate`. Without per-seed match rates, all survivors look identical and ML has no signal to learn from.

### The Fix

**`window_optimizer_integration_final.py` → v3.0**

`extract_survivors_from_result()` renamed to `extract_survivor_records()` — now returns `List[Dict[{seed, match_rate}]]` instead of `List[int]`. Legacy alias retained for compatibility. Accumulator now stores per-seed fields:

```python
# FIX (v3.0): per-seed match rate from GPU kernel
accumulator['bidirectional'].append({
    'seed': seed,
    'forward_match_rate': fwd_rate,        # 0.0-1.0 per seed
    'reverse_match_rate': rev_rate,        # 0.0-1.0 per seed
    'score': (fwd_rate + rev_rate) / 2.0,  # per-seed avg
    **metadata_base                         # trial context retained
})
```

**`convert_survivors_to_binary.py` → v3.1**

`forward_matches` and `reverse_matches` now map to `forward_match_rate` / `reverse_match_rate` (per-seed 0.0–1.0). Added percentage-based variance health check at conversion time — warns if unique value count < 10% of survivors, indicating old integration code was used.

### Caller Verification

Full grep on Zeus confirmed `extract_survivors_from_result()` is called by:
- `window_optimizer_integration_final.py` itself (patched)
- `prng_sweep_orchestrator.py`, `auto_fix_order.py`, `safe_three_lane_test.py`, `test_100k_bidirectional.py` — all use the legacy alias which returns `List[int]`, unchanged

`window_optimizer_integration_final_INTEGRATED.py` confirmed to be a stale generated artifact from `apply_integration.py` — not imported by anything active. Not patched.

### Deployment

```bash
# Backups created on Zeus before deployment
window_optimizer_integration_final.py.bak_20260221_pre_s103
convert_survivors_to_binary.py.bak_20260221_pre_s103
```

Files deployed via scp from ser8. Tested on Zeus:

**Test 1 — extract_survivor_records():**
```
Records returned: 4
  seed=1001  match_rate=0.82
  seed=1002  match_rate=0.51
  seed=1003  match_rate=0.93
  seed=1004  match_rate=0.67
Legacy alias returns: [1001, 1002, 1003, 1004]
PASS
```

**Test 2 — converter variance check:**
```
📊 forward_matches: min=0.5100 max=0.9300 unique=4
📊 reverse_matches: min=0.5500 max=0.8800 unique=4
✅ Good per-seed variance (4 unique values for 4 survivors)
✓ Conversion complete (v3.1.0)
PASS
```

---

## Step 2 v4.0 — Additional Blocker Identified (Feb 21)

After Step 1 fix, further analysis of `_vectorized_scoring_kernel` revealed:

```python
matches = (predictions == lottery_history_tensor.unsqueeze(0))
scores = matches.float().sum(dim=1) / history_len
```

`batch_score_vectorized` computes fraction of draw history positions where `prng(seed, i) == draw[i]` — a literal draw match rate. The sieve computes fraction of window positions where the seed's residue matches the draw residue. **Both are measuring the same thing** — how well the seed's sequence matches the draws.

**Team Beta's tautology warning is confirmed valid.** The proposed WSI objective optimizing `batch_score_vectorized` output would be a re-statement of the sieve's own ranking. Step 2 would optimize scorer params to agree with the sieve that created the pool — circular, no new information.

**Step 2 v4.0 implementation is blocked pending Team Beta ruling on a non-circular objective** that measures feature space quality independently of draw match rate.

---

## Team Beta Questions (Pending Ruling)

### Q1 — Composite objective tautology ruling *(primary ask, updated)*

Original proposal: `0.7 × WSI + 0.3 × IQR_Ratio`

**New concern (S103 Feb 21):** WSI uses `batch_score_vectorized` output as its separation signal. `batch_score_vectorized` computes literal draw match rate — same measurement as the sieve. This makes WSI tautological.

Team Beta to rule: what non-circular objective measures scorer parameter quality without restating the sieve ranking? Options to consider:
- Feature space variance / independence (PCA explained variance, pairwise correlation)
- Discriminative power on a held-out label not derived from the sieve
- IQR Ratio alone (on scorer features, not on raw scores)

### Q2 — Confirm draw history removal from Step 2 *(architectural confirmation)*

Confirm that `train_history.json` and `holdout_history.json` should be removed as Step 2 inputs entirely.

### Q3 — Remove neural network params from Step 2 search space

Confirm removal of `hidden_layers`, `dropout`, `learning_rate`, `batch_size` from `generate_scorer_jobs.py`. Rule on whether `temporal_num_windows` and `min_confidence_threshold` belong in Step 2.

### Q4 — S102 Carryover: `mod` field in `optimal_window_config.json` *(low priority)*

Should Window Optimizer (Step 1) write `mod` to its output config? Currently defaults to 1000 via fallback in Step 2.

---

## Work Completed This Session

| Item | Status |
|------|--------|
| Deep historical search — Chapter 3, whitepaper, changelogs | ✅ Complete |
| Confirmed original Chapter 3 design intent for Step 2 | ✅ Complete |
| Confirmed Chapter 13 redundancy with current Step 2 behavior | ✅ Complete |
| Recovered S101 changelog from Zeus | ✅ Complete |
| Exhaustive analysis of 7 objective function candidates | ✅ Complete |
| Selected composite: WSI (70%) + IQR Ratio (30%) | ✅ Complete |
| Team Beta proposal drafted (Q1–Q4) | ✅ Complete |
| NPZ field degeneracy investigation | ✅ Complete |
| Step 1 root cause confirmed in integration layer | ✅ Complete |
| `window_optimizer_integration_final.py` v3.0 patched + tested | ✅ Complete |
| `convert_survivors_to_binary.py` v3.1 patched + tested | ✅ Complete |
| Backups created, files deployed on Zeus | ✅ Complete |
| Caller verification — no breakage to downstream callers | ✅ Complete |
| WSI tautology risk identified and confirmed | ✅ Complete |
| Step 2 v4.0 correctly blocked pending TB ruling | ✅ Complete |

---

## Current System State

| Item | State |
|------|-------|
| `window_optimizer_integration_final.py` | v3.0 — deployed on Zeus |
| `convert_survivors_to_binary.py` | v3.1 — deployed on Zeus |
| `scorer_trial_worker.py` | v3.6 on Zeus + both rigs |
| `bidirectional_survivors_binary.npz` | Degenerate fields — needs Step 1 re-run |
| `bidirectional_survivors.json` | 37,846 survivors — needs Step 1 re-run |
| Scorer Meta-Optimizer (Step 2) | NOT running — blocked pending TB ruling |
| `optimal_window_config.json` | `prng_type=java_lcg`, `seed_count=100000` |

---

## Files Modified This Session

| File | Version | Change |
|------|---------|--------|
| `window_optimizer_integration_final.py` | v3.0 | Per-seed match rate fix |
| `convert_survivors_to_binary.py` | v3.1 | Map to per-seed match rates |
| `SESSION_CHANGELOG_20260221_S103.md` | NEW | This document |

---

## Next Session Starting Point

1. **Receive Team Beta ruling** on Q1 (tautology — non-circular objective) and Q2 (draw history removal)
2. **Re-run Step 1** with v3.0 integration to generate fresh NPZ with per-seed variance
3. **Verify NPZ** — confirm thousands of unique values in `forward_matches` / `reverse_matches`
4. **Implement Step 2 v4.0** only after TB ruling and Step 1 re-run confirmed

---

## Git Commit

```bash
cd ~/distributed_prng_analysis
git add window_optimizer_integration_final.py convert_survivors_to_binary.py
git add docs/SESSION_CHANGELOG_20260221_S103.md
git commit -m "fix+docs: S103 - Step 1 per-seed match rates fixed, Step 2 TB ruling pending

STEP 1 FIX (deployed Feb 21):
- window_optimizer_integration_final.py v3.0: extract_survivor_records()
  preserves per-seed match_rate from GPU sieve kernel. Accumulator stores
  forward_match_rate/reverse_match_rate per seed. score = avg(fwd+rev).
  Legacy alias extract_survivors_from_result() retained for compatibility.
- convert_survivors_to_binary.py v3.1: forward_matches/reverse_matches
  now map to per-seed match rates not trial-level aggregates.
  Percentage-based variance health check added.
- Tests: PASS on Zeus. Backups: *.bak_20260221_pre_s103

STEP 2 STATUS (blocked):
- Draw-history dependency confirmed as architectural failure
- Spearman confirmed misaligned with Step 2 purpose
- WSI+IQR composite designed but WSI identified as tautological
  (batch_score_vectorized measures same draw-match as sieve)
- Team Beta ruling required on non-circular objective before v4.0

PENDING: TB ruling on Q1 (non-circular objective), Q2 (draw history
removal), Q3 (NN params from search space), Q4 (mod field)."

git push origin main
git push public main
```

---

**END OF SESSION S103**
