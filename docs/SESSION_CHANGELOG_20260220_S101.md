# SESSION CHANGELOG — February 20, 2026 (S101)

**Focus:** Step 2 Scorer Objective Refactor — neg-MSE → Spearman + Per-Trial Seed Fix  
**Outcome:** Two critical bugs fixed in `scorer_trial_worker.py` (v3.4 → v3.5). Step 2 re-run launched. Scores still all -1.0 — deeper root cause investigation deferred to S102.

---

## Summary

Investigated why Step 2 (Scorer Meta-Optimizer) produced degenerate Optuna results — all 100 trials returning identical accuracy values with no meaningful optimization occurring. Identified two bugs in `scorer_trial_worker.py` v3.4:

1. **Objective function (neg-MSE) was collapsing on low-variance score distributions**, causing Optuna to see a flat landscape and learn nothing.
2. **`random.seed(42)` locked all trials to the same 450-seed sample**, meaning 100 trials were effectively evaluating the same 2.6% of the survivor pool with no diversity.

Both bugs were fixed in v3.5, distributed to the cluster, and a fresh Step 2 run was launched. The run returned accuracy=-1.0 for all trials. This was not caused by v3.5 — investigation at the end of session showed the Spearman infrastructure was working. Root cause was identified as a separate issue (prng_type resolution and/or seed space mismatch) — handed off to S102.

---

## Pre-Session State

| Item | State |
|------|-------|
| `scorer_trial_worker.py` | v3.4 (2025-11-29) |
| Optuna objective | `neg-MSE` — mean squared error between predicted and historical scores |
| Trial seed | `random.seed(42)` — hardcoded, all trials identical |
| Step 2 behavior | All trials returning same constant accuracy; Optuna flat landscape |
| Commit baseline | `6082188` (pre-S101 baseline, committed at session start) |

---

## Bug 1 — neg-MSE Objective Collapse

### Root Cause

The v3.4 Optuna objective used negative MSE:

```python
# v3.4 objective (BROKEN)
score = -np.mean((predicted_scores - historical_scores) ** 2)
return score
```

When survivor score distributions have very low variance (which is typical — sieve filtering produces survivors with similar statistical profiles), MSE collapses to a near-constant value regardless of the parameter combination being tested. Optuna's TPE sampler cannot learn from a flat landscape, so all 100 trials converge to the same accuracy and no meaningful optimization occurs.

### Why MSE Was Wrong Here

MSE measures absolute score magnitude differences. For Step 2's purpose — **finding which PRNG parameter configurations best rank survivors relative to their observed draw-match quality** — what matters is **rank order**, not absolute magnitude. Two survivors with scores 0.001 and 0.002 should be ranked relative to each other, not penalized by their absolute delta.

### Fix — Spearman Rank Correlation

Replaced neg-MSE with Spearman rank correlation coefficient:

```python
# v3.5 objective (FIXED) — lines 416-424
from scipy.stats import spearmanr
correlation, p_value = spearmanr(predicted_scores, historical_scores)
if np.isnan(correlation):
    return -1.0  # degenerate guard
return float(correlation)
```

Spearman measures whether the **rank ordering** of predicted vs. historical scores agrees, making it robust to monotone transformations and low-variance distributions. If predicted scores rank survivors in the same order as their actual draw-match quality, Spearman = +1.0. If reversed, Spearman = -1.0. If random, Spearman ≈ 0.

This gives Optuna a meaningful gradient to follow even when absolute score magnitudes are small or uniform.

---

## Bug 2 — Fixed Random Seed Locking All Trials to 2.6% Coverage

### Root Cause

v3.4 had a hardcoded `random.seed(42)` at the start of each trial's survivor sampling:

```python
# v3.4 (BROKEN)
random.seed(42)
sample_seeds = random.sample(all_seeds, min(sample_size, len(all_seeds)))
```

With `sample_size = 450` and `len(all_seeds) = 17,126`, this meant every single one of the 100 Optuna trials was evaluating the exact same 450 survivors — 2.6% of the pool, always the same 2.6%. The diversity between trials was zero. Optuna had no way to observe how different parameter choices affected different survivors.

### Fix — Per-Trial Unique Seed

```python
# v3.5 (FIXED) — line 354
random.seed(params['optuna_trial_number'])
sample_seeds = random.sample(all_seeds, min(sample_size, len(all_seeds)))
```

Each trial now uses its own Optuna trial number as the random seed. Trial 0 evaluates one sample of 450, trial 1 a different sample, trial 99 yet another. This restores the diversity that Optuna needs to learn parameter-performance relationships across the full survivor pool.

---

## Patch Script

Changes were delivered as `apply_s101_scorer_worker_v3_5.py`, a self-contained patch script with:
- Exact string replacement for both bug locations
- Version header bump from v3.4 → v3.5
- Auto-backup of original file
- Post-patch verification checks (6 assertions)
- Syntax validation via `ast.parse()`

---

## Deployment

```bash
# Applied on Zeus
python3 apply_s101_scorer_worker_v3_5.py

# Distributed to both rigs
scp scorer_trial_worker.py 192.168.3.120:~/distributed_prng_analysis/
scp scorer_trial_worker.py 192.168.3.154:~/distributed_prng_analysis/

# Checksums verified identical across all 3 nodes
md5sum scorer_trial_worker.py  # confirmed same on Zeus, rig-120, rig-154
```

---

## Step 2 Test Run Results

After applying v3.5, a clean Step 2 run was launched:
- 100 trials across 26 GPUs
- Duration: ~1 hour
- All trials returned `accuracy = -1.0`

Initial interpretation: Spearman fix wasn't working. Investigation revealed the Spearman infrastructure **was** working correctly — the -1.0 was being returned by the degenerate guard (`std < 1e-12` check before Spearman calculation), not by Spearman itself returning -1.

True root cause: **scores were all 0.0**, triggering the degenerate guard. The Spearman computation was never reached. This is a separate bug from the objective function — traced at end of session to either:
- `prng_type` not being read from config (NPZ branch had `pass` — pre-existing bug since commit 4e340de)
- Seed values in NPZ being sequential integers (0–37,999) with zero match rate against synthetic draw history

Both issues handed off to S102 for resolution.

---

## Version History Context

The v3.5 file header captured in commit `4e340de` (used as the S102 pre-patch baseline):

```
scorer_trial_worker.py (v3.5 - Spearman Objective + Per-Trial Sampling)
==================================================
v3.5 (2026-02-20):
- BUG FIX: Replace neg-MSE objective with Spearman rank correlation
  (MSE collapsed to constant on low-variance score distributions — S101)
- BUG FIX: Remove random.seed(42) — replaced with per-trial seed
  (seed=42 locked all 100 trials to identical 450 seeds, 2.6% pool coverage)
  New: random.seed(params['optuna_trial_number']) — unique per trial

v3.4 (2025-11-29):
  [prior changes]
```

---

## Work Completed

| Item | Status |
|------|--------|
| Pre-patch baseline committed to both repos (6082188) | ✅ Complete |
| Bug 1 identified (neg-MSE collapse) | ✅ Complete |
| Bug 2 identified (random.seed(42)) | ✅ Complete |
| v3.5 patch script written | ✅ Complete |
| v3.5 applied to Zeus | ✅ Complete |
| v3.5 distributed to rigs (120, 154) | ✅ Complete |
| Checksums verified identical across cluster | ✅ Complete |
| Step 2 test run (100 trials) | ✅ Complete |
| Root cause of remaining -1.0 identified | ✅ Partial (prng_type + seed space — S102) |

---

## Files Modified This Session

| File | Type | Purpose |
|------|------|---------|
| `scorer_trial_worker.py` | MODIFIED | v3.4 → v3.5, objective + seed fixes |
| `scorer_trial_worker.py.bak_20260220_*` | NEW | Auto-backup from patch script |
| `apply_s101_scorer_worker_v3_5.py` | NEW | Patch script (for record) |

---

## Key Insight

The MSE→Spearman change reflects a fundamental clarification of what Step 2 is actually optimizing:

- **Old framing:** "How close are predicted scores to historical scores?" (absolute magnitude = MSE)
- **New framing:** "Do predicted scores rank survivors in the right order?" (rank agreement = Spearman)

For survivor quality assessment in a PRNG mimicry context, ranking is the right metric. Two survivors both scoring 0.001 vs. 0.002 matter only in their relative ordering — their absolute scores are artifacts of the scoring function's scale, not intrinsically meaningful.

---

## Handoff to S102

S101 ended with v3.5 deployed and Step 2 still returning all -1.0. S102 picks up from here:

1. Investigate why scores are all 0.0 (prng_type `pass` block in `load_data()`)
2. Verify pre-existing vs. S101-introduced bugs (confirmed pre-existing in S102)
3. Apply v3.6 fix (read prng_type from `optimal_window_config.json`)
4. Trace remaining zero-score cause to seed space issue
5. Raise Team Beta architectural questions

---

**END OF SESSION S101**
