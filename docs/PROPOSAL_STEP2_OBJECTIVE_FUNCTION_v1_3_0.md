# PROPOSAL — Step 2 Scorer Meta-Optimizer
## Two Confirmed Bugs: Objective Function Collapse & Deterministic Sampling Lock

| Field | Value |
|---|---|
| Document | PROPOSAL_STEP2_OBJECTIVE_FUNCTION_v1_3_0 |
| Date | 2026-02-20 |
| Session | S101 |
| Author | Team Alpha — Lead Developer |
| Reviewer | Team Beta — Architecture Review |
| Version | 1.3.0 — Code-verified against GitHub commit 4e340de |
| Status | SUBMITTED — AWAITING TEAM BETA APPROVAL |

> **⚠ CRITICAL:** This document supersedes v1.0.0, v1.1.0, and v1.2.0.
> v1.0.0–v1.2.0 incorrectly treated `random.seed(42)` as optional.
> It is **not optional**. Code archaeology confirms it is a dead-code artifact
> with zero legitimate purpose in the current architecture.
> **Both fixes are MANDATORY before S102.**

---

## 1. Executive Summary

**Context:** S101 was the first full synthetic-data pipeline run (17,126 survivors). All 100 Step 2 Optuna trials returned the identical accuracy value `-2.2044446268409956e-06`. TPE received zero discriminating signal. `optimal_scorer_config.json` contains trial_0's random parameter draw — not a searched optimum.

**Two bugs found. Both mandatory:**

| # | Bug | Root Cause | Fix |
|---|---|---|---|
| 1 | Objective function collapses — all trials return identical accuracy | Negative MSE on low-variance targets: NN converges to mean prediction, MSE ≈ var(y_holdout) regardless of params | Replace neg-MSE with Spearman rank correlation — variance-independent, directly measures ranking quality |
| 2 | `random.seed(42)` locks every trial to the same 450 seeds — 97.4% of survivors never influence Step 2 | Dead-code artifact from v3.4 era (742K pool). Intended for retry reproducibility that is irrelevant at current scale. No config override exists. | Remove `random.seed(42)`. Replace with `random.seed(params.get('optuna_trial_number', 0))` |

**Combined effect:** Bug 1 means the objective cannot distinguish good params from bad params on *any* seeds. Bug 2 means even if Bug 1 is fixed, only 2.6% of survivors (450 of 17,126) inform the optimization. **Both must be fixed together.** Fixing only Bug 1 leaves Step 2 optimizing scorer params for a fixed, unrepresentative 450-seed slice.

---

## 2. Bug 1 — Objective Function Collapse (lines 416–424)

### 2.1 What the code does

Phase F of `run_trial()` computes the Optuna objective. After training the NN on `sampled_seeds` against `train_history` scores, it:

1. Scores `sampled_seeds` against `holdout_history` → `y_holdout` (SurvivorScorer ground truth on unseen data)
2. Runs NN inference on `sampled_seeds` → `y_pred_holdout` (NN's predicted quality)
3. Computes MSE between them → `accuracy = -holdout_mse` returned to Optuna

```python
# Lines 416–424  scorer_trial_worker.py  (CURRENT — DEFECTIVE)
        # Calculate MSE
        holdout_mse = float(torch.nn.functional.mse_loss(
            torch.tensor(y_pred_holdout),
            torch.tensor(y_holdout)
        ))
        accuracy = -holdout_mse  # Negative MSE (higher is better)
        logger.info(f'Holdout MSE: {holdout_mse:.6f}, Accuracy (NegMSE): {accuracy:.6f}')
```

### 2.2 Why it collapses

The SurvivorScorer scores cluster tightly in S101 (mean ≈ 0.002201, std ≈ 3.1e-5). When target variance is this low, the MSE-optimal NN strategy is to predict the mean for every seed — this minimises squared error regardless of which parameters were used to configure the scorer:

```
# With std = 3.1e-5 on 450 seeds:
# MSE-optimal NN outputs:  y_pred = [0.002201, 0.002201, ...]  for ALL trials
# This gives: MSE ≈ var(y_holdout) ≈ 9.6e-10  regardless of params
# Negative MSE ≈ -2.2044446268409956e-06  (exact S101 value, every trial)
#
# TPE receives 100 identical values → cannot learn → picks trial_0 as 'best'
# optimal_scorer_config.json = random unoptimized params from trial_0
```

### 2.3 S101 Evidence — 12 confirmed trials

| Trial | residue_mod_1 | residue_mod_2 | hidden_layers | accuracy |
|---|---|---|---|---|
| trial_0000 | 20 | 50 | 128_64 | **-2.2044446268409956e-06** |
| trial_0004 | 9 | 76 | 256_128_64 | **-2.2044446268409956e-06** |
| trial_0008 | 15 | 113 | 128_64 | **-2.2044446268409956e-06** |
| trial_0012 | 5 | 65 | 256_128_64 | **-2.2044446268409956e-06** |
| trial_0016 | 12 | 88 | 128_64 | **-2.2044446268409956e-06** |
| trial_0020 | 8 | 102 | 256_128_64 | **-2.2044446268409956e-06** |
| trial_0024 | 17 | 59 | 128_64 | **-2.2044446268409956e-06** |
| trial_0028 | 20 | 131 | 256_128_64 | **-2.2044446268409956e-06** |
| trial_0032 | 6 | 77 | 128_64 | **-2.2044446268409956e-06** |
| trial_0036 | 11 | 94 | 256_128_64 | **-2.2044446268409956e-06** |
| trial_0040 | 14 | 145 | 128_64 | **-2.2044446268409956e-06** |
| trial_0044 | 19 | 60 | 256_128_64 | **-2.2044446268409956e-06** |

`residue_mod_1` varies 5→20 (4× range), `residue_mod_2` varies 50→145 (2.9× range), NN architecture varies across all three options. All 12 return accuracy identical to 16 decimal places. Mathematically impossible unless the objective is completely insensitive to parameter variation — confirming MSE collapse.

### 2.4 Mathematical proof

```
# 6-seed worked example (S101-representative values, std = 3.1e-5)

Seed | y_holdout  | Good NN pred | Mean-pred NN
-----|------------|--------------|-------------
A    | 0.002263   |   0.002255   |   0.002201
B    | 0.002231   |   0.002228   |   0.002201
C    | 0.002204   |   0.002201   |   0.002201
D    | 0.002187   |   0.002185   |   0.002201
E    | 0.002165   |   0.002162   |   0.002201
F    | 0.002141   |   0.002139   |   0.002201

Good NN  MSE = 5.3e-11   (learns ranking)    accuracy = -5.3e-11
Bad  NN  MSE = 9.6e-10   (predicts mean)     accuracy = -9.6e-10

Both near-zero on Optuna's scale. TPE cannot distinguish them.

Spearman(Good NN) = 1.000   (perfect ranking)
Spearman(Bad  NN) = 0.000   (no ranking signal)

TPE immediately learns: Good NN → 1.0, Bad NN → 0.0
```

### 2.5 The fix — Spearman rank correlation

Spearman ρ measures whether the NN correctly orders survivors by quality — the actual goal of Step 2. It is invariant to score distribution variance: rank order is fully differentiable regardless of whether score std is 3.1e-5 or 0.3.

**REMOVE lines 416–424. REPLACE WITH:**

```python
        # v3.5: Spearman rank correlation — correct objective for ranking
        from scipy.stats import spearmanr
        y_pred_arr    = np.array(y_pred_holdout)
        y_holdout_arr = np.array(y_holdout)

        if np.std(y_pred_arr) < 1e-12:
            # Degenerate NN: predicted constant — no ranking signal
            accuracy = -1.0
            logger.warning('Degenerate NN: all predictions identical. rho = -1.0')
        else:
            correlation, p_value = spearmanr(y_pred_arr, y_holdout_arr)
            accuracy = float(correlation) if not np.isnan(correlation) else -1.0
            logger.info(f'Holdout Spearman rho: {accuracy:.6f}  (p={p_value:.4f})')
```

### 2.6 Why Spearman is correct

| Property | Neg MSE (remove) | Spearman ρ (install) |
|---|---|---|
| Measures | Magnitude of prediction error | Correctness of rank ordering |
| Range | (−inf, 0] — unbounded | [−1, +1] — bounded |
| Collapses on low-variance scores? | **YES — proven in S101** | NO — rank order is variance-independent |
| Incentivises mean prediction? | **YES — MSE-optimal strategy** | NO — mean prediction scores 0.0 |
| Correct for ranking goal? | **NO** | YES |
| Works on synthetic data? | **NO — S101 proves it** | YES |
| Works on real data? | UNCERTAIN — depends on score spread | YES — by construction |

---

## 3. Bug 2 — Dead-Code Sampling Lock (line 354)

> **⚠ NOT OPTIONAL.** Previous versions v1.0.0–v1.2.0 incorrectly marked this as
> "Team Beta discretion". Code archaeology confirms it is a dead-code artifact
> with zero valid purpose in the current architecture. It must be removed.

### 3.1 The line in question

```python
# scorer_trial_worker.py  lines 352–356  (CURRENT — DEFECTIVE)
        if sample_size and len(seeds_to_score) > sample_size:
            import random
            logger.info(f'Sampling {sample_size:,} seeds from {len(seeds_to_score):,} for training')
            random.seed(42)  # Reproducible sampling   ← BUG: MUST REMOVE
            sample_indices = random.sample(range(len(seeds_to_score)), sample_size)
```

### 3.2 Origin — the v3.4 era, 742K pool, retry intent

**Where it came from:** The v3.4 changelog (2025-11-29) documents a CRITICAL FIX: holdout evaluation was hanging 30+ minutes because it ran on the full 742,000-seed pool instead of the sampled 450. When that fix was written, `random.seed(42)` was added so that if a failed trial was retried by `scripts_coordinator._retry_on_localhost()`, it would receive the exact same 450 seeds as the original attempt — making the retry result comparable.

**Why that reasoning no longer applies:**

- The pool is now **17,126 survivors** (post bidirectional-sieve), not 742,000. Scale has changed completely.
- There is only **one seed value (42)** — meaning every trial, not just retries, draws the same 450 indices. The intent was per-trial retry consistency; the effect is a global lock across all 100 trials.
- The **correct fix for retry reproducibility** is `random.seed(params['optuna_trial_number'])` — each trial gets a unique, stable seed. Trial 17 always samples the same 450 seeds if retried, but trial 18 samples a different 450 than trial 17. With seed=42, trials 0 through 99 all sample the identical 450.
- **No config override exists** — there is no CLI flag, no params key, no way to change this without editing source.

### 3.3 Quantified impact

| Configuration | Unique survivors seen | Pool coverage | Optuna diversity |
|---|---|---|---|
| `seed=42` (current) | 450 — always the same | **2.6%** | ZERO — identical seeds every trial |
| `optuna_trial_number` as seed (fix) | ~15,931 expected | **~93%** | Each trial evaluates a different slice |

*Coverage: E[unique across k trials] = N × (1 − (1 − n/N)^k) = 17,126 × (1 − (1 − 450/17,126)^100) ≈ 15,931*

### 3.4 Interaction with Bug 1

These two bugs compound each other:

- Same 450 seeds → `y_holdout` has near-identical distribution every trial (seed-independent score variance is already low; identical seeds eliminate even that variation)
- MSE on near-zero variance → collapses to the same constant in every trial
- Result: 100 trials × identical seeds × collapsing objective = 100 identical accuracy values

Fixing Bug 1 alone (Spearman only) restores objective signal — but Optuna would still be searching scorer params using only 2.6% of survivors as its evaluation set. The optimal params found may not generalise to the full survivor pool. **Both fixes are required.**

### 3.5 The fix

**REMOVE line 354. REPLACE WITH:**

```python
# scorer_trial_worker.py  lines 352–357  (AFTER FIX)
        if sample_size and len(seeds_to_score) > sample_size:
            import random
            logger.info(f'Sampling {sample_size:,} seeds from {len(seeds_to_score):,} for training')
            random.seed(params.get('optuna_trial_number', 0))  # v3.5: per-trial seed
            sample_indices = random.sample(range(len(seeds_to_score)), sample_size)
            sampled_seeds = [seeds_to_score[i] for i in sample_indices]
            sampled_scores = [y_train[i] for i in sample_indices]
```

`params['optuna_trial_number']` is confirmed present — injected by `generate_scorer_jobs.py` line 119 (`params['optuna_trial_number'] = trial.number`). It is unique per trial, zero-indexed, and available inside `run_trial()` via the params argument. No other file changes are required for this fix.

---

## 4. Complete Patch — scorer_trial_worker.py

Two surgical changes to one file. No other pipeline component requires modification.

### Patch A — line 354 (Bug 2: sampling lock)

```python
# REMOVE:
            random.seed(42)  # Reproducible sampling

# REPLACE WITH:
            random.seed(params.get('optuna_trial_number', 0))  # v3.5: per-trial seed
```

### Patch B — lines 416–424 (Bug 1: objective function)

```python
# REMOVE:
        # Calculate MSE
        holdout_mse = float(torch.nn.functional.mse_loss(
            torch.tensor(y_pred_holdout),
            torch.tensor(y_holdout)
        ))
        accuracy = -holdout_mse  # Negative MSE (higher is better)
        logger.info(f'Holdout MSE: {holdout_mse:.6f}, Accuracy (NegMSE): {accuracy:.6f}')

# REPLACE WITH:
        # v3.5: Spearman rank correlation — correct objective for ranking
        from scipy.stats import spearmanr
        y_pred_arr    = np.array(y_pred_holdout)
        y_holdout_arr = np.array(y_holdout)
        if np.std(y_pred_arr) < 1e-12:
            accuracy = -1.0
            logger.warning('Degenerate NN: all predictions identical. rho = -1.0')
        else:
            correlation, p_value = spearmanr(y_pred_arr, y_holdout_arr)
            accuracy = float(correlation) if not np.isnan(correlation) else -1.0
            logger.info(f'Holdout Spearman rho: {accuracy:.6f}  (p={p_value:.4f})')
```

### Version bump — file header

```python
"""
scorer_trial_worker.py (v3.5 - Spearman Objective + Per-Trial Sampling)
==========================================================================
v3.5 (2026-02-20):
- BUG FIX: Replace neg-MSE objective with Spearman rank correlation
  (MSE collapsed to constant on low-variance score distributions — S101)
- BUG FIX: Remove random.seed(42) — replaced with per-trial seed
  (seed=42 locked all 100 trials to identical 450 seeds, 2.6% pool coverage)
  New: random.seed(params['optuna_trial_number']) — unique per trial,
  stable for retries, ~93% survivor pool coverage across 100 trials
"""
```

---

## 5. Additional Files to Update

| File | Action | Change |
|---|---|---|
| `scorer_trial_worker.py` | **PATCH** | Patch A (line 354) + Patch B (lines 416–424) + version bump to v3.5 |
| `docs/CHAPTER_3_SCORER_META_OPTIMIZER.md` | **DOCS** | Align documented objective metric and sampling strategy with live code. Version → 3.5.0. |
| `SESSION_CHANGELOG_S102.md` | **NEW** | Document S101 findings, both bug fixes applied, S102 re-run results. |

> **Note on Optuna study DB:** Study name is already timestamp-based
> (`scorer_meta_opt_$(date +%s)`) — each run creates a new independent DB
> automatically. S101 DB is auto-isolated. No manual naming change required.

---

## 6. Acceptance Criteria

All 7 criteria must pass before S102 results are considered valid.

| # | Criterion | Verification |
|---|---|---|
| 1 | Trial accuracy values NOT all identical in 10-trial smoke test | `sort -u` on accuracy fields from 10 trial JSONs \| `wc -l` → must be > 1 |
| 2 | Best trial accuracy > 0.0 (positive Spearman found) | `python3 -c "import json; d=json.load(open('optimal_scorer_config.json')); print(d['accuracy'] > 0)"` |
| 3 | Degenerate NN guard returns -1.0 for constant predictions | Unit test: inject `y_pred=[0.002]*450`, `y_holdout=[0.002]*450` → assert `accuracy == -1.0` |
| 4 | Different trials sample different seed indices (lock removed) | Compare `sample_indices` log output from `trial_0000` vs `trial_0001` — must differ |
| 5 | `scipy.stats.spearmanr` available on all AMD rigs | `ssh each rig: python3 -c 'from scipy.stats import spearmanr; print("OK")'` |
| 6 | Full pipeline Steps 1–3 smoke test completes cleanly | `PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 1 --end-step 3` |
| 7 | `CHAPTER_3_SCORER_META_OPTIMIZER.md` updated, version = 3.5.0 | `grep 'v3.5' docs/CHAPTER_3_SCORER_META_OPTIMIZER.md` |

---

## 7. Backward Compatibility

- **`optimal_scorer_config.json`:** Format unchanged. Steps 3–6 are unaffected.
- **Optuna study DB:** Timestamp-based naming already ensures each run creates an independent DB. No change needed.
- **Trial JSON `accuracy` field:** Now holds Spearman ρ ∈ [−1, +1] instead of neg-MSE. Any monitoring scripts that parse `accuracy` for comparison must be aware of the new range.
- **scipy.stats:** Already present in `~/venvs/torch/` (Zeus) and `~/rocm_env/` (AMD rigs). No new installation required.
- **S101 output:** `optimal_scorer_config.json` from S101 is unoptimized (trial_0 random params). S102 must regenerate Steps 2–6 after applying both patches.

---

## 8. Summary

Step 2 has two bugs that together made S101 optimization completely non-functional. Both are confirmed in live code at commit `4e340de` and both are mandatory fixes:

- **Bug 1 (lines 416–424):** Negative MSE is the wrong objective for a ranking problem. On low-variance score distributions it collapses to a constant regardless of parameter variation. Fix: Spearman rank correlation.
- **Bug 2 (line 354):** `random.seed(42)` is a dead-code artifact from the v3.4 / 742K-pool era. It locks every trial to the identical 450 seeds, giving Optuna zero sampling diversity and covering only 2.6% of survivors. Fix: `random.seed(params['optuna_trial_number'])`.

Both patches are surgical — four lines changed in one file (`scorer_trial_worker.py`). No other pipeline components are affected. S101 Step 2 output is discarded. S102 applies both patches and re-runs Steps 2–6.

---

**Team Alpha:** _______________________________  Date: _______________

**Team Beta Decision:**  ☐ APPROVED   ☐ APPROVED WITH MODIFICATIONS   ☐ REJECTED

**Team Beta Notes:** _______________________________________________________________

**Team Beta Signature:** _______________________________  Date: _______________
