# S111 Phase 1 Results — Team Beta Briefing
## Date: 2026-02-25 | Session: S111 (continued)

---

## 1. Phase 1 Run Complete — holdout_quality Target Live

Steps 5→6 re-ran with `holdout_quality` as the ML target (commit `1cb90aa`).
All 49,882 survivors have 100% `holdout_quality` coverage.

### Results vs Phase 0 Baseline

| Model     | Phase 0 R² (holdout_hits) | Phase 1 R² (holdout_quality) | Change |
|-----------|---------------------------|------------------------------|--------|
| LightGBM  | ~0.000264                 | **0.00265**                  | ~10×   |
| XGBoost   | —                         | 0.0022                       | —      |
| CatBoost  | ~0.000155                 | 0.0017                       | ~11×   |

**Winner:** LightGBM (R² = 0.00265). Signal classified as "weak" but directionally correct — the target switch from Poisson noise to continuous composite gave the models real gradient.

**However:** These trials were contaminated. Optuna resumed from trial 60 (the first 60 were optimized against the old `holdout_hits` target). TPE sampler was biased toward a stale landscape. Fresh 200-trial run pending.

---

## 2. holdout_quality Distribution — Narrow But Real

```
n:        49,882
min/max:  0.2120 → 0.2312  (range ≈ 0.019)
std:      0.00221
unique:   10,118
```

The target is continuous with 10K+ unique values (not collapsed), but the range is extremely narrow. This naturally caps achievable R² — even a good model predicting tiny deltas will show low R². **Rank-based metrics (Spearman, top-decile lift) should supplement R² going forward.**

---

## 3. Feature Importance — Corrected Column Mapping

TB correctly identified that the original Column_* mapping was off by the exclude-list offset. Corrected top features:

| Rank | Feature                    | Splits | % Total |
|------|----------------------------|--------|---------|
| 1    | residue_8_coherence        | 233    | 9.9%    |
| 2    | pred_std                   | 176    | 7.5%    |
| 3    | residue_125_kl_divergence  | 171    | 7.3%    |
| 4    | residual_max_abs           | 147    | 6.3%    |
| 5    | residue_8_kl_divergence    | 144    | 6.1%    |
| 6    | residue_125_coherence      | 132    | 5.6%    |
| 7    | residual_abs_mean          | 108    | 4.6%    |
| 8    | lane_consistency           | 94     | 4.0%    |
| 9    | pred_mean                  | 92     | 3.9%    |
| 10   | intersection_ratio         | 86     | 3.7%    |

- 25 of 62 features have zero importance (model ignores 40%)
- Gain values are tiny (~0.01) — consistent with weak-signal regime

---

## 4. Autocorrelation Diagnostics — LEAK DISCOVERED

`compute_autocorrelation_diagnostics()` output:

```
Feature                              r
─────────────────────────────────────────
actual_mean                      +1.0000  ← PROBLEM
pred_std                         +0.5009
pred_mean                        +0.5002
residual_mean                    +0.5002
residue_8_coherence              +0.2229
residue_8_kl_divergence          +0.2227
pred_min                         +0.1506
pred_max                         +0.1352
residue_125_kl_divergence        +0.0986
residue_125_coherence            +0.0984
residual_std                     +0.0943
residual_abs_mean                +0.0750
residue_1000_coherence           -0.0244
residue_1000_kl_divergence       -0.0244
residual_max_abs                 +0.0098
```

### The `actual_mean` Problem

**`actual_mean` shows r = +1.000 with holdout_quality — perfect correlation.**

Investigation reveals it's NOT a direct value leak:
- `actual_mean` ≈ 502.847 (mean of observed draw values)
- `holdout_quality` ≈ 0.222
- Completely different scales

**Root cause:** `actual_mean` is a **near-constant feature** — it's a property of the draw history, not the survivor. Every survivor sees the same draws. The tiny floating-point variations (std ≈ 1e-5 or less) happen to correlate perfectly with `holdout_quality` by numerical coincidence.

**Impact:** The model can overfit on floating-point noise that appears to be a perfect predictor. Even with this "free" signal available, R² is still only 0.003 — suggesting the model isn't heavily exploiting it yet, but it's a ticking bomb.

### Recommended Action: Exclude Near-Constant Draw-History Features

`actual_mean` and likely `actual_std` are properties of the draw history shared by ALL survivors. They should be added to the Step 5 exclude list alongside `score`, `confidence`, `holdout_hits`, `holdout_quality`.

**Proposed new exclude list:**
```python
exclude_features = [
    'score', 'confidence', 'holdout_hits', 'holdout_quality',
    'actual_mean', 'actual_std'  # S111: near-constant, draw-history properties
]
```

---

## 5. Guardrail Recommendations for TB Review

Based on what we found, the following guardrails would prevent similar issues:

### G1: Near-Constant Feature Detection (Step 5 preflight)
Flag or auto-exclude any feature with `std < threshold` (e.g., `std/mean < 1e-6` or `nunique < 10`). Near-constant features with floating-point noise create spurious correlations.

### G2: Perfect-Correlation Detection (Step 5 preflight)
If any feature has `|Pearson r| > 0.95` with the target, flag it for review. Could indicate target leakage or a degenerate feature.

### G3: Draw-History vs Survivor Feature Separation
Features that are properties of the draw history (same for all survivors) should be explicitly tagged in the feature schema and excluded by default. Currently no metadata distinguishes "per-survivor" from "per-dataset" features.

### G4: Optuna Study Staleness Detection
When the target field changes, Optuna studies optimized for the old target should be invalidated automatically. Currently, study reuse is keyed by `schema_hash + data_fingerprint` but NOT by `target_field`. Adding `target_field` to the study name would prevent stale-trial contamination.

### G5: Feature Importance in Sidecar (missing today)
`best_model.meta.json` should include feature importance. Currently absent — required a manual model reload to extract. Patch the save path to include it.

### G6: Rank Metrics Alongside R²
For narrow-band targets, R² understates model utility. Add Spearman rank correlation and top-decile lift to the sidecar metrics.

---

## 6. Signal Interpretation — What the Autocorrelation Tells Us

Excluding the `actual_mean` leak, the genuine persistence hierarchy is:

| Signal Band      | Features                          | r range    | Interpretation |
|------------------|-----------------------------------|------------|----------------|
| **Strong**       | pred_std, pred_mean, residual_mean| 0.50       | Prediction statistics persist across holdout |
| **Moderate**     | residue_8_coherence/kl            | 0.22       | Mod-8 structure persists (LCG low-bit pattern) |
| **Weak**         | residue_125_coherence/kl          | 0.10       | Mod-125 structure partially persists |
| **Near-zero**    | residue_1000_coherence/kl         | 0.02       | Mod-1000 is blind to the pattern |

**For a Java LCG test dataset, this makes sense:** low-order bits have the most algebraic structure, so mod-8 features detect the pattern most strongly, while mod-1000 washes it out.

**Battery-inspired features (v1.3 FINAL) would add:** spectral analysis, autocorrelation lags, bit-level balance, runs structure, and cumulative sum drift — all of which are designed to detect exactly the kind of algebraic structure an LCG produces. They represent orthogonal signal channels not covered by the current 62 features.

---

## 7. Next Steps (Pending TB Approval)

1. **Immediate:** Add `actual_mean`, `actual_std` to exclude list
2. **Immediate:** Archive stale Optuna studies, run fresh 200-trial Step 5
3. **After 200-trial results:** Evaluate whether battery features (Tier 1A) or target redesign (empirical weights / per-feature targets) is the higher-value next lever
4. **TB review requested:** Guardrails G1-G6 above

---

## Artifacts

- `models/reinforcement/best_model.meta.json` — Phase 1 sidecar (target: holdout_quality, R²: 0.00265)
- `diagnostics_outputs/holdout_feature_autocorr.json` — Autocorrelation diagnostics
- `results_archive/phase0_baseline/` — Phase 0 artifacts for comparison
- Git commits: `aaca35d` (S111 main), `1cb90aa` (hardcoded fix)
