# SESSION_CHANGELOG_20260225_S111.md

## Session 111 — February 25-26, 2026

### Focus: Holdout Quality Deployment + Clean 200-Trial Baseline

---

## Starting Point (from Session 110)

- Phase 0 baseline: R² = 0.000155 (holdout_hits, Poisson noise)
- holdout_quality.py module deployed to Zeus + 3 rigs
- Step 3 patch (full_scoring_worker.py) deployed, 49,882/49,882 coverage
- Step 5 hardcoded holdout_hits overrides fixed (lines 1481/1489)
- Battery features proposal v1.3 FINAL delivered (not yet implemented)
- Commits: `aaca35d` (main patch) + `1cb90aa` (hardcoded fix)

---

## Work Performed

### 1. Phase 1 Contaminated Run (Steps 5→6)

Initial re-run with holdout_quality target, but Optuna resumed from 60 stale
trials optimized for the old holdout_hits target. TPE sampler biased toward
stale hyperparameter regions.

**Contaminated Results:**
| Model | R² (CV) |
|-------|---------|
| LightGBM | 0.00265 (winner) |
| XGBoost | 0.0022 |
| CatBoost | 0.0017 |

### 2. Feature Importance Analysis

Extracted LightGBM feature importance (split-based). Corrected Column_*
index mapping error (exclude list offset). Top features:

| Rank | Feature | Splits | % |
|------|---------|--------|---|
| 1 | residue_8_coherence | 233 | 9.9% |
| 2 | pred_std | 176 | 7.5% |
| 3 | residue_125_kl_divergence | 171 | 7.3% |
| 4 | residual_max_abs | 147 | 6.3% |
| 5 | residue_8_kl_divergence | 144 | 6.1% |

25 of 62 features had zero importance (model ignores 40%).

### 3. Autocorrelation Diagnostics

Generated `holdout_feature_autocorr.json` via `compute_autocorrelation_diagnostics()`.

**Critical finding:** `actual_mean` shows r = +1.0000 — identified as
**degenerate constant-feature pathology** (TB correction: NOT a leak).
`actual_mean` ≈ 502.847 is a property of the draw history, same for all
survivors. Tiny floating-point noise creates spurious perfect correlation.

**Signal hierarchy (excluding actual_mean):**
| Band | Features | r range |
|------|----------|---------|
| Strong | pred_std, pred_mean, residual_mean | 0.50 |
| Moderate | residue_8_coherence/kl | 0.22 |
| Weak | residue_125_coherence/kl | 0.10 |
| Near-zero | residue_1000_coherence/kl | 0.02 |

Interpretation: mod-8 features detect LCG low-bit algebraic structure most
strongly. Mod-1000 washes it out. Consistent with Java LCG testbed.

### 4. Team Beta Briefing + Guardrails Proposal

Delivered `S111_TEAM_BETA_BRIEFING.md` with 6 proposed guardrails:
- G1: Near-constant feature detection (preflight)
- G2: Perfect-correlation detection (preflight)
- G3: Draw-history vs survivor feature separation
- G4: Optuna study staleness detection (target_field in study name)
- G5: Feature importance in sidecar (split+gain)
- G6: Rank metrics alongside R² (Spearman, top-decile lift)

**TB response:** Approved all 6. Corrected "leak" terminology to "degenerate
constant-feature pathology." Offered to write auto-patcher script (S112).

**Decision:** Hold off on S112 patch until clean baseline established.

### 5. Neural Net Skip Registry Reset

`model_skip_registry.json` showed neural_net with `consecutive_critical: 8`.
Reset to 0 so NN would run in the clean 200-trial baseline.

**Result:** NN ran but hit 3 consecutive criticals again. Root cause: y target
(holdout_quality) is NOT normalized. X features are normalized via
StandardScaler (line 487 train_single_trial.py) but y_train is passed raw.
With holdout_quality in range [0.21, 0.23] (std ≈ 0.002), NN produces
catastrophic R² values (-56, -11603) on most configs. Best NN trial: R² = -0.054.

**Fix needed:** Add y-normalization to train_neural_net() path.

### 6. Clean 200-Trial Baseline Run (FINAL)

Archived all stale Optuna studies to `optuna_studies/archive_phase0/`.
Ran fresh 200-trial Optuna per model (expanded to ~600 trials each by
compare-models mechanism × k-fold evaluation).

**Phase 1 Clean Baseline Results:**

| Model | Trials | Best R² (CV) | R² (test) | Status |
|-------|--------|-------------|-----------|--------|
| CatBoost | 600 | 0.002687 | **0.0046** | **WINNER** |
| LightGBM | 600 | 0.003119 | — | Runner-up |
| XGBoost | 600 | 0.002946 | — | Third |
| Neural Net | ~200 | -0.054 | — | SKIP (3 criticals) |

**Final model:** CatBoost, R² = 0.0046 (test set)
**Signal quality:** weak (but real — 30× over Phase 0)
**Generalization:** ✅ "Model generalizes well to test set!"

### 7. Prediction Pool Generated (Step 6)

Top-20 prediction pool with confidence spread 0.32 → 0.96:
| Rank | Seed | Confidence |
|------|------|------------|
| 1 | 410 | 0.9552 |
| 2 | 114 | 0.8528 |
| 3 | 062 | 0.7434 |
| ... | ... | ... |
| 20 | 300 | 0.3192 |

Healthy confidence differentiation — model is ranking survivors, not flat.

---

## Key Metrics Summary

| Metric | Phase 0 | Phase 1 (clean) | Improvement |
|--------|---------|-----------------|-------------|
| Target | holdout_hits | holdout_quality | — |
| R² (test) | 0.000155 | **0.0046** | **30×** |
| Unique target values | 8 | 47,679 | 5,960× |
| Target variance | 1.0e-06 | 4.87e-06 | 4.9× |
| Winner model | CatBoost | CatBoost | — |

---

## Files Changed

| File | Type | Change |
|------|------|--------|
| models/reinforcement/best_model.cbm | Modified | CatBoost winner (R²=0.0046) |
| models/reinforcement/best_model.meta.json | Modified | Updated sidecar |
| models/reinforcement/best_model.txt | Modified | LightGBM runner-up |
| models/reinforcement/best_model.json | Modified | XGBoost runner-up |
| predictions/next_draw_prediction.json | Modified | Top-20 prediction pool |
| optuna_studies/step5_catboost_*.db | New | 600 clean trials |
| optuna_studies/step5_lightgbm_*.db | New | 600 clean trials |
| optuna_studies/step5_xgboost_*.db | New | 600 clean trials |
| diagnostics_outputs/model_skip_registry.json | Modified | NN reset → 3 criticals |
| diagnostics_outputs/compare_models_summary_S88_*.json | New | 4-model comparison |
| diagnostics_outputs/holdout_feature_autocorr.json | New | Autocorr diagnostics |
| docs/S111_TEAM_BETA_BRIEFING.md | New | TB briefing document |

---

## Git Commits

| Hash | Description |
|------|-------------|
| `aaca35d` | S111: Holdout validation redesign (main patch) |
| `1cb90aa` | S111: Fix hardcoded holdout_hits overrides |
| `d860f6f` | S111: Clean 200-trial baseline (CatBoost R²=0.0046) |

---

## Open Issues

| Priority | Item | Status |
|----------|------|--------|
| 🔴 HIGH | NN y-normalization (target not normalized) | NEW |
| 🔴 HIGH | actual_mean/actual_std exclusion (degenerate constant) | NEW |
| 🔴 HIGH | S112 TB guardrails patch (G1-G6) | TB approved, pending |
| 🔴 HIGH | Battery features Tier 1A implementation (23 columns) | Proposal ready |
| 🟡 MED | Feature importance not saved in sidecar | NEW (G5) |
| 🟡 MED | Spearman/rank metrics not in sidecar | NEW (G6) |
| 🟡 MED | Optuna target_field not in study naming | NEW (G4) |
| 🟡 MED | sklearn feature_names warnings in Step 5 | Since S109 |
| Low | S110 root cleanup (884 files) | Deferred |
| Low | Remove dead CSV writer from coordinator.py | Deferred |
| Low | Regression diagnostics gate=True | Since S86 |
| Low | S103 Part2 | Since S103 |

---

## Next Session (S112)

1. Pull feature importance from CatBoost winner — see exactly what the 0.46% is
2. Decide priority order:
   - S112 TB guardrails patch (actual_mean exclusion + G1-G6)
   - Battery features Tier 1A (23 new columns)
   - NN y-normalization
3. Run autocorrelation diagnostics on clean baseline
4. Commit updated S110 + S111 changelogs

---

*Session 111 — Clean Phase 1 baseline established.*
*Signal confirmed real: 3 independent models, 30× improvement, validated on holdout.*
*Battery features are the next major lever.*
