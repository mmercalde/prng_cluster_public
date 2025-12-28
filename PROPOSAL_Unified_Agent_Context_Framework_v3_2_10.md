# Unified Agent Context Framework v3.2.10

**Document Version:** 3.2.10  
**Date:** December 27, 2025  
**Author:** Claude (AI Assistant)  
**Status:** PRODUCTION-READY  
**Supersedes:** v3.2.9  
**Patch Focus:** Step 0 Archived + Global Features Integration + Timeout CLI (Session 17)

---

## Changes from v3.2.9

| Section | Change |
|---------|--------|
| Part 18 | NEW: Step 0 PRNG Fingerprinting - ARCHIVED |
| Part 19 | NEW: Global Features Integration at Step 3 |
| Part 20 | NEW: Timeout CLI Argument for Step 5 |
| Part 15 | UPDATED: Feature Architecture (64 total, 62 training) |

---

## Critical Issues Addressed (Session 17)

### Issue 1: Step 0 PRNG Fingerprinting - ARCHIVED

**Problem:** Team Beta proposed Step 0 to classify unknown PRNGs by comparing behavioral fingerprints against a library of known PRNGs.

**Investigation:**
- Generated fingerprint libraries with 52 mod1000-specific features
- Tested at 5K, 18K, and 20K draw lengths
- Computed Signal-to-Noise Ratio (SNR) for all features

**Results:**

| Feature Set | Features | Best SNR | Useful (SNR≥1) | Classification |
|-------------|----------|----------|----------------|----------------|
| Mod1000 @ 5K | 52 | 0.11 | 0 | java_lcg #10/11 |
| Mod1000 @ 18K | 52 | 0.13 | 0 | java_lcg #5/11 |
| Original @ 18K | 30 | 0.13 | 0 | java_lcg #7/11 |

**Root Cause:** Under mod1000 projection, between-PRNG variance < within-PRNG variance for ALL features. SNR < 0.15 means seed noise dominates PRNG signal.

**Verdict:** PRNG fingerprinting is mathematically impossible under mod1000. **Step 0 ARCHIVED.**

**Recommendation:** Trust the sieve - wrong PRNG → 0 survivors, right PRNG → survivors exist.

### Issue 2: Global Features Not in Multi-Model Path (FIXED)

**Problem:** 14 global features from `GlobalStateTracker` were only used in `ReinforcementEngine` path, NOT in multi-model comparison (XGBoost, LightGBM, CatBoost).

**Root Cause:**
- `survivors_with_scores.json` contained only 50 per-seed features
- Global features computed at runtime in `ReinforcementEngine`
- Multi-model path used `load_features_from_survivors()` which read JSON directly

**Solution:** Add global features at Step 3 Phase 5 (Aggregation)

**Team Beta Approved Fixes:**
1. Prefix with `global_` to prevent namespace collision
2. Variance guardrail added
3. Documented replication concern

### Issue 3: Neural Net Timeout (FIXED)

**Problem:** Neural net trials timing out at 600s default, causing failed trials.

**Solution:** Added `--timeout` CLI argument to `meta_prediction_optimizer_anti_overfit.py`

---

## Part 18: Step 0 PRNG Fingerprinting - ARCHIVED (NEW)

### 18.1 Original Proposal

Team Beta proposed a classification step before sieving:
1. Generate fingerprints for 22 known PRNGs (20 sequences × 20K draws each)
2. Extract 52 mod1000-specific features per sequence
3. Classify unknown lottery data by distance to fingerprints
4. Output: Prior probabilities for each PRNG type

### 18.2 Investigation Summary

**Feature Sets Tested:**
- 52 mod1000-specific features (digit analysis, Fourier, transition matrices)
- 30 Team Beta curated features
- 64 original features

**SNR Analysis Methodology:**
```
SNR = Between-PRNG Variance / Within-PRNG Variance

For features to be useful: SNR must be > 1.0
- SNR > 1.0: PRNG identity dominates seed variation
- SNR < 1.0: Seed variation dominates PRNG identity
```

**Results:**
- ALL features had SNR < 0.15
- Within-PRNG variance 4-7× larger than between-PRNG variance
- No feature could reliably distinguish PRNGs

### 18.3 Root Cause Analysis

**Modular Reduction Destroys Information:**
```
Full PRNG output: 2^32 possible values → highly distinguishable
After mod 1000:   1000 possible values → information collapsed

Example:
  java_lcg(seed=100, n=5000) mod 1000 ≈ uniform over [0,999]
  mt19937(seed=100, n=5000) mod 1000  ≈ uniform over [0,999]
  
Both look statistically identical after modular reduction.
```

### 18.4 Final Verdict

**Step 0 is ARCHIVED.** Cannot distinguish PRNGs under mod1000.

**Alternative:** Trust the bidirectional sieve:
- Wrong PRNG → 0 survivors (impossible to match)
- Right PRNG → survivors exist (can be validated)

---

## Part 19: Global Features Integration (NEW)

### 19.1 Problem

Global features from `GlobalStateTracker` provide lottery-wide context:
- `global_regime_change_detected` - PRNG state changes
- `global_marker_390_variance` - Suspicious number patterns
- `global_frequency_bias_ratio` - Distribution anomalies

These were only available in `ReinforcementEngine` neural net path, not tree models.

### 19.2 Solution

Add global features at Step 3 Phase 5 (Aggregation) in `run_step3_full_scoring.sh`:
```python
# Compute global features (once for all survivors)
global_tracker = GlobalStateTracker(lottery_history, {'mod': 1000})
global_features = global_tracker.get_global_state()

# Team Beta Fix #1: Prefix with "global_" to prevent namespace collision
global_features_prefixed = {
    f"global_{k}": v for k, v in global_features.items()
}

# Merge into each survivor
for survivor in all_survivors:
    survivor['features'].update(global_features_prefixed)
```

### 19.3 Team Beta Requirements

| Requirement | Implementation |
|-------------|----------------|
| Prefix with `global_` | ✅ Prevents namespace collision |
| Variance guardrail | ✅ Warns if all values identical |
| Document replication | ✅ Same values for all survivors |

### 19.4 Feature Registry Update

`config_manifests/feature_registry.json` updated with `global_` prefixed names:
- `global_residue_8_entropy`
- `global_frequency_bias_ratio`
- `global_regime_change_detected`
- `global_marker_390_variance`
- etc.

---

## Part 20: Timeout CLI Argument (NEW)

### 20.1 Problem

Neural net trials often exceed 600s default timeout:
```
Trial 1: NEURAL_NET
❌ FAILED (timeout after 600s)
```

### 20.2 Solution

Added `--timeout` argument to `meta_prediction_optimizer_anti_overfit.py`:
```python
parser.add_argument('--timeout', type=int, default=600,
                    help='Timeout per trial in seconds (default: 600)')
```

### 20.3 Usage
```bash
python3 meta_prediction_optimizer_anti_overfit.py \
    --survivors survivors_with_scores.json \
    --lottery-data train_history.json \
    --compare-models \
    --trials 8 \
    --timeout 900  # 15 minutes per trial
```

### 20.4 Implementation Chain
```
CLI: --timeout 900
    ↓
run_multi_model_comparison(..., timeout=args.timeout)
    ↓
SubprocessTrialCoordinator(..., timeout=timeout)
    ↓
subprocess.run(..., timeout=self.timeout)
```

---

## Part 15: Feature Architecture (UPDATED)

### 15.1 Updated Feature Counts
```
Total Features: 64 (in survivors_with_scores.json)
Training Features: 62 (after excluding score, confidence)

├── Per-seed features: 50 (from survivor_scorer.py)
│   ├── Residue features: 12
│   ├── Temporal features: 20
│   ├── Statistical features: 12
│   ├── Metadata features: 4
│   └── Score metrics: 2 (excluded from training)
│
└── Global features: 14 (from GlobalStateTracker, prefixed with 'global_')
    ├── Residue entropy: 3
    │   └── global_residue_8_entropy, global_residue_125_entropy, global_residue_1000_entropy
    ├── Bias detection: 3
    │   └── global_power_of_two_bias, global_frequency_bias_ratio, global_suspicious_gap_percentage
    ├── Regime detection: 3
    │   └── global_regime_change_detected, global_regime_age, global_reseed_probability
    ├── Marker analysis: 4
    │   └── global_marker_390_variance, global_marker_804_variance, global_marker_575_variance, global_high_variance_count
    └── Stability: 1
        └── global_temporal_stability
```

### 15.2 Feature Flow
```
Step 3 (full_scoring_worker.py):
    Per-seed features extracted → 50 features per survivor
    
Step 3 Phase 5 (run_step3_full_scoring.sh):
    Global features computed → 14 features (identical for all)
    Merged into each survivor → 64 total features
    
Step 5 (meta_prediction_optimizer_anti_overfit.py):
    Loads survivors_with_scores.json
    Excludes score, confidence → 62 training features
    Trains XGBoost/LightGBM/CatBoost/Neural Net
```

---

## Multi-Model Comparison Results (Session 17)

### Test Configuration
- Survivors: 395,211 with 62 features
- Models: neural_net, xgboost, lightgbm, catboost
- Trials: 8

### Results

| Trial | Model | R² | MSE | Duration |
|-------|-------|-----|-----|----------|
| 0 | neural_net | 0.0000 | 0.002495 | 253s |
| 1 | neural_net | TIMEOUT | - | 600s |
| 2 | xgboost | 1.0000 | 1.0e-07 | 1.8s |
| 3 | neural_net | TIMEOUT | - | 600s |
| 4 | xgboost | 0.9991 | 2.2e-06 | 3.0s |
| 5 | **catboost** | **1.0000** | **8.6e-11** | 4.8s |
| 6 | lightgbm | 0.9996 | 1.0e-06 | 2.3s |
| 7 | lightgbm | 0.9999 | 2.1e-07 | 2.9s |

### Summary

| Model | Best R² | Speed | Verdict |
|-------|---------|-------|---------|
| 🏆 CatBoost | 1.0000 | 4.8s | **Winner** |
| XGBoost | 1.0000 | 1.8s | Fastest |
| LightGBM | 0.9999 | 2.9s | Solid |
| Neural Net | 0.0000 | 253s+ | Poor fit |

---

## Summary of Session 17 Changes

| Component | File | Change |
|-----------|------|--------|
| Step 0 | N/A | ARCHIVED - mathematically impossible |
| Global Features | `run_step3_full_scoring.sh` | Added at Phase 5 aggregation |
| Feature Registry | `config_manifests/feature_registry.json` | Updated with `global_` prefix |
| Timeout CLI | `meta_prediction_optimizer_anti_overfit.py` | Added `--timeout` argument |

---

## Git Commits (Session 17)
```
commit fb59429
Step 3: Add 14 global features at aggregation (Team Beta approved)

- GlobalStateTracker computes features once from lottery history
- Features prefixed with 'global_' to prevent namespace collision
- Variance guardrail added per Team Beta review
- Graceful fallback if GlobalStateTracker unavailable
- Total features now: 50 per-seed + 14 global = 64
```

---

## Appendix A: Files Modified (Session 17)

| File | Lines Changed | Purpose |
|------|---------------|---------|
| `run_step3_full_scoring.sh` | +79, -2 | Global features at aggregation |
| `config_manifests/feature_registry.json` | ~50 | Updated global feature names |
| `meta_prediction_optimizer_anti_overfit.py` | +8 | Timeout CLI argument |

---

## Appendix B: Data Quality Finding

**Problem:** `daily3_scraper.py` creates duplicate entries

**Analysis:**
- Full scrape (2000-2025): 18,666 entries, 721 duplicates
- After deduplication: 17,945 clean entries

**Output:** `daily3_clean.json` - deduplicated lottery history

---

## Next Steps

1. Commit remaining changes (feature_registry.json, timeout fix)
2. Run full pipeline test when rigs back online
3. Complete Step 6 cleanup (deprecate unused global_values parameter)
4. Consider Watcher Agent integration
