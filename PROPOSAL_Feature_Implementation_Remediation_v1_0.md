# PROPOSAL: Feature Implementation Remediation
## Version 1.0 | December 28, 2025
## Status: AWAITING TEAM BETA REVIEW

---

## Executive Summary

During Step 5/6 pipeline validation testing, **critical feature computation bugs** were discovered that rendered the ML training effectively useless. Of 64 defined features, only **2 features had actual variance** - the remaining 62 were either constant (same value for all survivors), hardcoded placeholders, or never computed.

This proposal documents the issues found, fixes already applied, and requests Team Beta input on architectural decisions for remaining remediation.

---

## Issue Discovery

### Symptom
- All 4 ML models achieved R² ≈ 1.0 (perfect fit) in under 3 minutes
- Prediction hit rate = random chance (no predictive lift)

### Root Cause Analysis
Feature importance analysis revealed:

```
Feature 38 (intersection_weight): 54.1% importance → VALUE = 0.0 for ALL survivors
Feature 5  (pred_mean):           45.9% importance → Only feature with real variance
All other 60 features:             0.0% importance → Constant or zero
```

**The model learned almost nothing because there was almost nothing to learn.**

---

## Feature Audit Results

### Schema Definition: 64 Features
- 50 per-seed features
- 14 global features

### Actual State Before Fixes

| Category | Count | Status | Root Cause |
|----------|-------|--------|------------|
| Working (have variance) | 21 | ✅ OK | Properly implemented |
| Never computed | 6 | ❌ BUG | Code sets to 0.0, never calculates |
| Missing input data | 16 | ❌ CONFIG | Forward/reverse/skip data not passed |
| Hardcoded placeholders | 5 | ❌ BUG | Values like `0.1`, `400`, `0` hardcoded |
| Duplicate of global | 2 | ❌ DESIGN | Same computation as global features |
| Global (constant by design) | 14 | ⚠️ OK | Same for all survivors (expected) |
| **TOTAL** | **64** | | |

---

## Fixes Already Applied (Session 18)

### Fix 1: Residual Features (6 features)
**File:** `survivor_scorer.py`

**Before:**
```python
for k in ['pred_min','pred_max','residual_mean','residual_std',
          'residual_abs_mean','residual_max_abs', ...]:
    features.setdefault(k, 0.0)  # Always 0.0!
```

**After:**
```python
features['pred_min'] = float(pred.float().min().item())
features['pred_max'] = float(pred.float().max().item())

residuals = (pred.float() - act.float())
features['residual_mean'] = float(residuals.mean().item())
features['residual_std'] = float(residuals.std().item())
features['residual_abs_mean'] = float(residuals.abs().mean().item())
features['residual_max_abs'] = float(residuals.abs().max().item())
```

**Status:** ✅ COMMITTED

### Fix 2: Random Forest Model Support
**Files:** `models/model_factory.py`, `models/wrappers/random_forest_wrapper.py`

Added `random_forest` to available models for Step 5/6 compatibility.

**Status:** ✅ COMMITTED

### Fix 3: Lottery Format Handling
**File:** `prediction_generator.py`

Fixed parsing of `[{date, session, draw}]` format to extract `[draw]` integers.

**Status:** ✅ COMMITTED

---

## Remaining Issues Requiring Team Beta Input

### Issue 1: Bidirectional Features Not Populated (10 features)

**Affected Features:**
- `intersection_weight`
- `survivor_overlap_ratio`
- `intersection_count`
- `intersection_ratio`
- `forward_only_count`
- `reverse_only_count`
- `forward_count`
- `reverse_count`
- `bidirectional_count`
- `bidirectional_selectivity`

**Root Cause:**
Step 3 script supports `--forward-survivors` and `--reverse-survivors` flags, but they were never passed during execution.

**Files Available:**
```
forward_survivors.json      752 MB   1,625,204 seeds
reverse_survivors.json    1,042 MB   2,251,069 seeds
bidirectional_survivors.json 160 MB   343,714 seeds
```

**Proposed Fix:**
Re-run Step 3 with flags:
```bash
./run_step3_full_scoring.sh \
    --survivors bidirectional_survivors.json \
    --lottery daily3_oldest_500.json \
    --forward-survivors forward_survivors.json \
    --reverse-survivors reverse_survivors.json
```

**Impact:** 10 features will gain variance

**Team Beta Decision Needed:**
- Approve re-run of Step 3 (2-3 hour cluster operation)
- Confirm this aligns with pipeline architecture

---

### Issue 2: Skip Metadata Not Passed (6 features)

**Affected Features:**
- `skip_entropy`
- `skip_mean`
- `skip_std`
- `skip_min`
- `skip_max`
- `skip_range`

**Root Cause:**
Skip values are computed during sieve (Step 2) but not exported or passed to scoring (Step 3).

**Proposed Fix Options:**

| Option | Effort | Description |
|--------|--------|-------------|
| A | Medium | Modify sieve to export skip metadata JSON, pass to scorer |
| B | Low | Defer - leave as 0.0 for now, implement in future sprint |
| C | Low | Remove from feature schema if not architecturally valuable |

**Team Beta Decision Needed:**
- Select Option A, B, or C
- If Option A: Define metadata schema for skip values

---

### Issue 3: Hardcoded Placeholder Values (5 features)

| Feature | Current Value | Problem |
|---------|---------------|---------|
| `confidence` | `0.1` | Hardcoded, meaningless |
| `total_predictions` | `400` | Hardcoded, should be `len(lottery_history)` |
| `best_offset` | `0` | Never computed |
| `survivor_velocity` | `0` | Requires window tracking (not implemented) |
| `velocity_acceleration` | `0` | Requires window tracking (not implemented) |

**Proposed Fix:**

```python
# confidence: Compute from match quality
features['confidence'] = features['exact_matches'] / features['total_predictions']

# total_predictions: Use actual count
features['total_predictions'] = float(len(lottery_history))

# best_offset: Compute optimal alignment
offsets = range(-10, 11)
best_off = max(offsets, key=lambda o: compute_match_rate(pred, act, o))
features['best_offset'] = float(best_off)
```

**For velocity features:**
- Requires architectural change to track window-level survivor counts
- Recommend: Set to 0.0 for v1.0, implement in v1.1

**Team Beta Decision Needed:**
- Approve quick fixes for `confidence`, `total_predictions`, `best_offset`
- Confirm deferral of velocity features to v1.1

---

### Issue 4: Duplicate Features (2 features)

| Per-Seed Feature | Global Feature | Duplication |
|------------------|----------------|-------------|
| `actual_mean` | `global_frequency_bias_ratio` (related) | Both derive from lottery |
| `actual_std` | (none, but lottery-derived) | Constant for all survivors |

**Root Cause:**
`actual_mean` and `actual_std` are computed from lottery history, not the seed. They have the same value for ALL survivors, making them useless for ranking.

**Proposed Fix Options:**

| Option | Description |
|--------|-------------|
| A | Remove from per-seed features, keep only in global |
| B | Rename to `lottery_mean`, `lottery_std` and document as context features |
| C | Redefine to compute something seed-specific (e.g., per-seed prediction stats) |

**Team Beta Decision Needed:**
- Select Option A, B, or C
- Consider backward compatibility implications

---

## Proposed Remediation Schedule

| Phase | Task | Effort | Features Fixed | Dependencies |
|-------|------|--------|----------------|--------------|
| 1 | Re-run Step 3 with bidirectional data | 2-3 hrs | 10 | Team Beta approval |
| 2 | Fix hardcoded placeholders | 30 min | 3 | None |
| 3 | Implement skip metadata pipeline | 2 hrs | 6 | Sieve modification |
| 4 | Address duplicate features | 30 min | 2 | Schema decision |
| 5 | Re-train ML with all models | 1 hr | - | Phases 1-4 complete |
| 6 | Validate hit rate improvement | 1 hr | - | Phase 5 complete |

**Total estimated effort:** 7-8 hours

---

## Expected Outcome

### Before Remediation
- 2 features with variance
- R² = 1.0 (false positive - learned 1 feature)
- Hit rate = random chance

### After Remediation
- 43+ features with variance
- R² likely 0.7-0.9 (realistic learning)
- Hit rate should exceed random if functional mimicry hypothesis is valid

---

## Files Modified/Created

### Already Modified (Session 18)
- `survivor_scorer.py` - Added residual computations
- `models/model_factory.py` - Added random_forest support
- `models/wrappers/random_forest_wrapper.py` - New file
- `prediction_generator.py` - Fixed lottery format parsing

### Pending Modification (Requires Approval)
- `survivor_scorer.py` - Fix hardcoded values
- `sieve_filter.py` - Export skip metadata (if Option A selected)
- `run_step3_full_scoring.sh` - Pass additional flags
- `full_scoring_worker.py` - Load skip metadata

---

## Request for Team Beta

1. **Review** this proposal and identified issues
2. **Approve** Phase 1 (re-run Step 3 with bidirectional data)
3. **Decide** on Issue 2 (skip metadata): Option A, B, or C
4. **Decide** on Issue 4 (duplicates): Option A, B, or C
5. **Confirm** velocity features deferred to v1.1

---

## Appendix: Feature Variance Analysis (Sample)

```
ZERO-VARIANCE FEATURES (constant for all 343K survivors): 36
  - confidence: 0.1 (hardcoded)
  - total_predictions: 400 (hardcoded)
  - best_offset: 0.0 (never computed)
  - actual_mean: 496.84 (lottery stat, not per-seed)
  - actual_std: 289.89 (lottery stat, not per-seed)
  - intersection_weight: 0.0 (data not passed)
  - intersection_count: 0.0 (data not passed)
  - skip_entropy: 0.0 (data not passed)
  - survivor_velocity: 0.0 (never implemented)
  ... (14 global features - constant by design)

WORKING FEATURES WITH VARIANCE: 28
  - pred_mean: 38,786 unique values ✓
  - residue_1000_coherence: 133,205 unique values ✓
  - pred_min: varies ✓ (just fixed)
  - pred_max: varies ✓ (just fixed)
  - residual_mean: varies ✓ (just fixed)
  ... etc
```

---

**Document Author:** Claude (Session 18)
**Review Requested From:** Team Beta
**Response Deadline:** Before next pipeline re-run
