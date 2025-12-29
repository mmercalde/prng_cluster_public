# Feature Remediation Implementation Outline
## Version 1.1 | December 28, 2025
## Status: APPROVED - READY FOR IMPLEMENTATION

---

## Approved Decisions Summary

| Issue | Decision | Owner |
|-------|----------|-------|
| Phase 1 | Re-run Step 3 with bidirectional inputs | Team Alpha |
| Issue 2 | MUST implement skip metadata pipeline | Team Alpha |
| Issue 3 | Remove ALL hardcoded values, implement velocity features | Team Alpha |
| Issue 4 | Option A - Remove duplicates from per-seed, keep as global context | Team Alpha |

---

## Implementation Phases

### Phase 0: Feature Integrity Harness (Team Beta Requirement)
**Estimated Time: 45 minutes**

Before any feature changes, create validation harness to prevent "garbage in, garbage out."

#### Tasks:
1. Create `feature_integrity_validator.py`
2. Implement variance report generator
3. Implement correlation scan (flag |corr| > 0.98 with label)
4. Implement shuffle-label leakage test
5. Output artifacts to `validation_reports/`

#### Acceptance Criteria:
- Per-seed features: ≥30 must have unique_count > 100 over 50k sample
- Zero-variance per-seed features: 0 (except explicitly marked "not implemented")
- Shuffle-label test: R² must drop to ±0.05 of 0

#### MANDATORY (Team Beta):
- **All leakage tests operate on the final Step-5 training label (post-aggregation), not intermediate scores.**

#### Files to Create:
```
feature_integrity_validator.py    # Main validation harness
validation_reports/               # Output directory for reports
```

---

### Phase 1: Re-run Step 3 with Bidirectional Inputs
**Estimated Time: 2-3 hours (cluster operation)**

#### Prerequisites:
- Phase 0 harness created
- Input files validated

#### Tasks:
1. Validate input file integrity (SHA256 hashes)
2. Log input contract to sidecar
3. Execute Step 3 with flags
4. Generate post-run variance report
5. Validate acceptance criteria before proceeding

#### Input Files:
| File | Expected Size | Seeds |
|------|---------------|-------|
| `bidirectional_survivors.json` | 160 MB | 343,714 |
| `forward_survivors.json` | 752 MB | 1,625,204 |
| `reverse_survivors.json` | 1,042 MB | 2,251,069 |
| `daily3_oldest_500.json` | 38 KB | 500 draws |

#### Command:
```bash
./run_step3_full_scoring.sh \
    --survivors bidirectional_survivors.json \
    --lottery daily3_oldest_500.json \
    --forward-survivors forward_survivors.json \
    --reverse-survivors reverse_survivors.json
```

#### Team Beta Guardrails:
- Log SHA256 for all input files
- Log seed counts and schema version
- Output per-feature variance report
- Block Step 5 if variance thresholds not met
- **Bidirectional features must be computed without modifying or filtering the survivor set used for scoring.**

#### Features Enabled (10):
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

---

### Phase 2: Implement Skip Metadata Pipeline
**Estimated Time: 2-3 hours**

This is CRITICAL - the whitepaper is built on this premise.

#### Architecture (Team Beta Preferred):
Export aggregate stats in Step 2, keep Step 3 lightweight.

#### Step 2 Output Schema (per-seed aggregate stats):
```json
{
  "schema_version": "skip_meta_v1",
  "run_id": "step2_20251228_HHMMSS",
  "mod": 1000,
  "prng_type": "java_lcg",
  "seeds": {
    "123456789": {
      "n": 4000,
      "mean": 0.73,
      "std": 1.1,
      "min": 0,
      "max": 6,
      "range": 6,
      "entropy": 1.42
    }
  }
}
```

#### Tasks:
1. **Modify `sieve_filter.py`** - Track skip values during sieve
2. **Add skip stats computation** - Compute mean/std/min/max/range/entropy per seed
3. **Export `skip_metadata.json`** - Write to file alongside survivors
4. **Modify `run_step3_full_scoring.sh`** - Add `--skip-metadata` flag
5. **Modify `full_scoring_worker.py`** - Load and use skip metadata
6. **Modify `survivor_scorer.py`** - Populate skip features from metadata

#### Files to Modify:
```
sieve_filter.py           # Export skip metadata during sieve
run_step3_full_scoring.sh # Add --skip-metadata flag
full_scoring_worker.py    # Load skip metadata file
survivor_scorer.py        # Use skip metadata in feature extraction
```

#### Features Enabled (6):
- `skip_entropy`
- `skip_mean`
- `skip_std`
- `skip_min`
- `skip_max`
- `skip_range`

---

### Phase 3: Remove Hardcoded Placeholders & Implement Velocity
**Estimated Time: 2 hours**

NO HARDCODED VALUES allowed.

#### 3A: Fix Simple Placeholders (30 min)

| Feature | Current | Fix |
|---------|---------|-----|
| `confidence` | `0.1` | `clamp(exact_matches / total_predictions, 0, 1)` |
| `total_predictions` | `400` | `len(lottery_history)` |
| `best_offset` | `0` | Compute optimal alignment (-10 to +10 range) |

#### Implementation:
```python
# confidence - with division-by-zero protection
if total_predictions > 0:
    features['confidence'] = min(1.0, max(0.0, 
        features['exact_matches'] / features['total_predictions']))
else:
    features['confidence'] = 0.0

# total_predictions - actual count
features['total_predictions'] = float(len(lottery_history))

# best_offset - optimal alignment (avoid leakage: use training segment only)
def compute_best_offset(pred, act, max_offset=10):
    best_off = 0
    best_rate = 0.0
    for off in range(-max_offset, max_offset + 1):
        if off < 0:
            p, a = pred[-off:], act[:off]
        elif off > 0:
            p, a = pred[:-off], act[off:]
        else:
            p, a = pred, act
        if len(p) > 0:
            rate = (p == a).float().mean().item()
            if rate > best_rate:
                best_rate = rate
                best_off = off
    return best_off

features['best_offset'] = float(compute_best_offset(pred, act))
```

#### 3B: Implement Window Tracking for Velocity Features (1.5 hours)

**Architecture Decision Required:**
Velocity features require window-level survivor count history. Two options:

**Option A: Compute in Step 2 (Sieve)**
- Track survivor counts per window during sieve
- Export as part of skip_metadata.json
- Step 3 reads and computes velocity/acceleration

**Option B: Compute in Step 3 (Scorer)**
- Pass window config to scorer
- Scorer re-computes survivor counts per window
- More compute but self-contained

**Recommended: Option A** - Compute where data naturally exists.

#### MANDATORY (Team Beta):
- **Window counts used for velocity/acceleration must be computed strictly within the same historical span available to the seed at that window — never using future draws relative to the scoring target.**

#### Skip Metadata Schema Extension:
```json
{
  "schema_version": "skip_meta_v1.1",
  "seeds": {
    "123456789": {
      "n": 4000,
      "mean": 0.73,
      "std": 1.1,
      "min": 0,
      "max": 6,
      "range": 6,
      "entropy": 1.42,
      "window_counts": [1200, 1150, 1100, 1050, 1000],
      "velocity": -50.0,
      "acceleration": 0.0
    }
  }
}
```

#### Velocity Computation:
```python
def compute_velocity_features(window_counts: List[int]) -> Tuple[float, float]:
    """
    Compute survivor velocity and acceleration from window counts.
    
    velocity = mean rate of change of survivor count
    acceleration = rate of change of velocity
    """
    if len(window_counts) < 2:
        return 0.0, 0.0
    
    import numpy as np
    counts = np.array(window_counts, dtype=float)
    
    # First derivative: velocity (change per window)
    velocity = np.diff(counts)
    mean_velocity = float(np.mean(velocity))
    
    # Second derivative: acceleration
    if len(velocity) >= 2:
        acceleration = np.diff(velocity)
        mean_acceleration = float(np.mean(acceleration))
    else:
        mean_acceleration = 0.0
    
    return mean_velocity, mean_acceleration
```

#### Features Enabled (5):
- `confidence` (fixed)
- `total_predictions` (fixed)
- `best_offset` (fixed)
- `survivor_velocity` (new implementation)
- `velocity_acceleration` (new implementation)

---

### Phase 4: Remove Duplicate Features from Per-Seed Vector
**Estimated Time: 1 hour**

#### Features to Relocate:
- `actual_mean` → Move to `context.lottery_stats.mean`
- `actual_std` → Move to `context.lottery_stats.std`

#### Backward Compatibility Strategy:
1. **v1.0.1 (Transition):**
   - Keep keys in per-seed JSON
   - Add `is_context: true` flag
   - Exclude from ML feature matrix (X)
   - Add parallel keys under `context.lottery_stats`

2. **v1.1.0 (Deprecation Warning):**
   - Log deprecation warning when old keys accessed
   - Documentation updated

3. **v2.0.0 (Removal):**
   - Remove from per-seed JSON entirely

#### Schema Change:
```json
// BEFORE (per-seed)
{
  "seed": 12345,
  "features": {
    "actual_mean": 496.84,    // REMOVE from here
    "actual_std": 289.89,     // REMOVE from here
    "pred_mean": 465.07,
    ...
  }
}

// AFTER (with context)
{
  "seed": 12345,
  "features": {
    "pred_mean": 465.07,
    ...
  },
  "context": {
    "lottery_stats": {
      "mean": 496.84,
      "std": 289.89
    },
    "is_context_fields": ["actual_mean", "actual_std"]
  }
}
```

#### Files to Modify:
```
survivor_scorer.py           # Remove from features dict, add to context
full_scoring_worker.py       # Handle context in output
feature_registry.json        # Update schema (mark as context)
config_manifests/            # Update feature schema version
```

#### Feature Count Change:
- Per-seed features: 50 → 48 (discriminative)
- Context features: +2 (lottery_stats)
- Global features: 14 (unchanged)
- **Total in JSON: Still 64** (backward compatible)
- **ML feature vector: 62** (48 per-seed + 14 global, minus 2 context-only)

---

### Phase 5: Validation & Re-training
**Estimated Time: 2 hours**

#### Tasks:
1. Run Phase 0 integrity harness on new features
2. Verify all acceptance criteria met
3. Re-train with all 5 models (including neural_net)
4. Run hit-rate test across 50 draws
5. Document results

#### Command:
```bash
python3 meta_prediction_optimizer_anti_overfit.py \
    --survivors survivors_with_scores.json \
    --lottery-data daily3_oldest_500.json \
    --compare-models \
    --trials 50 \
    --timeout 900 \
    --test-holdout 0.2
```

#### Expected Outcomes:
- R² < 1.0 (realistic learning curve)
- Feature importance distributed across multiple features
- Hit rate > random chance (if functional mimicry works)

---

## File Modification Summary

| File | Phase | Changes |
|------|-------|---------|
| `feature_integrity_validator.py` | 0 | NEW - validation harness |
| `sieve_filter.py` | 2 | Export skip metadata + window counts |
| `run_step3_full_scoring.sh` | 1,2 | Add hash logging, skip-metadata flag |
| `full_scoring_worker.py` | 1,2 | Load skip metadata, pass to scorer |
| `survivor_scorer.py` | 2,3,4 | Skip features, velocity, remove duplicates |
| `feature_registry.json` | 4 | Update schema, mark context fields |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking existing pipelines | Backward-compatible JSON schema |
| Label leakage | Shuffle-label test mandatory |
| Silent semantic changes | Schema versioning + hash validation |
| Compute explosion (skip arrays) | Export aggregates, not raw arrays |
| Window tracking overhead | Compute in Step 2 where data exists |

---

## Timeline Estimate

| Phase | Task | Time |
|-------|------|------|
| 0 | Feature integrity harness | 45 min |
| 1 | Re-run Step 3 with bidirectional | 2-3 hrs |
| 2 | Skip metadata pipeline | 2-3 hrs |
| 3 | Remove hardcoded + velocity | 2 hrs |
| 4 | Relocate duplicate features | 1 hr |
| 5 | Validation & re-training | 2 hrs |
| **TOTAL** | | **10-12 hrs** |

---

## Success Criteria

1. **Variance:** ≥43 per-seed features with unique_count > 100
2. **No Leakage:** Shuffle-label R² within ±0.05 of 0
3. **No Hardcoded Values:** Zero hardcoded placeholders in scorer
4. **Schema Integrity:** Hash validation passes across all steps
5. **Hit Rate:** Statistically significant improvement over random

---

## Approval

- [x] Team Beta reviewed and approved
- [ ] Implementation outline approved by project lead
- [ ] Progress document created
- [ ] Implementation begun

---

**Document Version:** 1.1
**Author:** Claude (Session 18)
**Approved By:** Team Beta, Project Lead
