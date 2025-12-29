# Feature Remediation Progress Tracker
## Version 1.0 | Started: December 28, 2025
## Status: IN PROGRESS

---

## Quick Status Dashboard

| Phase | Status | Progress | Blocking Issues |
|-------|--------|----------|-----------------|
| Phase 0 | ⬜ NOT STARTED | 0% | None |
| Phase 1 | ⬜ NOT STARTED | 0% | Depends on Phase 0 |
| Phase 2 | ⬜ NOT STARTED | 0% | Depends on Phase 1 |
| Phase 3 | ⬜ NOT STARTED | 0% | Can parallel with Phase 2 |
| Phase 4 | ⬜ NOT STARTED | 0% | Depends on Phase 3 |
| Phase 5 | ⬜ NOT STARTED | 0% | Depends on all above |

**Legend:** ⬜ Not Started | 🔄 In Progress | ✅ Complete | ❌ Blocked | ⚠️ Issues

---

## Phase 0: Feature Integrity Harness
**Status:** ⬜ NOT STARTED
**Estimated:** 45 min | **Actual:** -

### Tasks
| Task | Status | Notes |
|------|--------|-------|
| Create `feature_integrity_validator.py` | ⬜ | |
| Implement variance report generator | ⬜ | |
| Implement correlation scan | ⬜ | |
| Implement shuffle-label test | ⬜ | |
| Create `validation_reports/` directory | ⬜ | |
| Test on current survivors sample | ⬜ | |

### Acceptance Criteria
- [ ] Variance report generates successfully
- [ ] Correlation scan identifies any |corr| > 0.98
- [ ] Shuffle-label test framework working
- [ ] Output artifacts saved to validation_reports/

### Artifacts Produced
- [ ] `feature_integrity_validator.py`
- [ ] `validation_reports/variance_report_YYYYMMDD.json`
- [ ] `validation_reports/correlation_scan_YYYYMMDD.json`

---

## Phase 1: Re-run Step 3 with Bidirectional Inputs
**Status:** ⬜ NOT STARTED
**Estimated:** 2-3 hrs | **Actual:** -

### Pre-flight Checks
| Check | Status | Value |
|-------|--------|-------|
| `bidirectional_survivors.json` exists | ⬜ | |
| `forward_survivors.json` exists | ⬜ | |
| `reverse_survivors.json` exists | ⬜ | |
| SHA256 hashes logged | ⬜ | |
| Seed counts verified | ⬜ | |

### Tasks
| Task | Status | Notes |
|------|--------|-------|
| Validate input file integrity | ⬜ | |
| Add hash logging to run_step3 | ⬜ | |
| Execute Step 3 with flags | ⬜ | |
| Generate variance report | ⬜ | |
| Validate acceptance criteria | ⬜ | |

### Cluster Execution
| Node | Status | Chunks | GPU |
|------|--------|--------|-----|
| Zeus | ⬜ | - | RTX 3080 Ti x2 |
| rig-6600 | ⬜ | - | RX 6600 x12 |
| rig-6600b | ⬜ | - | RX 6600 x12 |

### Features Enabled (10)
| Feature | Before | After |
|---------|--------|-------|
| intersection_weight | 0.0 | ⬜ Pending |
| survivor_overlap_ratio | 0.0 | ⬜ Pending |
| intersection_count | 0.0 | ⬜ Pending |
| intersection_ratio | 0.0 | ⬜ Pending |
| forward_only_count | 0.0 | ⬜ Pending |
| reverse_only_count | 0.0 | ⬜ Pending |
| forward_count | constant | ⬜ Pending |
| reverse_count | constant | ⬜ Pending |
| bidirectional_count | constant | ⬜ Pending |
| bidirectional_selectivity | 0.0 | ⬜ Pending |

### Acceptance Criteria
- [ ] All 10 bidirectional features have variance
- [ ] No features with unique_count <= 1
- [ ] Hash validation logged to sidecar
- [ ] Variance report passes thresholds

---

## Phase 2: Skip Metadata Pipeline
**Status:** ⬜ NOT STARTED
**Estimated:** 2-3 hrs | **Actual:** -

### Architecture
- [ ] Schema design approved
- [ ] Export location determined: `skip_metadata.json`
- [ ] Transport mechanism: file-based

### Tasks
| Task | Status | Notes |
|------|--------|-------|
| Modify `sieve_filter.py` - track skips | ⬜ | |
| Add skip stats computation | ⬜ | |
| Export `skip_metadata.json` | ⬜ | |
| Modify `run_step3_full_scoring.sh` | ⬜ | |
| Modify `full_scoring_worker.py` | ⬜ | |
| Modify `survivor_scorer.py` | ⬜ | |
| Test skip feature computation | ⬜ | |

### Features Enabled (6)
| Feature | Before | After |
|---------|--------|-------|
| skip_entropy | 0.0 | ⬜ Pending |
| skip_mean | 0.0 | ⬜ Pending |
| skip_std | 0.0 | ⬜ Pending |
| skip_min | 0.0 | ⬜ Pending |
| skip_max | 0.0 | ⬜ Pending |
| skip_range | 0.0 | ⬜ Pending |

### Acceptance Criteria
- [ ] Skip metadata exports from Step 2
- [ ] Step 3 loads and uses skip metadata
- [ ] All 6 skip features have variance
- [ ] Schema version documented

---

## Phase 3: Remove Hardcoded & Implement Velocity
**Status:** ⬜ NOT STARTED
**Estimated:** 2 hrs | **Actual:** -

### 3A: Fix Simple Placeholders (30 min)
| Feature | Before | Fix | Status |
|---------|--------|-----|--------|
| confidence | 0.1 (hardcoded) | exact_matches/total_predictions | ⬜ |
| total_predictions | 400 (hardcoded) | len(lottery_history) | ⬜ |
| best_offset | 0 (never computed) | optimal alignment search | ⬜ |

### 3B: Implement Velocity Features (1.5 hrs)
| Task | Status | Notes |
|------|--------|-------|
| Design window tracking in Step 2 | ⬜ | |
| Extend skip_metadata schema | ⬜ | |
| Implement velocity computation | ⬜ | |
| Implement acceleration computation | ⬜ | |
| Pass to Step 3 scorer | ⬜ | |
| Test variance | ⬜ | |

### Features Enabled (5)
| Feature | Before | After |
|---------|--------|-------|
| confidence | 0.1 | ⬜ Computed |
| total_predictions | 400 | ⬜ Computed |
| best_offset | 0 | ⬜ Computed |
| survivor_velocity | 0 | ⬜ Computed |
| velocity_acceleration | 0 | ⬜ Computed |

### Acceptance Criteria
- [ ] Zero hardcoded values in survivor_scorer.py
- [ ] All 5 features have variance
- [ ] Division-by-zero handled
- [ ] Offset search capped at ±10

---

## Phase 4: Relocate Duplicate Features
**Status:** ⬜ NOT STARTED
**Estimated:** 1 hr | **Actual:** -

### Migration Plan
| Step | Status | Notes |
|------|--------|-------|
| Add context.lottery_stats structure | ⬜ | |
| Move actual_mean to context | ⬜ | |
| Move actual_std to context | ⬜ | |
| Add is_context flag | ⬜ | |
| Update feature_registry.json | ⬜ | |
| Test backward compatibility | ⬜ | |
| Exclude from ML feature matrix | ⬜ | |

### Schema Version
| Version | Changes |
|---------|---------|
| v1.0.0 | Current (features mixed) |
| v1.0.1 | Context fields marked, excluded from X |
| v1.1.0 | Deprecation warnings |
| v2.0.0 | Full removal (future) |

### Features Affected (2)
| Feature | Before | After |
|---------|--------|-------|
| actual_mean | per-seed (constant) | context.lottery_stats.mean |
| actual_std | per-seed (constant) | context.lottery_stats.std |

### Acceptance Criteria
- [ ] actual_mean/std in context structure
- [ ] Per-seed feature count: 48 (was 50)
- [ ] ML feature vector excludes context
- [ ] Backward compatibility test passes
- [ ] No silent semantic changes

---

## Phase 5: Validation & Re-training
**Status:** ⬜ NOT STARTED
**Estimated:** 2 hrs | **Actual:** -

### Pre-training Validation
| Check | Status | Result |
|-------|--------|--------|
| Variance report passes | ⬜ | |
| Correlation scan clean | ⬜ | |
| Shuffle-label R² near 0 | ⬜ | |
| Schema hash matches | ⬜ | |

### Training Run
| Model | Status | Val MSE | R² |
|-------|--------|---------|-----|
| LightGBM | ⬜ | - | - |
| XGBoost | ⬜ | - | - |
| CatBoost | ⬜ | - | - |
| Random Forest | ⬜ | - | - |
| Neural Net | ⬜ | - | - |

### Hit Rate Test (50 draws)
| Pool Size | Hits | Rate | vs Random | Lift |
|-----------|------|------|-----------|------|
| Top 10 | - | - | 1% | - |
| Top 100 | - | - | 10% | - |
| Top 500 | - | - | 50% | - |

### Acceptance Criteria
- [ ] R² < 1.0 (realistic learning)
- [ ] Feature importance distributed
- [ ] Hit rate > random (statistically significant)
- [ ] No leakage detected

---

## Issues & Blockers Log

| ID | Date | Description | Status | Resolution |
|----|------|-------------|--------|------------|
| - | - | - | - | - |

---

## Change Log

| Date | Phase | Change | Author |
|------|-------|--------|--------|
| 2025-12-28 | - | Document created | Claude |
| 2025-12-28 | - | Team Beta approval received | Team Beta |
| 2025-12-28 | - | Project lead direction confirmed | Michael |

---

## Notes & Decisions

### Team Beta Requirements
1. Hash-logged input contract for all steps
2. Post-run variance report mandatory
3. Shuffle-label leakage test required
4. Schema versioning enforced

### Project Lead Direction
1. Skip metadata pipeline MUST be implemented (not deferred)
2. Velocity features MUST be implemented (not deferred)
3. ALL hardcoded values must be removed
4. Careful backward compatibility for schema changes

---

**Last Updated:** December 28, 2025
**Next Review:** After Phase 0 completion
