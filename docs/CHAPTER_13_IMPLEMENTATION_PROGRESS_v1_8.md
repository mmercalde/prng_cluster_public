# Chapter 13 Implementation Progress

**Last Updated:** 2026-01-30  
**Document Version:** 1.9.0  
**Status:** Phases 1-9B.2 Complete → Phase 9B.3 (Future)  
**Team Beta Endorsement:** ✅ Approved (Phase 9B.2 complete)

---

## Overall Progress

| Phase | Status | Owner | Completion |
|-------|--------|-------|------------|
| 1. Draw Ingestion | ✅ Complete | Claude | 2026-01-12 |
| 2. Diagnostics Engine | ✅ Complete | Claude | 2026-01-12 |
| 3. Retrain Triggers | ✅ Complete | Claude | 2026-01-12 |
| 4. LLM Integration | ✅ Complete | Claude | 2026-01-12 |
| 5. Acceptance Engine | ✅ Complete | Claude | 2026-01-12 |
| 6. WATCHER Orchestration | ✅ Complete | Claude | 2026-01-12 |
| 7. Testing & Validation | 🟡 In Progress | TBD | — |
| 8. Selfplay Integration | ✅ Complete | Team Beta | 2026-01-30 |
| 9A. Chapter 13 ↔ Selfplay Hooks | ✅ Complete | Team Beta | 2026-01-30 |
| **9B.1 Policy Transform Module** | ✅ **COMPLETE** | Claude | **2026-01-30** |
| **9B.2 Policy-Conditioned Mode** | ✅ **COMPLETE** | Claude | **2026-01-30** |
| 9B.3 Policy Proposal Heuristics | 🔲 Not Started | TBD | — |

**Legend:** 🔲 Not Started | 🟡 In Progress | ✅ Complete | ❌ Blocked

---

## ⚠️ CRITICAL: Coordination Requirements

**GPU work MUST use existing coordinators to prevent ROCm/SSH storms.**

| Work Type | Direct SSH OK? | Use Coordinator? | Stagger Required? |
|-----------|----------------|------------------|-------------------|
| GPU Sieving (Outer Episode) | ❌ **NO** | ✅ **YES (mandatory)** | ✅ YES (0.3s) |
| CPU ML Training (Inner Episode) | ✅ YES | Optional | ❌ NO |

---

## Phase 9B.2: Policy-Conditioned Learning ✅ COMPLETE

**Completed:** 2026-01-30  
**Files:** 
- `policy_conditioned_episode.py` v1.0.0
- `selfplay_orchestrator.py` v1.1.0

### Core API

| Function | Purpose |
|----------|---------|
| `condition_episode(survivors, project_root)` | Apply active policy to survivors |
| `load_active_policy(project_root)` | Load from learned_policy_active.json |
| `create_policy_candidate(episode, parent, result)` | Create candidate with lineage |
| `emit_policy_candidate(candidate, project_root)` | Write to learned_policy_candidate.json |

### CLI Arguments Added to selfplay_orchestrator.py

| Flag | Purpose |
|------|---------|
| `--policy-conditioned` | Enable Phase 9B.2 conditioning |
| `--no-emit-candidate` | Disable candidate emission (testing) |
| `--project-root PATH` | Project root for policy files |

### Integration Complete

The following changes were applied to `selfplay_orchestrator.py`:

1. ✅ Added imports from `policy_conditioned_episode`
2. ✅ Added `--policy-conditioned` CLI flag
3. ✅ Added `--no-emit-candidate` CLI flag
4. ✅ Added `--project-root` CLI flag
5. ✅ Implemented `_load_active_policy()` method
6. ✅ Implemented `_condition_survivors()` method
7. ✅ Modified `_run_inner_episode()` to apply conditioning
8. ✅ Implemented `_emit_candidate_9b()` with lineage tracking
9. ✅ Updated summary to show conditioning stats
10. ✅ Bumped version to v1.1.0

### Test Results

```
Module tests: 41 passed, 0 failed
Dry run: ✅ Policy conditioning detected as ENABLED
```

---

## Phase 9B.1: Policy Transform Module ✅ COMPLETE

**Completed:** 2026-01-30  
**File:** `policy_transform.py` v1.0.0

### Core API

| Function | Purpose |
|----------|---------|
| `apply_policy(survivors, policy)` | Pure functional transform |
| `compute_policy_fingerprint(policy)` | SHA256[:16] for deduplication |
| `validate_policy_schema(policy)` | Schema validation |
| `create_empty_policy(policy_id)` | Factory for baseline |

### Transform Operations (Fixed Order)

| Order | Transform | Purpose | Safety |
|-------|-----------|---------|--------|
| 1 | filter | Remove below threshold | min_survivors floor |
| 2 | weight | Adjust scores | Normalized 0-1 |
| 3 | mask | Hide features | Forbids score/holdout_hits/seed |
| 4 | window | Restrict index range | ABSOLUTE_MIN check |

### Team Beta Decisions (Ratified)

| Decision | Ruling |
|----------|--------|
| Min survivors | Configurable + ABSOLUTE_MIN_SURVIVORS = 50 |
| Transform order | Fixed (not policy-specified) |
| Weight normalization | Always normalize to 0-1 |
| Mask scope | `features.*` only |
| Fingerprint scope | Include safety params, exclude metadata |

### Test Results

```
RESULTS: 20 passed, 0 failed
✅ ALL TESTS PASSED
```

### Policy Schema v1.0

```json
{
  "policy_id": "policy_selfplay_20260130_ep005",
  "parent_policy_id": "policy_selfplay_20260130_ep004",
  "created_at": "2026-01-30T15:30:00Z",
  "episode_number": 5,
  "transforms": {
    "filter": {
      "enabled": true,
      "field": "holdout_hits",
      "operator": "gte",
      "threshold": 0.5,
      "min_survivors": 60
    },
    "weight": {
      "enabled": true,
      "field": "features.temporal_stability_mean",
      "method": "linear",
      "params": {"alpha": 0.4}
    },
    "mask": {
      "enabled": true,
      "exclude_features": ["skip_entropy", "survivor_velocity"]
    },
    "window": {
      "enabled": false
    }
  },
  "fitness": null,
  "fingerprint": "37dac11f249885c2"
}
```

---

## Phase 9B.2: Policy-Conditioned Mode 🔲 NOT STARTED

**Purpose:** Integrate `apply_policy()` into selfplay_orchestrator.py

### Tasks

| Task | Status | Notes |
|------|--------|-------|
| Load `learned_policy_active.json` at episode start | 🔲 | Read from Chapter 13 |
| Call `apply_policy()` on survivors | 🔲 | Before inner episode training |
| Track policy lineage | 🔲 | parent_policy_id chain |
| Emit candidate with fingerprint | 🔲 | In learned_policy_candidate.json |
| `--policy-conditioned` CLI flag | 🔲 | Enable/disable feature |

### Integration Pattern

```python
# In selfplay_orchestrator.py

from policy_transform import apply_policy, compute_policy_fingerprint

def run_inner_episode(survivors, active_policy, episode_number):
    # Apply active policy to survivors
    result = apply_policy(survivors, active_policy)
    conditioned_survivors = result.survivors
    
    # Log transform
    for log_entry in result.transform_log:
        logger.info(f"[Episode {episode_number}] {log_entry}")
    
    # Train on conditioned data
    model_result = inner_episode_trainer.train(
        survivors=conditioned_survivors,
        model_type="catboost"
    )
    
    # New policy inherits from active
    new_policy = {
        "policy_id": f"policy_selfplay_{timestamp}_ep{episode_number:03d}",
        "parent_policy_id": active_policy.get("policy_id"),
        "transforms": propose_new_transforms(model_result),
        "fingerprint": compute_policy_fingerprint(new_policy)
    }
    
    return new_policy
```

---

## Phase 9A: Chapter 13 ↔ Selfplay Hooks ✅ COMPLETE

**Completed:** 2026-01-30  
**Commits:** 358e615, d90fcdc

### 9A.1 Acceptance Engine Extensions

| Task | Status | Details |
|------|--------|---------|
| `SelfplayCandidate` dataclass | ✅ | Schema validation |
| `validate_selfplay_candidate()` | ✅ | Fitness/R²/gap thresholds |
| `promote_candidate()` | ✅ | Writes `learned_policy_active.json` |
| `--validate-selfplay` CLI | ✅ | Test candidate validation |
| `--promote` CLI | ✅ | Execute promotion |
| `telemetry.record_promotion()` | ✅ | Audit trail |

### 9A Data Flow

```
Selfplay                     Chapter 13                    WATCHER
────────                     ──────────                    ───────
learned_policy_candidate.json
        │
        └──────────────────► validate_selfplay_candidate()
                                      │
                                      ▼
                             ACCEPT / REJECT / ESCALATE
                                      │
                                      ▼ (if ACCEPT)
                             learned_policy_active.json
                             telemetry.record_promotion()
                                      │
                                      ▼ (if retrain needed)
                             request_selfplay() ──────────► watcher_requests/*.json
```

---

## Phase 8: Selfplay Integration ✅ COMPLETE

**Completed:** 2026-01-30

| File | Version | Purpose |
|------|---------|---------|
| `selfplay_orchestrator.py` | 1.0.6 | Main orchestration |
| `inner_episode_trainer.py` | 1.0.3 | CPU tree model trainer |
| `modules/learning_telemetry.py` | 1.1.1 | Telemetry system |
| `configs/selfplay_config.json` | — | Configuration |

**Benchmark Results (Zeus, n_jobs=22):**

| Model | Time | R² | Fitness |
|-------|------|-----|---------|
| LightGBM | 760ms | 0.9999 | 0.8245 |
| XGBoost | 520ms | 1.0000 | 0.3500 |
| CatBoost | 13,200ms | 1.0000 | **0.8474** 🏆 |

---

## Session History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-12 | 1.0.0 | Initial document, Phases 1-6 complete |
| 2026-01-18 | 1.1.0 | Added Phase 7 testing framework |
| 2026-01-23 | 1.2.0 | NPZ v3.0 integration notes |
| 2026-01-27 | 1.3.0 | GPU stability improvements |
| 2026-01-29 | 1.5.0 | Phase 8 Selfplay architecture approved |
| 2026-01-30 | 1.6.0 | Phase 8 COMPLETE — Zeus integration verified |
| 2026-01-30 | 1.7.0 | Phase 9A COMPLETE — Hooks verified |
| **2026-01-30** | **1.8.0** | **Phase 9B.1 COMPLETE — Policy Transform Module** |
| **2026-01-30** | **1.8.1** | **Phase 9B.2 MODULES COMPLETE — Integration pending** |

---

## Files Reference

### Phase 9B.1 Files Created

| File | Purpose |
|------|---------|
| `policy_transform.py` | Core transform module (v1.0.0) |

### Phase 9B.2 Files Created

| File | Purpose |
|------|---------|
| `policy_conditioned_episode.py` | Episode conditioning (v1.0.0) |
| `PHASE_9B2_INTEGRATION_SPEC.md` | Integration guide for selfplay_orchestrator |
| `selfplay_orchestrator.py` | **Updated to v1.1.0** — Phase 9B.2 integration |

### Runtime Artifacts

| File | Written By | Read By |
|------|------------|---------|
| `learned_policy_candidate.json` | Selfplay | Chapter 13 |
| `learned_policy_active.json` | Chapter 13 | Pipeline, Selfplay (9B.2) |
| `telemetry/learning_health.jsonl` | Selfplay | Chapter 13, WATCHER |
| `telemetry/learning_health_latest.json` | Selfplay | Dashboards |
| `watcher_requests/*.json` | Chapter 13 | WATCHER |

---

## Critical Design Invariants

### Chapter 13 Invariant
**Chapter 13 v1 does not alter model weights directly. All learning occurs through controlled re-execution of Step 5 with expanded labels.**

### Selfplay Invariant
**GPU sieving work MUST use coordinator.py / scripts_coordinator.py. Direct SSH to rigs for GPU work is FORBIDDEN.**

### Learning Authority Invariant
**Learning is statistical (tree models + bandit). Verification is deterministic (Chapter 13). LLM is advisory only. Telemetry is observational only.**

### Policy Transform Invariant (New — Phase 9B.1)
**`apply_policy()` is pure functional: stateless, deterministic, never fabricates data. Same inputs always produce same outputs.**

---

*Update this document as implementation progresses.*
