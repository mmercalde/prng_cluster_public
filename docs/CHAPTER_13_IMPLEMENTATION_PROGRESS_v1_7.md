# Chapter 13 Implementation Progress

**Last Updated:** 2026-01-30  
**Document Version:** 1.7.0  
**Status:** Phases 1-8 Complete, Phase 9A Complete → Phase 9B Policy Conditioning  
**Team Beta Endorsement:** ✅ Approved (Phase 9A verified on Zeus)

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
| **9A. Chapter 13 ↔ Selfplay Hooks** | ✅ **COMPLETE** | Team Beta | **2026-01-30** |
| 9B. Policy-Conditioned Learning | 🔲 Not Started | TBD | — |

**Legend:** 🔲 Not Started | 🟡 In Progress | ✅ Complete | ❌ Blocked

---

## ⚠️ CRITICAL: Coordination Requirements

**GPU work MUST use existing coordinators to prevent ROCm/SSH storms.**

| Work Type | Direct SSH OK? | Use Coordinator? | Stagger Required? |
|-----------|----------------|------------------|-------------------|
| GPU Sieving (Outer Episode) | ❌ **NO** | ✅ **YES (mandatory)** | ✅ YES (0.3s) |
| CPU ML Training (Inner Episode) | ✅ YES | Optional | ❌ NO |

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

**CLI Usage:**
```bash
python3 chapter_13_acceptance.py --validate-selfplay learned_policy_candidate.json
python3 chapter_13_acceptance.py --promote learned_policy_candidate.json
```

### 9A.2 Diagnostics Engine Extensions

| Task | Status | Details |
|------|--------|---------|
| `LEARNING_TELEMETRY_FILE` constant | ✅ | `telemetry/learning_health_latest.json` |
| `load_learning_telemetry()` | ✅ | Safe JSON loader |
| `selfplay_health` in output | ✅ | Added to diagnostics dict |

### 9A.3 Triggers Engine Extensions

| Task | Status | Details |
|------|--------|---------|
| `TriggerType.SELFPLAY_RETRAIN` | ✅ | Enum (no auto-dispatch) |
| `TriggerAction.SELFPLAY` | ✅ | Enum (WATCHER dispatches) |
| `request_selfplay()` | ✅ | Creates request artifact |
| `should_request_selfplay()` | ✅ | Explicit triggers only |
| `watcher_requests/` directory | ✅ | Append-only audit |

**Contract Compliance (Team Beta Verified):**
- ✅ Chapter 13 requests, does not execute
- ✅ WATCHER remains sole execution gate
- ✅ No learning semantics in 9A
- ✅ Request files are append-only (unique IDs)

### 9A.4 Governance File

| Task | Status | Details |
|------|--------|---------|
| `watcher_policies.json` tracked | ✅ | Team Beta ruling |
| `selfplay` section added | ✅ | Validation thresholds |

---

## Phase 9A Data Flow
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

## Phase 9B: Policy-Conditioned Learning 🔲 NOT STARTED

**Purpose:** Policies decide what data is seen next, not outcomes.

### 9B Tasks (From Team Beta Briefing)

| Task | Status | Notes |
|------|--------|-------|
| `apply_policy(survivors, policy)` | 🔲 | Stateless, deterministic |
| Policy fingerprinting | 🔲 | Hash params, detect duplicates |
| `--policy-conditioned` mode | 🔲 | Sequential episode dependency |

### 9B Principles

- Policies filter/weight/mask/window — never fabricate data
- Episodes become sequential (policy_N influences Episode N+1)
- Fitness measures directional improvement
- All Phase 9A invariants remain locked

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
| **2026-01-30** | **1.7.0** | **Phase 9A COMPLETE — Hooks verified** |

---

## Files Reference

### Phase 9A Files Modified
| File | Changes |
|------|---------|
| `chapter_13_acceptance.py` | +validate_selfplay_candidate, +promote_candidate |
| `chapter_13_diagnostics.py` | +load_learning_telemetry, +selfplay_health |
| `chapter_13_triggers.py` | +request_selfplay, +should_request_selfplay |
| `watcher_policies.json` | +selfplay section (now tracked) |

### Runtime Artifacts
| File | Written By | Read By |
|------|------------|---------|
| `learned_policy_candidate.json` | Selfplay | Chapter 13 |
| `learned_policy_active.json` | Chapter 13 | Pipeline |
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

### Phase 9A Invariant (NEW)
**Chapter 13 requests selfplay via artifacts. WATCHER authorizes execution. Selfplay cannot promote itself.**

---

*Document maintained by Team Beta*
