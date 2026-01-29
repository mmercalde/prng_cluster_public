# Chapter 13 Implementation Progress

**Last Updated:** 2026-01-29  
**Document Version:** 1.5.0  
**Status:** Phases 1-6 Complete → Phase 7 Testing → Phase 8 Selfplay Integration  
**Team Beta Endorsement:** ✅ Approved (including Selfplay Architecture v1.1)

---

## Overall Progress

| Phase | Status | Owner | Target |
|-------|--------|-------|--------|
| 1. Draw Ingestion | ✅ Complete | Claude | Week 1 |
| 2. Diagnostics Engine | ✅ Complete | Claude | Week 1-2 |
| 3. Retrain Triggers | ✅ Complete | Claude | Week 2 |
| 4. LLM Integration | ✅ Complete | Claude | Week 3 |
| 5. Acceptance Engine | ✅ Complete | Claude | Week 3 |
| 6. WATCHER Orchestration | ✅ Complete | Claude | Week 4 |
| 7. Testing & Validation | 🟡 In Progress | TBD | Week 4 |
| **8. Selfplay Integration** | 🟡 **In Progress** | Team Beta | Week 5 |

**Legend:** 🔲 Not Started | 🟡 In Progress | ✅ Complete | ❌ Blocked

---

## ⚠️ CRITICAL: Coordination Requirements

**GPU work MUST use existing coordinators to prevent ROCm/SSH storms.**

| Work Type | Direct SSH OK? | Use Coordinator? | Stagger Required? |
|-----------|----------------|------------------|-------------------|
| GPU Sieving (Outer Episode) | ❌ **NO** | ✅ **YES (mandatory)** | ✅ YES (0.3s) |
| CPU ML Training (Inner Episode) | ✅ YES | Optional | ❌ NO |

See: `SELFPLAY_ARCHITECTURE_PROPOSAL_v1_0.md` for full details.

---

## Phase 8: Selfplay Integration 🟡 IN PROGRESS

**New in v1.5.0** — Integrates selfplay architecture approved by Team Beta on 2026-01-29.

### 8.1 Selfplay Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SELFPLAY ORCHESTRATOR                           │
│                                                                     │
│  CRITICAL: Does NOT directly SSH to rigs for GPU work!             │
│  Uses existing proven coordinators to prevent ROCm/SSH storms.     │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    OUTER EPISODE (Sieving)                          │
│                                                                     │
│  COORDINATION: Uses coordinator.py / scripts_coordinator.py        │
│  Framework: PyTorch (GPU vectorized operations)                    │
│                                                                     │
│  Zeus:      2× RTX 3080 Ti (CUDA)     → Sieving workers            │
│  rig-6600:  12× RX 6600 (ROCm)        → Sieving workers            │
│  rig-6600b: 12× RX 6600 (ROCm)        → Sieving workers            │
│                                                                     │
│  Total: 26 GPU workers (coordinated, not direct SSH)               │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    INNER EPISODE (ML Training)                      │
│                                                                     │
│  Models: LightGBM, XGBoost, CatBoost ONLY (NO neural_net)         │
│  Device: CPU only (GPU is 8-11x slower for tree models)           │
│                                                                     │
│  Zeus i9-9920X:    3× CPU workers (8 threads each) = ~30 models/s │
│  rig-6600 i5-9400: 2× CPU workers (3 threads each) = ~10 models/s │
│  rig-6600b i5-8400: 2× CPU workers (3 threads each) = ~11 models/s│
│                                                                     │
│  Total: 7 parallel CPU workers = ~50 models/sec                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 Selfplay Tasks

| Task | Status | Notes |
|------|--------|-------|
| Phase 1 verification complete | ✅ | Coordinators exist, packages verified |
| CPU throughput benchmarked | ✅ | Zeus: 10.6 models/sec |
| Learning Telemetry module | 🔲 | `modules/learning_telemetry.py` |
| Selfplay Orchestrator | 🔲 | `selfplay_orchestrator.py` |
| Inner Episode Trainer | 🔲 | `inner_episode_trainer.py` |
| Worker Pool Manager | 🔲 | `worker_pool_manager.py` |
| Configuration files | 🔲 | `configs/selfplay_config.json` |
| Chapter 13 telemetry hooks | 🔲 | Integration with diagnostics |
| End-to-end selfplay test | 🔲 | Full cycle validation |

### 8.3 Learning Telemetry (Observability Only)

**Purpose:** Provides visibility into learning progress WITHOUT controlling decisions.

```json
{
  "learning_health": {
    "timestamp": "2026-01-29T12:00:00Z",
    "inner_episode_throughput": 68.2,
    "policy_entropy": 0.41,
    "recent_reward_trend": "+3.2%",
    "last_promotion_days_ago": 4,
    "models_trained_total": 1247,
    "current_best_policy": "policy_v3_2_1",
    "survivor_count_avg": 2340,
    "training_time_avg_ms": 142
  }
}
```

| Metric | Healthy Range | Warning Threshold |
|--------|---------------|-------------------|
| `inner_episode_throughput` | 50-80 models/sec | < 30 |
| `policy_entropy` | 0.2-0.6 | < 0.1 (premature convergence) |
| `recent_reward_trend` | > -5% | < -10% (regression) |
| `last_promotion_days_ago` | < 14 days | > 21 (stalled) |

**Critical Constraint:** Telemetry is **READ-ONLY**. No system may use it for automated decisions.

### 8.4 Model Selection (Inner Episode)

| Model | Device | Use Case | Status |
|-------|--------|----------|--------|
| LightGBM | CPU | Default (fastest) | ✅ Approved |
| XGBoost | CPU | Alternative | ✅ Approved |
| CatBoost | CPU | Best accuracy | ✅ Approved |
| ~~Neural Net~~ | ~~GPU~~ | ~~FORBIDDEN~~ | ❌ **500,000x worse MSE** |

**Benchmark Results (January 29, 2026):**

| Rig | CPU (12 models) | GPU (12 models) | CPU Advantage |
|-----|-----------------|-----------------|---------------|
| rig-6600 | **1.12s** | 8.79s | **7.9x faster** |
| rig-6600b | **1.08s** | 11.92s | **11x faster** |
| Zeus | **1.14s** | — | **10.6 models/sec** |

### 8.5 Selfplay Files to Create

| File | Purpose | Status |
|------|---------|--------|
| `selfplay_orchestrator.py` | Main orchestration | 🔲 |
| `inner_episode_trainer.py` | CPU model trainer | 🔲 |
| `worker_pool_manager.py` | Worker management | 🔲 |
| `modules/learning_telemetry.py` | Telemetry module | 🔲 |
| `configs/selfplay_config.json` | Configuration | 🔲 |

### 8.6 Selfplay ↔ Chapter 13 Integration

| Chapter 13 Component | Selfplay Integration |
|---------------------|----------------------|
| `chapter_13_orchestrator.py` | Triggers selfplay outer episodes |
| `chapter_13_diagnostics.py` | Consumes learning telemetry (read-only) |
| `chapter_13_triggers.py` | May trigger selfplay retraining |
| `chapter_13_acceptance.py` | Validates selfplay policy promotions |

**Authority Model (Unchanged):**
- Learning happens **statistically** (tree models + bandit policy)
- Verification happens **deterministically** (Chapter 13)
- Telemetry happens **observationally** (no control path)
- LLM role remains **advisory only**

---

## Phase 1: Draw Ingestion ✅ COMPLETE

| Task | Status | Notes |
|------|--------|-------|
| Create `draw_ingestion_daemon.py` | ✅ | v1.0.0 - Directory watch + flag watch modes |
| Create `synthetic_draw_injector.py` | ✅ | v1.0.0 - Reads PRNG from config, uses registry |
| Create `watcher_policies.json` | ✅ | v1.0.0 - Full threshold config |
| Append-only history update logic | ✅ | Implemented in daemon |
| Fingerprint change detection | ✅ | SHA256-based detection |
| Test: Manual injection | ✅ | `--inject-one` mode ready |
| Test: Daemon injection | ✅ | `--daemon` mode ready |

**Blockers:** None  
**Completion Date:** 2026-01-12

### Phase 1 Deliverables

| File | Version | Lines | Description |
|------|---------|-------|-------------|
| `synthetic_draw_injector.py` | 1.0.0 | ~450 | Synthetic draw generation using config-based PRNG |
| `draw_ingestion_daemon.py` | 1.0.0 | ~450 | Draw monitoring and history management |
| `watcher_policies.json` | 1.0.0 | ~120 | Test mode settings and Chapter 13 thresholds |

---

## Phase 2: Diagnostics Engine ✅ COMPLETE

| Task | Status | Notes |
|------|--------|-------|
| Create `chapter_13_diagnostics.py` | ✅ | v1.0.0 - Core diagnostic generator |
| Prediction vs reality comparison | ✅ | Hit rate, rank, near-hits, coverage |
| Confidence calibration metrics | ✅ | Predicted vs actual correlation |
| Survivor performance tracking | ✅ | Hit/decay/reinforce candidates |
| Feature drift detection | ✅ | Entropy, turnover, schema hash |
| Generate `post_draw_diagnostics.json` | ✅ | Output artifact |
| Create `diagnostics_history/` archival | ✅ | Historical storage |
| Test: Diagnostic accuracy | ✅ | Validated with mock data |

**Blockers:** None  
**Completion Date:** 2026-01-12

---

## Phase 3: Retrain Trigger Logic ✅ COMPLETE

| Task | Status | Notes |
|------|--------|-------|
| Define thresholds in `watcher_policies.json` | ✅ | Done in Phase 1 |
| Create `chapter_13_triggers.py` | ✅ | v1.0.0 - Team Beta approved separation |
| Implement `should_retrain()` | ✅ | Quick boolean check |
| Implement `evaluate_triggers()` | ✅ | Full evaluation with metrics |
| Implement `execute_learning_loop()` | ✅ | Runs Steps 3→5→6 |
| Implement partial rerun logic | ✅ | Configurable step list |
| Implement cooldown enforcement | ✅ | CooldownState tracking |
| Human approval gate | ✅ | v1 requirement enforced |
| Test: Trigger conditions | ✅ | All 6 triggers implemented |

**Blockers:** None  
**Completion Date:** 2026-01-12

### Trigger Conditions Implemented

| Trigger | Threshold | Action |
|---------|-----------|--------|
| `consecutive_misses` | ≥5 | Learning Loop (3→5→6) |
| `confidence_drift` | correlation < 0.2 | Learning Loop |
| `hit_rate_collapse` | < 0.01 | Learning Loop |
| `n_draws_accumulated` | ≥10 | Learning Loop |
| `regime_shift` | decay > 0.5 AND churn > 0.4 | Full Pipeline (1→6) |
| `RETRAIN_RECOMMENDED` flag | From diagnostics | Learning Loop |

---

## Phase 4: LLM Integration ✅ COMPLETE

| Task | Status | Notes |
|------|--------|-------|
| Create `chapter_13_llm_advisor.py` | ✅ | v1.0.0 - LLM analysis module |
| Create `llm_proposal_schema.py` | ✅ | v1.0.0 - Dataclass models |
| Create `chapter_13.gbnf` | ✅ | v1.1 - Grammar constraint |
| System prompt template | ✅ | Strategist role with hard constraints |
| User prompt template | ✅ | Diagnostic analysis format |
| Integration with existing LLM infra | ✅ | LLMRouter + grammar |
| Test: DeepSeek grammar-constrained | ✅ | Verified working |
| Test: Claude backup fallback | ✅ | Verified working |
| Test: Heuristic fallback | ✅ | Verified working |

**Blockers:** None  
**Completion Date:** 2026-01-12

### LLM Role Enforced (Advisory Only)

| Allowed | Forbidden |
|---------|-----------|
| ✅ Interpret diagnostics | ❌ Modify files |
| ✅ Propose parameter adjustments | ❌ Execute code |
| ✅ Flag regime shifts | ❌ Apply parameters directly |
| ✅ Explain performance changes | ❌ Override WATCHER |

---

## Phase 5: Acceptance/Rejection Engine ✅ COMPLETE

| Task | Status | Notes |
|------|--------|-------|
| Create `chapter_13_acceptance.py` | ✅ | v1.0.0 - Validation engine |
| Implement `validate_proposal()` | ✅ | Full validation pipeline |
| Enforce 30% max delta | ✅ | `max_parameter_delta` check |
| Enforce 3 param max | ✅ | `max_parameters_per_proposal` |
| Enforce cooldown periods | ✅ | `ParameterHistory` tracking |
| Enforce frozen parameters | ✅ | Steps 1, 2, 4 params protected |
| Reversal detection | ✅ | `would_reverse()` check |
| Escalation logic | ✅ | Risk level, flags, failures |
| Create `acceptance_decisions.jsonl` | ✅ | Audit trail |

**Blockers:** None  
**Completion Date:** 2026-01-12

---

## Phase 6: WATCHER Orchestration ✅ COMPLETE

| Task | Status | Notes |
|------|--------|-------|
| Create `chapter_13_orchestrator.py` | ✅ | v1.0.0 - Main daemon |
| New draw detection (flag monitoring) | ✅ | Monitors `new_draw.flag` |
| Run diagnostics on trigger | ✅ | Calls `chapter_13_diagnostics` |
| Evaluate retrain triggers | ✅ | Uses `Chapter13TriggerManager` |
| Query LLM (optional) | ✅ | Uses `Chapter13LLMAdvisor` |
| Validate proposals | ✅ | Uses `Chapter13AcceptanceEngine` |
| Human approval gate | ✅ | v1 requirement enforced |

**Blockers:** None  
**Completion Date:** 2026-01-12

---

## Phase 7: Testing & Validation 🟡 IN PROGRESS

| Task | Status | Notes |
|------|--------|-------|
| Module import validation | ✅ | All modules import cleanly |
| Synthetic draw convergence test | 🔲 | True seed rises in rankings |
| Forced retrain validation | 🔲 | Steps 3→5→6 execute |
| Proposal rejection test | 🔲 | Bounds enforced |
| Divergence detection test | 🔲 | Halt on instability |
| Cooldown enforcement test | 🔲 | Thrashing prevented |
| Full autonomy test (100 draws) | 🔲 | Extended run |

**Blockers:** None

---

## Convergence Metrics

### Chapter 13 Metrics (Test Mode)

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Hit Rate (Top-20) | > 5% | - | 🔲 |
| Confidence Calibration | Correlation > 0.3 | - | 🔲 |
| Confidence trend | Increasing | - | 🔲 |

### Selfplay Metrics (Learning Telemetry)

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Inner episode throughput | ≥50 models/sec | - | 🔲 |
| Policy entropy | 0.2-0.6 | - | 🔲 |
| Reward trend | > -5% | - | 🔲 |
| Promotion frequency | < 14 days | - | 🔲 |

---

## Files Created/Modified

### Chapter 13 Core Files

| File | Status | Purpose |
|------|--------|---------|
| `draw_ingestion_daemon.py` | ✅ v1.0.0 | Draw monitoring |
| `synthetic_draw_injector.py` | ✅ v1.0.0 | Test mode |
| `watcher_policies.json` | ✅ v1.0.0 | Config |
| `chapter_13_diagnostics.py` | ✅ v1.0.0 | Diagnostic engine |
| `chapter_13_triggers.py` | ✅ v1.0.0 | Retrain triggers |
| `chapter_13_llm_advisor.py` | ✅ v1.0.0 | LLM integration |
| `llm_proposal_schema.py` | ✅ v1.0.0 | Schema models |
| `chapter_13.gbnf` | ✅ v1.1.0 | Grammar constraint |
| `chapter_13_acceptance.py` | ✅ v1.0.0 | Acceptance engine |
| `chapter_13_orchestrator.py` | ✅ v1.0.0 | Main orchestration daemon |

### Selfplay Files (Phase 8)

| File | Status | Purpose |
|------|--------|---------|
| `selfplay_orchestrator.py` | 🔲 | Main selfplay orchestration |
| `inner_episode_trainer.py` | 🔲 | CPU tree model trainer |
| `worker_pool_manager.py` | 🔲 | Worker management |
| `modules/learning_telemetry.py` | 🔲 | Telemetry module |
| `configs/selfplay_config.json` | 🔲 | Selfplay configuration |

### Documentation

| File | Status | Purpose |
|------|--------|---------|
| `SELFPLAY_ARCHITECTURE_PROPOSAL_v1_0.md` | ✅ | Approved architecture |
| `SELFPLAY_INTEGRATION_PROGRESS_v1_0.md` | ✅ | Implementation tracker |
| `CHAPTER_13_IMPLEMENTATION_PROGRESS_v1_5.md` | ✅ | This document |

---

## Critical Design Invariants

### Chapter 13 Invariant
**Chapter 13 v1 does not alter model weights directly. All learning occurs through controlled re-execution of Step 5 with expanded labels.**

### Selfplay Invariant
**GPU sieving work MUST use coordinator.py / scripts_coordinator.py. Direct SSH to rigs for GPU work is FORBIDDEN.**

### Learning Authority Invariant
**Learning is statistical (tree models + bandit). Verification is deterministic (Chapter 13). LLM is advisory only. Telemetry is observational only.**

---

## Session History

| Date | Event |
|------|-------|
| 2026-01-11 | Chapter 13 spec finalized (v1.1) |
| 2026-01-12 | Phases 1-6 implementation complete |
| 2026-01-13 | watcher_policies.json finalized |
| 2026-01-14 to 2026-01-17 | ROCm stability investigation (4.5× perf improvement) |
| 2026-01-18 | Resumed Chapter 13 testing, functional mimicry paradigm documented (v1.2) |
| 2026-01-28-29 | LightGBM GPU benchmarking, CPU wins confirmed (8-11x faster) |
| **2026-01-29** | **Selfplay Architecture Proposal v1.1 approved by Team Beta** |
| **2026-01-29** | **Phase 8 (Selfplay Integration) added, Phase 1 verification complete** |

---

## Deferred Extensions (Future Work)

| Extension | Description | Status |
|-----------|-------------|--------|
| #1 | Step-6 Backtesting Hooks | 🔲 Deferred |
| #2 | Confidence Calibration Curves | 🔲 Deferred |
| #3 | Autonomous Trigger Execution (no approval) | 🔲 Deferred |
| #4 | Convergence Dashboards | 🔲 Deferred |

---

## Next Actions

1. [x] ~~Phases 1-6 complete~~
2. [x] ~~Selfplay architecture proposal approved~~
3. [x] ~~Phase 1 selfplay verification (coordinators, packages, throughput)~~
4. [ ] Create `modules/learning_telemetry.py`
5. [ ] Create `inner_episode_trainer.py`
6. [ ] Create `selfplay_orchestrator.py`
7. [ ] Create `configs/selfplay_config.json`
8. [ ] Integrate telemetry with Chapter 13 diagnostics
9. [ ] End-to-end selfplay test
10. [ ] Phase 7: Full pipeline convergence test

---

*Update this document as implementation progresses.*
