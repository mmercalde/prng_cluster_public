# Selfplay Architecture Integration Progress
## Version 1.0 — January 29, 2026

**Status:** 🟡 IN PROGRESS  
**Last Updated:** 2026-01-29  
**Approved By:** Team Beta + Michael

---

## Quick Status Dashboard

| Component | Status | Owner | ETA |
|-----------|--------|-------|-----|
| Proposal Document | ✅ APPROVED | Team Beta | Done |
| Learning Telemetry | 🔲 NOT STARTED | — | — |
| Selfplay Orchestrator | 🔲 NOT STARTED | — | — |
| Inner Episode Trainer | 🔲 NOT STARTED | — | — |
| Coordinator Integration | 🔲 NOT STARTED | — | — |
| Configuration Files | 🔲 NOT STARTED | — | — |
| Testing & Validation | 🔲 NOT STARTED | — | — |
| Documentation | 🟡 IN PROGRESS | Claude | — |

**Legend:** ✅ Complete | 🟡 In Progress | 🔲 Not Started | ❌ Blocked

---

## Phase 1: Foundation (Prerequisites)

### 1.1 Verify Existing Infrastructure

| Task | Status | Notes |
|------|--------|-------|
| Confirm coordinator.py functional | 🔲 | Test with small job batch |
| Confirm scripts_coordinator.py functional | 🔲 | Test Step 3 execution |
| Verify ROCm stability on both rigs | 🔲 | Run rocm-smi, check for zombies |
| Verify CPU tree model packages | 🔲 | LightGBM 4.6.0, XGBoost 3.1.3, CatBoost 1.2.8 |
| Kill any zombie processes | 🔲 | `pkill -9 -f python3` on rigs |

**Validation Command:**
```bash
# On each rig
python3 -c "
import lightgbm, xgboost, catboost
print(f'LightGBM: {lightgbm.__version__}')
print(f'XGBoost: {xgboost.__version__}')
print(f'CatBoost: {catboost.__version__}')
"
```

### 1.2 Benchmark Verification

| Task | Status | Notes |
|------|--------|-------|
| Re-run CPU throughput test (12 models) | 🔲 | Expected: ~10-11 models/sec |
| Verify no GPU processes running | 🔲 | `rocm-smi` shows 0% usage |
| Document baseline metrics | 🔲 | Store in results/ |

---

## Phase 2: Core Components

### 2.1 Learning Telemetry Module

**File:** `modules/learning_telemetry.py`

| Task | Status | Notes |
|------|--------|-------|
| Create telemetry data structure | 🔲 | See schema below |
| Implement throughput tracker | 🔲 | Models/sec calculation |
| Implement policy entropy tracker | 🔲 | From bandit policy |
| Implement reward trend tracker | 🔲 | Rolling window average |
| Implement promotion tracker | 🔲 | Days since last promotion |
| Add JSON export | 🔲 | For dashboard/monitoring |
| Add logging integration | 🔲 | Write to learning_health.log |

**Telemetry Schema:**
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

### 2.2 Selfplay Orchestrator

**File:** `selfplay_orchestrator.py`

| Task | Status | Notes |
|------|--------|-------|
| Create main orchestrator class | 🔲 | Coordinates outer/inner episodes |
| Implement outer episode trigger | 🔲 | Calls coordinator.py |
| Implement inner episode trigger | 🔲 | Spawns CPU workers |
| Add Optuna integration | 🔲 | Parameter optimization |
| Add telemetry hooks | 🔲 | Update learning_health |
| Add Chapter 13 integration | 🔲 | Promotion gate checks |
| Add graceful shutdown | 🔲 | Cleanup on interrupt |

**Orchestrator Flow:**
```
┌─────────────────────────────────────────┐
│         SELFPLAY ORCHESTRATOR           │
├─────────────────────────────────────────┤
│ 1. Load configuration                   │
│ 2. Initialize telemetry                 │
│ 3. Start Optuna study                   │
│ 4. Loop:                                │
│    a. Generate trial parameters         │
│    b. Run outer episode (via coord.)    │
│    c. Run inner episode (CPU workers)   │
│    d. Calculate fitness                 │
│    e. Report to Optuna                  │
│    f. Update telemetry                  │
│    g. Check Chapter 13 promotion gate   │
│ 5. Export best parameters               │
└─────────────────────────────────────────┘
```

### 2.3 Inner Episode Trainer

**File:** `inner_episode_trainer.py`

| Task | Status | Notes |
|------|--------|-------|
| Create trainer class | 🔲 | Handles single model training |
| Implement LightGBM trainer | 🔲 | CPU, configurable threads |
| Implement XGBoost trainer | 🔲 | CPU, configurable threads |
| Implement CatBoost trainer | 🔲 | CPU, configurable threads |
| Add cross-validation | 🔲 | k=3 for selfplay speed |
| Add feature importance export | 🔲 | For analysis |
| Add model serialization | 🔲 | Save trained models |

**Trainer Configuration:**
```python
TRAINER_CONFIG = {
    "lightgbm": {
        "n_estimators": 100,
        "device": "cpu",
        "verbose": -1
    },
    "xgboost": {
        "n_estimators": 100,
        "tree_method": "hist",
        "verbosity": 0
    },
    "catboost": {
        "iterations": 100,
        "verbose": 0
    }
}
```

### 2.4 Worker Pool Manager

**File:** `worker_pool_manager.py`

| Task | Status | Notes |
|------|--------|-------|
| Create worker pool class | 🔲 | Manages CPU workers |
| Implement Zeus worker config | 🔲 | 3 workers × 8 threads |
| Implement rig worker config | 🔲 | 2 workers × 3 threads |
| Add job queue | 🔲 | Thread-safe queue |
| Add result aggregation | 🔲 | Collect from all workers |
| Add health monitoring | 🔲 | Detect stuck workers |

---

## Phase 3: Integration

### 3.1 Coordinator Integration

| Task | Status | Notes |
|------|--------|-------|
| Add selfplay job type to coordinator.py | 🔲 | Or use existing job types |
| Add selfplay job type to scripts_coordinator.py | 🔲 | For outer episodes |
| Test batching with selfplay jobs | 🔲 | Verify no SSH storms |
| Test stagger timing | 🔲 | 0.3s delay working |
| Test cooldown periods | 🔲 | Between batches |

### 3.2 Chapter 13 Integration

| Task | Status | Notes |
|------|--------|-------|
| Hook telemetry to diagnostics | 🔲 | learning_health visible |
| Implement promotion gate check | 🔲 | Chapter 13 authority |
| Add policy versioning | 🔲 | Track promoted policies |
| Test promotion workflow | 🔲 | End-to-end test |

### 3.3 WATCHER Agent Integration

| Task | Status | Notes |
|------|--------|-------|
| Add selfplay trigger to WATCHER | 🔲 | --run-selfplay flag |
| Add telemetry display | 🔲 | Show learning_health |
| Add selfplay status to dashboard | 🔲 | Web UI update |

---

## Phase 4: Configuration & Testing

### 4.1 Configuration Files

| File | Status | Notes |
|------|--------|-------|
| `configs/selfplay_config.json` | 🔲 | Main selfplay config |
| `configs/worker_allocation.json` | 🔲 | CPU thread allocation |
| `configs/telemetry_config.json` | 🔲 | Telemetry settings |

**selfplay_config.json Template:**
```json
{
  "outer_episode": {
    "coordinator": "scripts_coordinator.py",
    "seed_range": [0, 100000],
    "batch_size": 10000,
    "stagger_delay": 0.3,
    "cooldown": 2.0
  },
  "inner_episode": {
    "models": ["lightgbm", "xgboost", "catboost"],
    "device": "cpu",
    "k_folds": 3,
    "timeout_seconds": 60
  },
  "workers": {
    "zeus": {"cpu_workers": 3, "threads_per_worker": 8},
    "rig-6600": {"cpu_workers": 2, "threads_per_worker": 3},
    "rig-6600b": {"cpu_workers": 2, "threads_per_worker": 3}
  },
  "optuna": {
    "n_trials": 100,
    "study_name": "selfplay_optimization",
    "pruner": "median"
  },
  "telemetry": {
    "enabled": true,
    "log_interval_seconds": 60,
    "export_path": "results/learning_health.json"
  }
}
```

### 4.2 Testing Plan

| Test | Status | Expected Result |
|------|--------|-----------------|
| Unit: Inner episode trainer | 🔲 | Models train in <0.5s |
| Unit: Telemetry module | 🔲 | JSON export works |
| Integration: Orchestrator + Coordinator | 🔲 | No SSH storms |
| Integration: Orchestrator + Chapter 13 | 🔲 | Promotion gate works |
| End-to-end: Full selfplay cycle | 🔲 | 15-35 seconds per cycle |
| Stress: 100 Optuna trials | 🔲 | No crashes, stable throughput |

### 4.3 Validation Criteria

| Metric | Target | Status |
|--------|--------|--------|
| Inner episode throughput | ≥50 models/sec | 🔲 |
| Outer episode completion | 100% success | 🔲 |
| No ROCm/SSH storms | Zero incidents | 🔲 |
| Memory usage (rigs) | <4 GB | 🔲 |
| Selfplay cycle time | <60 seconds | 🔲 |

---

## Phase 5: Documentation & Rollout

### 5.1 Documentation

| Document | Status | Notes |
|----------|--------|-------|
| SELFPLAY_ARCHITECTURE_PROPOSAL_v1_0.md | ✅ | Approved |
| SELFPLAY_INTEGRATION_PROGRESS_v1_0.md | ✅ | This document |
| Update SYSTEM_ARCHITECTURE_REFERENCE.md | 🔲 | Add selfplay section |
| Update Chapter 13 docs | 🔲 | Add telemetry references |
| Create SELFPLAY_OPERATIONS_GUIDE.md | 🔲 | How to run/monitor |

### 5.2 Rollout Phases

| Phase | Description | Status |
|-------|-------------|--------|
| Phase A | Telemetry + Inner trainer only (Zeus local) | 🔲 |
| Phase B | Add coordinator integration (outer episodes) | 🔲 |
| Phase C | Full cluster selfplay (all nodes) | 🔲 |
| Phase D | Chapter 13 promotion integration | 🔲 |
| Phase E | Production selfplay enabled | 🔲 |

---

## Appendix A: File Inventory

### New Files to Create

```
prng_cluster_project/
├── selfplay_orchestrator.py          # Main orchestrator
├── inner_episode_trainer.py          # CPU model trainer
├── worker_pool_manager.py            # Worker management
├── modules/
│   └── learning_telemetry.py         # Telemetry module
├── configs/
│   ├── selfplay_config.json          # Main config
│   ├── worker_allocation.json        # Thread allocation
│   └── telemetry_config.json         # Telemetry settings
└── docs/
    ├── SELFPLAY_ARCHITECTURE_PROPOSAL_v1_0.md
    ├── SELFPLAY_INTEGRATION_PROGRESS_v1_0.md
    └── SELFPLAY_OPERATIONS_GUIDE.md
```

### Files to Modify

| File | Modification |
|------|--------------|
| coordinator.py | Add selfplay job type (if needed) |
| scripts_coordinator.py | Add selfplay job type (if needed) |
| agents/watcher_agent.py | Add --run-selfplay, telemetry display |
| Chapter 13 components | Add promotion gate hooks |

---

## Appendix B: Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| ROCm storm during outer episode | Low | High | Use coordinators (mandatory) |
| Memory exhaustion on rigs | Medium | Medium | Limit to 2 workers/rig |
| Stuck workers | Medium | Low | Timeout + cleanup |
| Optuna study corruption | Low | Medium | Checkpoint frequently |
| Chapter 13 promotion conflict | Low | High | Single-writer pattern |

---

## Appendix C: Command Reference

### Start Selfplay (Future)

```bash
# Full selfplay run
PYTHONPATH=. python3 selfplay_orchestrator.py \
    --config configs/selfplay_config.json \
    --trials 100

# Single cycle test
PYTHONPATH=. python3 selfplay_orchestrator.py \
    --config configs/selfplay_config.json \
    --trials 1 \
    --dry-run
```

### Monitor Telemetry

```bash
# View current learning health
cat results/learning_health.json | jq .

# Watch live updates
watch -n 5 'cat results/learning_health.json | jq .'
```

### Emergency Cleanup

```bash
# Kill all workers on rigs
ssh 192.168.3.152 'pkill -9 -f python3'
ssh 192.168.3.154 'pkill -9 -f python3'

# Verify cleanup
ssh 192.168.3.152 'rocm-smi'
ssh 192.168.3.154 'rocm-smi'
```

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-29 | 1.0 | Initial integration plan created |

---

**END OF INTEGRATION PROGRESS DOCUMENT**
