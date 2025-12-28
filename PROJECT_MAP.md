# 🗂 Distributed PRNG Analysis — Logical Project Map
📌 *AI-Friendly, Developer-Friendly — Updated December 27, 2025 (Session 17)*

> This structure explains **how your system is organized logically**,
> without physically changing file locations on disk.
> It is designed for Claude, ChatGPT, GitHub navigation, and future refactoring.

──────────────────────────────────────────────
🚀 CORE PIPELINE — 6-STEP EXECUTION FLOW
──────────────────────────────────────────────

| Step | Script | Purpose | Output |
|------|--------|---------|--------|
| ~~0~~ | ~~PRNG Fingerprinting~~ | ~~Classify unknown PRNGs~~ | **ARCHIVED** (Session 17) |
| 1 | `window_optimizer.py` | Bayesian window optimization | `bidirectional_survivors.json` |
| 2.5 | `generate_scorer_jobs.py` | Distributed scoring meta-optimizer | `optimal_scorer_config.json` |
| 3 | `run_step3_full_scoring.sh` | Full GPU scoring (64 features) | `survivors_with_scores.json` |
| 4 | `adaptive_meta_optimizer.py` | ML meta-optimizer | `reinforcement_engine_config.json` |
| 5 | `meta_prediction_optimizer_anti_overfit.py` | Multi-model training | `best_model.{cbm,json,pth,txt}` + sidecar |
| 6 | `prediction_generator.py` | Generate predictions | `predictions_*.json` |

### Step 0: PRNG Fingerprinting — ARCHIVED
Investigated in Session 17. **Mathematically impossible** under mod1000 projection:
- SNR < 0.15 for ALL features tested
- Within-PRNG variance dominates between-PRNG variance
- Alternative: Trust the sieve (wrong PRNG → 0 survivors)

──────────────────────────────────────────────
🧠 MULTI-MODEL ARCHITECTURE (v3.2.0)
──────────────────────────────────────────────

models/
├── __init__.py                    # Exports all model components
├── global_state_tracker.py        # 14 global features (GPU-neutral)
├── feature_schema.py              # Streaming schema derivation + hash
├── model_factory.py               # Model loader with sidecar support
├── model_selector.py              # Best model selection logic
└── wrappers/
    ├── base.py                    # ModelInterface protocol
    ├── neural_net_wrapper.py      # PyTorch NN (ROCm + CUDA)
    ├── xgboost_wrapper.py         # XGBoost (CUDA)
    ├── lightgbm_wrapper.py        # LightGBM (OpenCL)
    └── catboost_wrapper.py        # CatBoost (CUDA) 🏆 Session 17 winner

### Subprocess Isolation (OpenCL/CUDA Conflict Resolution)
```
Main Process (coordinator) - NO GPU imports
    │
    ├── Trial 0: subprocess → LightGBM (OpenCL) → exits
    ├── Trial 1: subprocess → PyTorch (CUDA) → exits  
    ├── Trial 2: subprocess → XGBoost (CUDA) → exits
    ├── Trial 3: subprocess → CatBoost (CUDA) → exits
    └── Trial N: Fresh GPU state each time
```

Files:
- `train_single_trial.py` - Isolated worker script
- `subprocess_trial_coordinator.py` - Coordinates subprocess execution

### Session 17 Multi-Model Results (62 features)
| Model | R² | MSE | Duration |
|-------|-----|-----|----------|
| CatBoost | 1.0000 | 8.6e-11 | 4.8s 🏆 |
| XGBoost | 1.0000 | 1.0e-07 | 1.8s |
| LightGBM | 0.9999 | 2.1e-07 | 2.9s |
| Neural Net | 0.0000 | 0.0025 | 253s+ |

──────────────────────────────────────────────
🤖 AI AGENT ARCHITECTURE
──────────────────────────────────────────────

agent_manifests/
├── window.json              # Step 1 manifest
├── scorer_meta.json         # Step 2.5 manifest
├── full_scoring.json        # Step 3 manifest
├── ml_meta.json             # Step 4 manifest
├── reinforcement.json       # Step 5 manifest (v1.5.0)
└── prediction.json          # Step 6 manifest (v1.5.0)

integration/
├── metadata_writer.py       # inject_agent_metadata() + lineage
├── context_builder.py       # Build LLM context from artifacts
└── artifact_handler.py      # JSON artifact I/O

watcher_agent.py             # Autonomous pipeline orchestration (WIP)

### Step 5 → Step 6 Handoff Protocol
```
Step 5 Output:
├── best_model.cbm (CatBoost wins - Session 17)
└── best_model.meta.json (sidecar)
    └── agent_metadata.run_id: "step5_20251226_235017"

Step 6 Input:
├── Reads sidecar → auto-detects model type
├── Extracts parent_run_id from sidecar
└── Outputs predictions with lineage chain
```

──────────────────────────────────────────────
📊 SCORING & FEATURES (Updated Session 17)
──────────────────────────────────────────────

survivor_scorer.py
│   • 50 per-seed features extraction
│   • _generate_sequence() - Dynamic PRNG lookup
│   • _coerce_seed_list() - Type-tolerant (int/dict)
│   • compute_dual_sieve_intersection() - Bidirectional filtering

models/global_state_tracker.py
│   • 14 global features (lottery-level statistics)
│   • SciPy fallback for entropy calculation
│   • GPU-neutral (importable anywhere)

run_step3_full_scoring.sh
│   • Phase 5 Aggregation: Merges global features
│   • GlobalStateTracker computed once (O(1))
│   • Features prefixed with 'global_' (Team Beta)

### Feature Architecture (64 total, 62 for training)
```
Total Features: 64 (in survivors_with_scores.json)
Training Features: 62 (after excluding score, confidence)

├── Per-seed features: 50 (from survivor_scorer.py)
│   ├── Residue features: 12
│   ├── Temporal features: 20
│   ├── Statistical features: 12
│   ├── Metadata features: 4 (skip_min, skip_max, bidirectional_count, bidirectional_selectivity)
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

### Feature Registry
`config_manifests/feature_registry.json` - Documents all 64 features with metadata

──────────────────────────────────────────────
🎯 STEP 6 OUTPUT CONTRACT (v2.2)
──────────────────────────────────────────────
```json
{
    "predictions": [521, 626, 415],
    "raw_scores": [0.127, 0.108, 0.057],           // Machine truth
    "confidence_scores": [0.79, 0.68, 0.32],       // Calibrated (sigmoid z-score)
    "confidence_scores_normalized": [1.0, 0.85, 0.45],  // Human display
    "metadata": {
        "score_stats": {
            "raw_min": 0.0001,
            "raw_max": 0.127,
            "raw_std": 0.034,
            "raw_unique": 10
        }
    },
    "agent_metadata": {
        "pipeline_step": 6,
        "parent_run_id": "step5_20251226_235017"   // Lineage
    }
}
```

──────────────────────────────────────────────
🖥 MULTI-NODE CLUSTER
──────────────────────────────────────────────

| Node | GPUs | Backend | Purpose |
|------|------|---------|---------|
| Zeus (primary) | 2× RTX 3080 Ti | CUDA | Orchestration, LLM inference |
| rig-6600 | 12× RX 6600 | ROCm | Worker Node 1 |
| rig-6600b | 12× RX 6600 | ROCm | Worker Node 2 |

**Total: 26 GPUs, ~285 TFLOPS**

coordinator.py (v1.8.2)
│   • Master controller for distributed execution
│   • SSH orchestration, GPU job scheduling
│   • ROCm/CUDA activation per node

distributed_worker.py (v1.8.0)
│   • Runs jobs on individual GPUs
│   • Pull-based job collection

scripts_coordinator.py
│   • Parallel execution within nodes (Session 16)
│   • ThreadPoolExecutor with GPU-aware workers

ROCm Activation:
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
source ~/tf/bin/activate
```

──────────────────────────────────────────────
📁 MODULES — ORCHESTRATION & ANALYTICS
──────────────────────────────────────────────

modules/
├── mt_pipeline.py           # High-level orchestrator
├── mt_engine_exact.py       # PRNG engine logic
├── database_manager.py      # Persistence storage
├── file_manager.py          # Safe JSON/CSV I/O
├── performance_analytics.py # GPU usage tracking
├── system_monitor.py        # Live resource monitoring
├── visualization_manager.py # 2D/3D visualizations
└── web_visualizer.py        # Web dashboard

──────────────────────────────────────────────
📘 DOCUMENTATION
──────────────────────────────────────────────

| File | Purpose |
|------|---------|
| `README.md` | Main project overview |
| `PROJECT_MAP.md` | This file - logical structure |
| `CURRENT_STATUS.txt` | Session-by-session progress |
| `IMPLEMENTATION_CHECKLIST.md` | Feature completion tracking |
| `PROPOSAL_Unified_Agent_Context_Framework_v3_2_10.md` | Latest architecture proposal |
| `COMPLETE_OPERATING_GUIDE_v1_1.md` | Full system documentation |
| `Multi-Model_Architecture_integration_autonomy.md` | Autonomy integration guide |

──────────────────────────────────────────────
⚙️ CONFIGURATION FILES
──────────────────────────────────────────────

distributed_config.json              # Node IPs, GPU mappings, SSH config
config_manifests/feature_registry.json  # Feature documentation (NEW Session 17)
agent_config.yaml                    # Meta-optimizer parameters
optimal_window_config.json           # Best window sizes (Optuna output)
prng_registry.py                    # 46 PRNG algorithm definitions

──────────────────────────────────────────────
📌 RECENT CHANGES (December 2025)
──────────────────────────────────────────────

### Session 17 (Dec 26-27)
- ❌ Step 0 PRNG Fingerprinting **ARCHIVED** (mathematically impossible under mod1000)
- ✅ Global features integrated at Step 3 Phase 5 aggregation
- ✅ Feature registry updated with `global_` prefix (Team Beta requirement)
- ✅ Added `--timeout` CLI argument to Step 5
- ✅ Multi-model test: CatBoost wins (R²=1.0, MSE=8.6e-11)
- ✅ Data quality: Found 721 duplicates in daily3.json

### Session 16 (Dec 25)
- ✅ Parallel execution in scripts_coordinator.py
- ✅ Step 4 --survivor-data argument fix
- ✅ Feature count alignment (48 per-seed features)

### Session 15 (Dec 24)
- ✅ Fixed confidence bug (was all 1.0, now differentiated)
- ✅ Added raw_scores, score_stats to output
- ✅ Implemented parent_run_id lineage

### Session 14 (Dec 23-24)
- ✅ GlobalStateTracker module (14 features)
- ✅ Type-tolerant intersection
- ✅ Model loading from sidecar

### Session 11-12 (Dec 22-23)
- ✅ Subprocess isolation for OpenCL/CUDA
- ✅ Multi-model architecture (4 ML models)
- ✅ Model checkpoint persistence
