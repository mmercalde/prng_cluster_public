# 🗂 Distributed PRNG Analysis — Logical Project Map
📌 *AI-Friendly, Developer-Friendly — Updated December 24, 2025*

> This structure explains **how your system is organized logically**,
> without physically changing file locations on disk.
> It is designed for Claude, ChatGPT, GitHub navigation, and future refactoring.

──────────────────────────────────────────────
🚀 CORE PIPELINE — 6-STEP EXECUTION FLOW
──────────────────────────────────────────────

| Step | Script | Purpose | Output |
|------|--------|---------|--------|
| 1 | `window_optimizer.py` | Bayesian window optimization | `bidirectional_survivors.json` |
| 2.5 | `generate_scorer_jobs.py` | Distributed scoring meta-optimizer | `optimal_scorer_config.json` |
| 3 | `generate_full_scoring_jobs.py` | Full GPU scoring (46 features) | `survivors_with_scores.json` |
| 4 | `adaptive_meta_optimizer.py` | ML meta-optimizer | `reinforcement_engine_config.json` |
| 5 | `meta_prediction_optimizer_anti_overfit.py` | Multi-model training | `best_model.{json,pth}` + sidecar |
| 6 | `prediction_generator.py` | Generate predictions | `predictions_*.json` |

──────────────────────────────────────────────
🧠 MULTI-MODEL ARCHITECTURE (v3.1.3)
──────────────────────────────────────────────

models/
├── __init__.py                    # Exports all model components
├── global_state_tracker.py        # NEW: 14 global features (GPU-neutral)
├── feature_schema.py              # Streaming schema derivation + hash
├── model_factory.py               # Model loader with sidecar support
├── model_selector.py              # Best model selection logic
└── wrappers/
    ├── base.py                    # ModelInterface protocol
    ├── neural_net_wrapper.py      # PyTorch NN (ROCm + CUDA)
    ├── xgboost_wrapper.py         # XGBoost (CUDA)
    ├── lightgbm_wrapper.py        # LightGBM (OpenCL)
    └── catboost_wrapper.py        # CatBoost (CUDA)

### Subprocess Isolation (OpenCL/CUDA Conflict Resolution)
```
Main Process (coordinator) - NO GPU imports
    │
    ├── Trial 0: subprocess → LightGBM (OpenCL) → exits
    ├── Trial 1: subprocess → PyTorch (CUDA) → exits  
    ├── Trial 2: subprocess → XGBoost (CUDA) → exits
    └── Trial N: Fresh GPU state each time
```

Files:
- `train_single_trial.py` - Isolated worker script
- `subprocess_trial_coordinator.py` - Coordinates subprocess execution

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
├── best_model.json (or .pth)
└── best_model.meta.json (sidecar)
    └── agent_metadata.run_id: "step5_20251223_171709"

Step 6 Input:
├── Reads sidecar → auto-detects model type
├── Extracts parent_run_id from sidecar
└── Outputs predictions with lineage chain
```

──────────────────────────────────────────────
📊 SCORING & FEATURES
──────────────────────────────────────────────

survivor_scorer.py
│   • 46 per-seed features extraction
│   • _generate_sequence() - Dynamic PRNG lookup
│   • _coerce_seed_list() - Type-tolerant (int/dict)
│   • compute_dual_sieve_intersection() - Bidirectional filtering

models/global_state_tracker.py
│   • 14 global features (lottery-level statistics)
│   • SciPy fallback for entropy calculation
│   • GPU-neutral (importable anywhere)

### Feature Architecture (62 total)
```
Per-seed features: 48 (from survivor_scorer.py)
├── actual_mean, actual_std, actual_min, actual_max
├── predicted_mean, predicted_std, predicted_min, predicted_max
├── mae, rmse, correlation, r_squared
├── skip_0_mae through skip_5_mae
└── ... (46 statistical features)

Global features: 14 (from GlobalStateTracker)
├── global_lottery_mean, global_lottery_std
├── global_lottery_skew, global_lottery_kurtosis
├── global_lottery_entropy (SciPy fallback)
└── global_draw_count, global_unique_ratio, etc.
```

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
        "parent_run_id": "step5_20251223_171709"   // Lineage
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

coordinator.py
│   • Master controller for distributed execution
│   • SSH orchestration, GPU job scheduling
│   • ROCm/CUDA activation per node

distributed_worker.py
│   • Runs jobs on individual GPUs
│   • Pull-based job collection

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
| `CURRENT_Status.txt` | Session-by-session progress |
| `IMPLEMENTATION_CHECKLIST.md` | Feature completion tracking |
| `PROPOSAL_Unified_Agent_Context_Framework_v3_2_8.md` | Latest architecture proposal |

──────────────────────────────────────────────
⚙️ CONFIGURATION FILES
──────────────────────────────────────────────

distributed_config.json     # Node IPs, GPU mappings, SSH config
agent_config.yaml           # Meta-optimizer parameters
optimal_window_config.json  # Best window sizes (Optuna output)
prng_registry.py           # 46 PRNG algorithm definitions

──────────────────────────────────────────────
📌 RECENT CHANGES (December 2025)
──────────────────────────────────────────────

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
