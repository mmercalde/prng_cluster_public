# Cluster Operating Manual - Session 17 Updates
## Changes to be applied to Cluster_operating_manual.txt

### Section 1.1 - System Overview (ADD after "NEW: Bidirectional Sieve Architecture")
```
NOTE: Step 0 (PRNG Fingerprinting) was investigated in Session 17 and ARCHIVED.
Mathematical analysis proved fingerprinting is impossible under mod1000 projection.
The bidirectional sieve remains the primary validation method.
```

### Section 2 - Pipeline Steps (UPDATE feature counts)
```
Step 3 now produces 64 ML features per survivor:
- 50 per-seed features (from survivor_scorer.py)
- 14 global features (from GlobalStateTracker, prefixed with 'global_')

Training uses 62 features (after excluding score, confidence).
```

### NEW Section - Multi-Model Architecture (ADD after Chapter 5)
```
Chapter 5.5: Multi-Model ML Architecture (Sessions 9-17)

Overview
Status: PRODUCTION READY - CatBoost wins (Session 17)

The system now supports 4 ML model types with subprocess isolation to prevent
GPU backend conflicts between CUDA and OpenCL.

Supported Models:
| Model     | Backend         | Session 17 Results |
|-----------|-----------------|-------------------|
| CatBoost  | CUDA           | R²=1.0000 🏆      |
| XGBoost   | CUDA           | R²=1.0000         |
| LightGBM  | OpenCL         | R²=0.9999         |
| Neural Net| PyTorch (CUDA) | R²=0.0000         |

Subprocess Isolation:
Each trial runs in a separate subprocess to prevent OpenCL/CUDA conflicts:

Main Process (subprocess_trial_coordinator.py)
    │ ← NO GPU imports!
    ├── Trial 0: subprocess → LightGBM (OpenCL) → exits
    ├── Trial 1: subprocess → PyTorch (CUDA) → exits
    ├── Trial 2: subprocess → XGBoost (CUDA) → exits
    └── Trial 3: subprocess → CatBoost (CUDA) → exits

CLI Usage:
python3 meta_prediction_optimizer_anti_overfit.py \
    --survivors survivors_with_scores.json \
    --lottery-data train_history.json \
    --compare-models \
    --trials 8 \
    --timeout 900  # NEW Session 17: configurable timeout

Key Files:
- meta_prediction_optimizer_anti_overfit.py: Main orchestrator
- subprocess_trial_coordinator.py: Subprocess isolation coordinator
- train_single_trial.py: Single trial worker
- models/wrappers/*.py: Model-specific wrappers
```

### NEW Section - Global Features (ADD to Chapter 16 or as new chapter)
```
Chapter 16.5: Global Features Integration (Session 17)

Overview
14 global features computed from lottery history are now integrated at Step 3
Phase 5 (Aggregation). These features capture lottery-wide patterns.

Global Features (prefixed with 'global_'):
| Category          | Features |
|-------------------|----------|
| Residue Entropy   | global_residue_8_entropy, global_residue_125_entropy, global_residue_1000_entropy |
| Bias Detection    | global_power_of_two_bias, global_frequency_bias_ratio, global_suspicious_gap_percentage |
| Regime Detection  | global_regime_change_detected, global_regime_age, global_reseed_probability |
| Marker Analysis   | global_marker_390_variance, global_marker_804_variance, global_marker_575_variance, global_high_variance_count |
| Stability         | global_temporal_stability |

Key Points:
- Computed once from lottery history (O(1) not O(N))
- Identical for all survivors in same dataset
- Prefixed with 'global_' per Team Beta requirement
- Feature registry: config_manifests/feature_registry.json

Integration Point:
run_step3_full_scoring.sh Phase 5 aggregation merges global features into
each survivor's features dict before saving survivors_with_scores.json.
```

### UPDATE Section - Feature Architecture (in Chapter 16)
```
Feature Architecture (Updated Session 17)
=========================================
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
    └── See Chapter 16.5 for full list
```

### ADD to Troubleshooting (Chapter 14)
```
"Neural net timeout" Error
Cause: Neural net trials exceeding default 600s timeout
Solution: Use --timeout argument:
  python3 meta_prediction_optimizer_anti_overfit.py --timeout 900

"OpenCL/CUDA conflict" Error
Cause: LightGBM (OpenCL) running after CUDA models in same process
Solution: Use --compare-models flag which enables subprocess isolation

"Step 0 PRNG fingerprinting not working"
Status: ARCHIVED (Session 17)
Reason: Mathematically impossible under mod1000 projection. SNR < 0.15 for all features.
Alternative: Trust the bidirectional sieve - wrong PRNG produces 0 survivors.
```

### ADD to Recent Changes section
```
Session 17 (Dec 26-27, 2025)
- ❌ Step 0 PRNG Fingerprinting ARCHIVED (mathematically impossible under mod1000)
- ✅ Global features (14) integrated at Step 3 Phase 5 aggregation
- ✅ Features prefixed with 'global_' per Team Beta requirement
- ✅ Added --timeout CLI argument to Step 5 (default: 600s)
- ✅ Multi-model test: CatBoost wins (R²=1.0, MSE=8.6e-11)
- ✅ Feature count updated: 64 total (62 for training)
- ✅ Feature registry: config_manifests/feature_registry.json
```
