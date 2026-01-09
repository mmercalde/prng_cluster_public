# Chapter 7: Prediction Generator (Step 6)

## PRNG Analysis Pipeline — Complete Operating Guide

**Version:** 1.0  
**File:** `prediction_generator.py`  
**Lines:** ~400  
**Purpose:** Generate final predictions using trained ML model

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Position](#2-architecture-position)
3. [Model Loading with Sidecar](#3-model-loading-with-sidecar)
4. [Feature Schema Validation](#4-feature-schema-validation)
5. [Survivor Ranking](#5-survivor-ranking)
6. [Prediction Pool Generation](#6-prediction-pool-generation)
7. [Next-Draw Prediction](#7-next-draw-prediction)
8. [Output Formats](#8-output-formats)
9. [CLI Interface](#9-cli-interface)
10. [Agent Integration](#10-agent-integration)
11. [Troubleshooting](#11-troubleshooting)
12. [Complete Method Reference](#12-complete-method-reference)

---

## 1. Overview

### 1.1 What the Prediction Generator Does

The prediction generator is **Step 6** — the final step that produces actionable predictions:

- **Loads trained model** from Step 5 using sidecar metadata
- **Validates feature schema** to ensure compatibility
- **Ranks survivors** by predicted quality
- **Generates prediction pools** (Tight/Balanced/Wide)
- **Produces next-draw predictions** with confidence scores

### 1.2 Key Features

| Feature | Description |
|---------|-------------|
| **Sidecar-Only Loading** | Model type from meta.json ONLY (never from extension) |
| **Feature Hash Validation** | FATAL error if schema mismatch |
| **Multi-Model Support** | Works with all 4 model types |
| **Prediction Pools** | Tight (20), Balanced (100), Wide (300) |
| **Confidence Scores** | Per-seed and per-prediction quality metrics |

### 1.3 Pipeline Position

```
Step 1: Window Optimizer
    ↓
Step 2.5: Scorer Meta-Optimizer
    ↓
Step 3: Full Scoring → survivors_with_scores.json
    ↓
Step 4: Adaptive Meta-Optimizer
    ↓
Step 5: Anti-Overfit Training → best_model.* + best_model.meta.json
    ↓
Step 6: PREDICTION GENERATOR ◄── THIS FILE
    ↓
Output: ranked_predictions.json, prediction_pools.json
```

---

## 2. Architecture Position

### 2.1 In the Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    STEP 6: PREDICTION GENERATOR                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INPUTS:                                                                 │
│  ├── models/reinforcement/best_model.*        (trained model)           │
│  ├── models/reinforcement/best_model.meta.json (sidecar metadata)       │
│  ├── forward_survivors.json                    (from Step 1)            │
│  └── lottery_history.json                      (recent draws)           │
│                                                                          │
│  PROCESSING:                                                             │
│  ├── Load model (type from meta.json ONLY)                              │
│  ├── Validate feature schema hash                                       │
│  ├── Extract features for forward survivors                             │
│  ├── Predict quality scores                                             │
│  ├── Rank by predicted quality                                          │
│  └── Build prediction pools                                             │
│                                                                          │
│  OUTPUTS:                                                                │
│  ├── ranked_predictions.json      (all survivors with predictions)      │
│  ├── prediction_pools.json        (Tight/Balanced/Wide pools)           │
│  └── next_draw_prediction.json    (aggregated next-draw forecast)       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
Step 5 Output                    Step 6 Processing                Step 6 Output
─────────────                    ─────────────────                ────────────
                                        
best_model.pth ─────┐                                            
                    │            ┌─────────────────┐             ranked_predictions.json
best_model.meta.json┼───────────►│  Model Loader   │                    │
                    │            │  (sidecar-only) │             ┌──────┴──────┐
forward_survivors ──┼───────────►├─────────────────┤             │             │
                    │            │ Feature Extract │──────►      │  Ranking    │
lottery_history ────┘            ├─────────────────┤             │             │
                                 │   Predictor     │──────►      └──────┬──────┘
                                 ├─────────────────┤                    │
                                 │   Pool Builder  │──────►      prediction_pools.json
                                 └─────────────────┘                    │
                                                                 next_draw_prediction.json
```

---

## 3. Model Loading with Sidecar

### 3.1 Critical Contract

**Model type is determined ONLY from `best_model.meta.json`. File extensions are NEVER used.**

```python
# CORRECT:
meta = load_json("best_model.meta.json")
model_type = meta["model_type"]  # "xgboost", "neural_net", etc.

# WRONG (FORBIDDEN):
if path.endswith(".pth"):
    model_type = "neural_net"  # NEVER DO THIS
```

### 3.2 load_best_model()

```python
from pathlib import Path
import json
from models import ModelFactory
from models.feature_schema import get_feature_schema_with_hash

def load_best_model(models_dir: str, survivors_file: str) -> tuple:
    """
    Load best model from Step 5 output with full validation.
    
    CRITICAL CONTRACTS:
    1. Model type from meta.json ONLY (never from file extension)
    2. Feature schema hash must match between training and runtime
    3. Fails fast on any validation error
    
    Args:
        models_dir: Directory containing best_model.meta.json
        survivors_file: Runtime survivors file for schema validation
    
    Returns:
        model: Loaded ModelInterface
        meta: Full metadata dict
    
    Raises:
        FileNotFoundError: If meta.json missing
        ValueError: If schema hash mismatch
    """
    meta_path = Path(models_dir) / "best_model.meta.json"
    
    # Contract 1: Meta file MUST exist
    if not meta_path.exists():
        raise FileNotFoundError(
            f"FATAL: Missing metadata sidecar: {meta_path}\n"
            f"Step 5 MUST generate best_model.meta.json.\n"
            f"Model type CANNOT be inferred from file extension."
        )
    
    with open(meta_path) as f:
        meta = json.load(f)
    
    # Contract 2: model_type field MUST exist
    if "model_type" not in meta:
        raise ValueError(
            f"FATAL: Missing 'model_type' in {meta_path}\n"
            f"Sidecar schema version may be outdated."
        )
    
    # Contract 3: Validate feature schema hash
    expected_hash = meta["feature_schema"]["feature_schema_hash"]
    runtime_schema = get_feature_schema_with_hash(survivors_file)
    runtime_hash = runtime_schema["feature_schema_hash"]
    
    if runtime_hash != expected_hash:
        raise ValueError(
            f"FATAL: Feature schema mismatch!\n"
            f"  Training hash: {expected_hash}\n"
            f"  Runtime hash:  {runtime_hash}\n"
            f"  Training file: {meta['feature_schema']['source_file']}\n"
            f"  Runtime file:  {survivors_file}\n"
            f"\n"
            f"Feature ordering has changed. Model predictions would be INVALID.\n"
            f"Retrain model or use matching survivors file."
        )
    
    # Load model (type from meta only)
    model_type = meta["model_type"]
    checkpoint_path = Path(models_dir) / meta["checkpoint_path"]
    
    model = ModelFactory.load(model_type, str(checkpoint_path))
    
    print(f"✅ Loaded {model_type} model from {checkpoint_path}")
    print(f"   Feature hash validated: {expected_hash}")
    
    return model, meta
```

### 3.3 Sidecar Metadata Structure

```json
{
  "schema_version": "3.1.2",
  "model_type": "xgboost",
  "checkpoint_path": "best_model.json",
  
  "feature_schema": {
    "source_file": "/absolute/path/to/survivors_with_scores.json",
    "feature_count": 50,
    "feature_names": ["actual_mean", "actual_std", "..."],
    "ordering": "lexicographic_by_key",
    "feature_schema_hash": "a1b2c3d4e5f67890"
  },
  
  "validation_metrics": {
    "mse": 0.0234,
    "mae": 0.1123,
    "r2": 0.847
  },
  
  "training_info": {
    "started_at": "2025-12-20T10:30:00Z",
    "completed_at": "2025-12-20T11:45:00Z",
    "k_folds": 5
  }
}
```

---

## 4. Feature Schema Validation

### 4.1 Why Validation Matters

```
TRAINING:                           PREDICTION:
  Feature 0: actual_mean              Feature 0: actual_mean     ✅ Match
  Feature 1: actual_std               Feature 1: best_offset     ❌ MISMATCH!
  Feature 2: best_offset              Feature 2: actual_std      ❌ MISMATCH!
  ...                                 ...

Without validation, model receives wrong features in wrong positions.
Predictions would be MEANINGLESS.
```

### 4.2 Hash Computation

```python
import hashlib

def get_feature_schema_with_hash(survivors_file: str) -> dict:
    """
    Get feature schema with validation hash.
    
    The hash is computed from sorted feature names, ensuring
    any change in feature set or ordering is detected.
    """
    schema = get_feature_schema_from_data(survivors_file)
    
    # Canonical hash from sorted feature names
    names_str = ",".join(sorted(schema["feature_names"]))
    schema["feature_schema_hash"] = hashlib.sha256(
        names_str.encode('utf-8')
    ).hexdigest()[:16]
    
    return schema
```

### 4.3 Validation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE SCHEMA VALIDATION                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Load expected_hash from best_model.meta.json                │
│                                                                  │
│  2. Compute runtime_hash from survivors file                    │
│                                                                  │
│  3. Compare:                                                     │
│     ┌─────────────────────────────────────────────┐             │
│     │  expected_hash == runtime_hash?              │             │
│     └─────────────────┬───────────────────────────┘             │
│                       │                                          │
│              ┌────────┴────────┐                                │
│              │ YES             │ NO                              │
│              ▼                 ▼                                 │
│     ┌─────────────┐   ┌─────────────────────────────────┐       │
│     │ ✅ PROCEED  │   │ ❌ FATAL ERROR                   │       │
│     │ Load model  │   │ "Feature schema mismatch!"       │       │
│     └─────────────┘   │ "Model predictions INVALID"      │       │
│                       │ EXIT with error code 1           │       │
│                       └─────────────────────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Survivor Ranking

### 5.1 Ranking Process

```python
def rank_survivors(
    model: ModelInterface,
    survivors: List[Dict],
    feature_names: List[str]
) -> List[Dict]:
    """
    Rank survivors by predicted quality.
    
    Args:
        model: Loaded ML model
        survivors: List of survivor dicts with features
        feature_names: Ordered feature names from schema
    
    Returns:
        Survivors sorted by predicted_quality (descending)
    """
    # Extract features in correct order
    X = np.array([
        [s['features'][f] for f in feature_names]
        for s in survivors
    ])
    
    # Predict quality
    predictions = model.predict(X)
    
    # Attach predictions to survivors
    for i, survivor in enumerate(survivors):
        survivor['predicted_quality'] = float(predictions[i])
    
    # Sort by predicted quality (best first)
    ranked = sorted(survivors, key=lambda s: s['predicted_quality'], reverse=True)
    
    # Add rank
    for i, survivor in enumerate(ranked):
        survivor['rank'] = i + 1
    
    return ranked
```

### 5.2 Ranking Output

```json
[
  {
    "seed": 12345678,
    "rank": 1,
    "predicted_quality": 0.923,
    "features": {...},
    "prng_type": "java_lcg"
  },
  {
    "seed": 87654321,
    "rank": 2,
    "predicted_quality": 0.891,
    "features": {...},
    "prng_type": "java_lcg"
  }
]
```

---

## 6. Prediction Pool Generation

### 6.1 Pool Definitions

| Pool | Size | Purpose | Expected Hit Rate |
|------|------|---------|-------------------|
| **Tight** | 20 | Highest confidence, maximum precision | Highest lift vs random |
| **Balanced** | 100 | Broader coverage, >60% weight | Hit@100 > 70% |
| **Wide** | 300 | Comprehensive coverage, >85% weight | Hit@300 > 90% |

### 6.2 build_prediction_pools()

```python
def build_prediction_pools(
    ranked_survivors: List[Dict],
    lottery_history: List[int],
    prng_type: str = 'java_lcg',
    mod: int = 1000
) -> Dict:
    """
    Build prediction pools from ranked survivors.
    
    Each pool contains:
    - Top-K survivors by predicted quality
    - Their next-draw predictions
    - Aggregated vote weights for each possible number
    """
    from survivor_scorer import SurvivorScorer
    
    scorer = SurvivorScorer(prng_type=prng_type, mod=mod)
    
    pools = {
        'tight': {'size': 20, 'survivors': [], 'predictions': {}},
        'balanced': {'size': 100, 'survivors': [], 'predictions': {}},
        'wide': {'size': 300, 'survivors': [], 'predictions': {}}
    }
    
    for pool_name, pool in pools.items():
        pool_survivors = ranked_survivors[:pool['size']]
        pool['survivors'] = pool_survivors
        
        # Aggregate predictions with weighted voting
        votes = {}  # number -> total weight
        
        for survivor in pool_survivors:
            seed = survivor['seed']
            weight = survivor['predicted_quality']
            
            # Get next prediction for this seed
            next_pred = scorer.predict_next(seed, lottery_history)
            
            if next_pred not in votes:
                votes[next_pred] = 0.0
            votes[next_pred] += weight
        
        # Normalize weights
        total_weight = sum(votes.values())
        pool['predictions'] = {
            num: weight / total_weight
            for num, weight in sorted(votes.items(), key=lambda x: -x[1])
        }
        
        # Summary stats
        pool['total_weight'] = total_weight
        pool['unique_predictions'] = len(votes)
        pool['top_prediction'] = max(votes.items(), key=lambda x: x[1])
    
    return pools
```

### 6.3 Pool Output Format

```json
{
  "tight": {
    "size": 20,
    "survivors": [...],
    "predictions": {
      "347": 0.15,
      "891": 0.12,
      "234": 0.09
    },
    "total_weight": 18.45,
    "unique_predictions": 15,
    "top_prediction": ["347", 0.15]
  },
  "balanced": {
    "size": 100,
    "survivors": [...],
    "predictions": {...},
    "total_weight": 85.23
  },
  "wide": {
    "size": 300,
    "survivors": [...],
    "predictions": {...},
    "total_weight": 245.67
  }
}
```

---

## 7. Next-Draw Prediction

### 7.1 Weighted Voting

```python
def generate_next_draw_prediction(
    pools: Dict,
    pool_weights: Dict[str, float] = None
) -> Dict:
    """
    Generate aggregated next-draw prediction from all pools.
    
    Args:
        pools: Output from build_prediction_pools()
        pool_weights: Relative weights for each pool
                     Default: {'tight': 0.5, 'balanced': 0.3, 'wide': 0.2}
    
    Returns:
        Next-draw prediction with confidence metrics
    """
    if pool_weights is None:
        pool_weights = {'tight': 0.5, 'balanced': 0.3, 'wide': 0.2}
    
    # Aggregate votes across pools
    combined_votes = {}
    
    for pool_name, pool in pools.items():
        pool_weight = pool_weights.get(pool_name, 0.0)
        
        for number, vote_weight in pool['predictions'].items():
            if number not in combined_votes:
                combined_votes[number] = 0.0
            combined_votes[number] += vote_weight * pool_weight
    
    # Sort by combined weight
    ranked_predictions = sorted(
        combined_votes.items(),
        key=lambda x: -x[1]
    )
    
    # Build output
    top_predictions = ranked_predictions[:20]
    total_weight = sum(v for _, v in combined_votes.items())
    
    return {
        'timestamp': datetime.now().isoformat(),
        'top_predictions': [
            {
                'number': num,
                'weight': weight,
                'confidence': weight / total_weight * 100
            }
            for num, weight in top_predictions
        ],
        'primary_prediction': {
            'number': ranked_predictions[0][0],
            'confidence': ranked_predictions[0][1] / total_weight * 100
        },
        'coverage': {
            'top_10_weight_pct': sum(v for _, v in ranked_predictions[:10]) / total_weight * 100,
            'top_50_weight_pct': sum(v for _, v in ranked_predictions[:50]) / total_weight * 100,
            'unique_numbers': len(combined_votes)
        },
        'pool_contributions': {
            pool_name: {
                'weight': pool_weights.get(pool_name, 0.0),
                'top_prediction': pool['top_prediction']
            }
            for pool_name, pool in pools.items()
        }
    }
```

### 7.2 Prediction Output

```json
{
  "timestamp": "2025-12-31T14:30:00",
  "top_predictions": [
    {"number": "347", "weight": 0.18, "confidence": 18.2},
    {"number": "891", "weight": 0.12, "confidence": 12.1},
    {"number": "234", "weight": 0.09, "confidence": 9.3}
  ],
  "primary_prediction": {
    "number": "347",
    "confidence": 18.2
  },
  "coverage": {
    "top_10_weight_pct": 65.3,
    "top_50_weight_pct": 89.2,
    "unique_numbers": 147
  },
  "pool_contributions": {
    "tight": {"weight": 0.5, "top_prediction": ["347", 0.15]},
    "balanced": {"weight": 0.3, "top_prediction": ["347", 0.11]},
    "wide": {"weight": 0.2, "top_prediction": ["891", 0.08]}
  }
}
```

---

## 8. Output Formats

### 8.1 File Outputs

| File | Contents | Size |
|------|----------|------|
| `ranked_predictions.json` | All survivors with predicted quality | ~50MB |
| `prediction_pools.json` | Tight/Balanced/Wide pools | ~5MB |
| `next_draw_prediction.json` | Aggregated prediction | ~10KB |

### 8.2 ranked_predictions.json

```json
{
  "metadata": {
    "generated_at": "2025-12-31T14:30:00Z",
    "model_type": "xgboost",
    "model_r2": 0.847,
    "total_survivors": 395211,
    "feature_schema_hash": "a1b2c3d4e5f67890"
  },
  "rankings": [
    {
      "rank": 1,
      "seed": 12345678,
      "predicted_quality": 0.923,
      "prng_type": "java_lcg",
      "next_prediction": 347
    }
  ]
}
```

### 8.3 prediction_pools.json

```json
{
  "metadata": {
    "generated_at": "2025-12-31T14:30:00Z",
    "lottery_history_length": 5000
  },
  "pools": {
    "tight": {...},
    "balanced": {...},
    "wide": {...}
  },
  "success_metrics": {
    "expected_hit_at_20": 0.15,
    "expected_hit_at_100": 0.70,
    "expected_hit_at_300": 0.90,
    "lift_vs_random": 15.2
  }
}
```

---

## 9. CLI Interface

### 9.1 Arguments

```bash
python3 prediction_generator.py [options]

Required:
  --models-dir DIR           Directory with best_model.* and meta.json
  --survivors-forward FILE   Forward survivors from Step 1
  --lottery-history FILE     Recent lottery draws

Optional:
  --output-dir DIR           Output directory (default: predictions/)
  --pool-sizes SIZES         Pool sizes as "tight,balanced,wide" (default: 20,100,300)
  --prng-type TYPE           PRNG algorithm (default: java_lcg)
  --mod MOD                  Modulo value (default: 1000)
  -h, --help                 Show help
```

### 9.2 Usage Examples

**Basic usage:**
```bash
python3 prediction_generator.py \
    --models-dir models/reinforcement \
    --survivors-forward forward_survivors.json \
    --lottery-history daily3.json
```

**Custom pool sizes:**
```bash
python3 prediction_generator.py \
    --models-dir models/reinforcement \
    --survivors-forward forward_survivors.json \
    --lottery-history daily3.json \
    --pool-sizes 10,50,200 \
    --output-dir my_predictions/
```

**Different PRNG:**
```bash
python3 prediction_generator.py \
    --models-dir models/reinforcement \
    --survivors-forward forward_survivors.json \
    --lottery-history daily3.json \
    --prng-type xorshift32
```

---

## 10. Agent Integration

### 10.1 Agent Manifest

**agent_manifests/prediction.json:**
```json
{
  "agent_id": "prediction_agent",
  "step_id": 6,
  "step_name": "prediction_generation",
  "script": "prediction_generator.py",
  
  "inputs": [
    {"name": "models_dir", "type": "directory", "required": true},
    {"name": "survivors_forward", "type": "file", "required": true},
    {"name": "lottery_history", "type": "file", "required": true}
  ],
  
  "outputs": [
    {"name": "ranked_predictions.json", "type": "file"},
    {"name": "prediction_pools.json", "type": "file"},
    {"name": "next_draw_prediction.json", "type": "file"}
  ],
  
  "success_criteria": {
    "min_survivors_ranked": 1000,
    "min_pool_coverage": 0.8
  },
  
  "follow_up_agent": null,
  
  "llm_prompts": {
    "analyze": "Analyze these prediction results and identify confidence patterns..."
  }
}
```

### 10.2 Watcher Agent Integration

```python
# In agents/watcher_agent.py

def handle_step6_completion(self, result_dir: str) -> AgentDecision:
    """Evaluate prediction generation results."""
    
    # Load prediction output
    pred_path = Path(result_dir) / 'next_draw_prediction.json'
    with open(pred_path) as f:
        prediction = json.load(f)
    
    # Check confidence
    top_confidence = prediction['primary_prediction']['confidence']
    
    if top_confidence < 5.0:
        return AgentDecision(
            decision='ESCALATE',
            confidence=0.8,
            reasoning=f'Low prediction confidence: {top_confidence:.1f}%'
        )
    
    # Query Math LLM for interpretation
    analysis = self.llm.calculate(f"""
        Analyze this prediction distribution:
        - Top prediction: {prediction['primary_prediction']}
        - Coverage: {prediction['coverage']}
        
        Is this a healthy prediction distribution?
    """)
    
    return AgentDecision(
        decision='PROCEED',
        confidence=0.9,
        reasoning=f'Prediction generated with {top_confidence:.1f}% top confidence'
    )
```

### 10.3 Agent Metadata Injection

```python
def generate_predictions_with_metadata(
    model, meta, survivors, lottery_history, output_dir
):
    """Generate predictions with agent metadata."""
    
    # Generate predictions
    ranked = rank_survivors(model, survivors, meta['feature_schema']['feature_names'])
    pools = build_prediction_pools(ranked, lottery_history)
    prediction = generate_next_draw_prediction(pools)
    
    # Inject agent metadata
    prediction['agent_metadata'] = {
        'pipeline_step': 6,
        'pipeline_step_name': 'prediction_generation',
        'run_id': f"step6_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'parent_run_id': meta.get('agent_metadata', {}).get('run_id'),
        'model_type': meta['model_type'],
        'model_r2': meta['validation_metrics']['r2'],
        'follow_up_agent': None  # Terminal step
    }
    
    # Save outputs
    save_json(prediction, output_dir / 'next_draw_prediction.json')
    
    return prediction
```

---

## 11. Troubleshooting

### 11.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Missing metadata sidecar" | Step 5 didn't generate meta.json | Re-run Step 5 with latest code |
| "Feature schema mismatch" | Different survivors file | Use same file as training |
| "Model type missing" | Old sidecar format | Re-run Step 5 |
| "Empty predictions" | No survivors passed | Check input file |
| "Low confidence" | Model not trained well | Check Step 5 metrics |

### 11.2 Debug Mode

```bash
# Verbose output
python3 prediction_generator.py \
    --models-dir models/reinforcement \
    --survivors-forward forward_survivors.json \
    --lottery-history daily3.json \
    --verbose \
    2>&1 | tee prediction_debug.log
```

### 11.3 Verify Model Loading

```bash
# Check sidecar contents
cat models/reinforcement/best_model.meta.json | jq .

# Verify feature hash
python3 -c "
from models.feature_schema import get_feature_schema_with_hash
schema = get_feature_schema_with_hash('forward_survivors.json')
print(f'Runtime hash: {schema[\"feature_schema_hash\"]}')
"
```

### 11.4 Validate Predictions

```bash
# Check prediction output
cat predictions/next_draw_prediction.json | jq '.primary_prediction'

# Verify pool sizes
cat predictions/prediction_pools.json | jq '.pools | keys'
```

---

## 12. Complete Method Reference

### 12.1 Model Loading

| Function | Purpose |
|----------|---------|
| `load_best_model(models_dir, survivors_file)` | Load model with validation |
| `validate_feature_schema_hash(expected, runtime)` | Check schema compatibility |
| `get_feature_schema_with_hash(file)` | Compute schema hash |

### 12.2 Ranking

| Function | Purpose |
|----------|---------|
| `rank_survivors(model, survivors, features)` | Rank by predicted quality |
| `extract_features_ordered(survivor, feature_names)` | Extract in correct order |

### 12.3 Pool Generation

| Function | Purpose |
|----------|---------|
| `build_prediction_pools(ranked, history)` | Build Tight/Balanced/Wide pools |
| `aggregate_votes(survivors, scorer)` | Weighted vote aggregation |
| `generate_next_draw_prediction(pools)` | Combined prediction |

### 12.4 Output

| Function | Purpose |
|----------|---------|
| `save_ranked_predictions(ranked, path)` | Save full rankings |
| `save_prediction_pools(pools, path)` | Save pool details |
| `save_next_draw(prediction, path)` | Save aggregated prediction |

---

## 13. Chapter Summary

**Chapter 16: Prediction Generator (Step 6)** covers the final pipeline step:

| Component | Lines | Purpose |
|-----------|-------|---------|
| Model loading | ~80 | Sidecar-only loading with validation |
| Feature validation | ~40 | Hash-based schema verification |
| Survivor ranking | ~60 | ML-based quality prediction |
| Pool generation | ~100 | Tight/Balanced/Wide pools |
| Next-draw prediction | ~60 | Weighted vote aggregation |

**Key Points:**
- Model type from meta.json ONLY (never from extension)
- Feature schema hash MUST match (FATAL if mismatch)
- Three prediction pools: Tight (20), Balanced (100), Wide (300)
- Weighted voting aggregates predictions
- Terminal step (no follow-up agent)

**Success Metrics:**
- Hit@100 > 70%
- Hit@300 > 90%
- Lift vs random: 10-20x

---

## Pipeline Complete

With Chapter 16, all 6 pipeline steps are now documented:

| Step | Chapter | Script |
|------|---------|--------|
| 1 | Chapter 1 | `window_optimizer.py` |
| 2.5 | Chapter 13 | `scorer_trial_worker.py` |
| 3 | Chapter 6, 8 | `survivor_scorer.py`, `full_scoring_worker.py` |
| 4 | Chapter 4 | `meta_prediction_optimizer.py` |
| 5 | Chapter 4, 10 | `meta_prediction_optimizer_anti_overfit.py` |
| **6** | **Chapter 16** | **`prediction_generator.py`** |

---

*End of Chapter 16: Prediction Generator (Step 6)*
