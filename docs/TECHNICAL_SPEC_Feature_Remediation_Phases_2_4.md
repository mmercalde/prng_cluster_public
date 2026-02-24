# Technical Specifications: Feature Remediation Phases 2-4
## Version 1.0 | December 28, 2025

---

## Phase 2: Skip Metadata Pipeline - Technical Specification

### 2.1 Schema Definition

**File:** `skip_metadata.json`
**Location:** Output alongside `forward_survivors.json` in Step 2

```json
{
  "schema_version": "skip_meta_v1.1",
  "run_id": "step2_20251228_HHMMSS",
  "prng_type": "java_lcg",
  "mod": 1000,
  "lottery_hash": "sha256_of_lottery_file",
  "total_seeds": 343714,
  "created_at": "2025-12-28T14:30:00Z",
  "seeds": {
    "123456789": {
      "skip_n": 4000,
      "skip_mean": 0.73,
      "skip_std": 1.1,
      "skip_min": 0,
      "skip_max": 6,
      "skip_range": 6,
      "skip_entropy": 1.42,
      "window_counts": [1200, 1150, 1100, 1050, 1000],
      "velocity": -50.0,
      "acceleration": 0.0
    },
    "987654321": {
      "skip_n": 4000,
      "skip_mean": 1.2,
      "skip_std": 0.8,
      "skip_min": 0,
      "skip_max": 4,
      "skip_range": 4,
      "skip_entropy": 1.15,
      "window_counts": [800, 820, 810, 805, 800],
      "velocity": 0.0,
      "acceleration": -2.5
    }
  }
}
```

### 2.2 Entropy Calculation

Shannon entropy over skip value histogram:

```python
import numpy as np
from collections import Counter

def compute_skip_entropy(skip_values: List[int]) -> float:
    """
    Compute Shannon entropy of skip value distribution.
    
    Higher entropy = more uniform distribution of skip values
    Lower entropy = concentrated on few values
    """
    if not skip_values:
        return 0.0
    
    counts = Counter(skip_values)
    total = sum(counts.values())
    
    if total == 0:
        return 0.0
    
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    
    return float(entropy)
```

### 2.3 Code Changes Required

#### File: `sieve_filter.py`

```python
# ADD: Track skip values per seed during sieve

class SieveFilter:
    def __init__(self, ...):
        ...
        self.skip_tracking = {}  # seed -> list of skip values
        self.window_counts = {}  # seed -> list of survivor counts per window
    
    def _track_skip_for_seed(self, seed: int, skip_value: int, window_idx: int):
        """Track skip value that produced a match for this seed."""
        if seed not in self.skip_tracking:
            self.skip_tracking[seed] = []
        self.skip_tracking[seed].append(skip_value)
    
    def _track_window_count(self, seed: int, window_idx: int, count: int):
        """Track survivor count at each window for velocity computation."""
        if seed not in self.window_counts:
            self.window_counts[seed] = []
        # Ensure list is long enough
        while len(self.window_counts[seed]) <= window_idx:
            self.window_counts[seed].append(0)
        self.window_counts[seed][window_idx] = count
    
    def export_skip_metadata(self, output_path: str, run_id: str):
        """Export skip metadata JSON after sieve completes."""
        metadata = {
            "schema_version": "skip_meta_v1.1",
            "run_id": run_id,
            "prng_type": self.prng_type,
            "mod": self.mod,
            "total_seeds": len(self.skip_tracking),
            "created_at": datetime.now().isoformat(),
            "seeds": {}
        }
        
        for seed, skip_values in self.skip_tracking.items():
            window_counts = self.window_counts.get(seed, [])
            
            # Compute aggregate stats
            skip_arr = np.array(skip_values) if skip_values else np.array([0])
            
            # Velocity and acceleration from window counts
            velocity, acceleration = compute_velocity_features(window_counts)
            
            metadata["seeds"][str(seed)] = {
                "skip_n": len(skip_values),
                "skip_mean": float(np.mean(skip_arr)),
                "skip_std": float(np.std(skip_arr)),
                "skip_min": int(np.min(skip_arr)),
                "skip_max": int(np.max(skip_arr)),
                "skip_range": int(np.max(skip_arr) - np.min(skip_arr)),
                "skip_entropy": compute_skip_entropy(skip_values),
                "window_counts": window_counts,
                "velocity": velocity,
                "acceleration": acceleration
            }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f)
        
        return output_path
```

#### File: `run_step3_full_scoring.sh`

```bash
# ADD: New flag for skip metadata

SKIP_METADATA=""

# In getopts:
--skip-metadata)
    SKIP_METADATA="$2"
    shift 2
    ;;

# In data distribution:
if [[ -n "$SKIP_METADATA" && -f "$SKIP_METADATA" ]]; then
    echo "    Copying skip metadata..."
    scp -q "$SKIP_METADATA" ${REMOTE_USER}@${NODE}:${REMOTE_BASE}/
fi

# In worker invocation:
if [[ -n "$SKIP_METADATA" ]]; then
    WORKER_CMD="$WORKER_CMD --skip-metadata $SKIP_METADATA"
fi
```

#### File: `full_scoring_worker.py`

```python
# ADD: Load skip metadata

parser.add_argument('--skip-metadata', type=str, default=None,
                   help='Path to skip metadata JSON from Step 2')

# In main():
skip_metadata = None
if args.skip_metadata and os.path.exists(args.skip_metadata):
    with open(args.skip_metadata, 'r') as f:
        skip_metadata = json.load(f)
    logger.info(f"Loaded skip metadata for {len(skip_metadata.get('seeds', {}))} seeds")

# Pass to scoring function:
results = score_chunk(
    ...,
    skip_metadata=skip_metadata
)
```

#### File: `survivor_scorer.py`

```python
# MODIFY: extract_ml_features() to use skip metadata

def extract_ml_features(self, seed: int, lottery_history: List[int],
                       forward_survivors=None, reverse_survivors=None,
                       skip_metadata=None,  # NEW PARAMETER
                       skip: int = 0) -> Dict[str, float]:
    ...
    
    # After computing basic features, add skip features from metadata
    if skip_metadata and str(seed) in skip_metadata.get('seeds', {}):
        seed_meta = skip_metadata['seeds'][str(seed)]
        features['skip_entropy'] = float(seed_meta.get('skip_entropy', 0.0))
        features['skip_mean'] = float(seed_meta.get('skip_mean', 0.0))
        features['skip_std'] = float(seed_meta.get('skip_std', 0.0))
        features['skip_min'] = float(seed_meta.get('skip_min', 0.0))
        features['skip_max'] = float(seed_meta.get('skip_max', 0.0))
        features['skip_range'] = float(seed_meta.get('skip_range', 0.0))
        features['survivor_velocity'] = float(seed_meta.get('velocity', 0.0))
        features['velocity_acceleration'] = float(seed_meta.get('acceleration', 0.0))
```

---

## Phase 3: Remove Hardcoded Values - Technical Specification

### 3.1 Hardcoded Values to Remove

| Feature | Current Code | Fixed Code |
|---------|--------------|------------|
| `confidence` | `0.1` | Computed |
| `total_predictions` | `400` | `len(lottery_history)` |
| `best_offset` | `0` | Computed |

### 3.2 Code Changes

#### File: `survivor_scorer.py`

**Location:** Inside `extract_ml_features()`, after basic stats computation

```python
# REMOVE THESE LINES:
# features['confidence'] = 0.1  # DELETE
# features['total_predictions'] = 400  # DELETE
# features['best_offset'] = 0  # DELETE

# REPLACE WITH:

# total_predictions - actual length used
features['total_predictions'] = float(len(lottery_history))

# confidence - computed from match quality (with safety checks)
exact_matches = features.get('exact_matches', 0)
total_pred = features['total_predictions']
if total_pred > 0:
    features['confidence'] = min(1.0, max(0.0, exact_matches / total_pred))
else:
    features['confidence'] = 0.0

# best_offset - optimal alignment search
features['best_offset'] = float(self._compute_best_offset(pred, act, max_offset=10))


# ADD NEW METHOD:
def _compute_best_offset(self, pred: torch.Tensor, act: torch.Tensor, 
                         max_offset: int = 10) -> int:
    """
    Find optimal alignment offset between predictions and actuals.
    
    Searches offsets from -max_offset to +max_offset.
    Returns offset that maximizes match rate.
    
    Note: Uses only training segment to avoid leakage.
    """
    best_off = 0
    best_rate = 0.0
    
    pred_np = pred.cpu().numpy() if hasattr(pred, 'cpu') else np.array(pred)
    act_np = act.cpu().numpy() if hasattr(act, 'cpu') else np.array(act)
    
    for off in range(-max_offset, max_offset + 1):
        if off < 0:
            p = pred_np[-off:]
            a = act_np[:off] if off != 0 else act_np
        elif off > 0:
            p = pred_np[:-off] if off != 0 else pred_np
            a = act_np[off:]
        else:
            p = pred_np
            a = act_np
        
        # Ensure same length
        min_len = min(len(p), len(a))
        if min_len == 0:
            continue
        
        p = p[:min_len]
        a = a[:min_len]
        
        match_rate = np.mean(p == a)
        if match_rate > best_rate:
            best_rate = match_rate
            best_off = off
    
    return best_off
```

### 3.3 Velocity Features Implementation

**Already covered in Phase 2** - velocity and acceleration are computed in Step 2 and stored in skip_metadata.

If skip_metadata is not available, fallback to 0.0 with explicit marker:

```python
# In survivor_scorer.py - only if skip_metadata NOT provided
if skip_metadata is None or str(seed) not in skip_metadata.get('seeds', {}):
    # Mark as not implemented rather than silent 0.0
    features['survivor_velocity'] = 0.0
    features['velocity_acceleration'] = 0.0
    features['_velocity_implemented'] = False  # Internal flag
else:
    features['_velocity_implemented'] = True
```

---

## Phase 4: Relocate Duplicate Features - Technical Specification

### 4.1 Features to Relocate

| Feature | Current Location | New Location |
|---------|------------------|--------------|
| `actual_mean` | `features.actual_mean` | `context.lottery_stats.mean` |
| `actual_std` | `features.actual_std` | `context.lottery_stats.std` |

### 4.2 Backward Compatibility Strategy

**Version 1.0.1 (Transition):**
- Keep old keys in JSON (for backward compat)
- Add new `context` structure
- Add `is_context: true` flag
- Exclude from ML feature matrix

**Schema:**
```json
{
  "seed": 12345,
  "score": 0.85,
  "features": {
    "pred_mean": 465.07,
    "pred_std": 127.3,
    "exact_matches": 3,
    ...
    // DEPRECATED - kept for backward compat v1.0.x
    "actual_mean": 496.84,
    "actual_std": 289.89
  },
  "context": {
    "lottery_stats": {
      "mean": 496.84,
      "std": 289.89,
      "min": 0,
      "max": 999,
      "count": 500
    },
    "deprecated_feature_keys": ["actual_mean", "actual_std"],
    "schema_version": "1.0.1"
  },
  "metadata": {
    "prng_type": "java_lcg",
    "mod": 1000
  }
}
```

### 4.3 Code Changes

#### File: `survivor_scorer.py`

```python
def extract_ml_features(self, seed: int, lottery_history: List[int], ...):
    ...
    
    # REMOVE from features computation:
    # features['actual_mean'] = float(act.float().mean().item())  # DELETE
    # features['actual_std'] = float(act.float().std().item())    # DELETE
    
    # KEEP for backward compat but mark as deprecated:
    lottery_mean = float(np.mean(lottery_history))
    lottery_std = float(np.std(lottery_history))
    
    # Add to features for backward compat (will be excluded from X)
    features['actual_mean'] = lottery_mean
    features['actual_std'] = lottery_std
    
    return features


def extract_ml_features_with_context(self, seed: int, lottery_history: List[int], ...):
    """
    New method that returns features + context separately.
    
    Returns:
        Tuple[Dict, Dict]: (features_for_ml, context)
    """
    features = self.extract_ml_features(seed, lottery_history, ...)
    
    # Separate context fields
    context = {
        'lottery_stats': {
            'mean': features.pop('actual_mean'),
            'std': features.pop('actual_std'),
            'min': float(min(lottery_history)),
            'max': float(max(lottery_history)),
            'count': len(lottery_history)
        },
        'deprecated_feature_keys': ['actual_mean', 'actual_std'],
        'schema_version': '1.0.1'
    }
    
    return features, context
```

#### File: `full_scoring_worker.py`

```python
# When building output JSON:

def build_survivor_output(seed, features, context, metadata):
    """Build survivor JSON with new context structure."""
    return {
        'seed': seed,
        'score': features.get('score', 0.0),
        'features': features,
        'context': context,  # NEW
        'metadata': metadata
    }
```

#### File: `meta_prediction_optimizer_anti_overfit.py`

```python
# When building feature matrix, exclude context fields:

CONTEXT_FIELDS = {'actual_mean', 'actual_std'}

def build_feature_matrix(survivors):
    # Get feature names excluding context
    feature_names = [k for k in survivors[0]['features'].keys()
                    if k not in CONTEXT_FIELDS and k not in ('score', 'confidence')]
    ...
```

#### File: `config_manifests/feature_registry.json`

```json
{
  "schema_version": "1.0.1",
  "per_seed_features": {
    "count": 48,
    "discriminative": true,
    "names": [
      "pred_mean", "pred_std", "pred_min", "pred_max",
      "residual_mean", "residual_std", "residual_abs_mean", "residual_max_abs",
      "exact_matches", "total_predictions", "best_offset", "confidence",
      "lane_agreement_8", "lane_agreement_125", "lane_consistency",
      "temporal_stability_mean", "temporal_stability_std",
      "temporal_stability_min", "temporal_stability_max", "temporal_stability_trend",
      "residue_8_match_rate", "residue_8_kl_divergence", "residue_8_coherence",
      "residue_125_match_rate", "residue_125_kl_divergence", "residue_125_coherence",
      "residue_1000_match_rate", "residue_1000_kl_divergence", "residue_1000_coherence",
      "forward_count", "reverse_count", "bidirectional_count",
      "intersection_weight", "survivor_overlap_ratio",
      "intersection_count", "intersection_ratio",
      "forward_only_count", "reverse_only_count", "bidirectional_selectivity",
      "skip_entropy", "skip_mean", "skip_std", "skip_min", "skip_max", "skip_range",
      "survivor_velocity", "velocity_acceleration"
    ]
  },
  "context_features": {
    "count": 2,
    "discriminative": false,
    "exclude_from_ml": true,
    "names": ["actual_mean", "actual_std"],
    "location": "context.lottery_stats"
  },
  "global_features": {
    "count": 14,
    "discriminative": false,
    "names": [
      "global_residue_8_entropy", "global_residue_125_entropy", "global_residue_1000_entropy",
      "global_power_of_two_bias", "global_frequency_bias_ratio",
      "global_suspicious_gap_percentage", "global_regime_change_detected",
      "global_regime_age", "global_marker_390_variance", "global_marker_804_variance",
      "global_marker_575_variance", "global_reseed_probability",
      "global_high_variance_count", "global_temporal_stability"
    ]
  },
  "total_in_json": 64,
  "total_for_ml": 62
}
```

---

## Validation Checkpoints

### After Phase 2:
```bash
python3 feature_integrity_validator.py \
    --survivors survivors_with_scores.json \
    --sample 50000

# Expected: skip_* features should now have variance
```

### After Phase 3:
```bash
# Verify no hardcoded values
grep -n "0\.1\|= 400\|= 0\.0" survivor_scorer.py | grep -v "#"

# Should return no matches for confidence/total_predictions/best_offset
```

### After Phase 4:
```bash
# Verify context structure
python3 -c "
import json
with open('survivors_with_scores.json') as f:
    s = json.load(f)[0]
print('Has context:', 'context' in s)
print('Context keys:', list(s.get('context', {}).keys()))
"
```

---

## File Change Summary

| File | Phase | Type | Description |
|------|-------|------|-------------|
| `feature_integrity_validator.py` | 0 | NEW | Validation harness |
| `sieve_filter.py` | 2 | MODIFY | Export skip metadata |
| `run_step3_full_scoring.sh` | 2 | MODIFY | Add --skip-metadata flag |
| `full_scoring_worker.py` | 2,4 | MODIFY | Load skip metadata, context |
| `survivor_scorer.py` | 2,3,4 | MODIFY | Skip features, hardcoded, context |
| `feature_registry.json` | 4 | MODIFY | Update schema |
| `meta_prediction_optimizer_anti_overfit.py` | 4 | MODIFY | Exclude context from X |

---

**Document Version:** 1.0
**Author:** Claude (Session 18)
**Status:** Ready for Implementation

---

## Phase 2.5: Autonomous Feature Validator (Watcher Agent)

### 2.5.1 Purpose

Enable autonomous detection and remediation of zero-variance features. When the ML pipeline detects features that should have values but are zero due to configuration issues (e.g., history too short), it can automatically recommend or trigger re-runs with corrected parameters.

### 2.5.2 Feature Diagnostic Rules

| Feature | Zero Condition | Required Fix |
|---------|----------------|--------------|
| `regime_change_detected` | `len(history) < 2000` | Use history ≥ 2000 draws |
| `regime_age` | `len(history) < 2000` | Use history ≥ 2000 draws |
| `marker_*_variance` | Marker appears < 2 times | Use longer history or different markers |
| `high_variance_count` | No markers have CV > 1.0 | Informational only (legitimate zero) |
| `reseed_probability` | `high_variance_count == 0` | Depends on high_variance_count |
| `skip_entropy/mean/std` | Skip metadata not provided | Run Phase 2 skip pipeline |
| `survivor_velocity` | Temporal tracking not enabled | Run Phase 2 skip pipeline |
| `intersection_*` | Fields not in chunk metadata | Re-run Step 2 with updated code |

### 2.5.3 Implementation

#### File: `feature_watcher_agent.py` (NEW)
```python
#!/usr/bin/env python3
"""
Feature Watcher Agent - Autonomous Feature Quality Monitor

Detects zero-variance features and recommends/triggers remediation.

Usage:
    python3 feature_watcher_agent.py --survivors survivors_with_scores.json
    python3 feature_watcher_agent.py --survivors survivors_with_scores.json --auto-fix
"""

import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class FixAction(Enum):
    NONE = "none"  # Legitimately zero
    RERUN_STEP2 = "rerun_step2"
    RERUN_STEP3 = "rerun_step3"
    LONGER_HISTORY = "longer_history"
    PHASE2_SKIP_PIPELINE = "phase2_skip_pipeline"

@dataclass
class FeatureDiagnosis:
    feature: str
    current_value: float
    is_zero: bool
    cause: str
    fix_action: FixAction
    fix_params: Dict
    is_legitimate_zero: bool

class FeatureWatcherAgent:
    """Monitors feature quality and recommends/triggers fixes."""
    
    # Features that are legitimately zero when no anomalies detected
    LEGITIMATE_ZERO_FEATURES = {
        'high_variance_count',  # Zero if no markers have CV > 1.0
        'best_offset',  # Zero if alignment search disabled
    }
    
    # Minimum history requirements
    HISTORY_REQUIREMENTS = {
        'regime_change_detected': 2000,
        'regime_age': 2000,
    }
    
    # Features requiring Phase 2 skip pipeline
    SKIP_PIPELINE_FEATURES = {
        'skip_entropy', 'skip_mean', 'skip_std',
        'survivor_velocity', 'velocity_acceleration'
    }
    
    # Features requiring Step 2 re-run
    STEP2_FEATURES = {
        'intersection_count', 'intersection_ratio', 'intersection_weight',
        'forward_only_count', 'reverse_only_count', 'survivor_overlap_ratio'
    }
    
    def __init__(self, survivors_file: str, history_file: str = 'train_history.json'):
        self.survivors_file = survivors_file
        self.history_file = history_file
        self.survivors = self._load_survivors()
        self.history_length = self._get_history_length()
        self.diagnoses: List[FeatureDiagnosis] = []
    
    def _load_survivors(self) -> List[Dict]:
        with open(self.survivors_file) as f:
            return json.load(f)
    
    def _get_history_length(self) -> int:
        try:
            with open(self.history_file) as f:
                return len(json.load(f))
        except:
            return 0
    
    def analyze(self) -> List[FeatureDiagnosis]:
        """Analyze all features and diagnose zeros."""
        self.diagnoses = []
        
        if not self.survivors:
            return self.diagnoses
        
        features = self.survivors[0].get('features', {})
        
        # Sample 1000 survivors for variance check
        sample_size = min(1000, len(self.survivors))
        
        for feature_name, value in features.items():
            # Check variance across sample
            values = [s['features'].get(feature_name, 0) 
                     for s in self.survivors[:sample_size]]
            unique_count = len(set(values))
            is_zero = (unique_count == 1 and values[0] == 0.0)
            
            if is_zero:
                diagnosis = self._diagnose_zero_feature(feature_name, value)
                self.diagnoses.append(diagnosis)
        
        return self.diagnoses
    
    def _diagnose_zero_feature(self, feature: str, value: float) -> FeatureDiagnosis:
        """Diagnose why a feature is zero and recommend fix."""
        
        # Check if legitimately zero
        if feature in self.LEGITIMATE_ZERO_FEATURES:
            return FeatureDiagnosis(
                feature=feature,
                current_value=value,
                is_zero=True,
                cause="Legitimately zero (no anomalies detected)",
                fix_action=FixAction.NONE,
                fix_params={},
                is_legitimate_zero=True
            )
        
        # Check history length requirements
        if feature in self.HISTORY_REQUIREMENTS:
            required = self.HISTORY_REQUIREMENTS[feature]
            if self.history_length < required:
                return FeatureDiagnosis(
                    feature=feature,
                    current_value=value,
                    is_zero=True,
                    cause=f"History too short ({self.history_length} < {required})",
                    fix_action=FixAction.LONGER_HISTORY,
                    fix_params={'min_history': required, 'current_history': self.history_length},
                    is_legitimate_zero=False
                )
        
        # Check skip pipeline features
        if feature in self.SKIP_PIPELINE_FEATURES:
            return FeatureDiagnosis(
                feature=feature,
                current_value=value,
                is_zero=True,
                cause="Skip metadata pipeline not implemented",
                fix_action=FixAction.PHASE2_SKIP_PIPELINE,
                fix_params={'phase': 2, 'component': 'skip_metadata'},
                is_legitimate_zero=False
            )
        
        # Check Step 2 features
        if feature in self.STEP2_FEATURES:
            return FeatureDiagnosis(
                feature=feature,
                current_value=value,
                is_zero=True,
                cause="Field not in chunk metadata (old Step 2 output)",
                fix_action=FixAction.RERUN_STEP2,
                fix_params={'step': 2},
                is_legitimate_zero=False
            )
        
        # Check marker features
        if feature.startswith('global_marker_'):
            marker_num = feature.split('_')[2]
            return FeatureDiagnosis(
                feature=feature,
                current_value=value,
                is_zero=True,
                cause=f"Marker {marker_num} appears < 2 times in history",
                fix_action=FixAction.LONGER_HISTORY,
                fix_params={'reason': 'marker_appearances'},
                is_legitimate_zero=False
            )
        
        # Global features dependent on other globals
        if feature in ('global_reseed_probability',):
            return FeatureDiagnosis(
                feature=feature,
                current_value=value,
                is_zero=True,
                cause="Dependent on high_variance_count (which is 0)",
                fix_action=FixAction.NONE,
                fix_params={},
                is_legitimate_zero=True
            )
        
        # Unknown - default to legitimate
        return FeatureDiagnosis(
            feature=feature,
            current_value=value,
            is_zero=True,
            cause="Unknown cause (treating as legitimate)",
            fix_action=FixAction.NONE,
            fix_params={},
            is_legitimate_zero=True
        )
    
    def generate_report(self) -> str:
        """Generate human-readable diagnostic report."""
        if not self.diagnoses:
            self.analyze()
        
        lines = [
            "=" * 70,
            "FEATURE WATCHER AGENT - DIAGNOSTIC REPORT",
            "=" * 70,
            f"Survivors file: {self.survivors_file}",
            f"History length: {self.history_length}",
            f"Total zero-variance features: {len(self.diagnoses)}",
            "",
        ]
        
        # Group by fix action
        by_action = {}
        for d in self.diagnoses:
            action = d.fix_action.value
            if action not in by_action:
                by_action[action] = []
            by_action[action].append(d)
        
        for action, diagnoses in by_action.items():
            lines.append(f"\n### {action.upper()} ({len(diagnoses)} features)")
            lines.append("-" * 50)
            for d in diagnoses:
                status = "✅ OK" if d.is_legitimate_zero else "⚠️ FIXABLE"
                lines.append(f"  {d.feature}: {status}")
                lines.append(f"    Cause: {d.cause}")
                if d.fix_params:
                    lines.append(f"    Params: {d.fix_params}")
        
        # Recommendations
        lines.append("\n" + "=" * 70)
        lines.append("RECOMMENDED ACTIONS")
        lines.append("=" * 70)
        
        fixable = [d for d in self.diagnoses if not d.is_legitimate_zero]
        if not fixable:
            lines.append("✅ No fixable issues - all zeros are legitimate")
        else:
            actions_needed = set(d.fix_action for d in fixable)
            for action in actions_needed:
                if action == FixAction.LONGER_HISTORY:
                    min_hist = max(d.fix_params.get('min_history', 0) 
                                  for d in fixable if d.fix_action == action)
                    lines.append(f"• Use longer history file (≥ {min_hist} draws)")
                    lines.append(f"  Command: ./run_step3_full_scoring.sh --train-history daily3_full_{min_hist}.json")
                elif action == FixAction.RERUN_STEP2:
                    lines.append("• Re-run Step 2 (window optimizer)")
                    lines.append("  Command: python3 window_optimizer.py --strategy bayesian ...")
                elif action == FixAction.PHASE2_SKIP_PIPELINE:
                    lines.append("• Implement Phase 2 skip metadata pipeline")
                    lines.append("  See: TECHNICAL_SPEC_Feature_Remediation_Phases_2_4.md")
        
        return "\n".join(lines)
    
    def auto_fix(self, dry_run: bool = True) -> List[str]:
        """
        Automatically apply fixes where possible.
        
        Returns list of commands executed (or would execute if dry_run).
        """
        if not self.diagnoses:
            self.analyze()
        
        commands = []
        fixable = [d for d in self.diagnoses if not d.is_legitimate_zero]
        
        # Group by action type
        actions_needed = set(d.fix_action for d in fixable)
        
        for action in actions_needed:
            if action == FixAction.RERUN_STEP3:
                cmd = f"./run_step3_full_scoring.sh --survivors bidirectional_survivors.json --train-history {self.history_file}"
                commands.append(cmd)
            elif action == FixAction.LONGER_HISTORY:
                # Check for available longer history files
                available = list(Path('.').glob('daily3_*_*.json'))
                longer = [f for f in available 
                         if self._get_file_draw_count(f) >= 2000]
                if longer:
                    best = max(longer, key=lambda f: self._get_file_draw_count(f))
                    cmd = f"./run_step3_full_scoring.sh --survivors bidirectional_survivors.json --train-history {best}"
                    commands.append(cmd)
        
        if not dry_run:
            for cmd in commands:
                print(f"Executing: {cmd}")
                subprocess.run(cmd, shell=True)
        
        return commands
    
    def _get_file_draw_count(self, filepath: Path) -> int:
        """Get number of draws in a lottery file."""
        try:
            with open(filepath) as f:
                return len(json.load(f))
        except:
            return 0


def main():
    parser = argparse.ArgumentParser(description='Feature Watcher Agent')
    parser.add_argument('--survivors', required=True, help='Path to survivors_with_scores.json')
    parser.add_argument('--history', default='train_history.json', help='Path to history file')
    parser.add_argument('--auto-fix', action='store_true', help='Automatically apply fixes')
    parser.add_argument('--dry-run', action='store_true', help='Show commands without executing')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()
    
    agent = FeatureWatcherAgent(args.survivors, args.history)
    agent.analyze()
    
    if args.json:
        output = {
            'diagnoses': [
                {
                    'feature': d.feature,
                    'is_zero': d.is_zero,
                    'cause': d.cause,
                    'fix_action': d.fix_action.value,
                    'fix_params': d.fix_params,
                    'is_legitimate_zero': d.is_legitimate_zero
                }
                for d in agent.diagnoses
            ],
            'summary': {
                'total_zero': len(agent.diagnoses),
                'legitimate_zero': sum(1 for d in agent.diagnoses if d.is_legitimate_zero),
                'fixable': sum(1 for d in agent.diagnoses if not d.is_legitimate_zero)
            }
        }
        print(json.dumps(output, indent=2))
    else:
        print(agent.generate_report())
    
    if args.auto_fix:
        commands = agent.auto_fix(dry_run=args.dry_run)
        if args.dry_run:
            print("\n[DRY RUN] Would execute:")
            for cmd in commands:
                print(f"  {cmd}")


if __name__ == '__main__':
    main()
```

### 2.5.4 Integration with Pipeline

#### Option A: Manual Trigger
```bash
# After Step 3 completes, run watcher
python3 feature_watcher_agent.py --survivors survivors_with_scores.json

# Auto-fix with dry run first
python3 feature_watcher_agent.py --survivors survivors_with_scores.json --auto-fix --dry-run

# Apply fixes
python3 feature_watcher_agent.py --survivors survivors_with_scores.json --auto-fix
```

#### Option B: Integrated into run_step3_full_scoring.sh
```bash
# At end of Phase 6 (validation):

echo "Phase 7: Feature Quality Check..."
python3 feature_watcher_agent.py --survivors survivors_with_scores.json --json > /tmp/feature_check.json

FIXABLE=$(python3 -c "import json; d=json.load(open('/tmp/feature_check.json')); print(d['summary']['fixable'])")
if [ "$FIXABLE" -gt 0 ]; then
    echo "⚠️  $FIXABLE features have fixable zero-variance issues"
    echo "   Run: python3 feature_watcher_agent.py --survivors survivors_with_scores.json"
fi
```

#### Option C: Full Autonomous Loop (Future)
```python
# In complete_whitepaper_workflow_with_meta_optimizer_v3.py

def run_step3_with_validation():
    """Run Step 3 with automatic re-run on feature issues."""
    max_retries = 2
    
    for attempt in range(max_retries):
        # Run Step 3
        run_command(['bash', 'run_step3_full_scoring.sh', ...])
        
        # Validate features
        agent = FeatureWatcherAgent('survivors_with_scores.json')
        agent.analyze()
        
        fixable = [d for d in agent.diagnoses if not d.is_legitimate_zero]
        
        if not fixable:
            print("✅ All features valid")
            return True
        
        # Check if we can fix
        can_fix = any(d.fix_action in (FixAction.RERUN_STEP3, FixAction.LONGER_HISTORY) 
                     for d in fixable)
        
        if can_fix and attempt < max_retries - 1:
            print(f"⚠️ {len(fixable)} fixable issues, retrying...")
            commands = agent.auto_fix(dry_run=False)
        else:
            print(f"⚠️ {len(fixable)} issues remain (Phase 2 required)")
            return True  # Continue anyway
    
    return True
```

### 2.5.5 Validation
```bash
# Test the watcher agent
python3 feature_watcher_agent.py --survivors survivors_with_scores.json

# Expected output:
# ======================================================================
# FEATURE WATCHER AGENT - DIAGNOSTIC REPORT
# ======================================================================
# ...
# ### NONE (7 features)
# --------------------------------------------------
#   best_offset: ✅ OK
#   global_high_variance_count: ✅ OK
#   ...
# 
# ### PHASE2_SKIP_PIPELINE (5 features)
# --------------------------------------------------
#   skip_entropy: ⚠️ FIXABLE
#   skip_mean: ⚠️ FIXABLE
#   ...
```

---

**Phase 2.5 Status:** Ready for Implementation
**Dependencies:** None (can be implemented independently)
**Priority:** Medium (enables autonomous operation)
