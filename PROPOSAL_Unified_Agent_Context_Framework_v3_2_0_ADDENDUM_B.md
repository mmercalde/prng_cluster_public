# v3.2.0 Addendum B: Steps 2-6 Manifest Implementation (December 6, 2025)

## Implementation Completed

Following the Step 1 pattern established in Addendum A, all remaining pipeline steps now have complete manifests with `parameter_bounds` sections. This completes TODO #1 from instructions.txt.

---

## Files Created

| File | Step | Description | Parameters |
|------|------|-------------|------------|
| `window_optimizer.json` | 1 | Bayesian window optimization | 10 params |
| `scorer_meta.json` | 2 | Distributed scorer meta-optimizer | 6 params |
| `full_scoring.json` | 3 | Full survivor scoring | 5 params |
| `ml_meta.json` | 4 | ML architecture optimizer | 13 params |
| `reinforcement.json` | 5 | K-fold anti-overfit training | 13 params |
| `prediction.json` | 6 | Final prediction generation | 11 params |

**Total: 58 configurable parameters across 6 pipeline steps**

---

## Manifest Schema (v1.3.0)

Each manifest includes the standard fields plus the new `parameter_bounds` section:

```json
{
  "agent_name": "step_name_agent",
  "description": "What this step does",
  "pipeline_step": 1-6,
  "version": "1.3.0",
  "inputs": ["required_files"],
  "outputs": ["produced_files"],
  "actions": [{...}],
  "follow_up_agents": ["next_agent"],
  "success_condition": "validation expression",
  "retry": 2,
  "parameter_bounds": {
    "param_name": {
      "type": "int|float|bool|choice",
      "min": 0,
      "max": 100,
      "default": 50,
      "choices": ["a", "b", "c"],
      "description": "What this parameter does",
      "optimized_by": "Optuna TPE|Manual|Fixed",
      "effect": "How changing this affects behavior"
    }
  }
}
```

---

## Parameter Summary by Step

### Step 1: Window Optimizer
*(Implemented in Addendum A)*

| Parameter | Type | Range | Default |
|-----------|------|-------|---------|
| window_size | int | 128-4096 | 512 |
| offset | int | 0-2000 | 0 |
| skip_min | int | 0-50 | 0 |
| skip_max | int | 20-500 | 100 |
| forward_threshold | float | 0.50-0.95 | 0.72 |
| reverse_threshold | float | 0.60-0.98 | 0.81 |
| window_trials | int | 10-200 | 50 |
| seed_count | int | 1M-5B | 50M |
| search_strategy | choice | bayesian/random/grid | bayesian |
| test_both_modes | bool | true/false | false |

### Step 2: Scorer Meta

| Parameter | Type | Range | Default |
|-----------|------|-------|---------|
| chunk_size | int | 100-10000 | 1000 |
| scorer_trials | int | 10-500 | 100 |
| match_weight | float | 0.1-2.0 | 1.0 |
| residue_weight | float | 0.0-1.5 | 0.5 |
| temporal_weight | float | 0.0-1.5 | 0.3 |
| min_score_threshold | float | 0.01-0.50 | 0.10 |

### Step 3: Full Scoring

| Parameter | Type | Range | Default |
|-----------|------|-------|---------|
| chunk_size | int | 500-20000 | 5000 |
| batch_size | int | 64-1024 | 256 |
| feature_extraction_depth | int | 46-128 | 64 |
| score_precision | int | 4-8 | 6 |
| parallel_workers | int | 1-26 | 26 |

### Step 4: ML Meta

| Parameter | Type | Range | Default |
|-----------|------|-------|---------|
| ml_trials | int | 20-500 | 100 |
| timeout_hours | float | 1.0-48.0 | 8.0 |
| parallel_jobs | int | 1-4 | 2 |
| min_layers | int | 1-5 | 2 |
| max_layers | int | 3-10 | 6 |
| min_neurons | int | 16-128 | 32 |
| max_neurons | int | 128-1024 | 512 |
| learning_rate_min | float | 1e-6 - 1e-4 | 1e-5 |
| learning_rate_max | float | 1e-3 - 1e-1 | 1e-2 |
| dropout_min | float | 0.0-0.3 | 0.1 |
| dropout_max | float | 0.3-0.7 | 0.5 |
| weight_decay_min | float | 1e-7 - 1e-5 | 1e-6 |
| weight_decay_max | float | 1e-4 - 1e-2 | 1e-3 |

### Step 5: Reinforcement

| Parameter | Type | Range | Default |
|-----------|------|-------|---------|
| k_folds | int | 3-15 | 5 |
| trials_per_fold | int | 10-100 | 50 |
| epochs_min | int | 10-50 | 20 |
| epochs_max | int | 50-500 | 200 |
| early_stopping_patience | int | 5-50 | 20 |
| batch_size | int | 16-256 | 64 |
| learning_rate | float | 1e-5 - 1e-2 | 1e-3 |
| lr_scheduler_factor | float | 0.1-0.9 | 0.5 |
| lr_scheduler_patience | int | 3-20 | 10 |
| gradient_clip_norm | float | 0.5-5.0 | 1.0 |
| validation_split | float | 0.10-0.30 | 0.20 |
| reinforce_weight | float | 0.0-1.0 | 0.5 |
| temporal_window | int | 10-200 | 50 |

### Step 6: Prediction

| Parameter | Type | Range | Default |
|-----------|------|-------|---------|
| pool_size | int | 5-100 | 20 |
| confidence_threshold | float | 0.50-0.95 | 0.70 |
| ensemble_mode | choice | single/weighted/voting/stacking | weighted |
| temporal_decay | float | 0.90-1.0 | 0.98 |
| drift_sensitivity | float | 0.0-1.0 | 0.3 |
| forward_weight | float | 0.3-0.7 | 0.5 |
| bidirectional_bonus | float | 1.0-3.0 | 1.5 |
| recency_window | int | 10-100 | 30 |
| min_survivor_history | int | 5-50 | 10 |
| skip_mode_preference | choice | constant/variable/adaptive/both | adaptive |
| explanation_verbosity | choice | minimal/standard/detailed/debug | standard |

---

## Deployment Instructions

```bash
# On Zeus, from the distributed_prng_analysis directory:
cd ~/distributed_prng_analysis

# Create directory if not exists
mkdir -p agent_manifests

# Extract downloaded archive
tar -xzvf agent_manifests_v1.3.0.tar.gz

# Verify
ls -la agent_manifests/
```

Expected output:
```
window_optimizer.json
scorer_meta.json
full_scoring.json
ml_meta.json
reinforcement.json
prediction.json
```

---

## Integration with ParameterContext

These manifests work with the `ParameterContext` class from the main proposal:

```python
from agents.parameters.parameter_context import ParameterContext
from agents.manifest.agent_manifest import AgentManifest

# Load manifest
manifest = AgentManifest.load('agent_manifests/reinforcement.json')

# Build parameter context with current values
context = ParameterContext.build(
    manifest=manifest,
    current_values={'k_folds': 5, 'learning_rate': 0.001}
)

# Generate prompt section for AI agent
prompt_section = context.to_prompt_section()

# Validate AI's suggested adjustments
errors = context.validate_adjustments({
    'k_folds': 10,           # valid
    'learning_rate': 0.1     # invalid - above max 0.01
})
```

---

## Git Commit Message

```
feat(manifests): Add parameter_bounds to Steps 2-6 (v1.3.0)

Following Step 1 pattern from Addendum A:
- scorer_meta.json: 6 parameters
- full_scoring.json: 5 parameters
- ml_meta.json: 13 parameters
- reinforcement.json: 13 parameters
- prediction.json: 11 parameters

Completes TODO #1 from instructions.txt
Reference: PROPOSAL v3.2.0 Addendum B
```

---

## Status

| Component | Status |
|-----------|--------|
| Step 1 (Window Optimizer) | ✅ Complete (Addendum A) |
| Step 2 (Scorer Meta) | ✅ Manifest Complete |
| Step 3 (Full Scoring) | ✅ Manifest Complete |
| Step 4 (ML Meta) | ✅ Manifest Complete |
| Step 5 (Reinforcement) | ✅ Manifest Complete |
| Step 6 (Prediction) | ✅ Manifest Complete |
| ParameterContext Integration | ✅ Ready |

---

**End of v3.2.0 Addendum B**
