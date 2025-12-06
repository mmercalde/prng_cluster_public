# v3.2.0 Addendum: Threshold Implementation (December 6, 2025)

## Implementation Completed

The v3.2.0 proposal has been partially implemented for Step 1 (Window Optimizer) as a proof-of-concept. This addendum documents what was done and serves as a pattern for other steps.

---

## Problem Solved

**Original Issue:** Sieve thresholds were hardcoded to 0.01 (1%), causing:
- 50,000 false positive survivors (should be ~1-10)
- AI agents couldn't adjust thresholds
- Forward/reverse sieves used same threshold (should differ: 0.72 vs 0.81)

**Root Cause:** Violated the configurability principle:
> "Nothing should be hardcoded - all parameters must remain adjustable for ML and AI applications."

---

## Implementation Details

### 1. SearchBounds Dataclass (window_optimizer.py)

Added threshold ranges for Optuna to explore:
```python
@dataclass
class SearchBounds:
    # ... existing fields ...
    
    # Threshold bounds for Optuna optimization
    min_forward_threshold: float = 0.50
    max_forward_threshold: float = 0.95
    min_reverse_threshold: float = 0.60
    max_reverse_threshold: float = 0.98
    
    # Optimized defaults (from prior Optuna runs)
    default_forward_threshold: float = 0.72
    default_reverse_threshold: float = 0.81
```

### 2. WindowConfig Dataclass (window_optimizer.py)

Added threshold fields to configuration:
```python
@dataclass
class WindowConfig:
    window_size: int
    offset: int
    sessions: List[str]
    skip_min: int
    skip_max: int
    forward_threshold: float = 0.72  # NEW
    reverse_threshold: float = 0.81  # NEW
```

### 3. Optuna Integration (window_optimizer_bayesian.py)

Added suggest_float() calls:
```python
forward_threshold = trial.suggest_float('forward_threshold',
                                       bounds.min_forward_threshold,
                                       bounds.max_forward_threshold)
reverse_threshold = trial.suggest_float('reverse_threshold',
                                       bounds.min_reverse_threshold,
                                       bounds.max_reverse_threshold)
```

### 4. CLI Arguments (window_optimizer.py)

Added manual override capability:
```python
parser.add_argument('--forward-threshold', type=float, default=None,
                   help='Forward sieve threshold (0.5-0.95). If not set, Optuna optimizes it.')
parser.add_argument('--reverse-threshold', type=float, default=None,
                   help='Reverse sieve threshold (0.6-0.98). If not set, Optuna optimizes it.')
```

### 5. Agent Manifest (agent_manifests/window_optimizer.json v1.3.0)

Added parameter_bounds section for AI awareness:
```json
{
  "version": "1.3.0",
  "parameter_bounds": {
    "forward_threshold": {
      "type": "float",
      "min": 0.50,
      "max": 0.95,
      "default": 0.72,
      "description": "Forward sieve match threshold. Higher = stricter filtering.",
      "optimized_by": "Optuna TPE",
      "effect": "Controls false positive rate in forward sieve. Values 0.70-0.85 typical."
    },
    "reverse_threshold": {
      "type": "float",
      "min": 0.60,
      "max": 0.98,
      "default": 0.81,
      "description": "Reverse sieve match threshold. Should be >= forward_threshold.",
      "optimized_by": "Optuna TPE",
      "effect": "Controls historical consistency check. Values 0.75-0.90 typical."
    }
  }
}
```

### 6. Integration Layer (window_optimizer_integration_final.py)

Updated to pass separate thresholds downstream:
```python
def run_bidirectional_test(...,
                          forward_threshold: float = 0.72,
                          reverse_threshold: float = 0.81,
                          ...):
```

---

## Configuration Flow (New Architecture)
```
1. Optuna Suggests (PRIMARY)
   └─ trial.suggest_float('forward_threshold', 0.50, 0.95)
   └─ trial.suggest_float('reverse_threshold', 0.60, 0.98)

2. CLI Override (MANUAL)
   └─ --forward-threshold 0.72
   └─ --reverse-threshold 0.81

3. Config File (STORED DEFAULTS)
   └─ optimal_window_config.json contains optimized values

4. Agent Manifest (AI AWARENESS)
   └─ parameter_bounds exposes ranges to AI agents
```

---

## Test Results

| Metric | Before (0.01) | After (Optuna) |
|--------|---------------|----------------|
| False positives | 50,000 | 1,639 |
| Reduction | - | 97% |
| Seed 12345 found | Buried in noise | ✅ Found |
| Optuna thresholds | N/A | FT=0.85, RT=0.96 |

---

## Files Modified

| File | Changes |
|------|---------|
| window_optimizer.py | SearchBounds, WindowConfig, CLI args |
| window_optimizer_bayesian.py | suggest_float(), WindowConfig class |
| window_optimizer_integration_final.py | Pass thresholds downstream |
| sieve_filter.py | dtype handling |
| agent_manifests/window_optimizer.json | v1.3.0 with parameter_bounds |

---

## Pattern for Other Steps

To add configurable parameters to Steps 2-6, follow this pattern:

1. **Identify optimizable parameters** (e.g., chunk_size, learning_rate, etc.)
2. **Add to Optuna search space** with suggest_int/suggest_float
3. **Add CLI arguments** for manual override
4. **Add parameter_bounds to manifest** with type, min, max, default, description
5. **Update integration layer** to pass parameters downstream

### Example for Step 2 (Scorer Meta):
```json
{
  "parameter_bounds": {
    "chunk_size": {
      "type": "int",
      "min": 100,
      "max": 10000,
      "default": 1000,
      "description": "Number of survivors per scoring job"
    },
    "trials": {
      "type": "int",
      "min": 2,
      "max": 500,
      "default": 100,
      "description": "Number of Optuna trials for scorer optimization"
    }
  }
}
```

---

## Git Commits

- `bd95d6a` - Make thresholds configurable for ML/AI optimization
- `eec22ea` - Fix chunk sizing (19K min) and progress display type hints
- `b45c8cc` - Update CURRENT_Status.txt with Session 3

---

## Status

| Component | Status |
|-----------|--------|
| Step 1 (Window Optimizer) | ✅ Complete with parameter_bounds |
| Steps 2-6 | 📝 Can follow same pattern when needed |
| Pydantic Framework Integration | ✅ Ready (ParameterContext supports this) |

---

**End of v3.2.0 Addendum**
