# Chapter 1: Window Optimizer (Step 1)

## PRNG Analysis Pipeline — Complete Operating Guide

**Version:** 2.0  
**File:** `window_optimizer.py`  
**Lines:** 868  
**Purpose:** Bayesian optimization of window parameters + survivor generation

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Data Structures](#3-data-structures)
4. [Search Bounds Configuration](#4-search-bounds-configuration)
5. [Scoring Functions](#5-scoring-functions)
6. [Search Strategies](#6-search-strategies)
7. [WindowOptimizer Class](#7-windowoptimizer-class)
8. [Bayesian Optimization Flow](#8-bayesian-optimization-flow)
9. [Run With Config Mode](#9-run-with-config-mode)
10. [CLI Interface](#10-cli-interface)
11. [Integration Layer](#11-integration-layer)
12. [Output Files](#12-output-files)
13. [Agent Metadata Injection](#13-agent-metadata-injection)
14. [Complete Method Reference](#14-complete-method-reference)

---

## 1. Overview

### 1.1 What Window Optimizer Does

The Window Optimizer is **Step 1** of the 6-step pipeline. It performs two critical functions:

1. **Parameter Optimization:** Uses Bayesian optimization (Optuna TPE) to find optimal window parameters
2. **Survivor Generation:** Runs real sieves across all 26 GPUs and accumulates survivors

### 1.2 Version 2.0 Features

```
NEW IN V2.0:
- --test-both-modes flag: Test constant AND variable skip patterns
- Survivors tagged with skip_mode metadata for ML feature engineering
- Backward compatible: defaults to constant skip only
```

### 1.3 Key Insight

**The optimizer doesn't run sieves directly.** It delegates to the integration layer (`window_optimizer_integration_final.py`) which coordinates with `coordinator.py` to run real sieves across all 26 GPUs.

### 1.4 Usage Examples

```bash
# Mode 1: Bayesian optimization (constant skip only)
python3 window_optimizer.py --strategy bayesian \
    --lottery-file lottery.json --trials 50

# Mode 2: Bayesian optimization (BOTH constant AND variable skip)
python3 window_optimizer.py --strategy bayesian \
    --lottery-file lottery.json --trials 50 --test-both-modes

# Mode 3: Run with existing optimal config
python3 window_optimizer.py --config-file optimal_window_config.json \
    --lottery-file lottery.json
```

---

## 2. Architecture

### 2.1 Component Hierarchy

```
window_optimizer.py
    │
    ├─→ WindowConfig, SearchBounds, TestResult (data structures)
    │
    ├─→ BayesianOptimization (strategy)
    │   └─→ window_optimizer_bayesian.py
    │       └─→ OptunaBayesianSearch (Optuna TPE)
    │
    ├─→ WindowOptimizer (main class)
    │   └─→ test_configuration()
    │       └─→ window_optimizer_integration_final.py
    │           └─→ run_bidirectional_test()
    │               └─→ coordinator.py
    │                   └─→ 26 GPUs execute sieves
    │
    └─→ Output files:
        ├─ optimal_window_config.json
        ├─ bidirectional_survivors.json
        ├─ forward_survivors.json
        ├─ reverse_survivors.json
        ├─ train_history.json
        └─ holdout_history.json
```

### 2.2 Execution Flow

```
main()
    │
    ├─→ args.strategy == 'bayesian':
    │   └─→ run_bayesian_optimization()
    │       ├─→ MultiGPUCoordinator()
    │       ├─→ add_window_optimizer_to_coordinator()
    │       ├─→ coordinator.optimize_window()
    │       │   └─→ [N trials of real sieve execution]
    │       ├─→ inject_agent_metadata()
    │       └─→ Save outputs
    │
    └─→ args.config_file:
        └─→ run_with_config()
            ├─→ Load optimal config
            ├─→ MultiGPUCoordinator()
            ├─→ add_window_optimizer_to_coordinator()
            ├─→ [iterations × run_bidirectional_test()]
            ├─→ Deduplicate survivors
            └─→ Save outputs
```

---

## 3. Data Structures

### 3.1 WindowConfig

```python
@dataclass
class WindowConfig:
    """Complete window and skip configuration for sieve execution"""
    
    window_size: int           # Size of temporal window (number of draws)
    offset: int                # Time offset from current draw
    sessions: List[str]        # ['midday', 'evening'] or subset
    skip_min: int              # Minimum skip for variable PRNGs
    skip_max: int              # Maximum skip for variable PRNGs
    forward_threshold: float = 0.40   # Forward sieve threshold
    reverse_threshold: float = 0.45   # Reverse sieve threshold
```

**Methods:**

| Method | Returns | Purpose |
|--------|---------|---------|
| `__hash__()` | `int` | Make hashable for sets/dicts |
| `description()` | `str` | Human-readable: `W512_O100_midday+evening_S0-50_FT0.4_RT0.45` |
| `to_dict()` | `Dict` | JSON serialization |

**Example:**

```python
config = WindowConfig(
    window_size=512,
    offset=100,
    sessions=['midday', 'evening'],
    skip_min=0,
    skip_max=50,
    forward_threshold=0.01,
    reverse_threshold=0.01
)
print(config.description())
# Output: W512_O100_midday+evening_S0-50_FT0.01_RT0.01
```

### 3.2 SearchBounds

```python
@dataclass
class SearchBounds:
    """Search space boundaries for optimization"""
    
    # Window parameters
    min_window_size: int = 2
    max_window_size: int = 500
    min_offset: int = 0
    max_offset: int = 100
    
    # Skip parameters
    min_skip_min: int = 0
    max_skip_min: int = 10
    min_skip_max: int = 10
    max_skip_max: int = 500
    
    # Threshold bounds (LOW for discovery)
    min_forward_threshold: float = 0.001
    max_forward_threshold: float = 0.10
    min_reverse_threshold: float = 0.001
    max_reverse_threshold: float = 0.10
    
    # Defaults
    default_forward_threshold: float = 0.01
    default_reverse_threshold: float = 0.01
    
    # Session options
    session_options: List[List[str]] = None  # Auto-initialized
```

**Key Methods:**

| Method | Returns | Purpose |
|--------|---------|---------|
| `from_config(path)` | `SearchBounds` | Load from distributed_config.json |
| `random_config()` | `WindowConfig` | Generate random config within bounds |
| `is_valid(config)` | `bool` | Validate config against bounds |

**Session Options (auto-initialized):**

```python
session_options = [
    ['midday', 'evening'],  # Both sessions
    ['midday'],             # Midday only
    ['evening']             # Evening only
]
```

### 3.3 TestResult

```python
@dataclass
class TestResult:
    """Result from testing a window configuration"""
    
    config: WindowConfig
    forward_count: int          # Survivors from forward sieve
    reverse_count: int          # Survivors from reverse sieve
    bidirectional_count: int    # Intersection (survived both)
    iteration: int              # Trial number
```

**Computed Properties:**

```python
@property
def precision(self) -> float:
    """Precision = bidirectional / forward"""
    return self.bidirectional_count / self.forward_count if self.forward_count > 0 else 0

@property
def recall(self) -> float:
    """Recall = bidirectional / reverse"""
    return self.bidirectional_count / self.reverse_count if self.reverse_count > 0 else 0
```

---

## 4. Search Bounds Configuration

### 4.1 Single Source of Truth

Search bounds are loaded from `distributed_config.json`:

```python
def load_search_bounds_from_config(config_path: str = "distributed_config.json") -> dict:
    """Load search bounds from config file"""
    defaults = {
        "window_size": {"min": 2, "max": 500},
        "offset": {"min": 0, "max": 100},
        "skip_min": {"min": 0, "max": 10},
        "skip_max": {"min": 10, "max": 500},
        "forward_threshold": {"min": 0.001, "max": 0.10, "default": 0.01},
        "reverse_threshold": {"min": 0.001, "max": 0.10, "default": 0.01}
    }
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        bounds = config.get("search_bounds", {})
        # Merge: config values override defaults
        for key in defaults:
            if key in bounds:
                defaults[key].update(bounds[key])
        return defaults
    except (FileNotFoundError, json.JSONDecodeError):
        return defaults  # Use defaults if config missing
```

### 4.2 distributed_config.json Structure

```json
{
    "search_bounds": {
        "window_size": {"min": 2, "max": 500},
        "offset": {"min": 0, "max": 100},
        "skip_min": {"min": 0, "max": 10},
        "skip_max": {"min": 10, "max": 500},
        "forward_threshold": {"min": 0.001, "max": 0.10, "default": 0.01},
        "reverse_threshold": {"min": 0.001, "max": 0.10, "default": 0.01}
    }
}
```

### 4.3 Threshold Philosophy

**CRITICAL INSIGHT:** Use LOW thresholds (0.001-0.10) for discovery!

```
The system is a behavioral fingerprint machine, NOT a filter.
Low thresholds maximize seed discovery.
Bidirectional intersection handles the actual filtering.
High thresholds (0.72+) would eliminate candidates prematurely.
```

---

## 5. Scoring Functions

### 5.1 Base Class

```python
class ScoringFunction(ABC):
    """Base class for scoring functions"""
    
    @abstractmethod
    def score(self, result: TestResult) -> float:
        """Score a test result. Higher is better."""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return name of scoring function"""
        pass
```

### 5.2 BidirectionalCountScorer (Default)

```python
class BidirectionalCountScorer(ScoringFunction):
    """Score = count of bidirectional survivors"""
    
    def score(self, result: TestResult) -> float:
        return float(result.bidirectional_count)
    
    def name(self) -> str:
        return "bidirectional_count"
```

**Rationale:** Seeds that survive BOTH forward AND reverse sieves are more likely to be temporally stable and produce good predictions.

---

## 6. Search Strategies

### 6.1 Strategy Base Class

```python
class SearchStrategy(ABC):
    """Base class for search strategies"""
    
    @abstractmethod
    def search(self,
               objective_function: Callable[[WindowConfig], TestResult],
               bounds: SearchBounds,
               max_iterations: int,
               scorer: ScoringFunction) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def name(self) -> str:
        pass
```

### 6.2 BayesianOptimization (Recommended)

```python
class BayesianOptimization(SearchStrategy):
    """Bayesian optimization using Optuna TPE"""
    
    def __init__(self, n_initial=5):
        self.n_initial = n_initial  # Startup trials before TPE kicks in
        
        if BAYESIAN_AVAILABLE:
            from window_optimizer_bayesian import OptunaBayesianSearch
            self.optuna_search = OptunaBayesianSearch(
                n_startup_trials=n_initial, 
                seed=None
            )
    
    def search(self, objective_function, bounds, max_iterations, scorer):
        if self.optuna_search:
            return self.optuna_search.search(
                objective_function, bounds, max_iterations, scorer
            )
        else:
            # Fallback to random search
            return RandomSearch().search(
                objective_function, bounds, max_iterations, scorer
            )
```

**How Optuna TPE Works:**

1. First `n_initial` trials: Random sampling
2. Subsequent trials: TPE (Tree-structured Parzen Estimator) suggests parameters
3. TPE models P(x|y) for good and bad trials, suggests x that maximizes EI

### 6.3 RandomSearch (Baseline)

```python
class RandomSearch(SearchStrategy):
    """Random search - samples configs uniformly"""
    
    def search(self, objective_function, bounds, max_iterations, scorer):
        results = []
        best_result = None
        best_score = float('-inf')
        
        for i in range(max_iterations):
            config = bounds.random_config()  # Uniform random
            result = objective_function(config)
            result.iteration = i
            score = scorer.score(result)
            results.append(result)
            
            if score > best_score:
                best_score = score
                best_result = result
                print(f"✨ NEW BEST [{i+1}/{max_iterations}]")
        
        return {
            'strategy': 'random_search',
            'best_config': best_result.config.to_dict(),
            'best_result': best_result.to_dict(),
            'best_score': best_score,
            'all_results': [r.to_dict() for r in results],
            'iterations': len(results)
        }
```

### 6.4 Other Strategies (Placeholders)

| Strategy | Status | Notes |
|----------|--------|-------|
| `GridSearch` | Placeholder | Not used in integrated mode |
| `EvolutionarySearch` | Placeholder | Not used in integrated mode |

---

## 7. WindowOptimizer Class

### 7.1 Constructor

```python
class WindowOptimizer:
    """Main optimizer that coordinates the search process"""
    
    def __init__(self, coordinator, dataset_path: str):
        self.coordinator = coordinator
        self.dataset_path = dataset_path
        self.test_cache = {}
        self.test_configuration_func = None  # Set by integration layer
```

### 7.2 test_configuration()

```python
def test_configuration(self, config: WindowConfig, 
                       seed_start: int = 0,
                       seed_count: int = 10_000_000) -> TestResult:
    """
    Test a configuration.
    
    This method is OVERRIDDEN by the integration layer to run real sieves.
    Thresholds come from config.forward_threshold and config.reverse_threshold.
    """
    if self.test_configuration_func:
        return self.test_configuration_func(config, seed_start, seed_count)
    
    # Fallback placeholder (never called in integrated mode)
    return TestResult(config=config, forward_count=0, 
                     reverse_count=0, bidirectional_count=0, iteration=0)
```

### 7.3 optimize()

```python
def optimize(self, strategy: SearchStrategy, bounds: SearchBounds,
             max_iterations: int = 50, scorer: ScoringFunction = None,
             seed_start: int = 0, seed_count: int = 10_000_000) -> Dict[str, Any]:
    """
    Run optimization using provided strategy.
    
    The strategy calls self.test_configuration() for each trial,
    which triggers real sieve execution via the integration layer.
    """
    if scorer is None:
        scorer = BidirectionalCountScorer()
    
    def objective(config: WindowConfig) -> TestResult:
        return self.test_configuration(config, seed_start, seed_count)
    
    return strategy.search(objective, bounds, max_iterations, scorer)
```

### 7.4 save_results()

```python
def save_results(self, results: Dict[str, Any], output_path: str):
    """Save optimization results to JSON"""
    output_dir = Path(output_path).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
```

---

## 8. Bayesian Optimization Flow

### 8.1 run_bayesian_optimization()

This is the main entry point for Bayesian optimization mode.

```python
def run_bayesian_optimization(
    lottery_file: str,
    trials: int,
    output_config: str,
    seed_count: int = 10_000_000,
    prng_type: str = 'java_lcg',
    test_both_modes: bool = False,    # NEW in V2.0
    strategy_name: str = 'bayesian'
) -> Dict[str, Any]:
```

### 8.2 Execution Flow

```
run_bayesian_optimization()
    │
    ├─→ Check COORDINATOR_AVAILABLE
    │
    ├─→ Import window_optimizer_integration_final
    │
    ├─→ Print configuration:
    │   ├─ Lottery file
    │   ├─ Trials count
    │   ├─ Seed count
    │   ├─ PRNG type
    │   └─ Mode (constant vs both)
    │
    ├─→ Initialize coordinator:
    │   coordinator = MultiGPUCoordinator(
    │       config_file="distributed_config.json",
    │       resume_policy="restart"
    │   )
    │
    ├─→ Add integration:
    │   add_window_optimizer_to_coordinator()
    │
    ├─→ Run optimization:
    │   results = coordinator.optimize_window(
    │       dataset_path=lottery_file,
    │       seed_start=0,
    │       seed_count=seed_count,
    │       prng_base=prng_type,
    │       test_both_modes=test_both_modes,
    │       strategy_name=strategy_name,
    │       max_iterations=trials,
    │       output_file='window_optimization_results.json'
    │   )
    │
    ├─→ Build optimal_config dict:
    │   {
    │       'window_size': ...,
    │       'offset': ...,
    │       'skip_min': ...,
    │       'skip_max': ...,
    │       'sessions': [...],
    │       'prng_type': ...,
    │       'test_both_modes': ...,
    │       'seed_count': ...,
    │       'optimization_score': ...,
    │       'forward_count': ...,
    │       'reverse_count': ...,
    │       'bidirectional_count': ...
    │   }
    │
    ├─→ Inject agent_metadata (for pipeline chaining)
    │
    ├─→ Save optimal_window_config.json
    │
    ├─→ Split lottery data:
    │   ├─ train_history.json (80%)
    │   └─ holdout_history.json (20%)
    │
    └─→ Return results
```

### 8.3 Test Both Modes (V2.0)

When `--test-both-modes` is enabled:

```python
if test_both_modes:
    print(f"Mode: TESTING BOTH CONSTANT AND VARIABLE SKIP")
    print(f"  - Will test {prng_type} (constant)")
    print(f"  - Will test {prng_type}_hybrid (variable)")
```

This runs sieves for both:
- **Constant skip:** `java_lcg` with fixed skip pattern
- **Variable skip:** `java_lcg_hybrid` with variable skip range

Survivors are tagged with `skip_mode` metadata for ML feature engineering.

---

## 9. Run With Config Mode

### 9.1 run_with_config()

For running sieves with an existing optimal configuration.

```python
def run_with_config(
    config_file: str,
    lottery_file: str,
    max_seeds: int,
    iterations: int,
    output_survivors: str = 'bidirectional_survivors.json',
    output_train: str = 'train_history.json',
    output_holdout: str = 'holdout_history.json'
) -> Dict[str, Any]:
```

### 9.2 Execution Flow

```
run_with_config()
    │
    ├─→ Load config from file
    │
    ├─→ Check test_both_modes from config
    │
    ├─→ Initialize coordinator
    │
    ├─→ Add integration layer
    │
    ├─→ Create WindowConfig object
    │
    ├─→ Initialize accumulator:
    │   accumulator = {
    │       'forward': [],
    │       'reverse': [],
    │       'bidirectional': []
    │   }
    │
    ├─→ For each iteration:
    │   result = run_bidirectional_test(
    │       coordinator=coordinator,
    │       config=window_config,
    │       dataset_path=lottery_file,
    │       seed_start=iteration * max_seeds,
    │       seed_count=max_seeds,
    │       prng_base=prng_type,
    │       test_both_modes=test_both_modes,
    │       forward_threshold=...,
    │       reverse_threshold=...,
    │       trial_number=iteration,
    │       accumulator=accumulator
    │   )
    │
    ├─→ Deduplicate survivors:
    │   def deduplicate(survivor_list):
    │       """Keep survivor with highest score per seed"""
    │       seed_map = {}
    │       for survivor in survivor_list:
    │           seed = survivor['seed']
    │           if seed not in seed_map or survivor['score'] > seed_map[seed]['score']:
    │               seed_map[seed] = survivor
    │       return list(seed_map.values())
    │
    ├─→ Save survivors:
    │   ├─ forward_survivors.json
    │   ├─ reverse_survivors.json
    │   └─ bidirectional_survivors.json
    │
    └─→ Split lottery data (80/20)
```

---

## 10. CLI Interface

### 10.1 Arguments

```python
parser = argparse.ArgumentParser(
    description='Window Optimizer - WITH VARIABLE SKIP SUPPORT (V2.0)'
)

# Mode selection
--strategy         # bayesian, random, grid, evolutionary
--config-file      # Run with existing config (skips optimization)

# Common parameters
--lottery-file     # Path to lottery data JSON (REQUIRED)

# Bayesian mode parameters
--trials           # Number of optimization trials (default: 50)
--output           # Output path for optimal config (default: optimal_window_config.json)

# Config mode parameters
--max-seeds        # Max seeds per iteration (default: 10,000,000)
--iterations       # Number of sieve iterations (default: 1)
--output-survivors # Output file for bidirectional survivors
--output-train     # Output file for training data
--output-holdout   # Output file for holdout data

# PRNG type
--prng-type        # PRNG from registry (default: java_lcg)

# Threshold parameters
--forward-threshold   # Override Optuna optimization (0.5-0.95)
--reverse-threshold   # Override Optuna optimization (0.6-0.98)

# NEW: Variable skip testing
--test-both-modes  # Test BOTH constant and variable skip patterns
```

### 10.2 Mode Decision Tree

```
main()
    │
    ├─→ args.strategy == 'bayesian':
    │   └─→ run_bayesian_optimization(strategy_name='bayesian')
    │
    ├─→ args.strategy == 'random':
    │   └─→ run_bayesian_optimization(strategy_name='random')
    │
    ├─→ args.strategy == 'grid':
    │   └─→ run_bayesian_optimization(strategy_name='grid')
    │
    ├─→ args.strategy == 'evolutionary':
    │   └─→ run_bayesian_optimization(strategy_name='evolutionary')
    │
    ├─→ args.config_file:
    │   └─→ run_with_config()
    │
    └─→ else:
        └─→ Print usage and exit(1)
```

---

## 11. Integration Layer

### 11.1 Key Import

```python
from window_optimizer_integration_final import add_window_optimizer_to_coordinator
```

### 11.2 What It Does

The integration layer (`window_optimizer_integration_final.py`) provides:

1. **`add_window_optimizer_to_coordinator()`** — Monkey-patches `optimize_window()` method onto coordinator
2. **`run_bidirectional_test()`** — Executes forward+reverse sieves and computes intersection
3. **Survivor accumulation** — Collects survivors across all trials with metadata

### 11.3 Integration Flow

```
window_optimizer.py                    window_optimizer_integration_final.py
       │                                              │
       │  add_window_optimizer_to_coordinator()      │
       │ ──────────────────────────────────────────→ │
       │                                              │
       │  coordinator.optimize_window(...)           │
       │ ──────────────────────────────────────────→ │
       │                                              │
       │                                              ├─→ Create WindowOptimizer
       │                                              │
       │                                              ├─→ Override test_configuration_func
       │                                              │
       │                                              ├─→ For each trial:
       │                                              │   └─→ run_bidirectional_test()
       │                                              │       ├─→ Forward sieve (coordinator)
       │                                              │       ├─→ Reverse sieve (coordinator)
       │                                              │       ├─→ Compute intersection
       │                                              │       └─→ Accumulate survivors
       │                                              │
       │  ←─────────────────────────────────────────  │
       │      Return results + accumulated survivors │
```

---

## 12. Output Files

### 12.1 Bayesian Mode Outputs

| File | Contents |
|------|----------|
| `optimal_window_config.json` | Best parameters + agent_metadata |
| `window_optimization_results.json` | Full trial history |
| `bidirectional_survivors.json` | Intersection survivors |
| `forward_survivors.json` | Forward sieve survivors |
| `reverse_survivors.json` | Reverse sieve survivors |
| `train_history.json` | 80% lottery data for training |
| `holdout_history.json` | 20% lottery data for validation |

### 12.2 optimal_window_config.json Structure

```json
{
    "window_size": 256,
    "offset": 50,
    "skip_min": 0,
    "skip_max": 30,
    "sessions": ["midday", "evening"],
    "prng_type": "java_lcg",
    "test_both_modes": false,
    "seed_count": 10000000,
    "optimization_score": 847.0,
    "forward_count": 12543,
    "reverse_count": 9876,
    "bidirectional_count": 847,
    "run_id": "step1_20251215_143052_12345",
    "agent_metadata": {
        "inputs": [{"file": "lottery.json", "required": true}],
        "outputs": ["optimal_window_config.json", "bidirectional_survivors.json", ...],
        "pipeline_step": 1,
        "follow_up_agent": "scorer_meta_agent",
        "confidence": 0.847,
        "suggested_params": {...},
        "reasoning": "Optimization found 847 survivors with score 847.0000"
    }
}
```

### 12.3 Survivor Record Structure

```json
{
    "seed": 12345678,
    "score": 0.85,
    "prng_type": "java_lcg",
    "skip_mode": "constant",
    "window_config": {
        "window_size": 256,
        "offset": 50,
        "skip_min": 0,
        "skip_max": 30
    },
    "trial_number": 23,
    "timestamp": "2025-12-15T14:30:52"
}
```

---

## 13. Agent Metadata Injection

### 13.1 Purpose

Agent metadata enables autonomous pipeline chaining by the Watcher Agent.

### 13.2 inject_agent_metadata()

```python
from integration.metadata_writer import inject_agent_metadata

optimal_config = inject_agent_metadata(
    optimal_config,
    inputs=[{"file": lottery_file, "required": True}],
    outputs=["optimal_window_config.json", "bidirectional_survivors.json",
             "train_history.json", "holdout_history.json"],
    pipeline_step=1,
    follow_up_agent="scorer_meta_agent",
    confidence=min(0.95, results['best_score'] * 10),
    suggested_params={
        "window_size": best_config['window_size'],
        "forward_threshold": 0.72,
        "reverse_threshold": 0.81,
        "k_folds": 5
    },
    reasoning=f"Optimization found {survivors_count} survivors with score {score:.4f}"
)
```

### 13.3 Metadata Fields

| Field | Type | Purpose |
|-------|------|---------|
| `inputs` | List[Dict] | Required input files |
| `outputs` | List[str] | Generated output files |
| `pipeline_step` | int | Step number (1 for window optimizer) |
| `follow_up_agent` | str | Next agent in pipeline |
| `confidence` | float | 0.0-1.0 confidence score |
| `suggested_params` | Dict | Parameters for next step |
| `reasoning` | str | Human-readable explanation |

---

## 14. Complete Method Reference

### 14.1 Module-Level Functions

| Function | Parameters | Returns | Purpose |
|----------|------------|---------|---------|
| `load_search_bounds_from_config()` | `config_path` | `dict` | Load bounds from JSON |
| `run_bayesian_optimization()` | `lottery_file, trials, ...` | `Dict` | Main Bayesian entry point |
| `run_with_config()` | `config_file, lottery_file, ...` | `Dict` | Run with existing config |
| `main()` | — | — | CLI entry point |

### 14.2 WindowConfig Methods

| Method | Returns | Purpose |
|--------|---------|---------|
| `__hash__()` | `int` | Hashable for sets/dicts |
| `description()` | `str` | Human-readable description |
| `to_dict()` | `Dict` | JSON serialization |

### 14.3 SearchBounds Methods

| Method | Returns | Purpose |
|--------|---------|---------|
| `from_config(path)` | `SearchBounds` | Load from config file |
| `random_config()` | `WindowConfig` | Generate random config |
| `is_valid(config)` | `bool` | Validate against bounds |

### 14.4 TestResult Properties

| Property | Type | Formula |
|----------|------|---------|
| `precision` | `float` | `bidirectional / forward` |
| `recall` | `float` | `bidirectional / reverse` |

### 14.5 WindowOptimizer Methods

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `test_configuration()` | `config, seed_start, seed_count` | `TestResult` | Test single config |
| `optimize()` | `strategy, bounds, max_iterations, ...` | `Dict` | Run optimization |
| `save_results()` | `results, output_path` | — | Save to JSON |

### 14.6 SearchStrategy Methods (Abstract)

| Method | Parameters | Returns |
|--------|------------|---------|
| `search()` | `objective_function, bounds, max_iterations, scorer` | `Dict[str, Any]` |
| `name()` | — | `str` |

---

## 15. Dependencies Summary

| Dependency | Required | Purpose |
|------------|----------|---------|
| `coordinator.py` | ✅ Yes | Sieve execution |
| `window_optimizer_bayesian.py` | ⚠️ Optional | Optuna TPE |
| `window_optimizer_integration_final.py` | ✅ Yes | Integration layer |
| `integration.metadata_writer` | ⚠️ Optional | Agent metadata |
| `distributed_config.json` | ⚠️ Optional | Search bounds |

---

## 16. Chapter Summary

**Chapter 1: Window Optimizer** covers Step 1 of the pipeline:

| Component | Lines | Purpose |
|-----------|-------|---------|
| Data structures | ~100 | WindowConfig, SearchBounds, TestResult |
| Scoring functions | ~30 | BidirectionalCountScorer |
| Search strategies | ~100 | Bayesian, Random, Grid, Evolutionary |
| WindowOptimizer class | ~50 | Main coordinator |
| run_bayesian_optimization() | ~100 | Bayesian entry point |
| run_with_config() | ~100 | Config mode entry point |
| CLI | ~100 | Argument parsing |

**Key Insight:** The window optimizer doesn't run sieves directly — it delegates to the integration layer which coordinates real 26-GPU sieve execution.

---

## Next Chapter

**Chapter 2: Sieve Filter (Step 2)** will cover:
- `sieve_filter.py` — GPU residue sieve implementation
- Forward/reverse sieve algorithms
- GPU memory management
- Residue set computations

---

*End of Chapter 1: Window Optimizer*
