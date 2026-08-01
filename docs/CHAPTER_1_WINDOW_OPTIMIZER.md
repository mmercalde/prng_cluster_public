# Chapter 1: Window Optimizer (Step 1)

## PRNG Analysis Pipeline — Complete Operating Guide

**Version:** 3.1  
**File:** `window_optimizer.py` + `window_optimizer_integration_final.py`  
**Lines:** ~868 + ~595  
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

### 1.2 Version History

```
VERSION 3.1 (S104, Feb 2026):
- RESTORED: 7 intersection fields accidentally omitted in v3.0 rewrite
  (intersection_count, intersection_ratio, forward_only_count,
   reverse_only_count, survivor_overlap_ratio, bidirectional_selectivity,
   intersection_weight)
- These fields represent ~32% of ML feature importance (Chapter 6/11)

VERSION 3.0 (S103, Feb 2026):
- Per-seed match rates: forward_matches/reverse_matches now store per-seed
  values from GPU sieve kernel, not trial-level aggregates
- extract_survivor_records() preserves individual match_rate per seed
- Legacy alias extract_survivors_from_result() retained for compatibility
- convert_survivors_to_binary.py v3.1: maps to per-seed match rates
- NPZ percentage-based variance health check added

VERSION 2.0:
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
        ├─ CERTIFIED NPZ GENERATION  ← canonical Steps 2-6 input (utils.run_finalizer)
        ├─ optimal_window_config.json
        ├─ bidirectional_survivors.json   (post-success SUMMARY — no seeds)
        ├─ forward_survivors.json         (count-only stub)
        ├─ reverse_survivors.json         (count-only stub)
        ├─ train_history.json
        └─ holdout_history.json
```

Two corrections to the diagram above, both material:

1. **`coordinator.py` is one of FOUR backends, not the path.** `run_bidirectional_test` opens
   with a cascade — RANGE-MINER first (`window_optimizer_integration_final.py:1167-1168`),
   then PWC, then ZMQ, then the legacy coordinator leg drawn here. Most production runs do
   not take the drawn path. `window_optimizer.py:1143-1154` enforces a mutex: at most one of
   `--use-persistent-workers` / `--use-zmq-sqlite` / `--use-range-miner`.
2. **The artifact that matters is the certified NPZ generation**, assembled by
   `utils.run_finalizer` (`:2490-2495`). See §12.1 — the three `*_survivors.json` files are
   summaries and stubs, not survivor data.

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

Live source: `window_optimizer.py:85-91`. The dataclass defaults shown above are the
*code* defaults; they are not the effective values a run uses. Effective thresholds are
resolved per trial by `resolve_directional_threshold()` (§7.2) and the sampled range comes
from `distributed_config.json` (§4.1). Do not read `0.40` / `0.45` as the operating point.

#### Why `skip_min` / `skip_max` exist — the physical model

> **This subsection is load-bearing.** In one session Team Alpha, Team Beta and Claude Code
> *independently* recommended removing `skip_min`/`skip_max` from variable-skip search,
> because no document any of them had read explained why skip exists. All three inferred
> intent from the current hybrid kernel signatures — which are the defect. The definitions
> above are kept **verbatim** for that reason.

The published draw sequence is **not** an uninterrupted PRNG output stream. Per the
*California State Lottery Daily & SuperLotto Plus Draw Procedures* (eff. 2021-06-09):

- **Two pre-test draws run before every live draw** on the selected equipment (§V: Pre-Test
  via `[Start Draw Session]`; *"Run Draw as Test"* is unchecked only afterwards). Pre-test
  outputs are generated, verified and certified — and **never published**.
- **Draw equipment is selected per session** by an RNG program, auditor-verified (§II).
  Midday and evening are separate sessions with separate equipment selection.
- The evening session draws **Daily 3, Daily 4, Fantasy 5 and Daily Derby together** — other
  games' outputs sit between the Daily 3 values that are observable.

**Therefore the observable sequence has real, structural gaps of unknown and varying size.**
Skip models those gaps. It is a **physical property of the data source**, not a tuning
convenience. A fixed stride asserts the gaps are constant, which the source document
contradicts.

| mode | assumption | kernels |
|---|---|---|
| constant skip | fixed stride *k* between observed outputs | 22 kernels, all declare `int skip_min, int skip_max` |
| variable skip (hybrid) | stride varies, e.g. `[5,5,3,7,5,5,8,4,5,5]` | 22 kernels, declare `skip_sequences` + `strategy_tolerances` |

#### DEFECT callout (Team Beta, verbatim)

```
DEFECT — current hybrid kernels do not execute the requested
skip_min/skip_max semantics and instead use a hard-coded stride.
Hybrid optimization results are non-certifying.
```

Anchors: hybrid kernels hardcode `int expected_skip = 5` (`prng_registry.py:805`, `:885`,
`:1027`, `:1159`) and neither hybrid signature declares `skip_min`/`skip_max`
(`prng_registry.py:1007-1013` forward, `:3172-3178` reverse). On the certifying miner route
the values survive argparse, config, coordinator, ledger, manifest, payload, worker unpack
and `BuildContext` (`miner/range_miner_worker.py:871`), then die one call before launch in
`_hybrid_prefix` (`:177-193`). Recorded as dead dimensions **D-1** / **D-2** (§3.1.1).

**Standing rule — the fix is WIRE-IN, not removal.** Absence of a working implementation is
not evidence of absent intent. The full transport chain exists and is intact; nobody builds
eight hops of plumbing for a value that was never meant to arrive. Before proposing to
remove, demote or simplify this parameter, cite this subsection and the draw-procedures
document. The purpose of this text is that no future reader re-derives "remove it."

**Methods:**

| Method | Returns | Purpose |
|--------|---------|---------|
| `__hash__()` | `int` | Make hashable for sets/dicts |
| `description()` | `str` | Human-readable, format string at `window_optimizer.py:101` |
| `to_dict()` | `Dict` | JSON serialization |

**Example** (values chosen inside the live bounds of §4.1 — the pre-correction example used
`window_size=512` and `0.25`, both of which are now **unreachable**: 512 exceeds the window
ceiling of 50 and 0.25 is below the threshold floor of 0.30):

```python
config = WindowConfig(
    window_size=12,
    offset=0,
    sessions=['midday'],
    skip_min=0,
    skip_max=16,
    forward_threshold=0.30,
    reverse_threshold=0.30
)
print(config.description())
# Output: W12_O0_midday_S0-16_FT0.3_RT0.3
```

#### 3.1.1 Dead dimensions D-1 … D-4

A **dead dimension** is a parameter the system samples or accepts but that never reaches the
code claiming to consume it. Each is a *defect to be wired in*, never a candidate for
removal.

| id | parameter | sampled / declared at | dies at | consequence |
|---|---|---|---|---|
| **D-1** | `skip_min`, `skip_max` — forward hybrid (`java_lcg_hybrid`) | `window_optimizer_bayesian.py:429-434`; carried on `WindowConfig` (`window_optimizer.py:88-89`) | `_hybrid_prefix` (`miner/range_miner_worker.py:177-193`) emits 13 args, neither of them. PWC route `sieve_gpu_worker.py:259-268` discards the generic prefix. Kernel hardcodes `expected_skip = 5` | Optuna tunes a knob wired to nothing. **Live on the certifying miner route.** OPEN |
| **D-2** | `skip_min`, `skip_max` — reverse hybrid (`java_lcg_hybrid_reverse`) | same | `_reverse_hybrid_tail` (`miner/range_miner_worker.py:200-202`) emits only `offset`; `sieve_gpu_worker.py:270-279` likewise | same class as D-1. OPEN |
| **D-3** | `offset` — forward hybrid, `java_lcg` only | `window_optimizer_bayesian.py:423-425` | `build_java_lcg` forward-hybrid branch returns `_hybrid_prefix + [a, c]`, in-source note *"ABI-critical, NO offset"*; PWC skips `sieve_gpu_worker.py:304` via the `continue` at `:293` | Family-specific — `build_lcg32`'s forward hybrid *does* pass `offset`. `java_lcg` is the TFM target family, so this is the consequential instance. OPEN |
| **D-4** | `--forward-threshold`, `--reverse-threshold` | declared `window_optimizer.py:1063-1066` | immediately — `args.forward_threshold` / `args.reverse_threshold` were never referenced after `parse_args()` | Operator-facing. Was a **silent no-op** on a run reporting success. **CLOSED as a silent defect: the flags now fail closed** (§10.1) |

Constant-skip is fully wired on all four `java_lcg` variants; the variable-skip path is where
the loss is.

### 3.2 SearchBounds

```python
@dataclass
class SearchBounds:
    """Search space boundaries for optimization"""
    
    # Window parameters
    min_window_size: int = 2
    max_window_size: int = 50     # S139: 500 -> 50
    min_offset: int = 0
    max_offset: int = 100
    
    # Skip parameters
    min_skip_min: int = 0
    max_skip_min: int = 10
    min_skip_max: int = 10
    max_skip_max: int = 250       # S139: 500 -> 250
    
    # Threshold bounds (LOW for discovery)
    min_forward_threshold: float = 0.40
    max_forward_threshold: float = 0.75
    min_reverse_threshold: float = 0.40
    max_reverse_threshold: float = 0.75
    
    # Defaults
    default_forward_threshold: float = 0.50
    default_reverse_threshold: float = 0.50
    
    # Session options
    session_options: List[List[str]] = None  # Auto-initialized
```

Live source: `window_optimizer.py:114-130`.

> **These are the CODE defaults, and they are NOT the effective search space.**
> `distributed_config.json` overrides them (§4.1), and it sets a *lower* threshold floor
> (`0.30`) and a *lower* threshold default (`0.30`) than the code does. Reading the numbers
> above as the operating bounds is the error this chapter previously made. **For effective
> values, see the extracted snapshot in §4.1.**

**Key Methods:**

| Method | Returns | Purpose |
|--------|---------|---------|
| `from_config(path)` | `SearchBounds` | Load from distributed_config.json (`window_optimizer.py:132`) |
| `random_config()` | `WindowConfig` | Generate random config within bounds (`:198`) |
| `is_valid(config)` | `bool` | Validate config against bounds (`:213`) |
| `validate_baseline_in_bounds()` | `None` | **Team Beta mandate** — raises `ValueError` when the baseline falls outside bounds (`:163-196`) |

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

### 4.1 Single Source of Truth — live authority, and a dated snapshot

**Precedence rule.** `distributed_config.json` → `search_bounds` **overrides** the code
defaults. The merge is a per-key `dict.update()` at `window_optimizer.py:57-61`, so config
wins key-by-key: a config block that supplies only `min` leaves `max` at the code default.

```python
def load_search_bounds_from_config(config_path: str = "distributed_config.json") -> dict:
    """Load search bounds from config file"""
    defaults = { ... }          # window_optimizer.py:46-53 — CODE defaults
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        bounds = config.get("search_bounds", {})
        for key in defaults:            # config values override defaults
            if key in bounds:
                defaults[key].update(bounds[key])
        return defaults
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"⚠️  Could not load search_bounds from {config_path}: {e}")
        print(f"   Using default bounds")     # window_optimizer.py:64-65
        return defaults
```

**Do not read numeric bounds out of this chapter.** Every numeric bound in the pre-correction
chapter was wrong — thresholds `[0.15, 0.60] default 0.25` against a live `[0.30, 0.75]
default 0.30`, the window ceiling 10× too large, the skip ceiling 2× too large. That is the
same class of error that produced the "~62 features" incident. Regenerate the snapshot below
instead of editing it by hand:

```bash
python3 scripts/extract_search_bounds_snapshot.py          # markdown block
python3 scripts/extract_search_bounds_snapshot.py --json   # machine-readable
```

**A date alone is insufficient provenance** — multiple code states share a date. The snapshot
therefore carries `repository_commit` (which tree) and `configuration_digest` (which
configuration bytes) as well.

<!-- BEGIN EXTRACTED BOUNDS SNAPSHOT — generated by scripts/extract_search_bounds_snapshot.py; do not hand-edit -->
```
Authority:
  distributed_config.json -> search_bounds (merged over code defaults by window_optimizer.load_search_bounds_from_config; config wins, window_optimizer.py:57-61)

Snapshot:
  generated_at         : 2026-08-01T01:33:18Z
  repository_commit    : 0c47fe34d0e276ea462bb6f2b5b972a9292f064d
  configuration_digest : sha256:6077bb1a6c7352bd21cbde736127394f464a082dbee6b098390e15dd1f2747cc
  status               : INFORMATIVE SNAPSHOT — NOT AUTHORITATIVE. Read the authority above for the binding values.

  extracted bounds:
    window_size        min=6, max=50, default=12
    offset             min=0, max=100
    skip_min           min=0, max=10
    skip_max           min=10, max=250
    forward_threshold  min=0.3, max=0.75, default=0.3
    reverse_threshold  min=0.3, max=0.75, default=0.3
```

**Provenance notes carried from `distributed_config.json`** (the only in-repo record of *why* these values are what they are):

- `window_size._calibration_note` — S148 Run-1 ruling: W12 is empirically preferred production baseline. Optuna still explores full [min,max] range. W12+T0.30 gives ~5 false fwd survivors/200k vs ~272 at W8/T0.25.
- `window_size._s172_note` — S172 (2026-04-30): min raised from 2 to 6 per TB ruling. W=2/3 produces ~39%/53% survivor rate by chance alone, regardless of threshold. Threshold bounds intentionally PRESERVED so Optuna can continue optimizing across [min, max].
<!-- END EXTRACTED BOUNDS SNAPSHOT -->

### 4.2 distributed_config.json Structure

The `search_bounds` block is the authority. Its **shape** is stable; its **values** are the
snapshot in §4.1 and must be read from there or from the file, never from a transcription.

```json
{
    "search_bounds": {
        "window_size":       {"min": …, "max": …, "default": …,
                              "_calibration_note": "…", "_s172_note": "…"},
        "offset":            {"min": …, "max": …},
        "skip_min":          {"min": …, "max": …},
        "skip_max":          {"min": …, "max": …},
        "forward_threshold": {"min": …, "max": …, "default": …},
        "reverse_threshold": {"min": …, "max": …, "default": …}
    }
}
```

The two `_note` keys are not decoration — they are the only in-repo record of *why* the
window floor is 6, and they are carried into the §4.1 snapshot for that reason. Do not strip
them when editing the config.

### 4.3 Threshold Philosophy

**CRITICAL INSIGHT:** target **1K–10K** bidirectional survivors
(`baselines/baseline_window_thresholds.json` → `expected_survivor_band`). For the sampled
range and baseline, see the §4.1 snapshot — not a number quoted here. See
`docs/THRESHOLD_GOVERNANCE.md`.

```
The system is a behavioral fingerprint machine, NOT a filter.
Low thresholds maximize seed discovery.
Bidirectional intersection handles the actual filtering.
```

This rationale is **correct and load-bearing**, and matches whitepaper §7: an exact sieve
eliminates all variance, leaving `{s*}` with no ranking, no gradients and **no learning
signal**. Loose thresholds deliberately admit a *manifold* of near-consistent seeds sharing
structured deviations that ML can learn to rank. Loose thresholds are a mathematical
necessity, not sloppiness. **A rewrite must keep this paragraph.**

**Correction — one claim here was falsified by measurement.** The pre-correction chapter
added *"High thresholds (0.72+) would eliminate candidates prematurely."* That was an
a-priori assumption. The live ceiling is `0.75`, so `0.72` is inside the sampled range, and
`baselines/baseline_window_thresholds.json` records the S148 empirical calibration
(2026-03-19): *"Known seed survives to threshold=0.75 — ceiling safe."* The claim is removed;
the surrounding rationale is not.

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
                       seed_count: int = 10_000_000,
                       optuna_trial=None) -> TestResult:      # S119
    """
    Test a configuration.
    
    This method is OVERRIDDEN by the integration layer to run real sieves.
    Thresholds come from config.forward_threshold and config.reverse_threshold.
    """
    if self.test_configuration_func:
        return self.test_configuration_func(config, seed_start, seed_count,
                                            optuna_trial=optuna_trial)
    
    # Fallback placeholder (never called in integrated mode)
    return TestResult(config=config, forward_count=0, 
                     reverse_count=0, bidirectional_count=0, iteration=0)
```

Live source: `window_optimizer.py:444-463`. Override: `window_optimizer_integration_final.py:2389`
(`optimizer.test_configuration = test_config`).

#### 7.2.1 INVARIANT — `resolve_directional_threshold()` is the single threshold authority

The sentence *"thresholds come from `config.forward_threshold` and `config.reverse_threshold`"*
is **true only because of a specific mechanism, and it has been false before.** A document
that states the outcome without the invariant cannot protect it — that is the same failure
mode as the skip-bound incident (§3.1).

**The authority** is `resolve_directional_threshold()`,
`window_optimizer_integration_final.py:210-236`. It is the *only* place a directional
threshold is resolved. Do not add a second resolution path.

| rule | why |
|---|---|
| Precedence **explicit > config > default** | one resolution, in the parent, never reinterpreted downstream |
| **`is None` is the SOLE fallback trigger** | **`0.0` is a legitimate threshold.** A truthiness test (`getattr(...) or default`) silently replaces it — the shape `s172_threshold_patch.py` FIX 2 used, and deliberately not reused here |
| **Fail closed:** raises `ThresholdResolutionError` when nothing resolves | it refuses to invent a value. Never substitute a constant, never clamp into range |
| Record **requested / payload / effective** separately | the effective value is read back off the real executor (`miner/range_miner_worker.py:784`, `:858-863`, `:913`), so provenance is observed, not asserted |

**Regression history — read this before touching the file.** Full trace:
`docs/THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md` (cited, not re-derived here).

| commit | date | event |
|---|---|---|
| `3fdf434` | 2026-04-30 | Optuna threshold-drop bug **fixed** |
| `2389b61` | 2026-07-07 | **silently reverted** — a Phase-0 PRNG-encoding commit rewrote `window_optimizer_integration_final.py` from a pre-fix copy. The commit message never mentions thresholds. A stale-copy overwrite |
| `8a55a68` | 2026-07-31 | **repaired**, both routes, via the single resolver above |

Between `2389b61` and `8a55a68` **every trial ran at the configured default `0.30/0.30`**
while the study recorded the sampled suggestion. Treat every threshold value recorded in
that window — study DBs, `step1_trial_history`, `optimal_window_config.json` — as
**non-executed**.

> **Gate design consequence.** `2389b61` reverted the fix by replacing the whole block, so a
> text-anchor check would have gone green. Any regression gate on this invariant must
> **execute the live call site**, not match text against it. That is what
> `tests/test_s172_threshold_propagation.py` and `tests/test_s172_phase5_d6_threshold_path.py`
> do.

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

# Threshold override flags — DECLARED BUT UNWIRED. These FAIL CLOSED.
--forward-threshold   # passing this aborts the run (see below)
--reverse-threshold   # passing this aborts the run (see below)

# NEW: Variable skip testing
--test-both-modes  # Test BOTH constant and variable skip patterns
```

This list is **partial** — `window_optimizer.py:1031-1139` declares 31 flags. Bringing the
full set into the chapter is a P1 item and is *not* done in this tranche; do not read the
block above as complete.

#### `--forward-threshold` / `--reverse-threshold` — dead dimension D-4, now fail-closed

The pre-correction chapter documented these as *"Override Optuna optimization (0.15-0.60)."*
**That override never existed.** They were declared at `window_optimizer.py:1063-1066` and
`args.forward_threshold` / `args.reverse_threshold` were never referenced after
`parse_args()`. An operator passing `--forward-threshold 0.6` got a **silent no-op on a run
that reported success** — the first operator-facing dead dimension (D-4, §3.1.1). Three
mutually inconsistent bound figures were in play at once: this chapter said `0.15-0.60`, the
`--help` text said `0.5-0.95` / `0.6-0.98`, and the effective bounds were `0.30-0.75`.

**Current behaviour:**

| invocation | result |
|---|---|
| flag absent | existing supported path, unchanged |
| flag present (any value, including `0.0`) | **explicit nonzero failure before coordinator construction**, diagnostic `WINDOW_OPTIMIZER_THRESHOLD_OVERRIDE_UNWIRED` |

The flags are **kept declared rather than deleted from argparse**, deliberately: the operator
intent they record is legitimate, and a named diagnostic tells the operator the capability is
*unwired* rather than misspelled. See `window_optimizer.py` for the recorded condition under
which they may return — they must feed the single `resolve_directional_threshold()` authority
(§7.2.1), preserve `0.0` via `is None`, and record requested/payload/effective. **They must
not create parallel threshold state.**

#### `--strategy random | grid | evolutionary` — fail closed (signature mismatch)

Only `--strategy bayesian` is functional. `WindowOptimizer.optimize` calls
`strategy.search(..., resume_study=, study_name=, trse_context_file=,
trial_history_context=)` (`window_optimizer.py:484-487`); only
`BayesianOptimization.search` accepts those kwargs (`:388-391`). The other three raise
`TypeError` on first call — verified by live `inspect.signature`:

```
RandomSearch          (self, objective_function, bounds, max_iterations, scorer)
GridSearch            (self, objective_function, bounds, max_iterations, scorer)
EvolutionarySearch    (self, objective_function, bounds, max_iterations, scorer)
BayesianOptimization  (self, objective_function, bounds, max_iterations, scorer,
                       resume_study=False, study_name='', trse_context_file=...,
                       trial_history_context=None)
```

**Root cause is code rot, not design.** The kwargs were added incrementally to
`BayesianOptimization` (S116/S121/S140b) and the sibling classes were never updated. The
`SearchStrategy` ABC (`window_optimizer.py:295-310`) still declares the *old* four-argument
convention, which is exactly why no signature check caught it. Per §0.4 these are **not**
candidates for deletion — the remedy is to bring them up to the calling convention.

Requesting one of the three now aborts with `WINDOW_OPTIMIZER_STRATEGY_UNSUPPORTED` naming
the missing kwargs, rather than letting `TypeError` escape mid-run.

**Related, and equally fail-closed:** a Bayesian request when Optuna is unavailable **fails**.
It does not fall back to random search. Team Beta: that is *semantic substitution, not
graceful degradation* — the operator asked for TPE and would have received uniform sampling
under the same label, with the study recording it as Bayesian.

> **Surface not corrected in this tranche:** `agent_manifests/window_optimizer.json`
> (`search_strategy.choices`) still advertises all four strategies to WATCHER. Its `default`
> is `bayesian`, and a request for any of the other three now fails closed at the CLI rather
> than crashing, so the manifest is misleading but no longer dangerous. Flagged for the
> manifest owner.

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

> **The canonical Step-1 → Steps-2–6 carrier is the certified NPZ generation**, produced by
> `utils.run_finalizer` (`window_optimizer_integration_final.py:2490-2495`). It is the one
> output that matters, and the pre-correction table had no row for it. The three
> `*_survivors.json` files are **not** the survivor data they appear to be.

| File | Contents | Status |
|------|----------|--------|
| **certified NPZ generation** (`utils.run_finalizer`) | the **22-array NPZ contract** plus sidecar; carries `artifact_sha256`, `sidecar_sha256`, `parent_generation_id` (`window_optimizer_integration_final.py:2596-2602`). Generations **chain** — the finalizer merges prior rows | **CANONICAL — this is what Steps 2–6 consume** |
| `optimal_window_config.json` | best parameters + `agent_metadata` (§12.2) | current |
| `window_optimization_results.json` | full trial history (`window_optimizer_integration_final.py:2450`) | current |
| `bidirectional_survivors.json` | **post-success SUMMARY of the certified generation** — generation IDs and sha256s, **no seeds** (`:2604-2631`). In-source: *"It is NO LONGER the canonical Steps 2-6 input… Steps 2-6 consume the canonical NPZ"* | **demoted — summary only** |
| `forward_survivors.json` | `{"survivor_count": N, "note": "Full survivors omitted — objects not retained"}` (`:2523-2532`) | **count-only stub** |
| `reverse_survivors.json` | as above | **count-only stub** |
| `train_history.json` | 80% lottery data for training (`window_optimizer.py:811-819`, `:1003-1011`) | current |
| `holdout_history.json` | 20% lottery data for validation | current |

**Why forward/reverse are count-only.** `accumulator['forward']` and `accumulator['reverse']`
are never appended to — only `accumulator['bidirectional']` is
(`window_optimizer_integration_final.py:1018-1019`, `:1538`, `:1640`). `[S166-ACCUM]`
(`:1529-1533`) replaced object retention with counters to stop a RAM bomb at 26-GPU scale.
**That change is deliberate and correct** — the canonical NPZ carries what downstream needs.
Do not "restore" full retention.

> **Known defect, separate ticket (not fixed in this tranche).** In `--config-file` mode
> `window_optimizer.py:960-976` still dedups those permanently-empty lists and writes `[]` to
> both files while printing `"✅ Saved 0 forward survivors"`. The Bayesian path degraded
> *honestly* (it writes a `note` explaining the omission); the config path degrades
> **silently**.

**Undocumented hard gate.** `--config-file` mode shells out to
`convert_survivors_to_binary.py` and raises `RuntimeError("Step 1 incomplete - NPZ conversion
required for Step 2")` on failure (`window_optimizer.py:978-988`). This is a release gate, not
a convenience step.

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

### 12.3 Survivor Record Structure (v3.1)

**The record is FLAT.** The pre-correction chapter nested the window parameters under a
`"window_config"` key and listed a `"timestamp"` field. Neither is real: `window_size`,
`offset`, `skip_min` and `skip_max` sit at top level in `metadata_base`
(`window_optimizer_integration_final.py:1505-1508`), and **no `timestamp` key is produced**
anywhere — not by `metadata_base` (`:1504-1527`) nor by the append (`:1538-1544`). Any
consumer written against the old shape would fail.

```json
{
    "seed": 12345678,
    "score": 0.85,
    "forward_match_rate": 0.75,
    "reverse_match_rate": 0.50,
    "prng_type": "java_lcg",
    "prng_base": "java_lcg",
    "skip_mode": "constant",

    "window_size": 4,
    "offset": 26,
    "skip_min": 1,
    "skip_max": 108,
    "skip_range": [1, 108],
    "sessions": ["midday"],

    "trial_number": 6,
    "forward_count": 8987,
    "reverse_count": 8855,
    "intersection_count": 8929,

    "bidirectional_count": 8929,
    "intersection_ratio": 0.488,
    "forward_only_count": 8987,
    "reverse_only_count": 8855,
    "survivor_overlap_ratio": 0.498,
    "bidirectional_selectivity": 1.007,
    "intersection_weight": 0.244
}
```

- `forward_match_rate` / `reverse_match_rate` are **per-seed** values read from the GPU sieve
  kernel via `forward_map` / `reverse_map` (`:1536-1541`), not trial-level aggregates.
- All 7 intersection fields are present (`:1520-1526`, constant; `:1624-1630`, variable).
- `intersection_count` duplicating `bidirectional_count` is **deliberate** — not a defect.
- `skip_range`, `sessions`, `prng_base`, `forward_count`, `reverse_count` and
  `intersection_count` were absent from the pre-correction example.

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

---

## Persistent Worker Call Chain (S130/S134/S135)

When --use-persistent-workers is set, window_optimizer_integration_final.py routes
through the run_trial_persistent() shim in persistent_worker_coordinator.py:669
instead of the standard coordinator path.

Call chain:
```
watcher_agent.py
  -> window_optimizer_integration_final.py  (use_persistent_workers=True)
    -> run_trial_persistent()  (persistent_worker_coordinator.py:669)
      -> PersistentWorkerCoordinator
            Zeus:    execute_local_sieve_job()  -> sieve_filter.py
            Remote:  _dispatch_to_worker()      -> sieve_gpu_worker.py --persistent
```

Invariant: persistent_worker_coordinator.py is STANDALONE.
Zero changes to coordinator.py, window_optimizer.py, or window_optimizer_integration_final.py.
The default subprocess path is completely untouched -- --use-persistent-workers is additive only.

### Optuna Resume

Active study: window_opt_1772507547.db (21 trials as of S132).
Flag: --resume-study --study-name window_opt_1772507547
Storage: JournalStorage (not SQLite). Trial-unique output paths prevent cross-trial collisions.

### enable_pruning / n_parallel Fix History (S116/S118/S123)

Both flags required fixes through the full call chain:
- CLI -> run_bayesian_optimization() signature (S116)
- -> optimize_window() signature -- enable_pruning was missing (S118)
- -> agent_manifests/window_optimizer.json args_map -- 4 keys missing (S123)

---

## Persistent Worker Mode — S146 Kernel Invariants

When `--use-persistent-workers` is active, Step 1 dispatches sieve jobs via
`persistent_worker_coordinator.py` → `sieve_gpu_worker.py`. The following invariants
were validated in S146 and must be preserved in any future modifications:

### Hybrid Kernel Arg Tails (CRITICAL)

```
Forward hybrid:  kernel_args = (..., threshold, a, c)
Reverse hybrid:  kernel_args = (..., threshold, offset)   # a,c hardcoded in kernel
```

These are **not interchangeable**. Passing `(threshold, a, c)` to a reverse hybrid kernel
causes an immediate crash.

### Threshold Invariant

Hybrid families use `phase2_threshold` for both kernel invocation and post-filter check.
Base threshold (`min_match_threshold`) is used only for constant-skip families.

### int32 Casts

All scalar kernel args must be explicitly cast: `cp.int32(n_seeds)`, `cp.int32(k)`,
`cp.int32(skip_min)`, `cp.int32(skip_max)`. ROCm/CuPy requires explicit types.

### Count Clamp (defensive)

```python
count = min(int(survivor_count_gpu[0].get()), n_seeds)
```

Applied to both hybrid and non-hybrid extraction paths to prevent buffer overrun on
corrupt kernel writes.

---

## Persistent Worker Mode — S146 Kernel Invariants

When `--use-persistent-workers` is active, Step 1 dispatches sieve jobs via
`persistent_worker_coordinator.py` → `sieve_gpu_worker.py`. The following invariants
were validated in S146 and must be preserved in any future modifications:

### Hybrid Kernel Arg Tails (CRITICAL)

```
Forward hybrid:  kernel_args = (..., threshold, a, c)
Reverse hybrid:  kernel_args = (..., threshold, offset)   # a,c hardcoded in kernel
```

These are **not interchangeable**. Passing `(threshold, a, c)` to a reverse hybrid kernel
causes an immediate crash.

### Threshold Invariant

Hybrid families use `phase2_threshold` for both kernel invocation and post-filter check.
Base threshold (`min_match_threshold`) is used only for constant-skip families.

### int32 Casts

All scalar kernel args must be explicitly cast: `cp.int32(n_seeds)`, `cp.int32(k)`,
`cp.int32(skip_min)`, `cp.int32(skip_max)`. ROCm/CuPy requires explicit types.

### Count Clamp (defensive)

```python
count = min(int(survivor_count_gpu[0].get()), n_seeds)
```

Applied to both hybrid and non-hybrid extraction paths to prevent buffer overrun on
corrupt kernel writes.
