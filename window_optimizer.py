#!/usr/bin/env python3
"""
Window Optimizer - WITH VARIABLE SKIP SUPPORT
==============================================
Version: 2.0
Date: 2025-11-15

NEW IN V2.0:
- Added --test-both-modes flag to test constant AND variable skip patterns
- Survivors now tagged with skip_mode metadata for ML feature engineering
- Backward compatible: defaults to constant skip only (original behavior)

Usage modes:
1. Bayesian Optimization - constant skip only (original):
   python3 window_optimizer.py --strategy bayesian --lottery-file lottery.json --trials 50

2. Bayesian Optimization - BOTH constant AND variable skip (NEW!):
   python3 window_optimizer.py --strategy bayesian --lottery-file lottery.json --trials 50 --test-both-modes

3. Run with existing optimal config:
   python3 window_optimizer.py --config-file optimal_window_config.json --lottery-file lottery.json

The key feature: This runs REAL sieves on all 26 GPUs!
"""

import json
import os
from datetime import datetime
import inspect
import sys
import argparse
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from integration.metadata_writer import inject_agent_metadata


class _MappingAttrView:
    """
    Attribute view over a plain dict.

    [S178 P0-3] ADAPTER ONLY. The single threshold authority,
    `window_optimizer_integration_final.resolve_directional_threshold()`, reads
    its config by attribute. Some call sites here hold a JSON dict instead of a
    `WindowConfig`. This wraps the dict so that ONE resolver can serve both —
    it performs no resolution, holds no defaults, and invents nothing. Missing
    keys raise `AttributeError`, so `getattr(view, name, None)` yields `None`
    and the resolver's `is None` fallback rule applies unchanged (0.0 survives).
    """

    __slots__ = ('_mapping',)

    def __init__(self, mapping: Dict[str, Any]):
        object.__setattr__(self, '_mapping', mapping)

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, '_mapping')[name]
        except KeyError:
            raise AttributeError(name)

# ============================================================================
# CONFIG LOADER - Single Source of Truth for Search Bounds
# ============================================================================
def load_search_bounds_from_config(config_path: str = "distributed_config.json") -> dict:
    """
    Load search bounds from distributed_config.json.
    Returns dict with all bounds, using safe defaults if config missing.
    """
    defaults = {
        "window_size": {"min": 2, "max": 50},   # S139: 500→50, short-term temporal confirmed
        "offset": {"min": 0, "max": 100},
        "skip_min": {"min": 0, "max": 10},
        "skip_max": {"min": 10, "max": 250},     # S139: 500→250, matches distributed_config.json
        "forward_threshold": {"min": 0.40, "max": 0.75, "default": 0.50},
        "reverse_threshold": {"min": 0.40, "max": 0.75, "default": 0.50}
    }
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        bounds = config.get("search_bounds", {})
        # Merge with defaults (config values override defaults)
        for key in defaults:
            if key in bounds:
                defaults[key].update(bounds[key])
        return defaults
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"⚠️  Could not load search_bounds from {config_path}: {e}")
        print(f"   Using default bounds")
        return defaults


# ============================================================================
# DATA STRUCTURES (Required by window_optimizer_integration_final.py)
# ============================================================================

@dataclass
class WindowConfig:
    """
    Complete window and skip configuration for sieve execution.
    
    Attributes:
        window_size: Size of the temporal window (number of draws)
        offset: Time offset from current draw
        sessions: Which lottery sessions to include ('midday', 'evening', or both)
        skip_min: Minimum skip value for variable skip PRNGs
        skip_max: Maximum skip value for variable skip PRNGs
    """
    window_size: int
    offset: int
    sessions: List[str]
    skip_min: int
    skip_max: int
    forward_threshold: float = 0.40
    reverse_threshold: float = 0.45

    def __hash__(self):
        """Make config hashable for use in sets/dicts"""
        return hash((self.window_size, self.offset, tuple(self.sessions),
                    self.skip_min, self.skip_max, self.forward_threshold, self.reverse_threshold))

    def description(self) -> str:
        """Human-readable description of config"""
        sess = '+'.join(self.sessions)
        return f"W{self.window_size}_O{self.offset}_{sess}_S{self.skip_min}-{self.skip_max}_FT{self.forward_threshold}_RT{self.reverse_threshold}"

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

@dataclass
class SearchBounds:
    """
    Search space boundaries for optimization.
    Values loaded from distributed_config.json via from_config() classmethod.
    """
    # Defaults (overridden by from_config)
    min_window_size: int = 2
    max_window_size: int = 50    # S139: 500→50, short-term temporal confirmed
    min_offset: int = 0
    max_offset: int = 100
    min_skip_min: int = 0
    max_skip_min: int = 10
    min_skip_max: int = 10
    max_skip_max: int = 250
    # Threshold bounds - LOW for discovery, not filtering
    min_forward_threshold: float = 0.40
    max_forward_threshold: float = 0.75
    min_reverse_threshold: float = 0.40
    max_reverse_threshold: float = 0.75
    # Defaults
    default_forward_threshold: float = 0.50
    default_reverse_threshold: float = 0.50
    session_options: List[List[str]] = None
    
    @classmethod
    def from_config(cls, config_path: str = "distributed_config.json") -> 'SearchBounds':
        """Create SearchBounds from config file."""
        cfg = load_search_bounds_from_config(config_path)
        return cls(
            min_window_size=cfg["window_size"]["min"],
            max_window_size=cfg["window_size"]["max"],
            min_offset=cfg["offset"]["min"],
            max_offset=cfg["offset"]["max"],
            min_skip_min=cfg["skip_min"]["min"],
            max_skip_min=cfg["skip_min"]["max"],
            min_skip_max=cfg["skip_max"]["min"],
            max_skip_max=cfg["skip_max"]["max"],
            min_forward_threshold=cfg["forward_threshold"]["min"],
            max_forward_threshold=cfg["forward_threshold"]["max"],
            min_reverse_threshold=cfg["reverse_threshold"]["min"],
            max_reverse_threshold=cfg["reverse_threshold"]["max"],
            default_forward_threshold=cfg["forward_threshold"].get("default", 0.50),
            default_reverse_threshold=cfg["reverse_threshold"].get("default", 0.50)
        )

    def __post_init__(self):
        """Initialize session options if not provided"""
        if self.session_options is None:
            self.session_options = [
                ['midday', 'evening'],  # Both sessions
                ['midday'],              # Midday only
                ['evening']              # Evening only
            ]


    def validate_baseline_in_bounds(self, baseline_path: str = "baselines/baseline_window_thresholds.json") -> bool:
        """
        Validate that baseline thresholds are within search bounds.
        Team Beta mandate: baseline must always be reachable.
        """
        import os
        if not os.path.exists(baseline_path):
            print(f"⚠️ Baseline file not found: {baseline_path}")
            return True  # No baseline = no constraint
        
        import json
        with open(baseline_path) as f:
            baseline = json.load(f)
        
        fwd = baseline.get('forward_threshold', 0.50)
        rev = baseline.get('reverse_threshold', 0.50)
        skip = baseline.get('skip_max', 200)
        
        errors = []
        if not (self.min_forward_threshold <= fwd <= self.max_forward_threshold):
            errors.append(f"forward_threshold {fwd} not in [{self.min_forward_threshold}, {self.max_forward_threshold}]")
        if not (self.min_reverse_threshold <= rev <= self.max_reverse_threshold):
            errors.append(f"reverse_threshold {rev} not in [{self.min_reverse_threshold}, {self.max_reverse_threshold}]")
        if skip > self.max_skip_max:
            errors.append(f"skip_max {skip} exceeds max {self.max_skip_max}")
        
        if errors:
            print("❌ BASELINE VALIDATION FAILED:")
            for e in errors:
                print(f"   {e}")
            raise ValueError("Baseline thresholds outside search bounds - fix config before proceeding")
        
        print("✅ Baseline validation passed - baseline is reachable within search bounds")
        return True

    def random_config(self) -> WindowConfig:
        """Generate random config within bounds (for random search)"""
        skip_min = random.randint(self.min_skip_min, self.max_skip_min)
        skip_max = random.randint(skip_min, self.max_skip_max)

        return WindowConfig(
            window_size=random.randint(self.min_window_size, self.max_window_size),
            offset=random.randint(self.min_offset, self.max_offset),
            sessions=random.choice(self.session_options),
            skip_min=skip_min,
            skip_max=skip_max,
            forward_threshold=round(random.uniform(self.min_forward_threshold, self.max_forward_threshold), 2),
            reverse_threshold=round(random.uniform(self.min_reverse_threshold, self.max_reverse_threshold), 2)
        )

    def is_valid(self, config: WindowConfig) -> bool:
        """Check if config is within bounds"""
        return (self.min_window_size <= config.window_size <= self.max_window_size and
                self.min_offset <= config.offset <= self.max_offset and
                self.min_skip_min <= config.skip_min <= self.max_skip_min and
                config.skip_min <= config.skip_max <= self.max_skip_max and
                self.min_forward_threshold <= config.forward_threshold <= self.max_forward_threshold and
                self.min_reverse_threshold <= config.reverse_threshold <= self.max_reverse_threshold and
                config.sessions in self.session_options)

@dataclass
class TestResult:
    """
    Result from testing a window configuration.
    
    Contains counts of survivors from forward, reverse, and bidirectional sieves.
    Note: When test_both_modes=True, these counts are for constant skip only.
    Variable skip counts are tracked separately in the accumulator.
    """
    config: WindowConfig
    forward_count: int
    reverse_count: int
    bidirectional_count: int
    iteration: int

    @property
    def precision(self) -> float:
        """Precision: bidirectional / forward"""
        return self.bidirectional_count / self.forward_count if self.forward_count > 0 else 0

    @property
    def recall(self) -> float:
        """Recall: bidirectional / reverse"""
        return self.bidirectional_count / self.reverse_count if self.reverse_count > 0 else 0

    def to_dict(self) -> Dict:
        """Convert to serializable dict"""
        return {
            'config': self.config.to_dict(),
            'forward_count': self.forward_count,
            'reverse_count': self.reverse_count,
            'bidirectional_count': self.bidirectional_count,
            'precision': self.precision,
            'recall': self.recall,
            'iteration': self.iteration
        }

# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

class ScoringFunction(ABC):
    """Base class for scoring functions that evaluate window configurations"""

    @abstractmethod
    def score(self, result: TestResult) -> float:
        """Score a test result. Higher is better."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Return name of scoring function"""
        pass

class BidirectionalCountScorer(ScoringFunction):
    """
    Score based on count of bidirectional survivors.
    
    This is the simplest scoring function - more bidirectional survivors = better.
    The rationale: seeds that survive both forward and reverse sieves are
    more likely to be temporally stable and produce good predictions.
    """
    def score(self, result: TestResult) -> float:
        return float(result.bidirectional_count)

    def name(self) -> str:
        return "bidirectional_count"

# ============================================================================
# SEARCH STRATEGIES
# ============================================================================

class StrategyContractError(RuntimeError):
    """
    [S178 P0-2] A requested search strategy cannot honour the optimizer's call
    convention, or cannot be provided at all. Raised instead of substituting a
    different algorithm behind the same request.
    """


# The keyword arguments `WindowOptimizer.optimize` forwards to `strategy.search`.
# Kept beside the strategies so the gate below reads like the call site it guards;
# `tests/test_chapter1_p0_corrections.py` proves this tuple equals the kwargs of
# the LIVE `strategy.search(...)` call, extracted from the AST of
# `WindowOptimizer.optimize`. A text anchor would not have caught 2389b61.
OPTIMIZE_FORWARDED_KWARGS = (
    'resume_study', 'study_name', 'trse_context_file', 'trial_history_context',
)


class SearchStrategy(ABC):
    """
    Base class for search strategies.

    STALE (audit db9782a, conflict C-3): the abstract signature below is the
    pre-S116 convention. The real convention adds OPTIMIZE_FORWARDED_KWARGS —
    see `WindowOptimizer.optimize`. Because this ABC was never updated, no
    signature check caught that three of the four strategies could not be
    called. `strategy_contract_gap()` now checks the concrete classes instead.
    """

    @abstractmethod
    def search(self,
               objective_function: Callable[[WindowConfig], TestResult],
               bounds: SearchBounds,
               max_iterations: int,
               scorer: ScoringFunction) -> Dict[str, Any]:
        """Run the search strategy"""
        pass

    @abstractmethod
    def name(self) -> str:
        """Return name of strategy"""
        pass

class RandomSearch(SearchStrategy):
    """Random search baseline - samples configs uniformly at random"""
    def search(self, objective_function, bounds, max_iterations, scorer):
        print(f"\n{'='*80}")
        print(f"🎲 RANDOM SEARCH")
        print(f"Max iterations: {max_iterations}")
        print(f"{'='*80}\n")

        results = []
        best_result = None
        best_score = float('-inf')

        for i in range(max_iterations):
            config = bounds.random_config()
            result = objective_function(config)
            result.iteration = i
            score = scorer.score(result)
            results.append(result)

            if score > best_score:
                best_score = score
                best_result = result
                print(f"✨ NEW BEST [{i+1}/{max_iterations}]: {config.description()}")
                print(f"   Bidirectional: {result.bidirectional_count}, Score: {score:.2f}\n")
            else:
                print(f"   [{i+1}/{max_iterations}] {config.description()}: {result.bidirectional_count}")

        return {
            'strategy': self.name(),
            'best_config': best_result.config.to_dict() if best_result else {},
            'best_result': best_result.to_dict() if best_result else {},
            'best_score': best_score,
            'all_results': [r.to_dict() for r in results],
            'iterations': len(results)
        }

    def name(self) -> str:
        return "random_search"

class GridSearch(SearchStrategy):
    """Grid search - not used in integrated mode"""
    def __init__(self, window_sizes=None, offsets=None, skip_ranges=None):
        self.window_sizes = window_sizes or [512, 768, 1024]
        self.offsets = offsets or [0, 100]
        self.skip_ranges = skip_ranges or [(0, 20), (0, 50)]

    def search(self, objective_function, bounds, max_iterations, scorer):
        # Placeholder - not used in integrated mode
        return {}

    def name(self) -> str:
        return "grid_search"

class BayesianOptimization(SearchStrategy):
    """
    Bayesian optimization using Optuna TPE.
    
    This is the recommended strategy - it learns from previous trials
    to intelligently explore the search space.
    """
    def __init__(self, n_initial=5, enable_pruning=False, n_parallel=1):  # S115 R3
        self.n_initial = n_initial
        self.enable_pruning = enable_pruning
        self.n_parallel = n_parallel
        self.optuna_search = None

        # Try to use real Optuna implementation
        if BAYESIAN_AVAILABLE:
            try:
                from window_optimizer_bayesian import OptunaBayesianSearch
                self.optuna_search = OptunaBayesianSearch(
                    n_startup_trials=n_initial, seed=None,
                    enable_pruning=enable_pruning, n_parallel=n_parallel)  # S115 R3
            except Exception as e:
                print(f"⚠️  Could not initialize Optuna: {e}")

    def search(self, objective_function, bounds, max_iterations, scorer,
               resume_study: bool = False, study_name: str = '',
               trse_context_file: str = 'trse_context.json',
               trial_history_context: dict = None):  # [S140b]
        """Run Bayesian optimization"""
        if self.optuna_search:
            # [S152] Wire _survivor_accumulator through to OptunaBayesianSearch
            # BayesianOptimization.search() delegates immediately — accumulator must
            # be copied onto the inner search object or getattr(self,...) finds None.
            if hasattr(self, '_survivor_accumulator'):
                self.optuna_search._survivor_accumulator = self._survivor_accumulator
            # Use real Optuna implementation
            return self.optuna_search.search(objective_function, bounds, max_iterations, scorer,
                                             resume_study=resume_study, study_name=study_name,
                                             trse_context_file=trse_context_file,
                                             trial_history_context=trial_history_context)
        else:
            # [S178 P0-2] FAIL CLOSED — was: RandomSearch().search(...).
            #
            # Team Beta: silently serving uniform random sampling for a
            # requested TPE search is "semantic substitution, not graceful
            # degradation". The operator asked for Bayesian optimization; the
            # study, step1_trial_history and optimal_window_config.json would
            # all have recorded the run as Bayesian while a different algorithm
            # chose every point. Refuse instead.
            raise StrategyContractError(
                "WINDOW_OPTIMIZER_BAYESIAN_UNAVAILABLE: Bayesian optimization was "
                "requested but OptunaBayesianSearch could not be constructed "
                "(optuna missing, or window_optimizer_bayesian failed to import). "
                "Refusing to substitute random search for a requested TPE search — "
                "that would record a Bayesian run that never happened. "
                "Install optuna, or request a supported strategy explicitly. "
                "No sieve was launched and no parameter was changed."
            )

    def name(self) -> str:
        return "bayesian_optimization"

class EvolutionarySearch(SearchStrategy):
    """Evolutionary algorithm - not used in integrated mode"""
    def __init__(self, population_size=10, mutation_rate=0.2):
        self.population_size = population_size
        self.mutation_rate = mutation_rate

    def search(self, objective_function, bounds, max_iterations, scorer):
        # Placeholder - not used in integrated mode
        return {}

    def name(self) -> str:
        return "evolutionary"


# ============================================================================
# [S178 P0-2] SEARCH-STRATEGY CALLING CONTRACT
# ============================================================================
# Three of the four documented strategies raised TypeError on their FIRST call
# — after the 26-GPU coordinator had already been constructed. `optimize()`
# forwards OPTIMIZE_FORWARDED_KWARGS; only BayesianOptimization.search accepts
# them. The siblings still declare the pre-S116 four-positional convention, and
# the stale SearchStrategy ABC is why no signature check caught it
# (audit db9782a, conflict C-3).
#
# Per the standing rule (tfm-project-facts §0.4) the three are NOT deleted.
# All four strategies were clearly meant to run; §6.4 of the chapter
# distinguishes "placeholder" from "working", a distinction that would be
# pointless if none ran. This is code rot, and the remedy is to bring the
# signatures up to the calling convention — at which point the gate below
# clears itself, because it is derived from LIVE signatures rather than from a
# hardcoded list of broken names.

STRATEGY_CLASSES = {
    'random':       RandomSearch,
    'grid':         GridSearch,
    'bayesian':     BayesianOptimization,
    'evolutionary': EvolutionarySearch,
}


def strategy_contract_gap(strategy_cls) -> tuple:
    """
    Which of OPTIMIZE_FORWARDED_KWARGS `strategy_cls.search` cannot accept.

    Returns a sorted tuple of missing keyword names; empty tuple means the
    strategy is callable by `WindowOptimizer.optimize`. A `**kwargs` catch-all
    counts as accepting everything.
    """
    params = inspect.signature(strategy_cls.search).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return ()
    return tuple(sorted(k for k in OPTIMIZE_FORWARDED_KWARGS if k not in params))


def require_supported_strategy(strategy_name: str) -> type:
    """
    Resolve a strategy name to its class, failing closed on both an unknown name
    and a known-but-uncallable strategy.

    An unknown name previously fell through to `RandomSearch()` at
    `window_optimizer_integration_final.py`'s strategy_map — which made the
    broken RandomSearch the default for every typo. Same anti-pattern as the
    Optuna fallback: a request silently becoming a different algorithm.
    """
    cls = STRATEGY_CLASSES.get(strategy_name)
    if cls is None:
        raise StrategyContractError(
            f"WINDOW_OPTIMIZER_STRATEGY_UNKNOWN: {strategy_name!r} is not a known "
            f"search strategy (known: {', '.join(sorted(STRATEGY_CLASSES))}). "
            f"Refusing to fall back to random search."
        )
    gap = strategy_contract_gap(cls)
    if gap:
        raise StrategyContractError(
            f"WINDOW_OPTIMIZER_STRATEGY_UNSUPPORTED: --strategy {strategy_name} "
            f"cannot be run. {cls.__name__}.search() does not accept "
            f"{', '.join(gap)}, which WindowOptimizer.optimize() forwards, so it "
            f"would raise TypeError on the first trial — after the coordinator was "
            f"built. Use --strategy bayesian. Per tfm-project-facts §0.4 this "
            f"strategy is GATED, NOT deleted: bring {cls.__name__}.search() up to "
            f"the calling convention and this gate clears itself."
        )
    return cls


# ============================================================================
# MAIN OPTIMIZER CLASS
# ============================================================================

class WindowOptimizer:
    """
    Main optimizer class that coordinates the search process.
    
    This class doesn't run sieves directly - it delegates to the
    integration layer (window_optimizer_integration_final.py) which
    runs real sieves via the coordinator.
    """

    def __init__(self, coordinator, dataset_path: str):
        self.coordinator = coordinator
        self.dataset_path = dataset_path
        self.test_cache = {}
        self.test_configuration_func = None

    def test_configuration(self, config: WindowConfig, seed_start: int = 0,
                          seed_count: int = 10_000_000,
                          optuna_trial=None) -> TestResult:  # S119
        """
        Test a configuration.
        This will be overridden by the integration layer to run real sieves.
        Thresholds are now taken from config.forward_threshold and config.reverse_threshold.
        """
        if self.test_configuration_func:
            return self.test_configuration_func(config, seed_start, seed_count,
                                                optuna_trial=optuna_trial)  # S119

        # Fallback placeholder (should never be called in integrated mode)
        return TestResult(
            config=config,
            forward_count=0,
            reverse_count=0,
            bidirectional_count=0,
            iteration=0
        )

    def optimize(self, strategy: SearchStrategy, bounds: SearchBounds,
                max_iterations: int = 50, scorer: ScoringFunction = None,
                seed_start: int = 0, seed_count: int = 10_000_000,
                resume_study: bool = False, study_name: str = '',
                trse_context_file: str = 'trse_context.json',
                trial_history_context: dict = None) -> Dict[str, Any]:  # [S140b]
        """
        Run optimization using the provided strategy.
        
        The strategy will call self.test_configuration() for each trial,
        which in turn calls run_bidirectional_test() from the integration layer.
        """
        if scorer is None:
            scorer = BidirectionalCountScorer()

        def objective(config: WindowConfig, optuna_trial=None) -> TestResult:  # S118
            return self.test_configuration(config, seed_start, seed_count,
                                           optuna_trial=optuna_trial)  # S118

        return strategy.search(objective, bounds, max_iterations, scorer,
                              resume_study=resume_study, study_name=study_name,
                              trse_context_file=trse_context_file,
                              trial_history_context=trial_history_context)  # [S140b]

    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save optimization results to JSON file"""
        output_dir = Path(output_path).parent
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

# ============================================================================
# IMPORTS (After data classes to avoid circular import issues)
# ============================================================================

# Import coordinator for real sieve execution
try:
    from coordinator import MultiGPUCoordinator
    COORDINATOR_AVAILABLE = True
except ImportError:
    COORDINATOR_AVAILABLE = False
    print("⚠️  Warning: coordinator.py not found")

# Import Bayesian optimization
try:
    from window_optimizer_bayesian import OptunaBayesianSearch
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("⚠️  Warning: Optuna Bayesian optimization not available")

# Import the integration layer that runs real sieves
# IMPORTANT: This must come AFTER WindowConfig/TestResult definitions

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def run_bayesian_optimization(
    lottery_file: str,
    trials: int,
    output_config: str,
    seed_count: int = 10_000_000,
    seed_start: int = 0,                   # [S140] coverage tracker base
    prng_type: str = 'java_lcg',
    test_both_modes: bool = False,
    strategy_name: str = 'bayesian',  # 'bayesian' or 'random'
    resume_study: bool = False,
    study_name: str = '',
    enable_pruning: bool = False,     # S115 R3
    n_parallel: int = 1,              # S115 M1
    trse_context_file: str = 'trse_context.json',  # S121 Step 0 context
    use_persistent_workers: bool = False,   # S134
    use_zmq_sqlite: bool = False,            # S158D
    pwc_transport: str = 'tcp',            # S162 TCP-PWC default
    pwc_min_workers: int = 24,             # S162 readiness gate
    worker_pool_size: int = 8,             # S134
    seed_cap_nvidia: int = 5_000_000,      # S137
    seed_cap_amd: int = 2_000_000,         # S137
    warm_start_window: int = None,         # [S166] explicit warm-start
    warm_start_offset: int = None,
    warm_start_skip_min: int = None,
    warm_start_skip_max: int = None,
    warm_start_fwd_thresh: float = None,
    warm_start_rev_thresh: float = None,
    warm_start_session_idx: int = None,    # [S166] session index for Optuna enqueue
    use_range_miner: bool = False,          # [S172 Phase 1]
    miner_stripe_size: int = 67_108_864,    # [S172 Phase 1]
    miner_substripes: int = 8,              # [S172 Phase 1]
    miner_output_dir: str = None,           # [S172 Phase 1]
) -> Dict[str, Any]:
    """
    Run Bayesian optimization to find optimal window parameters
    AND generate survivors during the process.
    
    NEW IN V2.0: Supports test_both_modes parameter!
    
    This is the INTEGRATED approach - optimization runs real sieves across
    all 26 GPUs and accumulates survivors with rich metadata.
    
    Args:
        lottery_file: Path to lottery data JSON
        trials: Number of Bayesian optimization trials
        output_config: Where to save optimal_window_config.json
        seed_count: Number of seeds to test per trial
        prng_type: Base PRNG name (e.g., 'java_lcg')
        test_both_modes: If True, test BOTH constant and variable skip (NEW!)
        
    Returns:
        Dictionary with optimization results
    """

    if not COORDINATOR_AVAILABLE:
        print("❌ Error: coordinator.py not available")
        print("   Cannot run sieves without coordinator")
        sys.exit(1)

    # Lazy import to avoid circular dependency
    try:
        from window_optimizer_integration_final import (
            add_window_optimizer_to_coordinator, run_bidirectional_test,
            resolve_directional_threshold, ThresholdResolutionError,   # [S178 P0-3]
        )
        integration_available = True
    except ImportError as e:
        integration_available = False
        print(f"⚠️  Warning: window_optimizer_integration_final.py import failed: {e}")

    if not integration_available:
        print("❌ Error: window_optimizer_integration_final.py not available")
        print("   This provides the integration between optimizer and coordinator")
        sys.exit(1)

    print("\n" + "="*80)
    print("BAYESIAN WINDOW OPTIMIZATION WITH REAL SIEVES")
    print("="*80)
    print(f"Lottery file: {lottery_file}")
    print(f"Trials: {trials}")
    print(f"Seed count: {seed_count:,}")
    print(f"PRNG type: {prng_type}")
    if test_both_modes:
        print(f"Mode: TESTING BOTH CONSTANT AND VARIABLE SKIP")  # NEW!
        print(f"  - Will test {prng_type} (constant)")
        print(f"  - Will test {prng_type}_hybrid (variable)")
    else:
        print(f"Mode: CONSTANT SKIP ONLY (original behavior)")
    print(f"Output: {output_config}")
    print("="*80 + "\n")

    # Initialize coordinator
    print("🔧 Initializing 26-GPU coordinator...")
    coordinator = MultiGPUCoordinator(config_file="distributed_config.json", resume_policy="restart")

    # S134: wire persistent worker flags onto coordinator so integration gate can read them
    coordinator.use_persistent_workers = use_persistent_workers
    coordinator.use_zmq_sqlite = use_zmq_sqlite
    coordinator.pwc_transport   = pwc_transport    # S162
    coordinator.pwc_min_workers = pwc_min_workers  # S162
    if use_zmq_sqlite:
        print(f"   [S158D] ZMQ-SQLite coordinator ENABLED")
    coordinator.worker_pool_size        = worker_pool_size
    if use_persistent_workers:
        print(f"   [S134] Persistent worker mode ENABLED (pool_size={worker_pool_size} per rig)")
    # S137: wire seed cap flags onto coordinator so integration final can read them
    coordinator.seed_cap_nvidia = seed_cap_nvidia
    coordinator.seed_cap_amd    = seed_cap_amd

    # [S172 Phase 1] Wire miner flags onto coordinator so the integration-final gate
    # (window_optimizer_integration_final.py:_use_miner) can read them.
    coordinator.use_range_miner   = use_range_miner
    coordinator.miner_stripe_size = miner_stripe_size
    coordinator.miner_substripes  = miner_substripes
    coordinator.miner_output_dir  = miner_output_dir
    if use_range_miner:
        print(f"   [S172 Phase 1] RANGE-MINER backend ENABLED "
              f"(stripe={miner_stripe_size}, substripes={miner_substripes})")
        print(f"   [S172 Phase 1] Miner output dir: "
              f"{miner_output_dir or 'auto (/dev/shm/prng/miner/ if writable, else ~/miner_output/)'}")

    # Add window optimizer to coordinator (this adds the optimize_window method)
    add_window_optimizer_to_coordinator()

    # Run optimization (this will run real sieves and accumulate survivors)
    print("\n🚀 Starting Bayesian optimization with real sieve execution...\n")

    results = coordinator.optimize_window(
        dataset_path=lottery_file,
        seed_start=seed_start,             # [S140] from coverage tracker
        seed_count=seed_count,
        prng_base=prng_type,
        test_both_modes=test_both_modes,  # NEW: Pass through to integration layer
        strategy_name=strategy_name,
        max_iterations=trials,
        output_file='window_optimization_results.json',
        resume_study=resume_study,
        study_name=study_name,
        enable_pruning=enable_pruning,  # S115 wire-up
        n_parallel=n_parallel,          # S115 wire-up
        trse_context_file=trse_context_file,  # S121 Step 0 context
        warm_start_window=warm_start_window,      # [S166]
        warm_start_offset=warm_start_offset,
        warm_start_skip_min=warm_start_skip_min,
        warm_start_skip_max=warm_start_skip_max,
        warm_start_fwd_thresh=warm_start_fwd_thresh,
        warm_start_rev_thresh=warm_start_rev_thresh,
        warm_start_session_idx=warm_start_session_idx,  # [S166]
    )

    # [S140] SEED COVERAGE WRITE-BACK — log this run's range to exhaustive_progress
    # Runs once per Step 1 completion. Enables WATCHER to advance seed_start next run.
    try:
        from database_system import DistributedPRNGDatabase as _DBM
        _db = _DBM()
        _best_result = results.get('best_result', {})
        _survivors = _best_result.get('bidirectional_survivors', [])
        _best_seed = None
        if _survivors and isinstance(_survivors[0], dict):
            _best_seed = _survivors[0].get('seed', None)
        elif _survivors and isinstance(_survivors[0], int):
            _best_seed = _survivors[0]
        _db.update_exhaustive_progress(
            search_id=f'step1_{prng_type}_{int(seed_start)}',
            prng_type=prng_type,
            mapping_type='bidirectional',
            seed_range_start=seed_start,
            seed_range_end=seed_start + seed_count,
            seeds_completed=seed_count,
            best_score=results.get('best_score'),
            best_seed=_best_seed
        )
        print(f'   [COVERAGE] Logged range {seed_start:,} → {seed_start + seed_count:,} '
              f'for {prng_type} (best_seed={_best_seed})')
    except Exception as _e:
        print(f'   [COVERAGE] Write-back failed (non-fatal): {_e}')

    # Save optimal config for downstream use
    best_config = results['best_config']

    # Extract survivor counts from results (nested in best_result)
    best_result = results.get('best_result', {})
    forward_count = best_result.get('forward_count', 0)
    reverse_count = best_result.get('reverse_count', 0)
    bidirectional_count = best_result.get('bidirectional_count', 0)
    
    optimal_config = {
        'window_size': best_config['window_size'],
        'offset': best_config['offset'],
        'skip_min': best_config['skip_min'],
        'skip_max': best_config['skip_max'],
        'sessions': best_config['sessions'],
        'prng_type': prng_type,
        'test_both_modes': test_both_modes,  # NEW: Record whether we tested both modes
        'seed_count': seed_count,
        'optimization_score': results['best_score'],
        # Survivor counts for watcher evaluation
        'forward_count': forward_count,
        'reverse_count': reverse_count,
        'bidirectional_count': bidirectional_count
    }
    
    # === MERGE INCREMENTAL OUTPUT FIELDS (Patch 2026-01-18) ===
    # Preserve fields from incremental saves (crash recovery data)
    if Path(output_config).exists():
        try:
            with open(output_config, 'r') as f:
                existing = json.load(f)
            # Preserve incremental tracking fields
            incremental_fields = ['status', 'completed_trials', 'total_trials', 
                                  'best_trial_number', 'best_value', 'best_bidirectional_count',
                                  'last_updated', 'last_trial_number', 'last_trial_value']
            for field in incremental_fields:
                if field in existing:
                    optimal_config[field] = existing[field]
            # Mark as complete since we finished successfully
            optimal_config['status'] = 'complete'
            optimal_config['completed_at'] = datetime.now().isoformat()
        except (json.JSONDecodeError, IOError):
            pass  # If file is corrupt, just use new config
    # === END MERGE ===

    # === [S178 P0-3] D-4: METADATA REPORTS WHAT EXECUTED — IT DOES NOT INVENT ===
    #
    # This block used to emit `best_config.get('forward_threshold', 0.72)` and
    # `... .get('reverse_threshold', 0.81)`. Those literals were a SECOND
    # threshold authority, independent of the configuration actually requested
    # and executed — the sixth instance of the dual-authority pattern, in the
    # very file repaired at 8a55a68. 0.81 also exceeded the live governance
    # ceiling of 0.75, but Team Beta ruled the ceiling violation the SYMPTOM;
    # the defect is dual authority.
    #
    # Resolution goes through the single authority,
    # resolve_directional_threshold() — no second resolution path. `explicit`
    # and `default` are deliberately left unset so that the ONLY source is the
    # winning trial's own config: `is None` is the sole fallback trigger, so a
    # legitimate 0.0 survives, and when nothing resolves the resolver raises
    # rather than inventing. On a raise the field is OMITTED. It is never
    # replaced by a constant and never clamped into range.
    _best_config_view = _MappingAttrView(best_config)
    _executed_thresholds = {}
    for _direction in ('forward', 'reverse'):
        try:
            _executed_thresholds[_direction] = resolve_directional_threshold(
                _best_config_view, _direction)
        except ThresholdResolutionError as _err:
            print(f"⚠️  [D-4] no authoritative executed {_direction} threshold "
                  f"for the winning trial — OMITTING the field rather than "
                  f"substituting a value: {_err}")

    # Provenance (observed/executed) and proposal (recommendation to the next
    # step) are separate concerns and separate fields. They currently carry the
    # same numbers because no governed threshold RECOMMENDER exists — inventing
    # a recommendation is precisely the defect being repaired here. When one is
    # introduced it writes `suggested_params`; `executed_thresholds` must keep
    # reporting what the sieve actually ran.
    if _executed_thresholds:
        optimal_config['executed_thresholds'] = {
            **{f"{_d}_threshold": _v for _d, _v in _executed_thresholds.items()},
            'source': 'winning_trial_window_config',
            'resolver': 'window_optimizer_integration_final.resolve_directional_threshold',
        }

    _suggested_params = {
        "window_size": best_config['window_size'],
        "k_folds": 5,
    }
    for _direction, _value in _executed_thresholds.items():
        _suggested_params[f"{_direction}_threshold"] = _value

    # Inject agent_metadata for pipeline chaining
    run_id = f"step1_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(results)) % 100000:05d}"
    optimal_config = inject_agent_metadata(
        optimal_config,
        inputs=[{"file": lottery_file, "required": True}],
        outputs=["optimal_window_config.json", "bidirectional_survivors.json",
                 "train_history.json", "holdout_history.json"],
        pipeline_step=1,
        follow_up_agent="scorer_meta_agent",
        confidence=min(0.95, results['best_score'] * 10) if results['best_score'] > 0 else 0.5,
        suggested_params=_suggested_params,
        reasoning=f"Optimization found {results.get('survivors_count', 'N/A')} survivors with score {results['best_score']:.4f}"
    )
    optimal_config["run_id"] = run_id

    with open(output_config, 'w') as f:
        json.dump(optimal_config, f, indent=2)

    print(f"\n✅ Optimal configuration saved to: {output_config}")

    # [S121] Feed winning window back into TRSE context as confirmed_windows entry.
    # Builds a regime→window lookup table over multiple runs.
    # Passive: never raises, never blocks pipeline if trse_context.json is absent.
    try:
        _trse_path = trse_context_file if trse_context_file else 'trse_context.json'
        if os.path.exists(_trse_path) and bidirectional_count > 0:
            with open(_trse_path, 'r') as _f:
                _ctx = json.load(_f)
            _entry = {
                "window_size":            best_config['window_size'],
                "offset":                 best_config['offset'],
                "skip_min":               best_config['skip_min'],
                "skip_max":               best_config['skip_max'],
                "bidirectional_survivors": bidirectional_count,
                "optimization_score":     results['best_score'],
                "regime_at_time":         _ctx.get('current_regime'),
                "regime_type":            _ctx.get('regime_type', 'unknown'),
                "regime_stable":          _ctx.get('regime_stable', False),
                "timestamp":              datetime.now().isoformat()
            }
            _confirmed = _ctx.get('confirmed_windows', [])
            _confirmed.append(_entry)
            # Keep last 50 confirmed entries — avoids unbounded growth
            _ctx['confirmed_windows'] = _confirmed[-50:]
            with open(_trse_path, 'w') as _f:
                json.dump(_ctx, _f, indent=2)
            print(f"   [TRSE] Confirmed window W{best_config['window_size']}_O{best_config['offset']} "
                  f"({bidirectional_count} survivors) logged to {_trse_path}")
    except Exception as _e:
        logger.warning(f"[TRSE] confirmed_windows update failed (non-fatal): {_e}")

    # Create train/holdout split from lottery data
    print("\n📊 Splitting lottery data for train/holdout...")
    with open(lottery_file, "r") as f:
        lottery_data = json.load(f)
        if isinstance(lottery_data, list) and len(lottery_data) > 0:
            if isinstance(lottery_data[0], dict) and "draw" in lottery_data[0]:
                full_history = [d["draw"] for d in lottery_data]
            else:
                full_history = lottery_data
        else:
            full_history = lottery_data
    split_point = int(len(full_history) * 0.8)
    train_data = full_history[:split_point]
    holdout_data = full_history[split_point:]
    with open("train_history.json", "w") as f:
        json.dump(train_data, f, indent=2)
    with open("holdout_history.json", "w") as f:
        json.dump(holdout_data, f, indent=2)
    print(f"✅ Saved {len(train_data)} training draws to train_history.json")
    print(f"✅ Saved {len(holdout_data)} holdout draws to holdout_history.json")


    return results


def run_with_config(
    config_file: str,
    lottery_file: str,
    max_seeds: int,
    iterations: int,
    output_survivors: str = 'bidirectional_survivors.json',
    output_train: str = 'train_history.json',
    output_holdout: str = 'holdout_history.json',
    use_persistent_workers: bool = False,   # [S170-PARITY] use_persistent_workers
    pwc_transport: str = 'tcp',             # [S170-PARITY] use_persistent_workers
    seed_cap_amd: int = 2_000_000,          # [S170-PARITY-2] execution sizing
    seed_cap_nvidia: int = 5_000_000,       # [S170-PARITY-2] execution sizing
    worker_pool_size: int = 8,              # [S170-PARITY-2] execution sizing
    min_workers: int = 1,                   # [S170-PARITY-2] execution sizing
) -> Dict[str, Any]:
    """
    Run sieves with an existing optimal configuration.

    This mode is for when you already have optimal_window_config.json
    and just want to generate survivors with those parameters.
    """

    if not COORDINATOR_AVAILABLE:
        print("❌ Error: coordinator.py not available")
        sys.exit(1)

    # Lazy import to avoid circular dependency
    try:
        from window_optimizer_integration_final import (
            add_window_optimizer_to_coordinator, run_bidirectional_test,
            resolve_directional_threshold, ThresholdResolutionError,   # [S178 P0-3]
        )
        integration_available = True
    except ImportError as e:
        integration_available = False
        print(f"⚠️  Warning: window_optimizer_integration_final.py import failed: {e}")

    if not integration_available:
        print("❌ Error: window_optimizer_integration_final.py not available")
        sys.exit(1)

    print("\n" + "="*80)
    print("RUNNING SIEVES WITH OPTIMAL CONFIGURATION")
    print("="*80)

    # Load config
    with open(config_file, 'r') as f:
        config = json.load(f)

    # NEW: Check if config has test_both_modes set
    test_both_modes = config.get('test_both_modes', False)

    print(f"Config file: {config_file}")
    print(f"Lottery file: {lottery_file}")
    print(f"Max seeds: {max_seeds:,}")
    print(f"Iterations: {iterations}")
    if test_both_modes:
        print(f"Mode: BOTH CONSTANT AND VARIABLE SKIP")
    else:
        print(f"Mode: CONSTANT SKIP ONLY")
    print(f"\nConfiguration:")
    print(f"  Window size: {config.get('window_size', 1024)}")
    print(f"  Offset: {config.get('offset', 100)}")
    print(f"  Skip range: [{config.get('skip_min', 0)}, {config.get('skip_max', 50)}]")
    print(f"  Sessions: {config.get('sessions', ['midday', 'evening'])}")
    print(f"  PRNG: {config.get('prng_type', 'java_lcg')}")
    print("="*80 + "\n")

    # Initialize coordinator
    print("🔧 Initializing coordinator...")
    coordinator = MultiGPUCoordinator(config_file="distributed_config.json", resume_policy="restart")

    # Add integration
    add_window_optimizer_to_coordinator()

    # [S170-PARITY] propagate persistent worker / transport — match Bayesian path
    # (lines 614-616). Without these, --config-file mode silently downgrades to
    # legacy SSH distribution regardless of CLI flags.
    coordinator.use_persistent_workers = use_persistent_workers
    coordinator.pwc_transport          = pwc_transport

    # [S170-PARITY-2] propagate execution sizing — match Bayesian/PWC path
    # Without these, --config-file mode silently falls back to default chunk caps
    # such as seed_cap_amd=2_000_000 despite CLI --seed-cap-amd 100000.
    coordinator.seed_cap_amd           = seed_cap_amd
    coordinator.seed_cap_nvidia        = seed_cap_nvidia
    coordinator.worker_pool_size       = worker_pool_size
    coordinator.min_workers            = min_workers

    # === [S178 P0-3] D-4: ONE threshold authority for this run ===
    #
    # Was: the WindowConfig below received NO thresholds (so it silently carried
    # the dataclass defaults 0.40/0.45) while the sibling kwargs on
    # run_bidirectional_test said `config.get('forward_threshold', 0.72)` /
    # `... 0.81`. Two authorities for one quantity in a single call, and 0.81
    # exceeded the live governance ceiling of 0.75.
    #
    # Now resolved ONCE, in the parent, through the single authority, and the
    # same value is placed on the WindowConfig and passed to the backend.
    # Precedence explicit > config > default; `is None` is the sole fallback
    # trigger, so a config carrying 0.0 keeps 0.0. The `default` is the governed
    # configuration value from distributed_config.json — not a magic constant —
    # and if the config file supplies neither and search_bounds is unreadable,
    # resolve_directional_threshold raises and the run fails closed rather than
    # sieving at an invented threshold.
    _bounds = SearchBounds.from_config()
    _config_view = _MappingAttrView(config)
    _forward_threshold = resolve_directional_threshold(
        _config_view, 'forward', default=_bounds.default_forward_threshold)
    _reverse_threshold = resolve_directional_threshold(
        _config_view, 'reverse', default=_bounds.default_reverse_threshold)
    print(f"  Thresholds (resolved): forward={_forward_threshold}, "
          f"reverse={_reverse_threshold}")

    # Create WindowConfig object
    window_config = WindowConfig(
        window_size=config.get('window_size', 1024),
        offset=config.get('offset', 100),
        sessions=config.get('sessions', ['midday', 'evening']),
        skip_min=config.get('skip_min', 0),
        skip_max=config.get('skip_max', 50),
        forward_threshold=_forward_threshold,
        reverse_threshold=_reverse_threshold,
    )

    # Run the sieves with accumulator
    print("\n🚀 Running sieves...\n")

    accumulator = {
        'forward': [],
        'reverse': [],
        'bidirectional': []
    }

    for iteration in range(iterations):
        print(f"\n--- Iteration {iteration + 1}/{iterations} ---")

        result = run_bidirectional_test(
            coordinator=coordinator,
            config=window_config,
            dataset_path=lottery_file,
            seed_start=iteration * max_seeds,
            seed_count=max_seeds,
            prng_base=config.get('prng_type', 'java_lcg'),
            test_both_modes=test_both_modes,  # NEW: Pass through from config
            forward_threshold=_forward_threshold,   # [S178 P0-3] same resolution
            reverse_threshold=_reverse_threshold,   # as window_config above
            trial_number=iteration,
            accumulator=accumulator
        )

    # Deduplicate and save survivors
    print("\n" + "="*80)
    print("SAVING SURVIVORS")
    print("="*80)

    def deduplicate(survivor_list):
        """Keep survivor with highest score for each unique seed"""
        seed_map = {}
        for survivor in survivor_list:
            seed = survivor['seed']
            if seed not in seed_map or survivor['score'] > seed_map[seed]['score']:
                seed_map[seed] = survivor
        return list(seed_map.values())

    forward_deduped = deduplicate(accumulator['forward'])
    reverse_deduped = deduplicate(accumulator['reverse'])
    bidirectional_deduped = deduplicate(accumulator['bidirectional'])

    # Save survivors
    with open('forward_survivors.json', 'w') as f:
        json.dump(sorted(forward_deduped, key=lambda x: x['seed']), f, indent=2)

    with open('reverse_survivors.json', 'w') as f:
        json.dump(sorted(reverse_deduped, key=lambda x: x['seed']), f, indent=2)

    with open(output_survivors, 'w') as f:
        json.dump(sorted(bidirectional_deduped, key=lambda x: x['seed']), f, indent=2)

    print(f"✅ Saved {len(forward_deduped):,} forward survivors")
    print(f"✅ Saved {len(reverse_deduped):,} reverse survivors")
    print(f"✅ Saved {len(bidirectional_deduped):,} bidirectional survivors to {output_survivors}")

    # Convert to NPZ binary format (required by Step 2)
    from subprocess import run, CalledProcessError
    try:
        run(
            ["python3", "convert_survivors_to_binary.py", output_survivors],
            check=True
        )
        print(f"✅ Converted to {output_survivors.replace('.json', '_binary.npz')}")
    except CalledProcessError as e:
        print(f"❌ NPZ conversion failed: {e}")
        raise RuntimeError("Step 1 incomplete - NPZ conversion required for Step 2")


    # Split lottery data for train/holdout
    print("\n📊 Splitting lottery data...")
    with open(lottery_file, 'r') as f:
        lottery_data = json.load(f)
        if isinstance(lottery_data, list) and len(lottery_data) > 0:
            if isinstance(lottery_data[0], dict) and 'draw' in lottery_data[0]:
                full_history = [d['draw'] for d in lottery_data]
            else:
                full_history = lottery_data
        else:
            full_history = lottery_data

    split_point = int(len(full_history) * 0.8)
    train_data = full_history[:split_point]
    holdout_data = full_history[split_point:]

    with open(output_train, 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(output_holdout, 'w') as f:
        json.dump(holdout_data, f, indent=2)

    print(f"✅ Saved {len(train_data)} training draws to {output_train}")
    print(f"✅ Saved {len(holdout_data)} holdout draws to {output_holdout}")
    print("="*80 + "\n")

    return {
        'forward_count': len(forward_deduped),
        'reverse_count': len(reverse_deduped),
        'bidirectional_count': len(bidirectional_deduped),
        'iterations': iterations
    }


def main():
    parser = argparse.ArgumentParser(
        description='Window Optimizer - WITH VARIABLE SKIP SUPPORT (V2.0)'
    )

    # Mode selection
    # [S178 P0-2] The three non-bayesian names are RETAINED as choices so that
    # requesting one produces the specific WINDOW_OPTIMIZER_STRATEGY_UNSUPPORTED
    # diagnostic (naming the signature mismatch) rather than argparse's generic
    # "invalid choice". They are gated, not deleted — see require_supported_strategy.
    parser.add_argument('--strategy', type=str, choices=['bayesian', 'random', 'grid', 'evolutionary'],
                       help='Optimization strategy. Only "bayesian" is functional; '
                            'random/grid/evolutionary FAIL CLOSED (their search() does not '
                            'accept the kwargs optimize() forwards — they would raise '
                            'TypeError on the first trial).')
    parser.add_argument('--config-file', type=str,
                       help='Run with existing optimal config (skips optimization)')

    # Common parameters
    parser.add_argument('--lottery-file', type=str, required=True,
                       help='Path to lottery data JSON file')

    # Bayesian mode parameters
    parser.add_argument('--trials', type=int, default=50,
                       help='Number of Bayesian optimization trials')
    parser.add_argument('--output', type=str, default='optimal_window_config.json',
                       help='Output path for optimal config (Bayesian mode)')

    # Config mode parameters
    parser.add_argument('--max-seeds', type=int, default=10_000_000,
                       help='Max seeds per iteration (config mode)')
    parser.add_argument('--iterations', type=int, default=1,
                       help='Number of sieve iterations (config mode)')
    parser.add_argument('--output-survivors', type=str, default='bidirectional_survivors.json',
                       help='Output file for bidirectional survivors')
    parser.add_argument('--output-train', type=str, default='train_history.json',
                       help='Output file for training data')
    parser.add_argument('--output-holdout', type=str, default='holdout_history.json',
                       help='Output file for holdout data')

    # PRNG type
    parser.add_argument('--prng-type', type=str, default='java_lcg',
                       help='PRNG type to use (any from prng_registry)')

    # [S178 P0-1] Threshold override flags — DECLARED BUT UNWIRED. See the
    # fail-closed gate after parse_args(); passing either aborts the run.
    parser.add_argument('--forward-threshold', type=float, default=None,
                       help='UNWIRED — passing this ABORTS the run '
                            '(WINDOW_OPTIMIZER_THRESHOLD_OVERRIDE_UNWIRED). It reaches '
                            'no sieve and never has. Set thresholds via '
                            'distributed_config.json search_bounds, or let Optuna sample.')
    parser.add_argument('--reverse-threshold', type=float, default=None,
                       help='UNWIRED — passing this ABORTS the run '
                            '(WINDOW_OPTIMIZER_THRESHOLD_OVERRIDE_UNWIRED). See '
                            '--forward-threshold.')

    # NEW: Variable skip testing flag
    parser.add_argument('--resume-study', action='store_true',
                       help='Resume most recent incomplete Optuna study DB instead of starting fresh. '
                            'Skips warm-start enqueue if study already has trials. '
                            'Default: False (fresh study every run).')
    parser.add_argument('--study-name', type=str, default='',
                       help='Optuna study DB name to resume (e.g. window_opt_1772507547). '
                            'Empty string = auto-select most recent incomplete study. '
                            'Only used when --resume-study is set.')
    parser.add_argument('--test-both-modes', action='store_true',
                       help='Test BOTH constant and variable skip patterns (NEW!)')
    # S115 R3: pruning + parallelism flags
    parser.add_argument('--enable-pruning', action='store_true', default=False,
                       help='Enable forward_count==0 pruning (~1.7x speedup alone).')
    parser.add_argument('--n-parallel', type=int, default=1,
                       help='Parallel partitions: 1=serial (default), 2=dual-partition split.')
    parser.add_argument('--trse-context', type=str, default='trse_context.json',
                       help='[S121] TRSE regime context file (Step 0 output). '
                            'If present and regime stable, narrows Step 1 search bounds. '
                            'Default: trse_context.json. Pass empty string to disable.')
    parser.add_argument('--use-persistent-workers', action='store_true', default=False,
                       help='[S134] Use persistent worker engine instead of subprocess sieve. '
                            'Workers stay alive across all 4 sieve passes per trial.')
    parser.add_argument('--use-zmq-sqlite', action='store_true', default=False,
                        help='Use ZMQ+SQLite coordinator (S158D — no persistent SSH pipes)')
    parser.add_argument('--pwc-transport', default='tcp',
                       help='[S162] PWC transport: ssh | tcp. tcp=10x faster, '
                            '26-GPU validated at 2.24M sps. (default: tcp)')
    parser.add_argument('--min-workers', type=int, default=24,
                       help='[S162] Minimum workers reaching ready state before dispatch. '
                            'Default 24 = full 3-rig AMD cluster.')
    parser.add_argument('--worker-pool-size', type=int, default=8,
                       help='[S134] Number of persistent workers to spawn per rig (default: 8).')
    parser.add_argument('--seed-cap-nvidia', type=int, default=5_000_000,
                       help='[S137] Max seeds per job chunk for NVIDIA GPUs (default: 5000000).')
    parser.add_argument('--seed-cap-amd', type=int, default=2_000_000,
                       help='[S137] Max seeds per job chunk for AMD GPUs (default: 2000000).')
    parser.add_argument('--seed-start', type=int, default=0,
                       help='[S140] Starting seed for search range. Set automatically by '
                            'WATCHER coverage tracker to advance into unexplored seed space. '
                            'Default 0.')
    parser.add_argument('--warm-start-window', type=int, default=None,
                       help='[S166] Warm-start: enqueue this window_size as trial 0.')
    parser.add_argument('--warm-start-offset', type=int, default=None,
                       help='[S166] Warm-start: enqueue this offset as trial 0.')
    parser.add_argument('--warm-start-skip-min', type=int, default=None,
                       help='[S166] Warm-start: skip_min for trial 0.')
    parser.add_argument('--warm-start-skip-max', type=int, default=None,
                       help='[S166] Warm-start: skip_max for trial 0.')
    parser.add_argument('--warm-start-fwd-thresh', type=float, default=None,
                       help='[S166] Warm-start: forward_threshold for trial 0.')
    parser.add_argument('--warm-start-rev-thresh', type=float, default=None,
                       help='[S166] Warm-start: reverse_threshold for trial 0.')
    parser.add_argument('--warm-start-session-idx', type=int, default=None,
                       help='[S166] Warm-start: session_idx for trial 0 (0=midday+evening, 1=midday, 2=evening).')

    # [S172 Phase 1] RANGE-MINER backend (mutually exclusive with --use-persistent-workers
    # and --use-zmq-sqlite). Phase 1 is scaffolding-only; enabling this flag will import
    # miner.range_miner_coordinator.run_trial_miner which raises NotImplementedError until
    # Phases 2-5 land. See docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md.
    parser.add_argument('--use-range-miner', action='store_true', default=False,
                        help='[S172] Use RANGE-MINER stripe backend (opt-in, mutex vs PWC/ZMQ)')
    parser.add_argument('--miner-stripe-size', type=int, default=67_108_864,
                        help='[S172 §6.2] Seeds per stripe per GPU (default 64M)')
    parser.add_argument('--miner-substripes', type=int, default=8,
                        help='[S172 §6.2] Sub-stripe count per stripe (default 8; sized to fit watchdog)')
    # [S172 Phase 1] Infrastructure-neutral output path — configurable to support LXC ramdisk
    # bind-mounts (/dev/shm/prng/miner/), VM disk paths, and bare-metal all identically.
    # None means "auto-detect: /dev/shm/prng/miner/ if writable else ~/miner_output/".
    parser.add_argument('--miner-output-dir', type=str, default=None,
                        help='[S172] Miner NPZ output directory (default: /dev/shm/prng/miner/ '
                             'if writable, else ~/miner_output/)')

    args = parser.parse_args()

    # ========================================================================
    # [S178 P0-1] DEAD DIMENSION D-4 — the threshold override flags fail closed
    # ========================================================================
    # `--forward-threshold` / `--reverse-threshold` were declared above and
    # `args.forward_threshold` / `args.reverse_threshold` were NEVER referenced
    # after parse_args(). An operator passing one received a SILENT NO-OP on a
    # run that reported success — the fifth dead dimension and the first
    # operator-facing one (audit db9782a §4, D-4). The chapter advertised them
    # as "Override Optuna optimization"; that override does not exist.
    #
    # They are KEPT IN ARGPARSE rather than deleted, deliberately. Deletion
    # would make them "unrecognized arguments" — also a nonzero failure, but a
    # diagnostic that says "you misspelled a flag" when the truth is "this
    # capability is unwired". Keeping the declaration preserves the record of
    # operator intent (tfm-project-facts §0.4: absence of an implementation is
    # not evidence of absent intent) and lets the failure name the real cause.
    #
    # CONDITION UNDER WHICH THEY MAY RETURN — all four, or not at all:
    #   1. they feed the SINGLE resolve_directional_threshold() authority
    #      established at 8a55a68 (window_optimizer_integration_final.py:210-236)
    #      as its `explicit` argument. They must NOT create parallel threshold
    #      state alongside WindowConfig / SearchBounds / the config file;
    #   2. 0.0 is preserved — `is None` is the sole fallback trigger, never
    #      truthiness (`getattr(...) or default` silently destroys 0.0);
    #   3. requested / payload / effective are recorded separately, per the D6
    #      read-back pattern (miner/range_miner_coordinator.py:1644-1652);
    #   4. this gate is deleted in the SAME change that wires them, never before.
    #
    # This runs before the backend mutex and long before MultiGPUCoordinator is
    # constructed, so nothing is allocated and no sieve is launched.
    _unwired_flags = [
        flag for flag, value in (
            ('--forward-threshold', args.forward_threshold),
            ('--reverse-threshold', args.reverse_threshold),
        ) if value is not None
    ]
    if _unwired_flags:
        parser.error(
            "WINDOW_OPTIMIZER_THRESHOLD_OVERRIDE_UNWIRED: "
            f"{', '.join(_unwired_flags)} reaches no sieve — it never has (dead "
            "dimension D-4). Honouring it would be a silent no-op on a run that "
            "reports success, so the run is refused instead. Set thresholds via "
            "distributed_config.json -> search_bounds, or let Optuna sample them. "
            "See docs/CHAPTER_1_WINDOW_OPTIMIZER.md §10.1."
        )

    # ========================================================================
    # [S178 P0-2] Unsupported search strategies fail closed
    # ========================================================================
    # `random`, `grid` and `evolutionary` raise TypeError on their first call,
    # AFTER the 26-GPU coordinator has been constructed. Fail here instead,
    # naming the cause (signature mismatch). Derived from live signatures, so
    # repairing a strategy clears this gate with no edit here.
    if args.strategy:
        try:
            require_supported_strategy(args.strategy)
        except StrategyContractError as _err:
            parser.error(str(_err))

    # [S172 Phase 1] Mutex validation: exactly one backend may be selected.
    _backends = [
        ('use_persistent_workers', args.use_persistent_workers),
        ('use_zmq_sqlite',         args.use_zmq_sqlite),
        ('use_range_miner',        args.use_range_miner),
    ]
    _enabled = [name for name, val in _backends if val]
    if len(_enabled) > 1:
        parser.error(
            f"only one of --use-persistent-workers, --use-zmq-sqlite, "
            f"--use-range-miner may be set (got: {', '.join('--' + n.replace('_','-') for n in _enabled)})"
        )

    # ========================================================================
    # [S172 Phase 6-P0.5] RUN-START DATASET AUTHORITY GATE
    # ========================================================================
    # Requirements 1-8 of the P0.5 brief converge on this one place, because it
    # is the last point at which nothing has been allocated: it runs after the
    # backend mutex and BEFORE MultiGPUCoordinator is constructed, before any
    # spool, before the first stripe assignment, before a single worker is
    # dispatched. Everything this gate refuses, it refuses for free.
    #
    #   * the pointer manifest daily3_current.json is resolved (req 1) and
    #     validated — a target outside the version grammar, a traversal, an
    #     absolute path or the bare alias itself is refused (req 8);
    #   * the resulting identity — manifest/version, ABSOLUTE path, sha256, size
    #     and record count — is frozen for the whole run (req 2), and every later
    #     consumer reads the freeze rather than the pointer, so a scrape landing
    #     mid-run cannot alter a run in progress (req 7);
    #   * args.lottery_file is REPLACED with the frozen absolute immutable path,
    #     so what gets dispatched is never the bare, mutable daily3.json (req 3)
    #     and never depends on any child process's CWD;
    #   * the fleet is verified per node, with each digest re-derived ON THE
    #     TARGET (req 5), and a failure raises here — before dispatch (req 4);
    #   * the frozen values are written to run provenance (req 6).
    #
    # Failure is `parser.error`, matching the S178 P0-1/P0-2 gates above: a run
    # that cannot establish which dataset it is running against does not start.
    from miner import dataset_authority as _dsauth
    from miner.range_miner_worker import DatasetProvisioningError as _DatasetProvErr

    #
    # [Beta P0.5 closure ruling] A MINER-BACKED run additionally hard-fails when
    # the provisioning manifest is missing, unreadable, invalid or empty: with no
    # manifest the system cannot establish which worker datasets must be
    # verified, and recording UNAVAILABLE and proceeding violates the authority
    # boundary. `remote_execution=True` is unconditional and is a statement of
    # fact, not of policy — BOTH sieve entry points construct the 26-GPU
    # MultiGPUCoordinator (:756 in run_bayesian_optimization, :1079 in
    # run_with_config), so no window-optimizer run is fleet-free. Declaring
    # otherwise for a single-GPU invocation would BE Beta's Q1 refinement, which
    # is explicitly not authorized.
    _p05_label = (f"window_opt_{args.prng_type}_"
                  f"{args.strategy or 'config'}_{os.getpid()}")
    try:
        _p05_frozen = _dsauth.run_start_dataset_gate(
            args.lottery_file,
            run_label=_p05_label,
            miner_backed=bool(getattr(args, 'use_range_miner', False)),
            remote_execution=True,
        )
    except (_dsauth.DatasetAuthorityError, _DatasetProvErr) as _p05_err:
        parser.error(f"DATASET_AUTHORITY_P0_5: {_p05_err}")

    if _p05_frozen.path != os.path.abspath(args.lottery_file):
        print(f"📌 [P0.5] dataset pointer resolved: {args.lottery_file} → "
              f"{_p05_frozen.path}")
    print(f"📌 [P0.5] dataset FROZEN for this run: {_p05_frozen.describe()}")
    args.lottery_file = _p05_frozen.path

    # Check mode
    if args.strategy == 'bayesian':
        # BAYESIAN OPTIMIZATION MODE
        if not BAYESIAN_AVAILABLE:
            print("❌ Error: Optuna not available for Bayesian optimization")
            print("   Install with: pip install optuna")
            sys.exit(1)

        results = run_bayesian_optimization(
            lottery_file=args.lottery_file,
            trials=args.trials,
            output_config=args.output,
            seed_count=args.max_seeds if args.max_seeds else 10_000_000,
            seed_start=getattr(args, 'seed_start', 0),                              # S140
            prng_type=args.prng_type,
            test_both_modes=args.test_both_modes,
            resume_study=getattr(args, 'resume_study', False),
            study_name=getattr(args, 'study_name', ''),
            enable_pruning=getattr(args, 'enable_pruning', False),
            n_parallel=getattr(args, 'n_parallel', 1),
            trse_context_file=getattr(args, 'trse_context', 'trse_context.json'),
            use_persistent_workers=getattr(args, 'use_persistent_workers', False),  # S134
            use_zmq_sqlite=getattr(args, 'use_zmq_sqlite', False),                      # S158D
            pwc_transport=getattr(args, 'pwc_transport', 'tcp'),                    # S162
            pwc_min_workers=getattr(args, 'min_workers', 24),                       # S162
            worker_pool_size=getattr(args, 'worker_pool_size', 8),                  # S134
            seed_cap_nvidia=getattr(args, 'seed_cap_nvidia', 5_000_000),            # S137
            seed_cap_amd=getattr(args, 'seed_cap_amd', 2_000_000),                  # S137
            warm_start_window=getattr(args, 'warm_start_window', None),             # S166
            warm_start_offset=getattr(args, 'warm_start_offset', None),
            warm_start_skip_min=getattr(args, 'warm_start_skip_min', None),
            warm_start_skip_max=getattr(args, 'warm_start_skip_max', None),
            warm_start_fwd_thresh=getattr(args, 'warm_start_fwd_thresh', None),
            warm_start_rev_thresh=getattr(args, 'warm_start_rev_thresh', None),
            warm_start_session_idx=getattr(args, 'warm_start_session_idx', None),
            # [S172 Phase 1]
            use_range_miner=getattr(args, 'use_range_miner', False),
            miner_stripe_size=getattr(args, 'miner_stripe_size', 67_108_864),
            miner_substripes=getattr(args, 'miner_substripes', 8),
            miner_output_dir=getattr(args, 'miner_output_dir', None),
        )

        print("\n✅ Bayesian optimization complete!")
        print(f"   Best score: {results['best_score']:.2f}")
        if args.test_both_modes:
            print(f"   Survivors generated for BOTH constant and variable skip")
        else:
            print(f"   Survivors generated for constant skip only")

    elif args.strategy == 'random':
        # RANDOM SEARCH MODE
        print("\n🎲 Running Random Search optimization...")
        print(f"   Trials: {args.trials}")
        print(f"   PRNG: {args.prng_type}")
        
        # Use same infrastructure as bayesian but with RandomSearch strategy
        results = run_bayesian_optimization(
            lottery_file=args.lottery_file,
            trials=args.trials,
            output_config=args.output,
            seed_count=args.max_seeds if args.max_seeds else 10_000_000,
            prng_type=args.prng_type,
            test_both_modes=args.test_both_modes,
            resume_study=getattr(args, 'resume_study', False),
            study_name=getattr(args, 'study_name', ''),
            strategy_name='random'  # Override to use RandomSearch
        )
        print("\n✅ Random search complete!")
        print(f"   Best score: {results['best_score']:.2f}")
    
    elif args.strategy == 'grid':
        # GRID SEARCH MODE
        print("\n📊 Running Grid Search optimization...")
        print(f"   Trials: {args.trials}")
        print(f"   PRNG: {args.prng_type}")
        
        results = run_bayesian_optimization(
            lottery_file=args.lottery_file,
            trials=args.trials,
            output_config=args.output,
            seed_count=args.max_seeds if args.max_seeds else 10_000_000,
            prng_type=args.prng_type,
            test_both_modes=args.test_both_modes,
            resume_study=getattr(args, 'resume_study', False),
            study_name=getattr(args, 'study_name', ''),
            strategy_name='grid'
        )
        print("\n✅ Grid search complete!")
        print(f"   Best score: {results['best_score']:.2f}")
    
    elif args.strategy == 'evolutionary':
        # EVOLUTIONARY SEARCH MODE
        print("\n🧬 Running Evolutionary Search optimization...")
        print(f"   Trials: {args.trials}")
        print(f"   PRNG: {args.prng_type}")
        
        results = run_bayesian_optimization(
            lottery_file=args.lottery_file,
            trials=args.trials,
            output_config=args.output,
            seed_count=args.max_seeds if args.max_seeds else 10_000_000,
            prng_type=args.prng_type,
            test_both_modes=args.test_both_modes,
            resume_study=getattr(args, 'resume_study', False),
            study_name=getattr(args, 'study_name', ''),
            strategy_name='evolutionary'
        )
        print("\n✅ Evolutionary search complete!")
        print(f"   Best score: {results['best_score']:.2f}")
        
    elif args.config_file:
        # RUN WITH EXISTING CONFIG MODE
        if not Path(args.config_file).exists():
            print(f"❌ Error: Config file not found: {args.config_file}")
            sys.exit(1)

        results = run_with_config(
            config_file=args.config_file,
            lottery_file=args.lottery_file,
            max_seeds=args.max_seeds,
            iterations=args.iterations,
            output_survivors=args.output_survivors,
            output_train=args.output_train,
            output_holdout=args.output_holdout,
            # [S170-PARITY] CLI passthrough — same defaults as Bayesian call site
            use_persistent_workers=getattr(args, 'use_persistent_workers', False),
            pwc_transport=getattr(args, 'pwc_transport', 'tcp'),
            # [S170-PARITY-2] CLI execution sizing passthrough
            seed_cap_amd=getattr(args, 'seed_cap_amd', 2_000_000),
            seed_cap_nvidia=getattr(args, 'seed_cap_nvidia', 5_000_000),
            worker_pool_size=getattr(args, 'worker_pool_size', 8),
            min_workers=getattr(args, 'min_workers', 1),
        )

        print("\n✅ Sieve execution complete!")
        print(f"   Bidirectional survivors: {results['bidirectional_count']:,}")

    else:
        print("❌ Error: Must specify either --strategy bayesian OR --config-file")
        print("\nUsage examples:")
        print("  1. Bayesian optimization - constant skip only (original):")
        print("     python3 window_optimizer.py --strategy bayesian --lottery-file lottery.json --trials 50")
        print("\n  2. Bayesian optimization - BOTH modes (NEW!):")
        print("     python3 window_optimizer.py --strategy bayesian --lottery-file lottery.json --trials 50 --test-both-modes")
        print("\n  3. Run with existing config:")
        print("     python3 window_optimizer.py --config-file optimal_window_config.json --lottery-file lottery.json")
        sys.exit(1)


if __name__ == '__main__':
    main()
