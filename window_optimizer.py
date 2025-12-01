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
import sys
import argparse
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

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

    def __hash__(self):
        """Make config hashable for use in sets/dicts"""
        return hash((self.window_size, self.offset, tuple(self.sessions),
                    self.skip_min, self.skip_max))

    def description(self) -> str:
        """Human-readable description of config"""
        sess = '+'.join(self.sessions)
        return f"W{self.window_size}_O{self.offset}_{sess}_S{self.skip_min}-{self.skip_max}"

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

@dataclass
class SearchBounds:
    """
    Search space boundaries for optimization.
    
    These define the valid ranges for each parameter that Optuna can explore.
    """
    min_window_size: int = 128
    max_window_size: int = 4096
    min_offset: int = 0
    max_offset: int = 2000
    min_skip_min: int = 0
    max_skip_min: int = 3
    min_skip_max: int = 0
    max_skip_max: int = 500
    session_options: List[List[str]] = None

    def __post_init__(self):
        """Initialize session options if not provided"""
        if self.session_options is None:
            self.session_options = [
                ['midday', 'evening'],  # Both sessions
                ['midday'],              # Midday only
                ['evening']              # Evening only
            ]

    def random_config(self) -> WindowConfig:
        """Generate random config within bounds (for random search)"""
        skip_min = random.randint(self.min_skip_min, self.max_skip_min)
        skip_max = random.randint(skip_min, self.max_skip_max)

        return WindowConfig(
            window_size=random.randint(self.min_window_size, self.max_window_size),
            offset=random.randint(self.min_offset, self.max_offset),
            sessions=random.choice(self.session_options),
            skip_min=skip_min,
            skip_max=skip_max
        )

    def is_valid(self, config: WindowConfig) -> bool:
        """Check if config is within bounds"""
        return (self.min_window_size <= config.window_size <= self.max_window_size and
                self.min_offset <= config.offset <= self.max_offset and
                self.min_skip_min <= config.skip_min <= self.max_skip_min and
                config.skip_min <= config.skip_max <= self.max_skip_max and
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

class SearchStrategy(ABC):
    """Base class for search strategies"""

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
    def __init__(self, n_initial=5):
        self.n_initial = n_initial
        self.optuna_search = None

        # Try to use real Optuna implementation
        if BAYESIAN_AVAILABLE:
            try:
                from window_optimizer_bayesian import OptunaBayesianSearch
                self.optuna_search = OptunaBayesianSearch(n_startup_trials=n_initial, seed=None)
            except Exception as e:
                print(f"⚠️  Could not initialize Optuna: {e}")

    def search(self, objective_function, bounds, max_iterations, scorer):
        """Run Bayesian optimization"""
        if self.optuna_search:
            # Use real Optuna implementation
            return self.optuna_search.search(objective_function, bounds, max_iterations, scorer)
        else:
            # Fallback to random search
            print("⚠️  Optuna not available, using random search fallback")
            return RandomSearch().search(objective_function, bounds, max_iterations, scorer)

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
                          seed_count: int = 10_000_000, threshold: float = 0.01) -> TestResult:
        """
        Test a configuration.
        This will be overridden by the integration layer to run real sieves.
        """
        if self.test_configuration_func:
            return self.test_configuration_func(config, seed_start, seed_count, threshold)

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
                threshold: float = 0.01) -> Dict[str, Any]:
        """
        Run optimization using the provided strategy.
        
        The strategy will call self.test_configuration() for each trial,
        which in turn calls run_bidirectional_test() from the integration layer.
        """
        if scorer is None:
            scorer = BidirectionalCountScorer()

        def objective(config: WindowConfig) -> TestResult:
            return self.test_configuration(config, seed_start, seed_count, threshold)

        return strategy.search(objective, bounds, max_iterations, scorer)

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
    prng_type: str = 'java_lcg',
    test_both_modes: bool = False  # NEW PARAMETER!
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
        from window_optimizer_integration_final import add_window_optimizer_to_coordinator
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
    coordinator = MultiGPUCoordinator(resume_policy="restart")

    # Add window optimizer to coordinator (this adds the optimize_window method)
    add_window_optimizer_to_coordinator()

    # Run optimization (this will run real sieves and accumulate survivors)
    print("\n🚀 Starting Bayesian optimization with real sieve execution...\n")

    results = coordinator.optimize_window(
        dataset_path=lottery_file,
        seed_start=0,
        seed_count=seed_count,
        prng_base=prng_type,
        test_both_modes=test_both_modes,  # NEW: Pass through to integration layer
        strategy_name='bayesian',
        max_iterations=trials,
        output_file='window_optimization_results.json'
    )

    # Save optimal config for downstream use
    best_config = results['best_config']

    optimal_config = {
        'window_size': best_config['window_size'],
        'offset': best_config['offset'],
        'skip_min': best_config['skip_min'],
        'skip_max': best_config['skip_max'],
        'sessions': best_config['sessions'],
        'prng_type': prng_type,
        'test_both_modes': test_both_modes,  # NEW: Record whether we tested both modes
        'seed_count': seed_count,
        'optimization_score': results['best_score']
    }

    with open(output_config, 'w') as f:
        json.dump(optimal_config, f, indent=2)

    print(f"\n✅ Optimal configuration saved to: {output_config}")

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
    output_holdout: str = 'holdout_history.json'
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
        from window_optimizer_integration_final import add_window_optimizer_to_coordinator
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
    coordinator = MultiGPUCoordinator(resume_policy="restart")

    # Add integration
    add_window_optimizer_to_coordinator()

    # Create WindowConfig object
    window_config = WindowConfig(
        window_size=config.get('window_size', 1024),
        offset=config.get('offset', 100),
        sessions=config.get('sessions', ['midday', 'evening']),
        skip_min=config.get('skip_min', 0),
        skip_max=config.get('skip_max', 50)
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
            threshold=0.01,
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
    parser.add_argument('--strategy', type=str, choices=['bayesian'],
                       help='Optimization strategy (currently only bayesian supported)')
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

    # NEW: Variable skip testing flag
    parser.add_argument('--test-both-modes', action='store_true',
                       help='Test BOTH constant and variable skip patterns (NEW!)')

    args = parser.parse_args()

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
            prng_type=args.prng_type,
            test_both_modes=args.test_both_modes  # NEW: Pass through
        )

        print("\n✅ Bayesian optimization complete!")
        print(f"   Best score: {results['best_score']:.2f}")
        if args.test_both_modes:
            print(f"   Survivors generated for BOTH constant and variable skip")
        else:
            print(f"   Survivors generated for constant skip only")

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
            output_holdout=args.output_holdout
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
