#!/usr/bin/env python3
"""
Window Optimizer Integration - Connects to actual coordinator
"""

from typing import Dict, Any, List
from window_optimizer import WindowConfig, TestResult

def extract_survivors_from_result(result: Dict[str, Any]) -> List[int]:
    """Extract survivor seeds from coordinator result"""
    survivors = []
    
    if 'results' in result:
        for job_result in result['results']:
            if 'survivors' in job_result:
                for survivor in job_result['survivors']:
                    seed = survivor.get('seed', survivor.get('id'))
                    if seed is not None:
                        survivors.append(seed)
            
            if 'per_family' in job_result:
                for family, family_data in job_result['per_family'].items():
                    if 'survivors' in family_data:
                        for survivor in family_data['survivors']:
                            seed = survivor.get('seed', survivor.get('id'))
                            if seed is not None:
                                survivors.append(seed)
    
    return list(set(survivors))

def run_bidirectional_test(coordinator,
                          config: WindowConfig,
                          dataset_path: str,
                          seed_start: int,
                          seed_count: int,
                          prng_base: str = 'java_lcg',
                          threshold: float = 0.01) -> TestResult:
    """Run forward + reverse sieve"""
    
    class Args:
        def __init__(self):
            self.target_file = dataset_path
            self.method = 'residue_sieve'
            self.seed_start = seed_start
            self.seeds = seed_count
            self.window_size = config.window_size
            self.offset = config.offset
            self.skip_min = config.skip_min
            self.skip_max = config.skip_max
            self.threshold = threshold
            self.resume_policy = 'restart'
            self.max_concurrent = 26
            
            if set(config.sessions) == {'midday', 'evening'}:
                self.session_filter = 'both'
            elif 'midday' in config.sessions:
                self.session_filter = 'midday'
            else:
                self.session_filter = 'evening'
    
    print(f"\n  Testing: {config.description()}")
    
    # FORWARD
    print(f"    Running FORWARD sieve ({prng_base})...")
    forward_args = Args()
    forward_args.prng_type = prng_base
    
    forward_result = coordinator.execute_distributed_analysis(
        forward_args.target_file,
        f'results/window_opt_forward_{config.window_size}_{config.offset}.json',
        forward_args,
        forward_args.seeds,
        1000,  # samples
        8,     # lmax
        50     # grid_size
    )
    
    forward_survivors = extract_survivors_from_result(forward_result)
    print(f"      Forward: {len(forward_survivors):,} survivors")
    
    # REVERSE
    reverse_prng = prng_base  # FIXED: Use same PRNG
    print(f"    Running REVERSE sieve ({reverse_prng})...")
    reverse_args = Args()
    reverse_args.prng_type = reverse_prng
    
    reverse_result = coordinator.execute_distributed_analysis(
        reverse_args.target_file,
        f'results/window_opt_reverse_{config.window_size}_{config.offset}.json',
        reverse_args,
        reverse_args.seeds,
        1000,
        8,
        50
    )
    
    reverse_survivors = extract_survivors_from_result(reverse_result)
    print(f"      Reverse: {len(reverse_survivors):,} survivors")
    
    # INTERSECTION
    forward_set = set(forward_survivors)
    reverse_set = set(reverse_survivors)
    bidirectional = forward_set & reverse_set
    
    print(f"      ✨ Bidirectional: {len(bidirectional):,} survivors")
    
    return TestResult(
        config=config,
        forward_count=len(forward_survivors),
        reverse_count=len(reverse_survivors),
        bidirectional_count=len(bidirectional),
        iteration=0
    )

def add_window_optimizer_to_coordinator():
    """Add window optimization to coordinator"""
    from coordinator import MultiGPUCoordinator
    from window_optimizer import (WindowOptimizer, SearchBounds,
                                  RandomSearch, GridSearch,
                                  BayesianOptimization, EvolutionarySearch,
                                  BidirectionalCountScorer)
    
    def optimize_window(self,
                       dataset_path: str,
                       seed_start: int = 0,
                       seed_count: int = 10_000_000,
                       prng_base: str = 'java_lcg',
                       strategy_name: str = 'bayesian',
                       max_iterations: int = 50,
                       output_file: str = 'window_optimization.json'):
        
        print(f"\n{'='*80}")
        print(f"WINDOW OPTIMIZATION")
        print(f"Dataset: {dataset_path}")
        print(f"PRNG: {prng_base} + {prng_base}_reverse")
        print(f"Seed range: {seed_start:,} → {seed_start + seed_count:,}")
        print(f"Strategy: {strategy_name}")
        print(f"Max iterations: {max_iterations}")
        print(f"{'='*80}\n")
        
        optimizer = WindowOptimizer(self, dataset_path)
        
        def test_config(config, ss=seed_start, sc=seed_count, th=0.01):
            return run_bidirectional_test(
                coordinator=self,
                config=config,
                dataset_path=dataset_path,
                seed_start=ss,
                seed_count=sc,
                prng_base=prng_base,
                threshold=th
            )
        
        optimizer.test_configuration = test_config
        
        strategy_map = {
            'random': RandomSearch(),
            'grid': GridSearch(
                window_sizes=[512, 768, 1024],
                offsets=[0, 100],
                skip_ranges=[(0, 20), (0, 50)]
            ),
            'bayesian': BayesianOptimization(n_initial=3),
            'evolutionary': EvolutionarySearch(population_size=10)
        }
        
        strategy = strategy_map.get(strategy_name, RandomSearch())
        
        bounds = SearchBounds(
            min_window_size=1,
            max_window_size=4096,
            min_offset=0,
            max_offset=500,
            min_skip_min=0,
            max_skip_min=50,
            min_skip_max=20,
            max_skip_max=200
        )
        
        results = optimizer.optimize(
            strategy=strategy,
            bounds=bounds,
            max_iterations=max_iterations,
            scorer=BidirectionalCountScorer(),
            seed_start=seed_start,
            seed_count=seed_count
        )
        
        optimizer.save_results(results, output_file)
        
        print(f"\n{'='*80}")
        print("OPTIMIZATION COMPLETE")
        print(f"Best configuration:")
        best = results['best_config']
        print(f"  Window size: {best['window_size']}")
        print(f"  Offset: {best['offset']}")
        print(f"  Sessions: {', '.join(best['sessions'])}")
        print(f"  Skip range: [{best['skip_min']}, {best['skip_max']}]")
        print(f"\nBest result:")
        print(f"  Bidirectional survivors: {results['best_result']['bidirectional_count']:,}")
        print(f"  Score: {results['best_score']:.2f}")
        print(f"{'='*80}\n")
        

    # === NEW: Save results in new format ===
    try:
        from integration.sieve_integration import save_bidirectional_sieve_results
        save_bidirectional_sieve_results(
            forward_survivors=[],  # Not available in this context
            reverse_survivors=[],  # Not available in this context
            intersection=[],  # Would need to pass bidirectional list here
            config={
                'prng_type': prng_base,
                'seed_start': seed_start,
                'seed_end': seed_start + seed_count,
                'total_seeds': seed_count,
                'window_size': best.get('window_size', 0),
                'offset': best.get('offset', 0),
                'skip_min': best.get('skip_min', 0),
                'skip_max': best.get('skip_max', 0),
                'threshold': 0.01,
                'dataset': dataset_path,
                'sessions': best.get('sessions', [])
            },
            run_id=f"window_opt_{prng_base}_{strategy_name}"
        )
    except Exception as e:
        print(f"Note: New results format unavailable: {e}")

        return results
    
    MultiGPUCoordinator.optimize_window = optimize_window
    print("✅ Window optimizer integrated into MultiGPUCoordinator")

