#!/usr/bin/env python3
"""
Window Optimizer Integration - WITH VARIABLE SKIP SUPPORT
==========================================================
Version: 2.0
Date: 2025-11-15

NEW IN V2.0:
- Supports testing BOTH constant and variable skip patterns in a single optimization run
- Adds skip_mode metadata to all survivors for ML feature engineering
- Backward compatible: test_both_modes defaults to False (original behavior)

ACCUMULATES ALL BIDIRECTIONAL SURVIVORS WITH RICH METADATA
Saves ALL survivors from ALL trials with window metadata for temporal diversity
"""

from typing import Dict, Any, List
import json
from window_optimizer import WindowConfig, TestResult

def extract_survivors_from_result(result: Dict[str, Any]) -> List[int]:
    """
    Extract survivor seeds from coordinator result.
    
    The coordinator returns results in various formats depending on the job type.
    This function handles all formats and extracts the unique survivor seeds.
    
    Args:
        result: Dictionary containing job results from coordinator
        
    Returns:
        List of unique survivor seed integers
    """
    survivors = []

    # Check if results contain survivors (format 1: direct results array)
    if 'results' in result:
        for job_result in result['results']:
            # Format 1a: Survivors directly in job result
            if 'survivors' in job_result:
                for survivor in job_result['survivors']:
                    seed = survivor.get('seed', survivor.get('id'))
                    if seed is not None:
                        survivors.append(seed)

            # Format 1b: Survivors grouped by PRNG family
            if 'per_family' in job_result:
                for family, family_data in job_result['per_family'].items():
                    if 'survivors' in family_data:
                        for survivor in family_data['survivors']:
                            seed = survivor.get('seed', survivor.get('id'))
                            if seed is not None:
                                survivors.append(seed)

    # Return unique survivors (remove duplicates)
    return list(set(survivors))


def run_bidirectional_test(coordinator,
                          config: WindowConfig,
                          dataset_path: str,
                          seed_start: int,
                          seed_count: int,
                          prng_base: str = 'java_lcg',
                          test_both_modes: bool = False,
                          forward_threshold: float = 0.01,
                          reverse_threshold: float = 0.01,
                          trial_number: int = 0,
                          accumulator: Dict[str, List] = None) -> TestResult:
    """
    Run forward + reverse sieve and ACCUMULATE survivors with metadata.
    
    NEW IN V2.0: Optionally tests BOTH constant and variable skip patterns!
    
    When test_both_modes=True, this function:
    1. Runs forward/reverse with constant skip (e.g., java_lcg)
    2. Runs forward/reverse with variable skip (e.g., java_lcg_hybrid)
    3. Tags all survivors with skip_mode metadata
    4. Accumulates BOTH sets into the same accumulator
    
    This allows the ML system to learn which skip pattern produces better survivors.
    
    Args:
        coordinator: MultiGPUCoordinator instance
        config: WindowConfig with window_size, offset, sessions, skip_min/max
        dataset_path: Path to lottery data JSON file
        seed_start: Starting seed value
        seed_count: Number of seeds to test
        prng_base: Base PRNG name (e.g., 'java_lcg', 'xorshift32')
        test_both_modes: If True, test BOTH constant and variable skip (NEW!)
        threshold: Match threshold for sieves
        trial_number: Current trial number (for metadata tracking)
        accumulator: Dict to accumulate survivors across trials
        
    Returns:
        TestResult with counts from the constant skip run
        (Variable skip counts are added to accumulator but not returned)
    """

    # ========================================================================
    # HELPER: Args Class for Coordinator
    # ========================================================================
    # The coordinator expects an args object with specific attributes
    # This class provides that interface
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
            self.threshold = forward_threshold  # Use forward threshold for forward sieve
            self.resume_policy = 'restart'
            self.max_concurrent = 26  # Use all 26 GPUs
            self.analysis_type = 'statistical'
            self.draw_match = None

            # Session filter determines which lottery draws to use
            if set(config.sessions) == {'midday', 'evening'}:
                self.session_filter = 'both'
            elif 'midday' in config.sessions:
                self.session_filter = 'midday'
            else:
                self.session_filter = 'evening'

    print(f"\n  Testing: {config.description()}")

    # ========================================================================
    # PART 1: CONSTANT SKIP TEST (Always runs)
    # ========================================================================
    # This is the original behavior - test with constant skip pattern
    
    print(f"    Running FORWARD sieve ({prng_base}) [CONSTANT SKIP]...")
    forward_args = Args()
    forward_args.step_name = f"Forward Sieve ({prng_base})"
    forward_args.prng_type = prng_base  # e.g., 'java_lcg'

    # Execute distributed sieve across all 26 GPUs
    forward_result = coordinator.execute_distributed_analysis(
        forward_args.target_file,
        f'results/window_opt_forward_{config.window_size}_{config.offset}.json',
        forward_args,
        forward_args.seeds,
        1000,  # samples
        8,     # lmax
        50     # grid_size
    )

    # Extract unique survivor seeds from results
    forward_survivors = extract_survivors_from_result(forward_result)
    print(f"      Forward: {len(forward_survivors):,} survivors")

    # REVERSE SIEVE (constant skip)
    reverse_prng = prng_base
    print(f"    Running REVERSE sieve ({reverse_prng}) [CONSTANT SKIP]...")
    reverse_args = Args()
    reverse_args.prng_type = reverse_prng
    reverse_args.threshold = reverse_threshold  # Use reverse threshold for reverse sieve

    reverse_args.step_name = f"Reverse Sieve ({reverse_prng})"
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

    # Find bidirectional survivors (seeds that survive BOTH forward and reverse)
    forward_set = set(forward_survivors)
    reverse_set = set(reverse_survivors)
    bidirectional_constant = forward_set & reverse_set

    print(f"      ✨ Bidirectional (constant): {len(bidirectional_constant):,} survivors")
    
    # Update dashboard with live trial stats
    if hasattr(coordinator, "_progress_writer") and coordinator._progress_writer:
        best_so_far = getattr(coordinator, "_best_bidirectional", 0)
        if len(bidirectional_constant) > best_so_far:
            coordinator._best_bidirectional = len(bidirectional_constant)
            best_so_far = len(bidirectional_constant)
        # Get accumulated totals if available
        acc_fwd = len(accumulator['forward']) if accumulator else 0
        acc_rev = len(accumulator['reverse']) if accumulator else 0
        acc_bid = len(accumulator['bidirectional']) if accumulator else 0
        coordinator._progress_writer.update_trial_stats(
            trial_num=trial_number,
            forward_survivors=len(forward_survivors),
            reverse_survivors=len(reverse_survivors),
            bidirectional=len(bidirectional_constant),
            best_bidirectional=best_so_far,
            config_desc=config.description(),
            accumulated_forward=acc_fwd,
            accumulated_reverse=acc_rev,
            accumulated_bidirectional=acc_bid
        )

    # ========================================================================
    # ACCUMULATE CONSTANT SKIP SURVIVORS WITH METADATA
    # ========================================================================
    # This metadata will be used by the ML system for feature engineering
    
    if accumulator is not None:
        # Prepare base metadata for this trial
        # NEW: Added prng_type, prng_base, skip_mode fields
        metadata_base = {
            'window_size': config.window_size,
            'offset': config.offset,
            'skip_min': config.skip_min,
            'skip_max': config.skip_max,
            'skip_range': config.skip_max - config.skip_min,
            'sessions': config.sessions,
            'trial_number': trial_number,
            'prng_base': prng_base,  # NEW: Base PRNG name (e.g., 'java_lcg')
        }

        # Metadata specific to constant skip
        # v1.9.1: Added 6 missing metadata fields for ML features
        union_size = len(forward_set | reverse_set)
        metadata_constant = {
            **metadata_base,
            'skip_mode': 'constant',  # NEW: Identifies this as constant skip
            'prng_type': prng_base,   # NEW: Full PRNG name (same as base for constant)
            'forward_count': len(forward_survivors),
            'reverse_count': len(reverse_survivors),
            'bidirectional_count': len(bidirectional_constant),
            'bidirectional_selectivity': len(forward_survivors) / max(len(reverse_survivors), 1),
            'score': len(bidirectional_constant),
            # v1.9.1: 6 new fields for ML feature completeness
            'intersection_count': len(bidirectional_constant),
            'intersection_ratio': len(bidirectional_constant) / max(union_size, 1),
            'forward_only_count': len(forward_set - reverse_set),
            'reverse_only_count': len(reverse_set - forward_set),
            'survivor_overlap_ratio': len(bidirectional_constant) / max(len(forward_set), 1),
            'intersection_weight': len(bidirectional_constant) / max(len(forward_set) + len(reverse_set), 1),
        }

        # Accumulate survivors with metadata
        # These will be saved to bidirectional_survivors.json at the end
        for seed in forward_survivors:
            accumulator['forward'].append({'seed': seed, **metadata_constant})

        for seed in reverse_survivors:
            accumulator['reverse'].append({'seed': seed, **metadata_constant})

        for seed in bidirectional_constant:
            accumulator['bidirectional'].append({'seed': seed, **metadata_constant})

    # ========================================================================
    # PART 2: VARIABLE SKIP TEST (NEW! Only if test_both_modes=True)
    # ========================================================================
    # This tests the same base PRNG but with variable skip pattern
    # For example: java_lcg becomes java_lcg_hybrid
    
    if test_both_modes and not prng_base.endswith('_hybrid'):
        # Construct the hybrid PRNG name
        prng_hybrid = f"{prng_base}_hybrid"
        
        print(f"\n    🔄 TESTING VARIABLE SKIP MODE...")
        print(f"    Running FORWARD sieve ({prng_hybrid}) [VARIABLE SKIP]...")
        
        forward_args_hybrid = Args()
        forward_args_hybrid.prng_type = prng_hybrid  # e.g., 'java_lcg_hybrid'
        forward_args_hybrid.step_name = f"Forward Sieve ({prng_hybrid}) [VARIABLE]"

        forward_result_hybrid = coordinator.execute_distributed_analysis(
            forward_args_hybrid.target_file,
            f'results/window_opt_forward_hybrid_{config.window_size}_{config.offset}.json',
            forward_args_hybrid,
            forward_args_hybrid.seeds,
            1000,
            8,
            50
        )

        forward_survivors_hybrid = extract_survivors_from_result(forward_result_hybrid)
        print(f"      Forward (variable): {len(forward_survivors_hybrid):,} survivors")

        # REVERSE SIEVE (variable skip)
        print(f"    Running REVERSE sieve ({prng_hybrid}) [VARIABLE SKIP]...")
        reverse_args_hybrid = Args()
        reverse_args_hybrid.threshold = reverse_threshold  # Use reverse threshold for reverse sieve
        reverse_args_hybrid.step_name = f"Reverse Sieve ({prng_hybrid}) [VARIABLE]"
        reverse_args_hybrid.prng_type = prng_hybrid

        reverse_result_hybrid = coordinator.execute_distributed_analysis(
            reverse_args_hybrid.target_file,
            f'results/window_opt_reverse_hybrid_{config.window_size}_{config.offset}.json',
            reverse_args_hybrid,
            reverse_args_hybrid.seeds,
            1000,
            8,
            50
        )

        reverse_survivors_hybrid = extract_survivors_from_result(reverse_result_hybrid)
        print(f"      Reverse (variable): {len(reverse_survivors_hybrid):,} survivors")

        # Find bidirectional survivors for variable skip
        forward_set_hybrid = set(forward_survivors_hybrid)
        reverse_set_hybrid = set(reverse_survivors_hybrid)
        bidirectional_variable = forward_set_hybrid & reverse_set_hybrid

        print(f"      ✨ Bidirectional (variable): {len(bidirectional_variable):,} survivors")

        # ====================================================================
        # ACCUMULATE VARIABLE SKIP SURVIVORS WITH METADATA
        # ====================================================================
        # These get added to the SAME accumulator as constant skip survivors
        # They are distinguished by the skip_mode='variable' field
        
        if accumulator is not None:
            # Metadata specific to variable skip
            # v1.9.1: Added 6 missing metadata fields for ML features
            union_size_hybrid = len(forward_set_hybrid | reverse_set_hybrid)
            metadata_variable = {
                **metadata_base,
                'skip_mode': 'variable',  # NEW: Identifies this as variable skip
                'prng_type': prng_hybrid, # NEW: Full PRNG name (e.g., 'java_lcg_hybrid')
                'forward_count': len(forward_survivors_hybrid),
                'reverse_count': len(reverse_survivors_hybrid),
                'bidirectional_count': len(bidirectional_variable),
                'bidirectional_selectivity': len(forward_survivors_hybrid) / max(len(reverse_survivors_hybrid), 1),
                'score': len(bidirectional_variable),
                # v1.9.1: 6 new fields for ML feature completeness
                'intersection_count': len(bidirectional_variable),
                'intersection_ratio': len(bidirectional_variable) / max(union_size_hybrid, 1),
                'forward_only_count': len(forward_set_hybrid - reverse_set_hybrid),
                'reverse_only_count': len(reverse_set_hybrid - forward_set_hybrid),
                'survivor_overlap_ratio': len(bidirectional_variable) / max(len(forward_set_hybrid), 1),
                'intersection_weight': len(bidirectional_variable) / max(len(forward_set_hybrid) + len(reverse_set_hybrid), 1),
            }

            # Accumulate variable skip survivors
            # These will be in the SAME bidirectional_survivors.json file
            for seed in forward_survivors_hybrid:
                accumulator['forward'].append({'seed': seed, **metadata_variable})

            for seed in reverse_survivors_hybrid:
                accumulator['reverse'].append({'seed': seed, **metadata_variable})

            for seed in bidirectional_variable:
                accumulator['bidirectional'].append({'seed': seed, **metadata_variable})

    # ========================================================================
    # PRINT ACCUMULATOR STATUS
    # ========================================================================
    # Show running totals across all trials
    
    if accumulator is not None:
        print(f"      📊 Accumulated totals:")
        print(f"         Forward: {len(accumulator['forward'])} total")
        print(f"         Reverse: {len(accumulator['reverse'])} total")
        print(f"         Bidirectional: {len(accumulator['bidirectional'])} total")

    # ========================================================================
    # RETURN RESULT
    # ========================================================================
    # Note: We only return the constant skip counts in the TestResult
    # Variable skip counts are in the accumulator for later analysis
    
    return TestResult(
        config=config,
        forward_count=len(forward_survivors),
        reverse_count=len(reverse_survivors),
        bidirectional_count=len(bidirectional_constant),
        iteration=trial_number
    )


def add_window_optimizer_to_coordinator():
    """
    Add window optimization method to coordinator.
    
    This function monkey-patches the MultiGPUCoordinator class to add
    the optimize_window() method, which runs Bayesian optimization
    with real sieves executing on all 26 GPUs.
    """
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
                       test_both_modes: bool = False,  # NEW PARAMETER!
                       strategy_name: str = 'bayesian',
                       max_iterations: int = 50,
                       output_file: str = 'window_optimization.json'):
        """
        Run window optimization with real sieve execution.
        
        NEW IN V2.0: Supports test_both_modes parameter!
        
        Args:
            dataset_path: Path to lottery data JSON
            seed_start: Starting seed value
            seed_count: Number of seeds to test per trial
            prng_base: Base PRNG name (e.g., 'java_lcg')
            test_both_modes: If True, test BOTH constant and variable skip (NEW!)
            strategy_name: Optimization strategy ('bayesian', 'random', etc.)
            max_iterations: Number of optimization trials
            output_file: Where to save optimization results
        """

        print(f"\n{'='*80}")
        print(f"WINDOW OPTIMIZATION WITH SURVIVOR ACCUMULATION")
        print(f"Dataset: {dataset_path}")
        print(f"PRNG: {prng_base}")
        if test_both_modes:
            print(f"Mode: TESTING BOTH CONSTANT AND VARIABLE SKIP")  # NEW!
            print(f"  Constant: {prng_base}")
            print(f"  Variable: {prng_base}_hybrid")
        else:
            print(f"Mode: CONSTANT SKIP ONLY")
        print(f"Seed range: {seed_start:,} → {seed_start + seed_count:,}")
        print(f"Strategy: {strategy_name}")
        print(f"Max iterations: {max_iterations}")
        print(f"{'='*80}\n")

        # ====================================================================
        # CREATE SURVIVOR ACCUMULATOR
        # ====================================================================
        # This accumulates ALL survivors from ALL trials
        # Survivors will have metadata including skip_mode, prng_type, etc.
        survivor_accumulator = {
            'forward': [],
            'reverse': [],
            'bidirectional': []
        }

        optimizer = WindowOptimizer(self, dataset_path)
        # Define search bounds - loaded from distributed_config.json
        bounds = SearchBounds.from_config()

        # Track trial number for metadata
        trial_counter = {'count': 0}

        def test_config(config, ss=seed_start, sc=seed_count, ft=bounds.default_forward_threshold, rt=bounds.default_reverse_threshold):
            """
            Wrapper function that Optuna calls for each trial.
            This passes through to run_bidirectional_test with test_both_modes.
            """
            trial_counter['count'] += 1
            return run_bidirectional_test(
                coordinator=self,
                config=config,
                dataset_path=dataset_path,
                seed_start=ss,
                seed_count=sc,
                prng_base=prng_base,
                test_both_modes=test_both_modes,  # NEW: Pass through
                forward_threshold=ft,
                reverse_threshold=rt,
                trial_number=trial_counter['count'],
                accumulator=survivor_accumulator
            )

        optimizer.test_configuration = test_config

        # Setup optimization strategy
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


        # ====================================================================
        # RUN OPTIMIZATION
        # ====================================================================
        # This will call test_config() for each trial
        # Which will call run_bidirectional_test()
        # Which will accumulate survivors with metadata
        
        results = optimizer.optimize(
            strategy=strategy,
            bounds=bounds,
            max_iterations=max_iterations,
            scorer=BidirectionalCountScorer(),
            seed_start=seed_start,
            seed_count=seed_count
        )

        optimizer.save_results(results, output_file)

        # ====================================================================
        # PRINT OPTIMIZATION SUMMARY
        # ====================================================================
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

        # ====================================================================
        # SAVE ALL ACCUMULATED SURVIVORS WITH METADATA
        # ====================================================================
        # NEW: Survivors now have skip_mode, prng_type, prng_base fields
        
        print(f"\n{'='*80}")
        print("SAVING ALL ACCUMULATED SURVIVORS WITH METADATA")
        print(f"{'='*80}")

        try:
            # Deduplicate survivors while preserving best metadata
            def deduplicate_survivors(survivor_list):
                """
                Keep survivor with highest score for each unique seed.
                If a seed appears in both constant and variable mode,
                keep the one with the higher score.
                """
                seed_map = {}
                for survivor in survivor_list:
                    seed = survivor['seed']
                    if seed not in seed_map or survivor['score'] > seed_map[seed]['score']:
                        seed_map[seed] = survivor
                return list(seed_map.values())

            # Deduplicate each category
            forward_deduped = deduplicate_survivors(survivor_accumulator['forward'])
            reverse_deduped = deduplicate_survivors(survivor_accumulator['reverse'])
            bidirectional_deduped = deduplicate_survivors(survivor_accumulator['bidirectional'])

            # Save forward survivors with metadata
            with open('forward_survivors.json', 'w') as f:
                json.dump(sorted(forward_deduped, key=lambda x: x['seed']), f, indent=2)
            print(f"✅ Saved forward_survivors.json:")
            print(f"   Total: {len(forward_deduped)} unique seeds with metadata")
            print(f"   (Accumulated from {len(survivor_accumulator['forward'])} total across trials)")

            # Save reverse survivors with metadata
            with open('reverse_survivors.json', 'w') as f:
                json.dump(sorted(reverse_deduped, key=lambda x: x['seed']), f, indent=2)
            print(f"✅ Saved reverse_survivors.json:")
            print(f"   Total: {len(reverse_deduped)} unique seeds with metadata")
            print(f"   (Accumulated from {len(survivor_accumulator['reverse'])} total across trials)")

            # Save bidirectional survivors with metadata
            with open('bidirectional_survivors.json', 'w') as f:
                json.dump(sorted(bidirectional_deduped, key=lambda x: x['seed']), f, indent=2)
            print(f"✅ Saved bidirectional_survivors.json:")
            print(f"   Total: {len(bidirectional_deduped)} unique seeds with metadata")
            print(f"   (Accumulated from {len(survivor_accumulator['bidirectional'])} total across trials)")

            # Print metadata sample
            if bidirectional_deduped:
                print(f"\n📊 Sample survivor with metadata:")
                sample = bidirectional_deduped[0]
                print(f"   Seed: {sample['seed']}")
                print(f"   Skip mode: {sample.get('skip_mode', 'N/A')}")  # NEW!
                print(f"   PRNG type: {sample.get('prng_type', 'N/A')}")  # NEW!
                print(f"   Window: {sample['window_size']}, Offset: {sample['offset']}")
                print(f"   Skip range: [{sample['skip_min']}, {sample['skip_max']}]")
                print(f"   Sessions: {sample['sessions']}")
                print(f"   Trial: {sample['trial_number']}, Score: {sample['score']}")

            # NEW: Print skip mode distribution
            if test_both_modes:
                constant_count = sum(1 for s in bidirectional_deduped if s.get('skip_mode') == 'constant')
                variable_count = sum(1 for s in bidirectional_deduped if s.get('skip_mode') == 'variable')
                print(f"\n📈 Skip Mode Distribution:")
                print(f"   Constant skip: {constant_count} survivors")
                print(f"   Variable skip: {variable_count} survivors")

            print(f"{'='*80}\n")

        except Exception as e:
            print(f"⚠️  Error saving survivors with metadata: {e}")
            import traceback
            traceback.print_exc()

        # Keep existing integration save (if available)
        try:
            from integration.sieve_integration import save_bidirectional_sieve_results
            save_bidirectional_sieve_results(
                forward_survivors=[],
                reverse_survivors=[],
                intersection=[],
                config={
                    'prng_type': prng_base,
                    'seed_start': seed_start,
                    'seed_end': seed_start + seed_count,
                    'total_seeds': seed_count,
                    'window_size': best.get('window_size', 0),
                    'offset': best.get('offset', 0),
                    'skip_min': best.get('skip_min', 0),
                    'skip_max': best.get('skip_max', 0),
                    'forward_threshold': best.get('forward_threshold', 0.01),
                    'reverse_threshold': best.get('reverse_threshold', 0.01),
                    'dataset': dataset_path,
                    'sessions': best.get('sessions', [])
                },
                run_id=f"window_opt_{prng_base}_{strategy_name}"
            )
        except Exception as e:
            print(f"Note: New results format unavailable: {e}")

        return results

    # Monkey-patch the coordinator class
    MultiGPUCoordinator.optimize_window = optimize_window
    print("✅ Window optimizer integrated into MultiGPUCoordinator")
