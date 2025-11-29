#!/usr/bin/env python3
"""
Window Size Optimization Module - WITH PROPER REVERSE SIEVE
Passes exact skip values from forward sieve to reverse sieve
"""

import time
import json
import os
from typing import Dict, List
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class WindowEvaluation:
    """Single evaluation - ML feature vector"""
    window_size: int
    forward_survivors: int
    verified_survivors: int
    runtime: float
    signal_strength: float
    match_rate: float = 0.0
    success: bool = True
    timestamp: float = 0.0
    prng: str = ""
    
    def to_dict(self):
        return asdict(self)

class WindowOptimizer:
    """Discovers optimal window sizes using FORWARD + REVERSE sieves"""
    
    def __init__(self, coordinator, test_seeds: int = 1_000_000,
                 archive_dir: str = 'window_optimization_archive'):
        self.coordinator = coordinator
        self.test_seeds = test_seeds
        self.archive_dir = archive_dir
        os.makedirs(archive_dir, exist_ok=True)
        self.cache = {}
    
    def evaluate_window(self, prng: str, window: int, 
                       use_all_gpus: bool = False) -> WindowEvaluation:
        """
        Evaluate one window using BOTH forward and reverse sieves
        FIXED: Now passes exact skip values to reverse sieve!
        """
        
        cache_key = f"{prng}_{window}_{self.test_seeds}_{'all' if use_all_gpus else '1'}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # FORWARD SIEVE - Find candidates
        class ForwardArgs:
            target_file = self.coordinator.current_target_file or 'daily3.json'
            method = 'residue_sieve'
            prng_type = prng
            window_size = window
            seeds = self.test_seeds
            seed_start = 0
            offset = 0
            skip_min = 0
            skip_max = 20
            threshold = 0.01
            phase1_threshold = 0.01
            phase2_threshold = 0.50
            session_filter = 'both'
            hybrid = '_hybrid' in prng
            gpu_id = None
        
        forward_args = ForwardArgs()
        
        start = time.time()
        try:
            # PHASE 1: Forward sieve
            forward_jobs = self.coordinator._create_sieve_jobs(forward_args)
            forward_survivors = []
            
            if use_all_gpus:
                for job, worker in forward_jobs:
                    result = self.coordinator.execute_gpu_job(job, worker)
                    if hasattr(result, 'results') and result.results:
                        if 'survivors' in result.results:
                            forward_survivors.extend(result.results['survivors'])
            else:
                if forward_jobs:
                    job, worker = forward_jobs[0]
                    result = self.coordinator.execute_gpu_job(job, worker)
                    if hasattr(result, 'results') and result.results:
                        if 'survivors' in result.results:
                            forward_survivors = result.results['survivors']
            
            forward_count = len(forward_survivors)
            
            # PHASE 2: Reverse sieve on survivors WITH EXACT SKIP VALUES
            verified_survivors = 0
            
            if forward_survivors:
                # For each survivor, test ONLY its best_skip value
                # Group by skip value for efficiency
                survivors_by_skip = {}
                for s in forward_survivors:
                    skip = s.get('best_skip', 0)
                    if skip not in survivors_by_skip:
                        survivors_by_skip[skip] = []
                    survivors_by_skip[skip].append(s)
                
                # Test each skip group separately
                for skip_val, survivors_group in survivors_by_skip.items():
                    # Create args for this specific skip
                    class ReverseArgs:
                        target_file = self.coordinator.current_target_file or 'daily3.json'
                        method = 'reverse_sieve'
                        prng_type = prng
                        window_size = window
                        threshold = 0.01
                        session_filter = 'both'
                        hybrid = '_hybrid' in prng
                        gpu_id = None
                        # CRITICAL: Set skip_min = skip_max = the exact skip value!
                        skip_min = skip_val
                        skip_max = skip_val
                        offset = 0
                    
                    reverse_args = ReverseArgs()
                    
                    # Prepare candidate seeds for this skip
                    candidate_seeds = [
                        {
                            'seed': s['seed'],
                            'skip': skip_val,
                            'match_rate': s.get('match_rate', 0)
                        }
                        for s in survivors_group
                    ]
                    
                    # Create reverse sieve jobs
                    reverse_jobs = self.coordinator._create_reverse_sieve_jobs(
                        reverse_args, 
                        candidate_seeds
                    )
                    
                    # Execute reverse sieve jobs
                    for job, worker in reverse_jobs:
                        result = self.coordinator.execute_gpu_job(job, worker)
                        if hasattr(result, 'results') and result.results:
                            if 'verified_survivors' in result.results:
                                verified_survivors += len(result.results['verified_survivors'])
                            elif 'survivors' in result.results:
                                verified_survivors += len(result.results['survivors'])
            
            runtime = time.time() - start
            signal = verified_survivors / runtime if runtime > 0 else 0
            match_rate = verified_survivors / self.test_seeds if self.test_seeds > 0 else 0
            
            eval_result = WindowEvaluation(
                window_size=window,
                forward_survivors=forward_count,
                verified_survivors=verified_survivors,
                runtime=runtime,
                signal_strength=signal,
                match_rate=match_rate,
                success=True,
                timestamp=time.time(),
                prng=prng
            )
            
        except Exception as e:
            print(f"    ERROR evaluating window {window}: {e}")
            import traceback
            traceback.print_exc()
            eval_result = WindowEvaluation(
                window_size=window,
                forward_survivors=0,
                verified_survivors=0,
                runtime=0,
                signal_strength=0,
                match_rate=0,
                success=False,
                timestamp=time.time(),
                prng=prng
            )
        
        self.cache[cache_key] = eval_result
        return eval_result
    
    def adaptive_search(self, prng: str, verbose: bool = True,
                       use_all_gpus: bool = False,
                       min_window: int = 50,
                       max_window: int = 3000,
                       initial_step: int = 50) -> Dict:
        """Adaptive search with fine granularity where needed"""
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Adaptive Window Optimization: {prng}")
            print(f"Mode: FORWARD + REVERSE SIEVE (exact skip matching)")
            print(f"Range: {min_window} - {max_window}")
            print(f"Initial step: {initial_step}")
            print(f"Test seeds: {self.test_seeds:,}")
            print(f"GPU mode: {'ALL 26 GPUs' if use_all_gpus else '1 GPU (fast)'}")
            print(f"{'='*70}")
        
        evaluations = []
        
        # Phase 1: Coarse scan
        if verbose:
            print(f"\nPhase 1: Coarse scan (step={initial_step})")
        
        for window in range(min_window, max_window + 1, initial_step):
            result = self.evaluate_window(prng, window, use_all_gpus)
            evaluations.append(result)
            
            if verbose:
                status = "✓" if result.verified_survivors > 0 else " "
                verification_rate = (result.verified_survivors / result.forward_survivors * 100 
                                   if result.forward_survivors > 0 else 0)
                print(f"  [{status}] Window {window:>4}: "
                      f"{result.forward_survivors:>5} forward → "
                      f"{result.verified_survivors:>5} verified ({verification_rate:.1f}%), "
                      f"{result.runtime:.1f}s, signal={result.signal_strength:.2f}")
        
        # Find survivor regions
        survivor_regions = [e.window_size for e in evaluations if e.verified_survivors > 0]
        
        if not survivor_regions:
            if verbose:
                print(f"\n  ⚠️  No verified survivors in coarse scan")
            
            best = evaluations[0]
            return {
                'prng': prng,
                'optimal_window': best.window_size,
                'signal_strength': 0,
                'verified_survivors': 0,
                'confidence': 'NONE',
                'tests_performed': len(evaluations),
                'evaluations': [e.to_dict() for e in evaluations]
            }
        
        # Phase 2: Fine scan around survivors
        if verbose:
            print(f"\nPhase 2: Fine scan (step=1) around {len(survivor_regions)} regions")
        
        for center in survivor_regions:
            scan_start = max(min_window, center - 25)
            scan_end = min(max_window, center + 25)
            
            if verbose:
                print(f"  Scanning {scan_start}-{scan_end} around window={center}")
            
            for window in range(scan_start, scan_end + 1, 1):
                if not any(e.window_size == window for e in evaluations):
                    result = self.evaluate_window(prng, window, use_all_gpus)
                    evaluations.append(result)
                    
                    if verbose and result.verified_survivors > 0:
                        verification_rate = (result.verified_survivors / result.forward_survivors * 100 
                                           if result.forward_survivors > 0 else 0)
                        print(f"    Window {window:>4}: "
                              f"{result.forward_survivors:>5} → {result.verified_survivors:>5} "
                              f"({verification_rate:.1f}%), signal={result.signal_strength:.2f}")
        
        # Find optimal
        all_survivors = [e for e in evaluations if e.verified_survivors > 0]
        best = max(all_survivors, key=lambda e: e.signal_strength)
        
        if verbose:
            verification_rate = (best.verified_survivors / best.forward_survivors * 100 
                               if best.forward_survivors > 0 else 0)
            print(f"\n  ✅ Optimal window: {best.window_size}")
            print(f"     Signal strength: {best.signal_strength:.2f}")
            print(f"     Forward survivors: {best.forward_survivors}")
            print(f"     Verified survivors: {best.verified_survivors} ({verification_rate:.1f}%)")
            print(f"     Tests performed: {len(evaluations)}")
        
        return {
            'prng': prng,
            'optimal_window': best.window_size,
            'signal_strength': best.signal_strength,
            'forward_survivors': best.forward_survivors,
            'verified_survivors': best.verified_survivors,
            'match_rate': best.match_rate,
            'confidence': 'HIGH',
            'runtime_total': sum(e.runtime for e in evaluations),
            'tests_performed': len(evaluations),
            'evaluations': [e.to_dict() for e in evaluations]
        }

def test_window_optimizer_standalone(prng: str = 'lcg32', 
                                    test_seeds: int = 1_000_000,
                                    use_all_gpus: bool = False):
    """Standalone test"""
    import sys
    sys.path.insert(0, '.')
    
    from coordinator import MultiGPUCoordinator
    
    print("🔧 WINDOW OPTIMIZER - EXACT SKIP MATCHING")
    print("="*70)
    print(f"Testing PRNG: {prng}")
    print(f"Test seeds: {test_seeds:,}")
    print(f"GPU mode: {'ALL 26 GPUs' if use_all_gpus else '1 GPU (fast)'}")
    print(f"Verification: Forward + Reverse with EXACT skip values")
    print("="*70)
    
    coordinator = MultiGPUCoordinator('distributed_config.json')
    coordinator.current_target_file = 'daily3.json'
    
    optimizer = WindowOptimizer(coordinator, test_seeds=test_seeds)
    result = optimizer.adaptive_search(prng, verbose=True, use_all_gpus=use_all_gpus)
    
    output_file = f'window_test_{prng}_verified.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ Test complete! Results: {output_file}")
    print(f"\nSummary:")
    print(f"  Optimal window: {result['optimal_window']}")
    print(f"  Verified survivors: {result['verified_survivors']}")
    print(f"  Signal strength: {result['signal_strength']:.2f}")
    
    return result

if __name__ == "__main__":
    import sys
    prng = sys.argv[1] if len(sys.argv) > 1 else 'lcg32'
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
    use_all = '--all-gpus' in sys.argv
    test_window_optimizer_standalone(prng, seeds, use_all)
