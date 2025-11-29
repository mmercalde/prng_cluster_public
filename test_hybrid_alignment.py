#!/usr/bin/env python3
"""
Cross-validation: Forward HYBRID vs Reverse HYBRID Alignment Test
Tests variable skip pattern matching between both sieves
"""

import sys
import json
from prng_registry import mt19937_cpu

def generate_variable_skip_data():
    """Generate test draws with VARIABLE skip pattern"""
    known_seed = 12345
    # Variable skip pattern: [5,5,3,7,5,5,8,4,5,5...]
    skip_pattern = [5,5,3,7,5,5,8,4,5,5,6,5,5,3,9,5,5,5,4,7,5,5,5,6,5,5,3,8,5,5]
    k = len(skip_pattern)
    
    # Generate with variable skip
    total_needed = sum(skip_pattern) + k
    all_outputs = mt19937_cpu(known_seed, total_needed, skip=0)
    
    draws = []
    idx = 0
    for skip in skip_pattern:
        idx += skip  # Skip
        draws.append(all_outputs[idx] % 1000)
        idx += 1
    
    return known_seed, skip_pattern, draws


def test_forward_hybrid():
    """Run forward sieve in HYBRID mode"""
    print("="*70)
    print("TESTING FORWARD HYBRID SIEVE")
    print("="*70)
    
    known_seed, skip_pattern, draws = generate_variable_skip_data()
    
    print(f"\nKnown seed: {known_seed}")
    print(f"Skip pattern: {skip_pattern[:15]}...")
    print(f"Draws: {draws[:10]}...")
    
    # Import strategies
    from hybrid_strategy import get_all_strategies
    strategies = get_all_strategies()
    
    # Create hybrid job
    job = {
        'job_id': 'hybrid_forward_test',
        'dataset_path': 'test_hybrid_draws.json',
        'seed_start': 10000,
        'seed_end': 15000,
        'window_size': len(draws),
        'skip_range': [0, 20],
        'min_match_threshold': 0.01,
        'phase1_threshold': 0.001,
        'phase2_threshold': 0.50,
        'prng_families': ['mt19937'],
        'hybrid': True,
        'strategies': [
            {'name': s.name, 'max_consecutive_misses': s.max_consecutive_misses, 
             'skip_tolerance': s.skip_tolerance} 
            for s in strategies
        ],
        'sessions': ['midday'],
        'offset': 0
    }
    
    # Save test draws
    test_data = [{'draw': d, 'session': 'midday'} for d in draws]
    with open('test_hybrid_draws.json', 'w') as f:
        json.dump(test_data, f)
    
    # Save job
    with open('hybrid_forward_test_job.json', 'w') as f:
        json.dump(job, f, indent=2)
    
    print(f"\nRunning forward HYBRID sieve...")
    print(f"  Seed range: {job['seed_start']} - {job['seed_end']}")
    print(f"  Strategies: {len(strategies)}")
    
    # Execute
    import subprocess
    result = subprocess.run(
        ['python3', 'sieve_filter.py', '--job-file', 'hybrid_forward_test_job.json', '--gpu-id', '0'],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"\n❌ Forward hybrid sieve failed!")
        print(result.stderr)
        return None
    
    # Parse result
    result_data = json.loads(result.stdout)
    
    survivors = result_data.get('survivors', [])
    print(f"\n✅ Forward hybrid sieve complete")
    print(f"   Phase 1 survivors: {result_data.get('per_family', {}).get('mt19937_hybrid', {}).get('phase1', {}).get('survivors', 0)}")
    print(f"   Phase 2 survivors: {len(survivors)}")
    
    # Find our known seed
    found = None
    for s in survivors:
        if s['seed'] == known_seed:
            found = s
            break
    
    if found:
        print(f"\n🎯 FOUND KNOWN SEED in forward hybrid:")
        print(f"   Seed: {found['seed']}")
        print(f"   Match rate: {found['match_rate']:.3f}")
        print(f"   Strategy: {found.get('strategy_name', 'unknown')}")
        print(f"   Skip pattern: {found.get('skip_pattern', [])[:15]}...")
        return found
    else:
        print(f"\n⚠️  Known seed NOT found in forward hybrid survivors")
        if survivors:
            print(f"   Top survivor: seed={survivors[0]['seed']}, rate={survivors[0]['match_rate']:.3f}")
        return None


def test_reverse_hybrid():
    """Run reverse sieve in HYBRID mode"""
    print("\n" + "="*70)
    print("TESTING REVERSE HYBRID SIEVE")
    print("="*70)
    
    known_seed, skip_pattern, draws = generate_variable_skip_data()
    
    print(f"\nKnown seed: {known_seed}")
    print(f"Skip pattern: {skip_pattern[:15]}...")
    print(f"Draws: {draws[:10]}...")
    
    # Import strategies
    from hybrid_strategy import get_all_strategies, StrategyConfig
    strategies = get_all_strategies()
    
    candidate_seeds = [12345, 12340, 12350, 12300, 12400]
    
    print(f"\nTesting {len(candidate_seeds)} candidate seeds with {len(strategies)} strategies")
    
    # Use the reverse hybrid kernel
    import cupy as cp
    exec(open('reverse_kernels_addition.py').read(), globals())
    
    kernel = cp.RawKernel(MT19937_HYBRID_REVERSE_KERNEL, 'mt19937_hybrid_reverse_sieve')
    
    n_candidates = len(candidate_seeds)
    candidate_seeds_gpu = cp.array(candidate_seeds, dtype=cp.uint32)
    residues = cp.array(draws, dtype=cp.uint32)
    survivors = cp.zeros(n_candidates, dtype=cp.uint32)
    match_rates = cp.zeros(n_candidates, dtype=cp.float32)
    skip_sequences = cp.zeros(n_candidates * 512, dtype=cp.uint32)
    strategy_ids = cp.zeros(n_candidates, dtype=cp.uint32)
    survivor_count = cp.zeros(1, dtype=cp.uint32)
    
    # Strategy parameters
    strategy_max_misses = cp.array([s.max_consecutive_misses for s in strategies], dtype=cp.int32)
    strategy_tolerances = cp.array([s.skip_tolerance for s in strategies], dtype=cp.int32)
    
    kernel(
        (1,), (256,),
        (candidate_seeds_gpu, residues, survivors, match_rates, skip_sequences,
         strategy_ids, survivor_count, cp.int32(n_candidates), cp.int32(len(draws)),
         strategy_max_misses, strategy_tolerances, cp.int32(len(strategies)),
         cp.float32(0.50), cp.int32(0))
    )
    
    count = int(survivor_count[0].get())
    
    print(f"\n✅ Reverse hybrid sieve complete")
    print(f"   Found {count} survivors")
    
    if count > 0:
        found = None
        for i in range(count):
            seed = int(survivors[i].get())
            if seed == known_seed:
                strat_id = int(strategy_ids[i].get())
                skip_seq_flat = skip_sequences.get()
                skip_seq = skip_seq_flat[i*512:(i+1)*512][:len(draws)].tolist()
                
                found = {
                    'seed': seed,
                    'match_rate': float(match_rates[i].get()),
                    'strategy_name': strategies[strat_id].name if strat_id < len(strategies) else 'unknown',
                    'skip_pattern': skip_seq
                }
                break
        
        if found:
            print(f"\n🎯 FOUND KNOWN SEED in reverse hybrid:")
            print(f"   Seed: {found['seed']}")
            print(f"   Match rate: {found['match_rate']:.3f}")
            print(f"   Strategy: {found.get('strategy_name', 'unknown')}")
            print(f"   Skip pattern: {found.get('skip_pattern', [])[:15]}...")
            return found
        else:
            print(f"\n⚠️  Known seed survived but not in results")
            return None
    else:
        print(f"\n❌ No survivors found in reverse hybrid")
        return None


def main():
    print("\n🔬 FORWARD HYBRID vs REVERSE HYBRID ALIGNMENT TEST")
    print("Validating variable skip pattern matching\n")
    
    try:
        forward_result = test_forward_hybrid()
        reverse_result = test_reverse_hybrid()
        
        print("\n" + "="*70)
        print("HYBRID ALIGNMENT TEST RESULTS")
        print("="*70)
        
        if forward_result and reverse_result:
            match_rate_match = abs(forward_result['match_rate'] - reverse_result['match_rate']) < 0.05
            strategy_match = forward_result.get('strategy_name') == reverse_result.get('strategy_name')
            
            # Compare skip patterns (first 15 elements)
            f_pattern = forward_result.get('skip_pattern', [])[:15]
            r_pattern = reverse_result.get('skip_pattern', [])[:15]
            pattern_match = f_pattern == r_pattern
            
            print(f"\nForward: seed={forward_result['seed']}, rate={forward_result['match_rate']:.3f}, strategy={forward_result.get('strategy_name')}")
            print(f"         skip={f_pattern}")
            print(f"Reverse: seed={reverse_result['seed']}, rate={reverse_result['match_rate']:.3f}, strategy={reverse_result.get('strategy_name')}")
            print(f"         skip={r_pattern}")
            
            print(f"\n{'✅' if match_rate_match else '❌'} Match rates align: {match_rate_match}")
            print(f"{'✅' if strategy_match else '⚠️ '} Strategy match: {strategy_match}")
            print(f"{'✅' if pattern_match else '❌'} Skip patterns align: {pattern_match}")
            
            if match_rate_match and pattern_match:
                print(f"\n🎉 HYBRID ALIGNMENT SUCCESS - Both sieves found variable skip pattern!")
                return 0
            else:
                print(f"\n⚠️  PARTIAL ALIGNMENT - Some differences in hybrid results")
                return 1
        else:
            print(f"\n❌ TEST FAILED - One or both hybrid sieves didn't find the known seed")
            print(f"   Forward found: {forward_result is not None}")
            print(f"   Reverse found: {reverse_result is not None}")
            return 1
            
    except Exception as e:
        print(f"\n💥 TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
