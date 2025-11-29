#!/usr/bin/env python3
"""
COMPLETE Forward vs Reverse Hybrid Variable Skip Alignment Test
"""

import subprocess
import json
import cupy as cp
from prng_registry import mt19937_cpu

# Generate variable skip test data
known_seed = 12345
skip_pattern = [5,5,3,7,5,5,8,4,5,5,6,5,5,3,9,5,5,5,4,7,5,5,5,6,5,5,3,8,5,5]
total_needed = sum(skip_pattern) + len(skip_pattern)
all_outputs = mt19937_cpu(known_seed, total_needed, skip=0)
draws = []
idx = 0
for skip in skip_pattern:
    idx += skip
    draws.append(all_outputs[idx] % 1000)
    idx += 1

print("="*70)
print("FORWARD + REVERSE HYBRID ALIGNMENT TEST")
print("="*70)
print(f"\nKnown seed: {known_seed}")
print(f"Skip pattern: {skip_pattern[:15]}...")
print(f"Draws: {draws[:10]}...")

# ============================================================================
# TEST 1: FORWARD HYBRID
# ============================================================================

print("\n" + "="*70)
print("FORWARD HYBRID SIEVE")
print("="*70)

test_data = [{'draw': d, 'session': 'midday'} for d in draws]
with open('test_hybrid_draws.json', 'w') as f:
    json.dump(test_data, f)

from hybrid_strategy import get_all_strategies
strategies = get_all_strategies()

strategies_data = []
for s in strategies:
    strategies_data.append({
        'name': s.name,
        'max_consecutive_misses': s.max_consecutive_misses,
        'skip_tolerance': s.skip_tolerance,
        'enable_reseed_search': s.enable_reseed_search,
        'skip_learning_rate': s.skip_learning_rate,
        'breakpoint_threshold': s.breakpoint_threshold
    })

job = {
    'job_id': 'forward_test',
    'dataset_path': 'test_hybrid_draws.json',
    'seed_start': 12345,
    'seed_end': 12346,
    'window_size': len(draws),
    'skip_range': [0, 10],
    'phase1_threshold': 0.0,
    'phase2_threshold': 0.50,
    'prng_families': ['mt19937'],
    'hybrid': True,
    'strategies': strategies_data,
    'sessions': ['midday'],
    'offset': 0
}

with open('forward_test_job.json', 'w') as f:
    json.dump(job, f, indent=2)

result = subprocess.run(
    ['python3', 'sieve_filter.py', '--job-file', 'forward_test_job.json', '--gpu-id', '0'],
    capture_output=True,
    text=True
)

forward_result = None
if result.returncode == 0:
    data = json.loads(result.stdout)
    survivors = data.get('survivors', [])
    print(f"✅ Forward found {len(survivors)} survivors")
    for s in survivors:
        if s['seed'] == known_seed:
            forward_result = s
            print(f"   🎯 Seed {s['seed']}: rate={s['match_rate']:.3f}")
            print(f"      Strategy: {s.get('strategy_name', 'unknown')}")
            print(f"      Pattern: {s.get('skip_pattern', [])[:15]}")
            break
else:
    print(f"❌ Forward failed!")

# ============================================================================
# TEST 2: REVERSE HYBRID
# ============================================================================

print("\n" + "="*70)
print("REVERSE HYBRID SIEVE")
print("="*70)

exec(open('reverse_kernels_addition.py').read(), globals())

kernel = cp.RawKernel(MT19937_HYBRID_REVERSE_KERNEL, 'mt19937_hybrid_reverse_sieve')

candidate_seeds = [12345]  # Only test our known seed
n_candidates = len(candidate_seeds)
candidate_seeds_gpu = cp.array(candidate_seeds, dtype=cp.uint32)
residues = cp.array(draws, dtype=cp.uint32)
survivors = cp.zeros(n_candidates, dtype=cp.uint32)
match_rates = cp.zeros(n_candidates, dtype=cp.float32)
skip_sequences = cp.zeros(n_candidates * 512, dtype=cp.uint32)
strategy_ids = cp.zeros(n_candidates, dtype=cp.uint32)
survivor_count = cp.zeros(1, dtype=cp.uint32)

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

reverse_result = None
if count > 0:
    print(f"✅ Reverse found {count} survivors")
    for i in range(count):
        seed = int(survivors[i].get())
        if seed == known_seed:
            strat_id = int(strategy_ids[i].get())
            skip_seq_flat = skip_sequences.get()
            skip_seq = skip_seq_flat[i*512:(i+1)*512][:len(draws)].tolist()
            
            reverse_result = {
                'seed': seed,
                'match_rate': float(match_rates[i].get()),
                'strategy_name': strategies[strat_id].name,
                'skip_pattern': skip_seq
            }
            
            print(f"   🎯 Seed {seed}: rate={reverse_result['match_rate']:.3f}")
            print(f"      Strategy: {reverse_result['strategy_name']}")
            print(f"      Pattern: {reverse_result['skip_pattern'][:15]}")
            break
else:
    print(f"❌ Reverse found 0 survivors")

# ============================================================================
# COMPARISON
# ============================================================================

print("\n" + "="*70)
print("ALIGNMENT RESULTS")
print("="*70)

if forward_result and reverse_result:
    rate_match = abs(forward_result['match_rate'] - reverse_result['match_rate']) < 0.01
    strategy_match = forward_result.get('strategy_name') == reverse_result.get('strategy_name')
    
    f_pattern = forward_result.get('skip_pattern', [])[:15]
    r_pattern = reverse_result.get('skip_pattern', [])[:15]
    pattern_match = f_pattern == r_pattern
    
    print(f"\nForward: rate={forward_result['match_rate']:.3f}, strategy={forward_result.get('strategy_name')}")
    print(f"         pattern={f_pattern}")
    print(f"\nReverse: rate={reverse_result['match_rate']:.3f}, strategy={reverse_result.get('strategy_name')}")
    print(f"         pattern={r_pattern}")
    
    print(f"\n{'✅' if rate_match else '❌'} Match rates align")
    print(f"{'✅' if strategy_match else '⚠️ '} Strategies match")
    print(f"{'✅' if pattern_match else '❌'} Skip patterns align")
    
    if rate_match and pattern_match:
        print(f"\n🎉 PERFECT ALIGNMENT - Forward and Reverse Hybrid are consistent!")
    else:
        print(f"\n⚠️  MISALIGNMENT detected")
else:
    print(f"\n❌ ALIGNMENT TEST FAILED")
    print(f"   Forward found: {forward_result is not None}")
    print(f"   Reverse found: {reverse_result is not None}")

