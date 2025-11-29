#!/usr/bin/env python3
"""
Integration test for reverse_sieve_filter.py module
Tests the job file interface and coordinator compatibility
"""

import subprocess
import json
from prng_registry import mt19937_cpu

print("="*70)
print("REVERSE_SIEVE_FILTER.PY MODULE INTEGRATION TEST")
print("="*70)

# Generate test data with known seed
known_seed = 12345
skip = 5
k = 30

total_needed = k * (skip + 1)
all_outputs = mt19937_cpu(known_seed, total_needed, skip=0)

draws = []
idx = 0
for i in range(k):
    idx += skip
    draws.append(all_outputs[idx] % 1000)
    idx += 1

print(f"\nKnown seed: {known_seed}")
print(f"Skip: {skip}")
print(f"Draws: {draws[:10]}...")

# Save test draws
test_data = [{'draw': d, 'session': 'midday'} for d in draws]
with open('test_module_draws.json', 'w') as f:
    json.dump(test_data, f)

# ============================================================================
# TEST 1: Fixed Skip Mode
# ============================================================================

print("\n" + "="*70)
print("TEST 1: FIXED SKIP MODE")
print("="*70)

# Create job file
job = {
    'job_id': 'reverse_fixed_test',
    'dataset_path': 'test_module_draws.json',
    'candidate_seeds': [12345, 12340, 12350],  # Include known seed + noise
    'window_size': 30,
    'skip_range': [0, 10],
    'min_match_threshold': 0.90,
    'prng_families': ['mt19937'],
    'sessions': ['midday'],
    'offset': 0,
    'hybrid': False
}

with open('test_fixed_job.json', 'w') as f:
    json.dump(job, f, indent=2)

print("\nRunning reverse_sieve_filter.py (fixed skip)...")
result = subprocess.run(
    ['python3', 'reverse_sieve_filter.py', '--job-file', 'test_fixed_job.json', '--gpu-id', '0'],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    data = json.loads(result.stdout)
    survivors = data.get('survivors', [])
    
    print(f"✅ Fixed skip test passed")
    print(f"   Candidates tested: {data['candidates_tested']}")
    print(f"   Survivors found: {len(survivors)}")
    
    if survivors:
        for s in survivors:
            print(f"   Seed {s['seed']}: rate={s['match_rate']:.3f}, skip={s['best_skip']}")
            if s['seed'] == known_seed:
                if s['match_rate'] == 1.0 and s['best_skip'] == skip:
                    print(f"   🎯 PERFECT MATCH - Found known seed with correct skip!")
                    test1_pass = True
                else:
                    print(f"   ⚠️ Found known seed but metrics unexpected")
                    test1_pass = False
        
        if not any(s['seed'] == known_seed for s in survivors):
            print(f"   ⚠️ Known seed not in survivors")
            test1_pass = False
    else:
        print(f"   ⚠️ No survivors found (threshold may be too high)")
        test1_pass = False
else:
    print(f"❌ Test failed with return code {result.returncode}")
    print(f"STDERR: {result.stderr}")
    test1_pass = False

# ============================================================================
# TEST 2: Hybrid Mode
# ============================================================================

print("\n" + "="*70)
print("TEST 2: HYBRID MODE")
print("="*70)

# Generate variable skip data
skip_pattern = [5,5,3,7,5,5,8,4,5,5,6,5,5,3,9,5,5,5,4,7,5,5,5,6,5,5,3,8,5,5]
total_needed = sum(skip_pattern) + len(skip_pattern)
all_outputs = mt19937_cpu(known_seed, total_needed, skip=0)

draws = []
idx = 0
for skip in skip_pattern:
    idx += skip
    draws.append(all_outputs[idx] % 1000)
    idx += 1

# Save hybrid test draws
test_data = [{'draw': d, 'session': 'midday'} for d in draws]
with open('test_hybrid_module_draws.json', 'w') as f:
    json.dump(test_data, f)

# Get strategies
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

# Create hybrid job
job = {
    'job_id': 'reverse_hybrid_test',
    'dataset_path': 'test_hybrid_module_draws.json',
    'candidate_seeds': [12345],
    'window_size': 30,
    'phase2_threshold': 0.50,
    'prng_families': ['mt19937'],
    'sessions': ['midday'],
    'offset': 0,
    'hybrid': True,
    'strategies': strategies_data
}

with open('test_hybrid_job.json', 'w') as f:
    json.dump(job, f, indent=2)

print("\nRunning reverse_sieve_filter.py (hybrid mode)...")
result = subprocess.run(
    ['python3', 'reverse_sieve_filter.py', '--job-file', 'test_hybrid_job.json', '--gpu-id', '0'],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    data = json.loads(result.stdout)
    survivors = data.get('survivors', [])
    
    print(f"✅ Hybrid test passed")
    print(f"   Candidates tested: {data['candidates_tested']}")
    print(f"   Survivors found: {len(survivors)}")
    
    if survivors:
        for s in survivors:
            print(f"   Seed {s['seed']}: rate={s['match_rate']:.3f}")
            print(f"   Strategy: {s.get('strategy_name', 'unknown')}")
            print(f"   Pattern: {s.get('skip_pattern', [])[:10]}")
            
            if s['seed'] == known_seed and s['match_rate'] == 1.0:
                print(f"   🎯 PERFECT MATCH - Found known seed with variable skip!")
                test2_pass = True
            else:
                test2_pass = False
    else:
        print(f"   ⚠️ No survivors found")
        test2_pass = False
else:
    print(f"❌ Test failed with return code {result.returncode}")
    print(f"STDERR: {result.stderr}")
    test2_pass = False

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)
print(f"Fixed skip mode: {'✅ PASS' if test1_pass else '❌ FAIL'}")
print(f"Hybrid mode: {'✅ PASS' if test2_pass else '❌ FAIL'}")

if test1_pass and test2_pass:
    print(f"\n🎉 ALL TESTS PASSED - reverse_sieve_filter.py is working correctly!")
    print(f"✅ Job file interface works")
    print(f"✅ Coordinator-compatible output format")
    print(f"✅ Ready for Step 3 (ML features)")
else:
    print(f"\n⚠️ SOME TESTS FAILED - needs debugging")

