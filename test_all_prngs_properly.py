#!/usr/bin/env python3
"""
Test all PRNGs with their correct test data
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from prng_registry import mt19937_cpu, xorshift32_cpu
import json

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()

# Define tests with CORRECT data for each PRNG
tests = [
    {
        'name': 'MT19937 - Constant Skip',
        'prng': 'mt19937',
        'file': 'test_known.json',  # Already exists, seed 54321, skip 1
        'seed': 54321,
        'window': 10,
        'hybrid': False,
        'skip_max': 5
    },
    {
        'name': 'Xorshift32 - Variable Skip (Hybrid)',
        'prng': 'xorshift32_hybrid',
        'file': 'test_xorshift32_hybrid_dist.json',  # seed 54321, variable skip
        'seed': 54321,
        'window': 512,
        'hybrid': True,
        'skip_max': 10
    }
]

# Create xorshift32 constant skip test data
print("Creating xorshift32 constant skip test data...")
seed_xor = 12345
skip_xor = 5
n_draws = 100

all_outputs = xorshift32_cpu(seed_xor, n_draws * (skip_xor + 1), skip=0)
draws_xor = []
idx = 0
for i in range(n_draws):
    idx += skip_xor
    draws_xor.append(all_outputs[idx] % 1000)
    idx += 1

test_data_xor = [{'draw': d, 'session': 'midday', 'timestamp': 5000000 + i} 
                 for i, d in enumerate(draws_xor)]

with open('test_xorshift32_const_skip5.json', 'w') as f:
    json.dump(test_data_xor, f)

print(f"Created test_xorshift32_const_skip5.json: seed={seed_xor}, skip={skip_xor}, draws={n_draws}")
print(f"First 10: {draws_xor[:10]}")

tests.append({
    'name': 'Xorshift32 - Constant Skip',
    'prng': 'xorshift32',
    'file': 'test_xorshift32_const_skip5.json',
    'seed': seed_xor,
    'window': 100,
    'hybrid': False,
    'skip_max': 10
})

results = []

for test in tests:
    print(f"\n{'='*70}")
    print(f"TEST: {test['name']}")
    print(f"{'='*70}")
    print(f"PRNG: {test['prng']}, Seed: {test['seed']}, File: {test['file']}")
    
    class Args:
        target_file = test['file']
        window_size = test['window']
        seeds = 100000
        seed_start = 0
        threshold = 0.01 if not test['hybrid'] else 0.50
        skip_min = 0
        skip_max = test['skip_max']
        offset = 0
        prng_type = test['prng']
        hybrid = test['hybrid']
        phase1_threshold = 0.01
        phase2_threshold = 0.50
        gpu_id = 0
    
    args = Args()
    coordinator.current_target_file = args.target_file
    
    jobs = coordinator._create_sieve_jobs(args)
    print(f"Created {len(jobs)} jobs")
    
    # Find and run job with seed
    found = False
    for job, worker in jobs:
        seed_start, seed_end = job.seeds
        if seed_start <= test['seed'] < seed_end:
            print(f"Running job {job.job_id} on {worker.node.hostname} GPU{worker.gpu_id}")
            result = coordinator.execute_gpu_job(job, worker)
            
            if result.success and result.results:
                survivors = result.results.get('survivors', [])
                match = [s for s in survivors if s['seed'] == test['seed']]
                
                if match:
                    print(f"✅ PASSED - Found seed {test['seed']}")
                    print(f"   Match rate: {match[0].get('match_rate', 0):.1%}")
                    if 'best_skip' in match[0]:
                        print(f"   Skip: {match[0]['best_skip']}")
                    if 'skip_pattern' in match[0]:
                        print(f"   Pattern (first 10): {match[0]['skip_pattern'][:10]}")
                    results.append({'test': test['name'], 'status': 'PASSED'})
                    found = True
                else:
                    print(f"❌ FAILED - Seed not found ({len(survivors)} other survivors)")
                    results.append({'test': test['name'], 'status': 'FAILED'})
                    found = True
            else:
                print(f"❌ FAILED - Job error: {result.error}")
                results.append({'test': test['name'], 'status': 'FAILED'})
                found = True
            break
    
    if not found:
        print(f"❌ FAILED - No job contained seed {test['seed']}")
        results.append({'test': test['name'], 'status': 'FAILED'})

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

for r in results:
    icon = "✅" if r['status'] == 'PASSED' else "❌"
    print(f"{icon} {r['test']}")

passed = sum(1 for r in results if r['status'] == 'PASSED')
print(f"\nPassed: {passed}/{len(results)}")

sys.exit(0 if passed == len(results) else 1)

