#!/usr/bin/env python3
"""
Comprehensive test of MT19937 and Xorshift32 - all modes
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from prng_registry import mt19937_cpu, xorshift32_cpu
import json

print("="*70)
print("COMPREHENSIVE PRNG TESTING")
print("="*70)

# Test configurations
tests = [
    {
        'name': 'MT19937 - Constant Skip (Forward)',
        'prng': 'mt19937',
        'seed': 12345,
        'skip_type': 'constant',
        'skip_value': 5,
        'n_draws': 100,
        'hybrid': False,
        'cpu_func': mt19937_cpu
    },
    {
        'name': 'MT19937 - Variable Skip (Forward Hybrid)',
        'prng': 'mt19937_hybrid',
        'seed': 54321,
        'skip_type': 'variable',
        'skip_pattern': [5,5,3,7,5,5,8,4,5,5],
        'n_draws': 100,  # 10 * 10
        'hybrid': True,
        'cpu_func': mt19937_cpu
    },
    {
        'name': 'Xorshift32 - Constant Skip (Forward)',
        'prng': 'xorshift32',
        'seed': 98765,
        'skip_type': 'constant',
        'skip_value': 7,
        'n_draws': 100,
        'hybrid': False,
        'cpu_func': xorshift32_cpu
    },
    {
        'name': 'Xorshift32 - Variable Skip (Forward Hybrid)',
        'prng': 'xorshift32_hybrid',
        'seed': 54321,
        'skip_type': 'variable',
        'skip_pattern': [5,5,3,7,5,5,8,4,5,5],
        'n_draws': 100,
        'hybrid': True,
        'cpu_func': xorshift32_cpu
    }
]

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()

results = []

for test_config in tests:
    print(f"\n{'='*70}")
    print(f"TEST: {test_config['name']}")
    print(f"{'='*70}")
    
    # Generate test data
    seed = test_config['seed']
    cpu_func = test_config['cpu_func']
    
    if test_config['skip_type'] == 'constant':
        skip = test_config['skip_value']
        n_draws = test_config['n_draws']
        
        total_outputs = n_draws * (skip + 1)
        all_outputs = cpu_func(seed, total_outputs, skip=0)
        
        draws = []
        idx = 0
        for i in range(n_draws):
            idx += skip
            draws.append(all_outputs[idx] % 1000)
            idx += 1
            
        print(f"Generated {n_draws} draws, constant skip={skip}")
        
    else:  # variable
        base_pattern = test_config['skip_pattern']
        skip_pattern = base_pattern * 10  # 100 draws
        
        total_needed = sum(skip_pattern) + len(skip_pattern)
        all_outputs = cpu_func(seed, total_needed, skip=0)
        
        draws = []
        idx = 0
        for skip in skip_pattern:
            idx += skip
            draws.append(all_outputs[idx] % 1000)
            idx += 1
            
        print(f"Generated {len(draws)} draws, variable pattern={base_pattern}")
    
    print(f"First 10 draws: {draws[:10]}")
    print(f"Known seed: {seed}")
    
    # Create test file
    test_data = [{'draw': d, 'session': 'midday', 'timestamp': 5000000 + i} 
                 for i, d in enumerate(draws)]
    
    test_file = f"test_{test_config['prng']}_{'var' if test_config['skip_type'] == 'variable' else 'const'}.json"
    with open(test_file, 'w') as f:
        json.dump(test_data, f)
    
    # Configure job
    class Args:
        target_file = test_file
        window_size = 512
        seeds = 100000
        seed_start = 0
        threshold = 0.01
        skip_min = 0
        skip_max = 15
        offset = 0
        prng_type = test_config['prng']
        hybrid = test_config['hybrid']
        phase1_threshold = 0.01
        phase2_threshold = 0.50
        gpu_id = 0
    
    args = Args()
    coordinator.current_target_file = args.target_file
    
    # Run forward sieve
    print(f"\nRunning forward sieve...")
    jobs = coordinator._create_sieve_jobs(args)
    
    # Find job with our seed
    target_job = None
    target_worker = None
    for job, worker in jobs:
        seed_start, seed_end = job.seeds
        if seed_start <= seed < seed_end:
            target_job = job
            target_worker = worker
            break
    
    if not target_job:
        print(f"❌ Seed {seed} not in any job range!")
        results.append({'test': test_config['name'], 'status': 'FAILED', 'reason': 'seed not in range'})
        continue
    
    result = coordinator.execute_gpu_job(target_job, target_worker)
    
    if not result.success:
        print(f"❌ Job failed: {result.error}")
        results.append({'test': test_config['name'], 'status': 'FAILED', 'reason': result.error})
        continue
    
    survivors = result.results.get('survivors', [])
    found = any(s['seed'] == seed for s in survivors)
    
    if found:
        survivor = [s for s in survivors if s['seed'] == seed][0]
        match_rate = survivor.get('match_rate', 0)
        print(f"✅ FORWARD: Found seed {seed}, match rate: {match_rate:.1%}")
        
        # Test reverse sieve
        print(f"\nRunning reverse sieve...")
        args.threshold = 0.50
        rev_jobs = coordinator._create_reverse_sieve_jobs(args, [seed])
        
        if rev_jobs:
            rev_job, rev_worker = rev_jobs[0]
            rev_result = coordinator.execute_gpu_job(rev_job, rev_worker)
            
            if rev_result.success:
                rev_survivors = rev_result.results.get('survivors', [])
                rev_found = any(s['seed'] == seed for s in rev_survivors)
                
                if rev_found:
                    rev_survivor = [s for s in rev_survivors if s['seed'] == seed][0]
                    rev_match = rev_survivor.get('match_rate', 0)
                    print(f"✅ REVERSE: Confirmed seed {seed}, match rate: {rev_match:.1%}")
                    results.append({
                        'test': test_config['name'],
                        'status': 'PASSED',
                        'forward_match': match_rate,
                        'reverse_match': rev_match
                    })
                else:
                    print(f"❌ REVERSE: Did not confirm seed {seed}")
                    results.append({
                        'test': test_config['name'],
                        'status': 'PARTIAL',
                        'forward_match': match_rate,
                        'reverse_match': 0
                    })
            else:
                print(f"❌ REVERSE: Job failed: {rev_result.error}")
                results.append({
                    'test': test_config['name'],
                    'status': 'PARTIAL',
                    'forward_match': match_rate,
                    'reason': 'reverse failed'
                })
    else:
        print(f"❌ FORWARD: Seed {seed} not found")
        results.append({'test': test_config['name'], 'status': 'FAILED', 'reason': 'seed not found'})

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

for r in results:
    status_icon = "✅" if r['status'] == 'PASSED' else "⚠️" if r['status'] == 'PARTIAL' else "❌"
    print(f"{status_icon} {r['test']}: {r['status']}")
    if 'forward_match' in r:
        print(f"   Forward: {r['forward_match']:.1%}, Reverse: {r.get('reverse_match', 0):.1%}")

passed = sum(1 for r in results if r['status'] == 'PASSED')
print(f"\nTotal: {passed}/{len(results)} tests passed")

sys.exit(0 if passed == len(results) else 1)

