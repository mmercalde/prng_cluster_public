#!/usr/bin/env python3
"""
Test ALL PRNGs across the full 26-GPU distributed cluster
Generate separate test data for each PRNG
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from prng_registry import KERNEL_REGISTRY
import json

print("="*70)
print("26-GPU DISTRIBUTED MULTI-PRNG TEST")
print("="*70)

# Test each PRNG with its own generated data
test_prngs = ['xorshift32', 'pcg32', 'lcg32', 'mt19937', 'xorshift64']

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()

print(f"✅ {len(workers)} GPUs ready across 3 nodes")
print("="*70)

results = {}
import subprocess

for prng in test_prngs:
    print(f"\n🔬 Testing {prng} across 26 GPUs...")
    
    # Get the CPU reference function for this PRNG
    prng_config = KERNEL_REGISTRY.get(prng)
    if not prng_config or 'cpu_reference' not in prng_config:
        print(f"   ⚠️ No CPU reference for {prng}, skipping...")
        continue
    
    cpu_func = prng_config['cpu_reference']
    
    # Generate test data using this PRNG
    known_seed = 88675
    skip = 5
    n_draws = 2000
    total_outputs = n_draws * (skip + 1)
    
    all_outputs = cpu_func(known_seed, total_outputs, skip=0)
    
    draws = []
    for i in range(n_draws):
        idx = i * (skip + 1) + skip
        draws.append(all_outputs[idx] % 1000)
    
    test_data = [{'draw': d, 'session': 'midday', 'timestamp': 3000000 + i} for i, d in enumerate(draws)]
    
    # Save test file
    test_file = f'test_multi_prng_{prng}.json'
    with open(test_file, 'w') as f:
        json.dump(test_data, f)
    
    print(f"   Generated {len(draws)} draws using {prng}")
    
    # Copy to remote nodes
    for host in ['192.168.3.120', '192.168.3.154']:
        subprocess.run(['scp', test_file, f'{host}:/home/michael/distributed_prng_analysis/'], 
                       capture_output=True)
    
    # Create jobs for this PRNG
    class Args:
        target_file = test_file
        window_size = 512
        seeds = 100000
        seed_start = 0
        threshold = 0.01
        skip_min = 0
        skip_max = 10
        offset = 0
        prng_type = prng  # TEST THIS PRNG
        hybrid = False
        phase1_threshold = 0.01
        phase2_threshold = 0.50
        gpu_id = 0
    
    args = Args()
    coordinator.current_target_file = args.target_file
    
    jobs = coordinator._create_sieve_jobs(args)
    print(f"   Created {len(jobs)} jobs")
    
    # Execute jobs
    survivors = []
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def execute_job(job_worker):
        job, worker = job_worker
        result = coordinator.execute_gpu_job(job, worker)
        return result
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(execute_job, jw): jw for jw in jobs}
        
        for future in as_completed(futures):
            result = future.result()
            if result.success and result.results:
                job_survivors = result.results.get('survivors', [])
                survivors.extend(job_survivors)
    
    # Check if seed 12345 was found
    found_12345 = any(s['seed'] == known_seed for s in survivors)
    
    if found_12345:
        seed_12345 = [s for s in survivors if s['seed'] == known_seed][0]
        detected_skip = seed_12345.get('best_skip', -1)
        print(f"   ✅ Found seed {known_seed} with skip={detected_skip}")
        results[prng] = 'PASS' if detected_skip == 5 else f'WRONG_SKIP({detected_skip})'
    else:
        print(f"   ❌ Did NOT find seed {known_seed}")
        results[prng] = 'FAIL'

print("\n" + "="*70)
print("FINAL RESULTS - 26-GPU MULTI-PRNG TEST")
print("="*70)
for prng, status in results.items():
    emoji = "✅" if status == "PASS" else "❌"
    print(f"{emoji} {prng:20} {status}")

passed = sum(1 for s in results.values() if s == 'PASS')
print(f"\nPassed: {passed}/{len(results)}")

if passed == len(results):
    print("\n🎉🎉🎉 ALL PRNGs WORK ACROSS 26 GPUs!")
    print("✅ PRNG configurability fully verified!")
    sys.exit(0)
else:
    print("\n⚠️ Some PRNGs failed")
    sys.exit(1)

