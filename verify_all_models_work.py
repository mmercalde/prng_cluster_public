#!/usr/bin/env python3
"""
Verify ALL PRNG models still work after xorshift32_hybrid changes
Uses test_known.json (seed 54321, skip=1, 10 draws)
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()

# Test all PRNGs with test_known.json
prngs_to_test = [
    'mt19937',
    'xorshift32', 
    'pcg32',
    'lcg32',
    'xorshift64'
]

results = []

for prng in prngs_to_test:
    print(f"\n{'='*70}")
    print(f"Testing {prng.upper()}")
    print(f"{'='*70}")
    
    class Args:
        target_file = 'test_known.json'
        window_size = 10
        seeds = 100000
        seed_start = 0
        threshold = 0.01
        skip_min = 0
        skip_max = 5
        offset = 0
        prng_type = prng
        hybrid = False
        phase1_threshold = 0.01
        phase2_threshold = 0.50
        gpu_id = 0
    
    args = Args()
    coordinator.current_target_file = args.target_file
    
    jobs = coordinator._create_sieve_jobs(args)
    
    # Run job containing seed 54321
    found_seed = False
    for job, worker in jobs:
        seed_start, seed_end = job.seeds
        if seed_start <= 54321 < seed_end:
            result = coordinator.execute_gpu_job(job, worker)
            
            if result.success and result.results:
                survivors = result.results.get('survivors', [])
                if any(s['seed'] == 54321 for s in survivors):
                    survivor = [s for s in survivors if s['seed'] == 54321][0]
                    match_rate = survivor.get('match_rate', 0)
                    skip = survivor.get('best_skip', 'N/A')
                    print(f"✅ PASSED: Found seed 54321")
                    print(f"   Match rate: {match_rate:.1%}, Skip: {skip}")
                    results.append({'prng': prng, 'status': 'PASSED'})
                    found_seed = True
                else:
                    print(f"❌ FAILED: Seed 54321 not found")
                    print(f"   Found {len(survivors)} other seeds")
                    results.append({'prng': prng, 'status': 'FAILED'})
            else:
                print(f"❌ FAILED: Job error: {result.error}")
                results.append({'prng': prng, 'status': 'FAILED'})
            break
    
    if not found_seed and not any(r['prng'] == prng for r in results):
        print(f"❌ FAILED: No job contained seed 54321")
        results.append({'prng': prng, 'status': 'FAILED'})

print(f"\n{'='*70}")
print("VERIFICATION SUMMARY")
print(f"{'='*70}")

for r in results:
    icon = "✅" if r['status'] == 'PASSED' else "❌"
    print(f"{icon} {r['prng']}: {r['status']}")

passed = sum(1 for r in results if r['status'] == 'PASSED')
print(f"\nTotal: {passed}/{len(results)} models working")

if passed == len(results):
    print("\n🎉 ALL EXISTING MODELS STILL WORK!")
    sys.exit(0)
else:
    print("\n❌ SOME MODELS BROKEN!")
    sys.exit(1)

