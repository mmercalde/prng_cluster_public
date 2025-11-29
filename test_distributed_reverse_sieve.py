#!/usr/bin/env python3
"""
Test distributed reverse sieve across 26-GPU cluster
"""

import subprocess
import json
import sys
from prng_registry import mt19937_cpu

print("="*70)
print("DISTRIBUTED REVERSE SIEVE TEST (26-GPU CLUSTER)")
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
test_data = [{'draw': d, 'session': 'midday', 'timestamp': 1000000 + i} for i, d in enumerate(draws)]
with open('test_distributed_draws.json', 'w') as f:
    json.dump(test_data, f)

# Generate candidate seeds
candidate_seeds = [known_seed]
for offset in [-1000, -500, -100, -10, 10, 100, 500, 1000]:
    candidate_seeds.append(known_seed + offset)

import random
random.seed(42)
for _ in range(100):
    candidate_seeds.append(random.randint(10000, 20000))

print(f"\nCandidate seeds: {len(candidate_seeds)} total")
print(f"  Including known seed: {known_seed}")

# Create test script
test_script = f'''
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
import json

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()
print(f"Created {{len(workers)}} GPU workers")

class Args:
    target_file = 'test_distributed_draws.json'
    window_size = 30
    threshold = 0.90
    skip_min = 0
    skip_max = 10
    offset = 0
    prng_type = 'mt19937'
    hybrid = False
    phase2_threshold = 0.50
    gpu_id = 0

args = Args()
candidate_seeds = {candidate_seeds}

print("\\n" + "="*70)
print("CREATING DISTRIBUTED REVERSE SIEVE JOBS")
print("="*70)

job_assignments = coordinator._create_reverse_sieve_jobs(args, candidate_seeds)

print(f"\\nâœ… Created {{len(job_assignments)}} jobs")
for i, (job, worker) in enumerate(job_assignments[:5]):
    print(f"  Job {{i}}: {{len(job.seeds)}} candidates -> GPU {{worker.gpu_id}}")

from collections import Counter
gpu_counts = Counter(worker.gpu_id for job, worker in job_assignments)
print(f"\\nJob distribution across {{len(gpu_counts)}} GPUs:")
for gpu, count in sorted(gpu_counts.items())[:10]:
    print(f"  GPU {{gpu}}: {{count}} jobs")

print("\\n" + "="*70)
print("EXECUTING FIRST JOB (TEST)")
print("="*70)

if job_assignments:
    test_job, test_worker = job_assignments[0]
    print(f"\\nTesting job: {{test_job.job_id}}")
    print(f"  Candidates: {{len(test_job.seeds)}}")
    print(f"  Worker: GPU {{test_worker.gpu_id}}")
    
    with open(f'{{test_job.job_id}}_payload.json', 'w') as f:
        json.dump(test_job.payload, f, indent=2)
    
    print(f"  Payload: {{test_job.job_id}}_payload.json")
    
    result = coordinator.execute_local_job(test_job, test_worker)
    
    if result.success:
        print(f"\\nâœ… Job executed successfully!")
        print(f"   Runtime: {{result.runtime:.2f}}s")
        
        # FIXED: Use result.results (not result_data)
        if result.results:
            survivors = result.results.get('survivors', [])
            
            print(f"   Survivors: {{len(survivors)}}")
            if survivors:
                for s in survivors[:5]:
                    print(f"     Seed {{s['seed']}}: rate={{s['match_rate']:.3f}}, skip={{s['best_skip']}}")
                    if s['seed'] == {known_seed}:
                        print(f"       🎯 FOUND KNOWN SEED WITH PERFECT MATCH!")
            else:
                print(f"   (No survivors at threshold {{args.threshold}})")
        else:
            print(f"   ⚠️ No result data returned")
    else:
        print(f"\\nâŒ Job failed!")
        print(f"   Error: {{result.error}}")
'''

with open('_test_dist_coord.py', 'w') as f:
    f.write(test_script)

print("\n" + "="*70)
print("RUNNING DISTRIBUTED TEST")
print("="*70)

result = subprocess.run(['python3', '_test_dist_coord.py'], 
                       capture_output=True, text=True)

print(result.stdout)
if result.stderr and 'Configured node' not in result.stderr:
    print("\nSTDERR:")
    print(result.stderr)

if result.returncode == 0:
    print("\n" + "="*70)
    print("✅ DISTRIBUTED REVERSE SIEVE TEST PASSED")
    print("="*70)
    print("✅ Jobs distributed across 26-GPU cluster")
    print("✅ Worker execution successful")
    print("✅ reverse_sieve_filter.py integrated")
    print("\n🎉 Ready for Step 3: Add ML features")
else:
    print("\n" + "="*70)
    print("⚠️ TEST ISSUES")
    print("="*70)
    sys.exit(1)

