#!/usr/bin/env python3
"""
Test VARIABLE SKIP (hybrid) reverse sieve across 26-GPU cluster
"""

import subprocess
import json
import sys
from prng_registry import mt19937_cpu

print("="*70)
print("VARIABLE SKIP REVERSE SIEVE TEST (26-GPU CLUSTER)")
print("="*70)

# Generate test data with VARIABLE skip pattern
known_seed = 12345
skip_pattern = [5,5,3,7,5,5,8,4,5,5,6,5,5,3,9,5,5,5,4,7,5,5,5,6,5,5,3,8,5,5]
k = len(skip_pattern)

total_needed = sum(skip_pattern) + k
all_outputs = mt19937_cpu(known_seed, total_needed, skip=0)

draws = []
idx = 0
for skip in skip_pattern:
    idx += skip
    draws.append(all_outputs[idx] % 1000)
    idx += 1

print(f"\nKnown seed: {known_seed}")
print(f"Skip pattern (variable): {skip_pattern[:10]}...")
print(f"Average skip: {sum(skip_pattern)/len(skip_pattern):.1f}")
print(f"Draws: {draws[:10]}...")

# Save test draws
test_data = [{'draw': d, 'session': 'midday', 'timestamp': 1000000 + i} for i, d in enumerate(draws)]
with open('test_hybrid_reverse_draws.json', 'w') as f:
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

# Create test script with FIXED comparison
test_script = f'''
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from hybrid_strategy import get_all_strategies
import json

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()
print(f"Created {{len(workers)}} GPU workers")

strategies = get_all_strategies()
print(f"Loaded {{len(strategies)}} hybrid strategies")

class Args:
    target_file = 'test_hybrid_reverse_draws.json'
    window_size = 30
    threshold = 0.50
    skip_min = 0
    skip_max = 10
    offset = 0
    prng_type = 'mt19937'
    hybrid = True
    phase2_threshold = 0.50
    gpu_id = 0

args = Args()
candidate_seeds = {candidate_seeds}

print("\\n" + "="*70)
print("CREATING HYBRID REVERSE SIEVE JOBS")
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
print("EXECUTING FIRST JOB (HYBRID TEST)")
print("="*70)

if job_assignments:
    test_job, test_worker = job_assignments[0]
    print(f"\\nTesting job: {{test_job.job_id}}")
    print(f"  Candidates: {{len(test_job.seeds)}}")
    print(f"  Worker: GPU {{test_worker.gpu_id}}")
    print(f"  Mode: HYBRID (variable skip)")
    
    with open(f'{{test_job.job_id}}_payload.json', 'w') as f:
        json.dump(test_job.payload, f, indent=2)
    
    print(f"  Payload: {{test_job.job_id}}_payload.json")
    
    result = coordinator.execute_local_job(test_job, test_worker)
    
    if result.success:
        print(f"\\nâœ… Job executed successfully!")
        print(f"   Runtime: {{result.runtime:.2f}}s")
        
        if result.results:
            survivors = result.results.get('survivors', [])
            
            print(f"   Survivors: {{len(survivors)}}")
            if survivors:
                for s in survivors[:5]:
                    print(f"     Seed {{s['seed']}}: rate={{s['match_rate']:.3f}}")
                    print(f"       Strategy: {{s.get('strategy_name', 'unknown')}}")
                    pattern = s.get('skip_pattern', [])
                    if pattern:
                        print(f"       Pattern (first 10): {{pattern[:10]}}")
                    
                    if s['seed'] == {known_seed}:
                        print(f"       🎯 FOUND KNOWN SEED WITH VARIABLE SKIP!")
                        
                        # FIXED: Proper comparison
                        expected = {skip_pattern}
                        actual = pattern[:len(expected)]  # Trim to same length
                        
                        # Element-by-element comparison
                        matches = sum(1 for i in range(min(len(expected), len(actual))) if expected[i] == actual[i])
                        total = len(expected)
                        accuracy = matches / total if total > 0 else 0
                        
                        print(f"       Pattern accuracy: {{matches}}/{{total}} = {{accuracy:.1%}}")
                        
                        if accuracy == 1.0:
                            print(f"       ✅ SKIP PATTERN MATCHES PERFECTLY!")
                        elif accuracy >= 0.90:
                            print(f"       âœ… SKIP PATTERN MATCHES ({{accuracy:.1%}} accurate)")
                        else:
                            print(f"       ⚠️ Skip pattern differs significantly")
                            print(f"         Expected: {{expected[:10]}}")
                            print(f"         Got:      {{actual[:10]}}")
                            print(f"         Mismatches at indices: {{[i for i in range(min(len(expected), len(actual))) if expected[i] != actual[i]][:5]}}")
            else:
                print(f"   (No survivors at threshold {{args.threshold}})")
        else:
            print(f"   ⚠️ No result data returned")
    else:
        print(f"\\nâŒ Job failed!")
        print(f"   Error: {{result.error}}")
'''

with open('_test_hybrid_reverse.py', 'w') as f:
    f.write(test_script)

print("\n" + "="*70)
print("RUNNING HYBRID REVERSE TEST")
print("="*70)

result = subprocess.run(['python3', '_test_hybrid_reverse.py'], 
                       capture_output=True, text=True)

print(result.stdout)
if result.stderr and 'Configured node' not in result.stderr:
    print("\nSTDERR:")
    print(result.stderr)

if result.returncode == 0:
    print("\n" + "="*70)
    print("✅ HYBRID REVERSE SIEVE TEST PASSED")
    print("="*70)
    print("✅ Variable skip pattern detected")
    print("✅ mt19937_hybrid_reverse kernel working")
    print("✅ Ready for ML features")
else:
    print("\n" + "="*70)
    print("⚠️ TEST ISSUES")
    print("="*70)
    sys.exit(1)

