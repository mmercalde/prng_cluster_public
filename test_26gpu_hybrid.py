#!/usr/bin/env python3
"""
Test VARIABLE SKIP (hybrid mode) across all 26 GPUs
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from prng_registry import mt19937_cpu
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

print("="*70)
print("26-GPU HYBRID (VARIABLE SKIP) TEST")
print("="*70)

# Generate test data with VARIABLE skip
known_seed = 54321
base_pattern = [5,5,3,7,5,5,8,4,5,5,6,5,5,3,9,5,5,5,4,7,5,5,5,6,5,5,3,8,5,5]
skip_pattern = base_pattern * 67  # 2010 skips total
k = len(skip_pattern)

total_needed = sum(skip_pattern) + k
all_outputs = mt19937_cpu(known_seed, total_needed, skip=0)

draws = []
idx = 0
for skip in skip_pattern:
    idx += skip
    draws.append(all_outputs[idx] % 1000)
    idx += 1

print(f"Known seed: {known_seed}")
print(f"Skip pattern (first 10): {skip_pattern[:10]}")
print(f"Average skip: {sum(skip_pattern)/len(skip_pattern):.1f}")
print(f"Total draws: {len(draws)}")
print(f"First 10: {draws[:10]}")

test_data = [{'draw': d, 'session': 'midday', 'timestamp': 3000000 + i} for i, d in enumerate(draws)]
with open('test_26gpu_hybrid.json', 'w') as f:
    json.dump(test_data, f)

# Copy to remote nodes
print("\nCopying test file to remote nodes...")
for host in ['192.168.3.120', '192.168.3.154']:
    result = subprocess.run(['scp', 'test_26gpu_hybrid.json', f'{host}:/home/michael/distributed_prng_analysis/'], capture_output=True)
    print(f"  ✅ {host}" if result.returncode == 0 else f"  ❌ {host}")

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()

class Args:
    target_file = 'test_26gpu_hybrid.json'
    window_size = 512
    seeds = 100000
    seed_start = 0
    threshold = 0.01
    skip_min = 0
    skip_max = 10
    offset = 0
    prng_type = 'mt19937'
    hybrid = True  # HYBRID MODE!
    phase1_threshold = 0.01
    phase2_threshold = 0.50
    gpu_id = 0

args = Args()
coordinator.current_target_file = args.target_file

print("\n" + "="*70)
print("FORWARD HYBRID SIEVE - ALL 26 GPUs")
print("="*70)

forward_jobs = coordinator._create_sieve_jobs(args)
print(f"Created {len(forward_jobs)} jobs")

all_survivors = []

def execute_job(job_worker):
    job, worker = job_worker
    seed_start, seed_end = job.seeds
    result = coordinator.execute_gpu_job(job, worker)
    return job.job_id, worker, seed_start, seed_end, result

print(f"\n🚀 Executing all {len(forward_jobs)} jobs...")

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(execute_job, jw): jw for jw in forward_jobs}
    
    completed = 0
    for future in as_completed(futures):
        job_id, worker, seed_start, seed_end, result = future.result()
        completed += 1
        
        if result.success and result.results:
            survivors = result.results.get('survivors', [])
            all_survivors.extend(survivors)
            
            has_54321 = any(s['seed'] == 54321 for s in survivors)
            marker = "🎯" if has_54321 else "✅"
            print(f"{marker} [{completed}/{len(forward_jobs)}] {job_id} on {worker.node.hostname} GPU{worker.gpu_id}: {len(survivors)} survivors [{seed_start}-{seed_end})")
            
            if has_54321:
                known = [s for s in survivors if s['seed'] == 54321][0]
                pattern = known.get('skip_pattern', [])
                print(f"     🎯 FOUND SEED 54321! Pattern (first 10): {pattern[:10]}")
        else:
            print(f"❌ [{completed}/{len(forward_jobs)}] {job_id}: {result.error}")

print(f"\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Total survivors: {len(all_survivors)}")

known = [s for s in all_survivors if s['seed'] == 54321]
if known:
    pattern = known[0].get('skip_pattern', [])
    expected = skip_pattern[:50]
    matches = sum(1 for i in range(min(50, len(pattern))) if pattern[i] == expected[i])
    
    print(f"\n🎯 SEED 54321 FOUND!")
    print(f"   Pattern (first 10): {pattern[:10]}")
    print(f"   Expected (first 10): {expected[:10]}")
    print(f"   Accuracy: {matches/50:.1%} ({matches}/50)")
    
    if matches >= 45:
        print(f"\n✅✅✅ HYBRID MODE WORKING ACROSS 26 GPUs!")
        sys.exit(0)
else:
    print(f"\n❌ Seed 54321 not found")
    print(f"Top survivors: {[(s['seed'], s.get('match_rate', 0)) for s in all_survivors[:5]]}")

sys.exit(1)

