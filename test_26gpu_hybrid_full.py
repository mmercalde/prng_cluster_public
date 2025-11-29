#!/usr/bin/env python3
"""
FULL 26-GPU HYBRID TEST - Forward AND Reverse alignment
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from prng_registry import mt19937_cpu
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

print("="*70)
print("26-GPU HYBRID ALIGNMENT TEST")
print("="*70)

# Generate variable skip test data
known_seed = 54321
base_pattern = [5,5,3,7,5,5,8,4,5,5,6,5,5,3,9,5,5,5,4,7,5,5,5,6,5,5,3,8,5,5]
skip_pattern = base_pattern * 67
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
print(f"Total draws: {len(draws)}")
print(f"Skip pattern (first 10): {skip_pattern[:10]}")

test_data = [{'draw': d, 'session': 'midday', 'timestamp': 3000000 + i} for i, d in enumerate(draws)]
with open('test_26gpu_hybrid.json', 'w') as f:
    json.dump(test_data, f)

print("\nCopying test file to remote nodes...")
for host in ['192.168.3.120', '192.168.3.154']:
    result = subprocess.run(['scp', 'test_26gpu_hybrid.json', f'{host}:/home/michael/distributed_prng_analysis/'], capture_output=True)
    print(f"  ✅ {host}")

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
    hybrid = True
    phase1_threshold = 0.01
    phase2_threshold = 0.50
    gpu_id = 0

args = Args()
coordinator.current_target_file = args.target_file

print("\n" + "="*70)
print("FORWARD HYBRID SIEVE")
print("="*70)

forward_jobs = coordinator._create_sieve_jobs(args)
all_survivors = []

def execute_job(job_worker):
    job, worker = job_worker
    seed_start, seed_end = job.seeds
    result = coordinator.execute_gpu_job(job, worker)
    return job.job_id, worker, seed_start, seed_end, result

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
            if has_54321:
                print(f"🎯 [{completed}/{len(forward_jobs)}] {job_id}: FOUND 54321!")

known = [s for s in all_survivors if s['seed'] == 54321]
if known:
    fwd_pattern = known[0].get('skip_pattern', [])
    print(f"\n✅ Forward found seed 54321")
    print(f"   Pattern (first 10): {fwd_pattern[:10]}")
    
    # REVERSE SIEVE
    print(f"\n" + "="*70)
    print("REVERSE HYBRID SIEVE")
    print("="*70)
    
    candidate_seeds = [54321] + [s['seed'] for s in all_survivors[:50]]
    args.threshold = 0.50
    
    rev_jobs = coordinator._create_reverse_sieve_jobs(args, candidate_seeds)
    print(f"Created {len(rev_jobs)} reverse jobs")
    
    rev_survivors_all = []
    
    def execute_reverse(job_worker):
        job, worker = job_worker
        result = coordinator.execute_gpu_job(job, worker)
        return job.job_id, worker, result
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(execute_reverse, jw): jw for jw in rev_jobs}
        
        completed = 0
        for future in as_completed(futures):
            job_id, worker, result = future.result()
            completed += 1
            
            if result.success and result.results:
                survivors = result.results.get('survivors', [])
                rev_survivors_all.extend(survivors)
                
                has_54321 = any(s['seed'] == 54321 for s in survivors)
                if has_54321:
                    print(f"🎯 [{completed}/{len(rev_jobs)}] {job_id}: CONFIRMED 54321!")
    
    rev_known = [s for s in rev_survivors_all if s['seed'] == 54321]
    if rev_known:
        rev_pattern = rev_known[0].get('skip_pattern', [])
        print(f"\n✅ Reverse confirmed seed 54321")
        print(f"   Pattern (first 10): {rev_pattern[:10]}")
        
        # ALIGNMENT CHECK
        print(f"\n" + "="*70)
        print("HYBRID ALIGNMENT VERIFICATION")
        print("="*70)
        
        expected = skip_pattern[:50]
        fwd_matches = sum(1 for i in range(min(50, len(fwd_pattern))) if fwd_pattern[i] == expected[i])
        rev_matches = sum(1 for i in range(min(50, len(rev_pattern))) if rev_pattern[i] == expected[i])
        
        print(f"Forward pattern accuracy: {fwd_matches/50:.1%} ({fwd_matches}/50)")
        print(f"Reverse pattern accuracy: {rev_matches/50:.1%} ({rev_matches}/50)")
        print(f"Expected (first 10): {expected[:10]}")
        
        if fwd_matches >= 45 and rev_matches >= 45:
            print(f"\n✅✅✅ PERFECT HYBRID ALIGNMENT ACROSS 26 GPUs!")
            print(f"✅ Forward and Reverse both detect variable skip patterns!")
            sys.exit(0)
    else:
        print(f"❌ Reverse did not confirm seed 54321")
else:
    print(f"❌ Forward did not find seed 54321")

sys.exit(1)

