#!/usr/bin/env python3
"""
Test xorshift32 across 26 GPUs - Constant Skip
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from prng_registry import xorshift32_cpu
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

print("="*70)
print("XORSHIFT32 - 26-GPU DISTRIBUTED TEST (Constant Skip)")
print("="*70)

# Generate test data
known_seed = 12345
skip = 5
n_draws = 2000

total_outputs = n_draws * (skip + 1)
all_outputs = xorshift32_cpu(known_seed, total_outputs, skip=0)

draws = []
idx = 0
for i in range(n_draws):
    idx += skip
    draws.append(all_outputs[idx] % 1000)
    idx += 1

print(f"Known seed: {known_seed}")
print(f"Skip: {skip}")
print(f"Total draws: {len(draws)}")
print(f"First 10: {draws[:10]}")

test_data = [{'draw': d, 'session': 'midday', 'timestamp': 1000000 + i} 
             for i, d in enumerate(draws)]

with open('test_xorshift32.json', 'w') as f:
    json.dump(test_data, f)

# Copy to remote nodes
print("\nCopying test file to remote nodes...")
for host in ['192.168.3.120', '192.168.3.154']:
    result = subprocess.run(
        ['scp', 'test_xorshift32.json', 
         f'{host}:/home/michael/distributed_prng_analysis/'],
        capture_output=True
    )
    print(f"  ✅ {host}" if result.returncode == 0 else f"  ❌ {host}")

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()
print(f"\nTotal workers: {len(workers)} across {len(coordinator.nodes)} nodes")

class Args:
    target_file = 'test_xorshift32.json'
    window_size = 512
    seeds = 20000
    seed_start = 0
    threshold = 0.01
    skip_min = 0
    skip_max = 10
    offset = 0
    prng_type = 'xorshift32'
    hybrid = False
    phase2_threshold = 0.50
    gpu_id = 0

args = Args()
coordinator.current_target_file = args.target_file

print("\n" + "="*70)
print("FORWARD SIEVE - xorshift32")
print("="*70)

forward_jobs = coordinator._create_sieve_jobs(args)
print(f"Created {len(forward_jobs)} jobs")

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
            
            has_12345 = any(s['seed'] == 12345 for s in survivors)
            marker = "🎯" if has_12345 else "✅"
            print(f"{marker} [{completed}/{len(forward_jobs)}] {job_id} on {worker.node.hostname} GPU{worker.gpu_id}: {len(survivors)} survivors")
            
            if has_12345:
                known = [s for s in survivors if s['seed'] == 12345][0]
                print(f"     🎯 FOUND seed 12345! rate={known['match_rate']:.3f}, skip={known['best_skip']}")

print(f"\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Total survivors: {len(all_survivors)}")

known = [s for s in all_survivors if s['seed'] == 12345]
if known:
    print(f"\n🎯 SEED 12345 FOUND!")
    print(f"   Match rate: {known[0]['match_rate']:.3f}")
    print(f"   Best skip: {known[0]['best_skip']}")
    
    if known[0]['best_skip'] == 5:
        print(f"\n✅✅✅ XORSHIFT32 WORKS ACROSS 26 GPUs!")
        sys.exit(0)
else:
    print(f"\n❌ Seed 12345 not found")

sys.exit(1)

