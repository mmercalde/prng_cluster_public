#!/usr/bin/env python3
"""
FULL 26-GPU ALIGNMENT TEST - Execute ALL jobs, show ALL results
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from prng_registry import mt19937_cpu
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

print("="*70)
print("FULL 26-GPU ALIGNMENT TEST")
print("="*70)

# Generate LARGE test data
known_seed = 12345
constant_skip = 5
k = 2000

total_needed = k * (constant_skip + 1)
all_outputs = mt19937_cpu(known_seed, total_needed, skip=0)

draws = []
idx = 0
for i in range(k):
    idx += constant_skip
    draws.append(all_outputs[idx] % 1000)
    idx += 1

print(f"Known seed: {known_seed}")
print(f"Skip: {constant_skip}")
print(f"Total draws: {len(draws)}")
print(f"First 10: {draws[:10]}")

test_data = [{'draw': d, 'session': 'midday', 'timestamp': 1000000 + i} for i, d in enumerate(draws)]
with open('test_26gpu_large.json', 'w') as f:
    json.dump(test_data, f)

print("\nCopying test file to remote nodes...")
remote_nodes = [
    ('192.168.3.120', '/home/michael/distributed_prng_analysis/'),
    ('192.168.3.154', '/home/michael/distributed_prng_analysis/')
]

for host, path in remote_nodes:
    print(f"  Copying to {host}...")
    result = subprocess.run(
        ['scp', 'test_26gpu_large.json', f'{host}:{path}'],
        capture_output=True
    )
    if result.returncode == 0:
        print(f"    ✅ {host}")

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()
print(f"\nTotal workers: {len(workers)} across {len(coordinator.nodes)} nodes")

class Args:
    target_file = 'test_26gpu_large.json'
    window_size = 512
    seeds = 20000
    seed_start = 0
    threshold = 0.01
    skip_min = 0
    skip_max = 10
    offset = 0
    prng_type = 'mt19937'
    hybrid = False
    phase2_threshold = 0.50
    gpu_id = 0

args = Args()
coordinator.current_target_file = args.target_file

print("\n" + "="*70)
print(f"FORWARD SIEVE - ALL {len(workers)} GPUs")
print(f"Threshold: {args.threshold*100}%, Window: {args.window_size} draws")
print("="*70)

forward_jobs = coordinator._create_sieve_jobs(args)
print(f"Created {len(forward_jobs)} jobs")

from collections import Counter
node_dist = Counter(w.node.hostname for j, w in forward_jobs)
print(f"Job distribution:")
for node, count in node_dist.items():
    print(f"  {node}: {count} jobs")

print(f"\n🚀 Executing all {len(forward_jobs)} jobs across 26 GPUs...")

all_survivors = []

def execute_forward_job(job_worker):
    job, worker = job_worker
    seed_start, seed_end = job.seeds
    result = coordinator.execute_gpu_job(job, worker)
    return job.job_id, worker, seed_start, seed_end, result

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(execute_forward_job, jw): jw for jw in forward_jobs}
    
    completed = 0
    for future in as_completed(futures):
        job_id, worker, seed_start, seed_end, result = future.result()
        completed += 1
        
        if result.success and result.results:
            survivors = result.results.get('survivors', [])
            all_survivors.extend(survivors)
            
            has_12345 = any(s['seed'] == 12345 for s in survivors)
            marker = "🎯" if has_12345 else "✅"
            print(f"{marker} [{completed}/{len(forward_jobs)}] {job_id} on {worker.node.hostname} GPU{worker.gpu_id}: {len(survivors)} survivors [{seed_start}-{seed_end})")
            
            if has_12345:
                known = [s for s in survivors if s['seed'] == 12345][0]
                print(f"     🎯 FOUND SEED 12345! rate={known['match_rate']:.3f}, skip={known['best_skip']}")
        else:
            print(f"❌ [{completed}/{len(forward_jobs)}] {job_id} FAILED: {result.error}")

print(f"\n" + "="*70)
print(f"FORWARD SIEVE COMPLETE")
print("="*70)
print(f"Total survivors across all GPUs: {len(all_survivors)}")

known = [s for s in all_survivors if s['seed'] == 12345]
if known:
    print(f"\n🎯 SEED 12345 FOUND!")
    print(f"   Match rate: {known[0]['match_rate']:.3f}")
    print(f"   Best skip: {known[0]['best_skip']}")
    
    if known[0]['best_skip'] == 5:
        print(f"   ✅ Skip matches expected!")
        
        print(f"\n" + "="*70)
        print("REVERSE SIEVE - ALL GPUs")
        print("="*70)
        
        candidate_seeds = [12345] + [s['seed'] for s in all_survivors[:100]]
        args.threshold = 0.90
        
        rev_jobs = coordinator._create_reverse_sieve_jobs(args, candidate_seeds)
        print(f"Created {len(rev_jobs)} reverse jobs")
        
        rev_node_dist = Counter(w.node.hostname for j, w in rev_jobs)
        print(f"Reverse job distribution:")
        for node, count in rev_node_dist.items():
            print(f"  {node}: {count} jobs")
        
        print(f"\n🚀 Executing all {len(rev_jobs)} reverse jobs...")
        
        rev_survivors_all = []
        
        def execute_reverse_job(job_worker):
            job, worker = job_worker
            # Reverse jobs have individual seeds, not ranges
            result = coordinator.execute_gpu_job(job, worker)
            return job.job_id, worker, result
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(execute_reverse_job, jw): jw for jw in rev_jobs}
            
            completed = 0
            for future in as_completed(futures):
                job_id, worker, result = future.result()
                completed += 1
                
                if result.success and result.results:
                    survivors = result.results.get('survivors', [])
                    rev_survivors_all.extend(survivors)
                    
                    has_12345 = any(s['seed'] == 12345 for s in survivors)
                    marker = "🎯" if has_12345 else "✅"
                    print(f"{marker} [{completed}/{len(rev_jobs)}] {job_id} on {worker.node.hostname} GPU{worker.gpu_id}: {len(survivors)} survivors")
                    
                    if has_12345:
                        rev_known = [s for s in survivors if s['seed'] == 12345][0]
                        print(f"     🎯 CONFIRMED SEED 12345! skip={rev_known['best_skip']}")
        
        rev_known = [s for s in rev_survivors_all if s['seed'] == 12345]
        if rev_known:
            print(f"\n" + "="*70)
            print("26-GPU ALIGNMENT VERIFICATION")
            print("="*70)
            print(f"Forward skip: {known[0]['best_skip']}")
            print(f"Reverse skip: {rev_known[0]['best_skip']}")
            
            if known[0]['best_skip'] == rev_known[0]['best_skip'] == 5:
                print(f"\n✅✅✅ PERFECT ALIGNMENT ACROSS 26-GPU CLUSTER!")
                sys.exit(0)
else:
    print(f"\n❌ Seed 12345 not found")

sys.exit(1)

