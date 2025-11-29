#!/usr/bin/env python3
"""
Simple alignment test - FORCE seed 12345 into local job range
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from prng_registry import mt19937_cpu
import json

print("="*70)
print("ALIGNMENT TEST - Forcing Local Execution")
print("="*70)

# Generate test data
known_seed = 12345
constant_skip = 5
k = 100

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
print(f"Draws: {draws[:10]}")

test_data = [{'draw': d, 'session': 'midday', 'timestamp': 1000000 + i} for i, d in enumerate(draws)]
with open('test_alignment.json', 'w') as f:
    json.dump(test_data, f)

coordinator = MultiGPUCoordinator('distributed_config.json')

# HACK: Temporarily remove remote workers so ALL jobs are local
original_nodes = coordinator.nodes
coordinator.nodes = [n for n in coordinator.nodes if n.hostname in ['localhost', '127.0.0.1']]

workers = coordinator.create_gpu_workers()
print(f"Workers (local only): {len(workers)}")

class Args:
    target_file = 'test_alignment.json'
    window_size = 50
    seeds = 20000  # 0-20k includes seed 12345
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
print("FORWARD SIEVE TEST (Local GPUs Only)")
print("="*70)

# Create jobs with ONLY local workers
forward_jobs = coordinator._create_sieve_jobs(args)
print(f"Created {len(forward_jobs)} jobs across {len(workers)} local GPUs")

# Show job ranges
for i, (job, worker) in enumerate(forward_jobs[:5]):
    seed_start, seed_end = job.seeds
    print(f"Job {i}: seeds [{seed_start}, {seed_end})")

# Find and execute job containing seed 12345
for i, (job, worker) in enumerate(forward_jobs):
    seed_start, seed_end = job.seeds
    
    if seed_start <= 12345 < seed_end:
        print(f"\n✅ Job {i} contains seed 12345: [{seed_start}, {seed_end})")
        print(f"   Executing on {worker.node.hostname} GPU {worker.gpu_id}...")
        
        result = coordinator.execute_local_job(job, worker)
        
        if result.success and result.results:
            survivors = result.results.get('survivors', [])
            print(f"✅ Job completed: {len(survivors)} survivors")
            
            known = [s for s in survivors if s['seed'] == 12345]
            if known:
                print(f"\n🎯 FOUND SEED 12345!")
                print(f"   Match rate: {known[0]['match_rate']:.3f}")
                print(f"   Best skip: {known[0]['best_skip']}")
                
                if known[0]['best_skip'] == 5:
                    print(f"   ✅ Forward skip matches expected (5)!")
                    
                    # TEST REVERSE
                    print(f"\n" + "="*70)
                    print("REVERSE SIEVE TEST")
                    print("="*70)
                    
                    candidate_seeds = [12345] + [s['seed'] for s in survivors[:20]]
                    args.threshold = 0.90
                    
                    rev_jobs = coordinator._create_reverse_sieve_jobs(args, candidate_seeds)
                    print(f"Created {len(rev_jobs)} reverse jobs")
                    
                    # Execute first reverse job that contains seed 12345
                    for rev_job, rev_worker in rev_jobs:
                        if 12345 in rev_job.seeds:
                            print(f"Testing reverse job with {len(rev_job.seeds)} candidates...")
                            
                            rev_result = coordinator.execute_local_job(rev_job, rev_worker)
                            
                            if rev_result.success and rev_result.results:
                                rev_survivors = rev_result.results.get('survivors', [])
                                print(f"✅ Reverse completed: {len(rev_survivors)} survivors")
                                
                                rev_known = [s for s in rev_survivors if s['seed'] == 12345]
                                if rev_known:
                                    print(f"\n🎯 REVERSE CONFIRMED SEED 12345!")
                                    print(f"   Skip: {rev_known[0]['best_skip']}")
                                    
                                    print(f"\n" + "="*70)
                                    print("ALIGNMENT VERIFICATION")
                                    print("="*70)
                                    print(f"Forward skip: {known[0]['best_skip']}")
                                    print(f"Reverse skip: {rev_known[0]['best_skip']}")
                                    print(f"Expected: {constant_skip}")
                                    
                                    if known[0]['best_skip'] == rev_known[0]['best_skip'] == constant_skip:
                                        print(f"\n✅✅✅ PERFECT ALIGNMENT!")
                                        sys.exit(0)
                                    else:
                                        print(f"\n⚠️ ALIGNMENT MISMATCH")
                                else:
                                    print(f"⚠️ Seed 12345 not in reverse survivors")
                            else:
                                print(f"❌ Reverse failed: {rev_result.error}")
                            break
                else:
                    print(f"⚠️ Forward skip mismatch: got {known[0]['best_skip']}, expected 5")
            else:
                print(f"⚠️ Seed 12345 not found in survivors")
                print(f"   Top 5 survivors: {[(s['seed'], s['best_skip']) for s in survivors[:5]]}")
        else:
            print(f"❌ Job failed: {result.error}")
        
        break

print("\n⚠️ Test incomplete")
sys.exit(1)

