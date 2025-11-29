#!/usr/bin/env python3
"""
Test VARIABLE SKIP alignment - hybrid mode
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from prng_registry import mt19937_cpu
import json

print("="*70)
print("HYBRID ALIGNMENT TEST - Variable Skip")
print("="*70)

# Generate test data with VARIABLE skip
known_seed = 54321
base_pattern = [5,5,3,7,5,5,8,4,5,5,6,5,5,3,9,5,5,5,4,7,5,5,5,6,5,5,3,8,5,5]
skip_pattern = base_pattern * 3 + base_pattern[:10]  # 100 skips
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
print(f"Draws: {draws[:10]}")

test_data = [{'draw': d, 'session': 'midday', 'timestamp': 2000000 + i} for i, d in enumerate(draws)]
with open('test_hybrid_align.json', 'w') as f:
    json.dump(test_data, f)

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.nodes = [n for n in coordinator.nodes if n.hostname in ['localhost', '127.0.0.1']]
workers = coordinator.create_gpu_workers()

class Args:
    target_file = 'test_hybrid_align.json'
    window_size = 50
    seeds = 100000
    seed_start = 0
    threshold = 0.01
    skip_min = 0
    skip_max = 10
    offset = 0
    prng_type = 'mt19937'
    hybrid = True  # HYBRID MODE
    phase1_threshold = 0.01
    phase2_threshold = 0.50
    gpu_id = 0

args = Args()
coordinator.current_target_file = args.target_file

print("\n" + "="*70)
print("FORWARD HYBRID SIEVE")
print("="*70)

forward_jobs = coordinator._create_sieve_jobs(args)
print(f"Created {len(forward_jobs)} jobs")

for i, (job, worker) in enumerate(forward_jobs):
    seed_start, seed_end = job.seeds
    
    if seed_start <= 54321 < seed_end:
        print(f"\n✅ Job {i} contains seed 54321: [{seed_start}, {seed_end})")
        
        result = coordinator.execute_local_job(job, worker)
        
        if result.success and result.results:
            survivors = result.results.get('survivors', [])
            print(f"✅ Forward completed: {len(survivors)} survivors")
            
            known = [s for s in survivors if s['seed'] == 54321]
            if known:
                fwd_pattern = known[0].get('skip_pattern', [])
                print(f"\n🎯 FOUND SEED 54321!")
                print(f"   Match rate: {known[0]['match_rate']:.3f}")
                print(f"   Pattern (first 10): {fwd_pattern[:10]}")
                
                # REVERSE TEST
                print(f"\n" + "="*70)
                print("REVERSE HYBRID SIEVE")
                print("="*70)
                
                candidate_seeds = [54321] + [s['seed'] for s in survivors[:20]]
                args.threshold = 0.50
                
                rev_jobs = coordinator._create_reverse_sieve_jobs(args, candidate_seeds)
                print(f"Created {len(rev_jobs)} reverse jobs")
                
                for rev_job, rev_worker in rev_jobs:
                    if 54321 in rev_job.seeds:
                        print(f"Testing reverse job...")
                        
                        rev_result = coordinator.execute_local_job(rev_job, rev_worker)
                        
                        if rev_result.success and rev_result.results:
                            rev_survivors = rev_result.results.get('survivors', [])
                            print(f"✅ Reverse completed: {len(rev_survivors)} survivors")
                            
                            rev_known = [s for s in rev_survivors if s['seed'] == 54321]
                            if rev_known:
                                rev_pattern = rev_known[0].get('skip_pattern', [])
                                print(f"\n🎯 REVERSE CONFIRMED SEED 54321!")
                                print(f"   Pattern (first 10): {rev_pattern[:10]}")
                                
                                # Compare patterns
                                expected = skip_pattern[:50]
                                fwd_pat = fwd_pattern[:50]
                                rev_pat = rev_pattern[:50]
                                
                                fwd_matches = sum(1 for i in range(min(len(expected), len(fwd_pat))) if fwd_pat[i] == expected[i])
                                rev_matches = sum(1 for i in range(min(len(expected), len(rev_pat))) if rev_pat[i] == expected[i])
                                
                                fwd_acc = fwd_matches / len(expected)
                                rev_acc = rev_matches / len(expected)
                                
                                print(f"\n" + "="*70)
                                print("HYBRID ALIGNMENT VERIFICATION")
                                print("="*70)
                                print(f"Forward accuracy: {fwd_acc:.1%} ({fwd_matches}/{len(expected)})")
                                print(f"Reverse accuracy: {rev_acc:.1%} ({rev_matches}/{len(expected)})")
                                
                                if fwd_acc >= 0.90 and rev_acc >= 0.90:
                                    print(f"\n✅✅✅ HYBRID ALIGNMENT VERIFIED!")
                                    sys.exit(0)
                                else:
                                    print(f"\n⚠️ Accuracy below 90%")
                        break
        break

sys.exit(1)

