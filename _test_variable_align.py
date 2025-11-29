
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from hybrid_strategy import get_all_strategies
import json

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()
strategies = get_all_strategies()

class Args:
    target_file = 'test_variable_align.json'
    window_size = 50
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

# CRITICAL FIX: Set coordinator's current_target_file
coordinator.current_target_file = args.target_file

print("\nSTEP 1: Running FORWARD HYBRID sieve...")
forward_jobs = coordinator._create_sieve_jobs(args)
print(f"Created {len(forward_jobs)} forward jobs")

if forward_jobs:
    fwd_job, fwd_worker = forward_jobs[0]
    fwd_result = coordinator.execute_local_job(fwd_job, fwd_worker)
    
    if fwd_result.success and fwd_result.results:
        fwd_survivors = fwd_result.results.get('survivors', [])
        print(f"Forward hybrid found {len(fwd_survivors)} survivors")
        
        fwd_known = [s for s in fwd_survivors if s['seed'] == 54321]
        if fwd_known:
            fwd_pattern = fwd_known[0].get('skip_pattern', [])
            print(f"✅ Forward found seed {fwd_known[0]['seed']}")
            print(f"   Pattern (first 10): {fwd_pattern[:10]}")
        else:
            print(f"⚠️ Forward did not find known seed 54321")
        
        print(f"\nSTEP 2: Running REVERSE HYBRID sieve...")
        candidate_seeds = [s['seed'] for s in fwd_survivors[:50]]
        
        if 54321 not in candidate_seeds:
            candidate_seeds.append(54321)
        
        print(f"Testing {len(candidate_seeds)} candidates in reverse...")
        
        args.threshold = 0.50
        rev_jobs = coordinator._create_reverse_sieve_jobs(args, candidate_seeds)
        
        if rev_jobs:
            rev_job, rev_worker = rev_jobs[0]
            rev_result = coordinator.execute_local_job(rev_job, rev_worker)
            
            if rev_result.success and rev_result.results:
                rev_survivors = rev_result.results.get('survivors', [])
                print(f"Reverse hybrid found {len(rev_survivors)} survivors")
                
                rev_known = [s for s in rev_survivors if s['seed'] == 54321]
                if rev_known:
                    rev_pattern = rev_known[0].get('skip_pattern', [])
                    print(f"✅ Reverse found seed {rev_known[0]['seed']}")
                    print(f"   Pattern (first 10): {rev_pattern[:10]}")
                    
                    # ALIGNMENT CHECK
                    if fwd_known and rev_known:
                        expected = [5, 5, 3, 7, 5, 5, 8, 4, 5, 5, 6, 5, 5, 3, 9, 5, 5, 5, 4, 7, 5, 5, 5, 6, 5, 5, 3, 8, 5, 5, 5, 5, 3, 7, 5, 5, 8, 4, 5, 5, 6, 5, 5, 3, 9, 5, 5, 5, 4, 7]
                        fwd_pat = fwd_pattern[:50]
                        rev_pat = rev_pattern[:50]
                        
                        fwd_matches = sum(1 for i in range(min(len(expected), len(fwd_pat))) if fwd_pat[i] == expected[i])
                        rev_matches = sum(1 for i in range(min(len(expected), len(rev_pat))) if rev_pat[i] == expected[i])
                        
                        fwd_accuracy = fwd_matches / len(expected) if len(expected) > 0 else 0
                        rev_accuracy = rev_matches / len(expected) if len(expected) > 0 else 0
                        
                        print(f"\n🎯 ALIGNMENT CHECK:")
                        print(f"   Forward accuracy: {fwd_accuracy:.1%} ({fwd_matches}/{len(expected)})")
                        print(f"   Reverse accuracy: {rev_accuracy:.1%} ({rev_matches}/{len(expected)})")
                        
                        if fwd_accuracy >= 0.90 and rev_accuracy >= 0.90:
                            print(f"   ✅ BOTH ALIGNED!")
                        else:
                            print(f"   ⚠️ ALIGNMENT ISSUES")
                else:
                    print(f"❌ Reverse did not confirm known seed")
            else:
                print(f"❌ Reverse hybrid failed: {rev_result.error}")
    else:
        print(f"❌ Forward hybrid failed: {fwd_result.error if fwd_result else 'unknown'}")
