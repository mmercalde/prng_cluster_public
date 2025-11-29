
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
import json

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()

class Args:
    target_file = 'test_constant_align.json'
    window_size = 50
    seeds = 100000
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

print("\nSTEP 1: Running FORWARD sieve (seed range 0-100k)...")
forward_jobs = coordinator._create_sieve_jobs(args)
print(f"Created {len(forward_jobs)} forward jobs")

# Execute FIRST LOCAL JOB ONLY (to avoid remote execution issues in test)
local_jobs = [(j, w) for j, w in forward_jobs if w.node.hostname in ['localhost', '127.0.0.1']]
if not local_jobs:
    print("❌ No local jobs found!")
    sys.exit(1)

print(f"Found {len(local_jobs)} local jobs")

# Execute first local job
fwd_job, fwd_worker = local_jobs[0]
print(f"Testing local job: {fwd_job.job_id} on {fwd_worker.node.hostname}")

fwd_result = coordinator.execute_local_job(fwd_job, fwd_worker)

if fwd_result.success and fwd_result.results:
    fwd_survivors = fwd_result.results.get('survivors', [])
    print(f"Forward sieve found {len(fwd_survivors)} survivors (from first local job)")
    
    fwd_known = [s for s in fwd_survivors if s['seed'] == 12345]
    if fwd_known:
        print(f"✅ Forward found seed {fwd_known[0]['seed']} with skip={fwd_known[0]['best_skip']}")
    else:
        print(f"⚠️ Forward did not find known seed 12345 in first local job")
        print(f"   (Seed 12345 may be in a different job's range)")
    
    print(f"\nSTEP 2: Running REVERSE sieve...")
    candidate_seeds = [s['seed'] for s in fwd_survivors[:50]]
    
    if 12345 not in candidate_seeds:
        candidate_seeds.append(12345)
    
    print(f"Testing {len(candidate_seeds)} candidates in reverse...")
    
    args.threshold = 0.90
    rev_jobs = coordinator._create_reverse_sieve_jobs(args, candidate_seeds)
    
    # Get first local reverse job
    local_rev_jobs = [(j, w) for j, w in rev_jobs if w.node.hostname in ['localhost', '127.0.0.1']]
    if local_rev_jobs:
        rev_job, rev_worker = local_rev_jobs[0]
        print(f"Testing local reverse job: {rev_job.job_id}")
        
        rev_result = coordinator.execute_local_job(rev_job, rev_worker)
        
        if rev_result.success and rev_result.results:
            rev_survivors = rev_result.results.get('survivors', [])
            print(f"Reverse sieve found {len(rev_survivors)} survivors")
            
            rev_known = [s for s in rev_survivors if s['seed'] == 12345]
            if rev_known:
                print(f"✅ Reverse confirmed seed {rev_known[0]['seed']} with skip={rev_known[0]['best_skip']}")
                
                if fwd_known and rev_known:
                    fwd_skip = fwd_known[0]['best_skip']
                    rev_skip = rev_known[0]['best_skip']
                    if fwd_skip == rev_skip == 5:
                        print(f"\n🎯 PERFECT ALIGNMENT!")
                        print(f"   Forward skip: {fwd_skip}")
                        print(f"   Reverse skip: {rev_skip}")
                        print(f"   Expected: 5")
                        print(f"   ✅ ALL MATCH!")
                    else:
                        print(f"\n⚠️ ALIGNMENT ISSUE!")
            else:
                print(f"⚠️ Reverse did not confirm known seed")
        else:
            print(f"❌ Reverse sieve failed: {rev_result.error}")
else:
    print(f"❌ Forward sieve failed: {fwd_result.error if fwd_result else 'unknown'}")
