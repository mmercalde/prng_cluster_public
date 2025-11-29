
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from hybrid_strategy import get_all_strategies
import json

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()
print(f"Created {len(workers)} GPU workers")

strategies = get_all_strategies()
print(f"Loaded {len(strategies)} hybrid strategies")

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
candidate_seeds = [12345, 11345, 11845, 12245, 12335, 12355, 12445, 12845, 13345, 11824, 10409, 14506, 14012, 13657, 12286, 11679, 18935, 11424, 19674, 16912, 10520, 10488, 11535, 13582, 13811, 18279, 19863, 10434, 19195, 13257, 18928, 16873, 13611, 17359, 19654, 14557, 10106, 12615, 16924, 15574, 14552, 12547, 13527, 15514, 11674, 11519, 16224, 11584, 15881, 15635, 19891, 14333, 10711, 17527, 18785, 12045, 16201, 11291, 19044, 14803, 15925, 19459, 13150, 11139, 10750, 13733, 14741, 11307, 13814, 11654, 16227, 14554, 17428, 15977, 12664, 16065, 15820, 13432, 14374, 11169, 19980, 12803, 18751, 14010, 12677, 17573, 16216, 14422, 19125, 13598, 15313, 10916, 13752, 10525, 15168, 16572, 14386, 11084, 13456, 19292, 15155, 13483, 18179, 16482, 17517, 12340, 14339, 12287, 14040]

print("\n" + "="*70)
print("CREATING HYBRID REVERSE SIEVE JOBS")
print("="*70)

job_assignments = coordinator._create_reverse_sieve_jobs(args, candidate_seeds)

print(f"\nâœ… Created {len(job_assignments)} jobs")
for i, (job, worker) in enumerate(job_assignments[:5]):
    print(f"  Job {i}: {len(job.seeds)} candidates -> GPU {worker.gpu_id}")

from collections import Counter
gpu_counts = Counter(worker.gpu_id for job, worker in job_assignments)
print(f"\nJob distribution across {len(gpu_counts)} GPUs:")
for gpu, count in sorted(gpu_counts.items())[:10]:
    print(f"  GPU {gpu}: {count} jobs")

print("\n" + "="*70)
print("EXECUTING FIRST JOB (HYBRID TEST)")
print("="*70)

if job_assignments:
    test_job, test_worker = job_assignments[0]
    print(f"\nTesting job: {test_job.job_id}")
    print(f"  Candidates: {len(test_job.seeds)}")
    print(f"  Worker: GPU {test_worker.gpu_id}")
    print(f"  Mode: HYBRID (variable skip)")
    
    with open(f'{test_job.job_id}_payload.json', 'w') as f:
        json.dump(test_job.payload, f, indent=2)
    
    print(f"  Payload: {test_job.job_id}_payload.json")
    
    result = coordinator.execute_local_job(test_job, test_worker)
    
    if result.success:
        print(f"\nâœ… Job executed successfully!")
        print(f"   Runtime: {result.runtime:.2f}s")
        
        if result.results:
            survivors = result.results.get('survivors', [])
            
            print(f"   Survivors: {len(survivors)}")
            if survivors:
                for s in survivors[:5]:
                    print(f"     Seed {s['seed']}: rate={s['match_rate']:.3f}")
                    print(f"       Strategy: {s.get('strategy_name', 'unknown')}")
                    pattern = s.get('skip_pattern', [])
                    if pattern:
                        print(f"       Pattern (first 10): {pattern[:10]}")
                    
                    if s['seed'] == 12345:
                        print(f"       🎯 FOUND KNOWN SEED WITH VARIABLE SKIP!")
                        
                        # FIXED: Proper comparison
                        expected = [5, 5, 3, 7, 5, 5, 8, 4, 5, 5, 6, 5, 5, 3, 9, 5, 5, 5, 4, 7, 5, 5, 5, 6, 5, 5, 3, 8, 5, 5]
                        actual = pattern[:len(expected)]  # Trim to same length
                        
                        # Element-by-element comparison
                        matches = sum(1 for i in range(min(len(expected), len(actual))) if expected[i] == actual[i])
                        total = len(expected)
                        accuracy = matches / total if total > 0 else 0
                        
                        print(f"       Pattern accuracy: {matches}/{total} = {accuracy:.1%}")
                        
                        if accuracy == 1.0:
                            print(f"       ✅ SKIP PATTERN MATCHES PERFECTLY!")
                        elif accuracy >= 0.90:
                            print(f"       âœ… SKIP PATTERN MATCHES ({accuracy:.1%} accurate)")
                        else:
                            print(f"       ⚠️ Skip pattern differs significantly")
                            print(f"         Expected: {expected[:10]}")
                            print(f"         Got:      {actual[:10]}")
                            print(f"         Mismatches at indices: {[i for i in range(min(len(expected), len(actual))) if expected[i] != actual[i]][:5]}")
            else:
                print(f"   (No survivors at threshold {args.threshold})")
        else:
            print(f"   ⚠️ No result data returned")
    else:
        print(f"\nâŒ Job failed!")
        print(f"   Error: {result.error}")
