
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator

# Initialize coordinator
coordinator = MultiGPUCoordinator('distributed_config.json')

# Create mock args object
class Args:
    target_file = 'test_distributed_draws.json'
    window_size = 30
    threshold = 0.90
    skip_min = 0
    skip_max = 10
    offset = 0
    prng_type = 'mt19937'
    hybrid = False
    phase2_threshold = 0.50
    gpu_id = 0

args = Args()

# Candidate seeds
candidate_seeds = [12345, 11345, 11845, 12245, 12335, 12355, 12445, 12845, 13345, 11824, 10409, 14506, 14012, 13657, 12286, 11679, 18935, 11424, 19674, 16912, 10520, 10488, 11535, 13582, 13811, 18279, 19863, 10434, 19195, 13257, 18928, 16873, 13611, 17359, 19654, 14557, 10106, 12615, 16924, 15574, 14552, 12547, 13527, 15514, 11674, 11519, 16224, 11584, 15881, 15635, 19891, 14333, 10711, 17527, 18785, 12045, 16201, 11291, 19044, 14803, 15925, 19459, 13150, 11139, 10750, 13733, 14741, 11307, 13814, 11654, 16227, 14554, 17428, 15977, 12664, 16065, 15820, 13432, 14374, 11169, 19980, 12803, 18751, 14010, 12677, 17573, 16216, 14422, 19125, 13598, 15313, 10916, 13752, 10525, 15168, 16572, 14386, 11084, 13456, 19292, 15155, 13483, 18179, 16482, 17517, 12340, 14339, 12287, 14040]

print(f"\n{'='*70}")
print(f"CREATING DISTRIBUTED REVERSE SIEVE JOBS")
print(f"{'='*70}")

# Create reverse sieve jobs
jobs = coordinator._create_reverse_sieve_jobs(args, candidate_seeds)

print(f"\n✅ Created {len(jobs)} jobs")
for i, job in enumerate(jobs[:5]):
    print(f"  Job {i}: {len(job.seeds)} candidates -> {job.assigned_worker.gpu_id}")

# Show distribution
print(f"\nJob distribution across GPUs:")
from collections import Counter
gpu_counts = Counter(job.assigned_worker.gpu_id for job in jobs)
for gpu, count in sorted(gpu_counts.items())[:10]:
    print(f"  GPU {gpu}: {count} jobs")

print(f"\n{'='*70}")
print(f"EXECUTING DISTRIBUTED REVERSE SIEVE")
print(f"{'='*70}")

# Execute jobs (this will distribute across 26 GPUs)
# For now, just execute first job as a test
if jobs:
    test_job = jobs[0]
    print(f"\nTesting first job: {test_job.job_id}")
    print(f"  Candidates: {len(test_job.seeds)}")
    print(f"  Worker: {test_job.assigned_worker.node.hostname} GPU {test_job.assigned_worker.gpu_id}")
    
    # Save job payload to file
    import json
    with open(f'{test_job.job_id}_payload.json', 'w') as f:
        json.dump(test_job.payload, f, indent=2)
    
    print(f"  Payload saved to: {test_job.job_id}_payload.json")
    
    # Execute locally for testing
    result = coordinator.execute_local_job(test_job, test_job.assigned_worker)
    
    if result.success:
        print(f"\n✅ Job executed successfully!")
        print(f"   Runtime: {result.runtime:.2f}s")
        
        # Parse result
        result_data = json.loads(result.raw_output)
        survivors = result_data.get('survivors', [])
        
        print(f"   Survivors: {len(survivors)}")
        if survivors:
            for s in survivors[:3]:
                print(f"     Seed {s['seed']}: rate={s['match_rate']:.3f}, skip={s['best_skip']}")
    else:
        print(f"\n❌ Job failed!")
        print(f"   Error: {result.error}")
