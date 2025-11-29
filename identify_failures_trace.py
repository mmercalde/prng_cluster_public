import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer
from reverse_sieve_filter import GPUReverseSieve, load_draws_from_daily3

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

# --- 1. CAPTURE FORWARD SURVIVORS ---
forward_survivors = []
original_execute = coordinator.execute_gpu_job
def capture_forward(job, worker):
    result = original_execute(job, worker)
    if job.search_type == 'residue_sieve' and result.results:
        if 'survivors' in result.results:
            forward_survivors.extend(result.results['survivors'])
    return result
coordinator.execute_gpu_job = capture_forward

# Run the single window (only needs to run forward to capture survivors)
# Note: This will still execute the reverse sieve, but we focus on capturing forward_survivors.
optimizer = WindowOptimizer(coordinator, test_seeds=1_000_000)
result = optimizer.evaluate_window('lcg32', 768, use_all_gpus=True)

# --- 2. MANUALLY VERIFY LOCALLY (Bypassing Remote Bug) ---
draws = load_draws_from_daily3('daily3.json', 768, ['midday', 'evening'], 0)
sieve = GPUReverseSieve(gpu_id=0) # Use the local GPU for reliable verification

print(f"\nForward Sieve found {len(forward_survivors)} unique seeds (Expected 25):")

failed_seeds = []
for s in forward_survivors:
    # CRITICAL: Convert forward sieve's 'best_skip' key to the 'skip' key the manual run expects.
    candidate = {'seed': s['seed'], 'skip': s['best_skip']}
    
    # Run the reverse sieve locally with the single seed/skip
    result = sieve.run_reverse_sieve([candidate], 'lcg32', draws, (0,20), 0.01, 0)
    
    # Check for failure (0 survivors)
    if len(result['survivors']) == 0:
        failed_seeds.append(s['seed'])
        print(f"  ❌ FAIL: seed={s['seed']}, skip={s['best_skip']}, match_rate={s['match_rate']:.6f} ({s['matches']}/{s['total']} matches)")
    else:
        print(f"  ✅ PASS: seed={s['seed']}, skip={s['best_skip']}, match_rate={s['match_rate']:.6f}")

print(f"\n=============================================")
print(f"Total Forward Survivors: {len(forward_survivors)}")
print(f"Total Seeds that FAILED LOCAL Verification: {len(failed_seeds)}")
print(f"FAILING SEEDS (These should NOT be here): {failed_seeds}")
