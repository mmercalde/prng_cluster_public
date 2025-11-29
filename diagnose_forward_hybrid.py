import subprocess
import json
from prng_registry import mt19937_cpu

# Generate variable skip test data
known_seed = 12345
skip_pattern = [5,5,3,7,5,5,8,4,5,5,6,5,5,3,9,5,5,5,4,7,5,5,5,6,5,5,3,8,5,5]
total_needed = sum(skip_pattern) + len(skip_pattern)
all_outputs = mt19937_cpu(known_seed, total_needed, skip=0)
draws = []
idx = 0
for skip in skip_pattern:
    idx += skip
    draws.append(all_outputs[idx] % 1000)
    idx += 1

print(f"Testing seed {known_seed} with {len(draws)} draws")
print(f"Skip pattern: {skip_pattern[:10]}...")

# Save draws
test_data = [{'draw': d, 'session': 'midday'} for d in draws]
with open('test_hybrid_draws.json', 'w') as f:
    json.dump(test_data, f)

from hybrid_strategy import get_all_strategies
strategies = get_all_strategies()

# Serialize strategies with ALL fields
strategies_data = []
for s in strategies:
    strategies_data.append({
        'name': s.name,
        'max_consecutive_misses': s.max_consecutive_misses,
        'skip_tolerance': s.skip_tolerance,
        'enable_reseed_search': s.enable_reseed_search,
        'skip_learning_rate': s.skip_learning_rate,
        'breakpoint_threshold': s.breakpoint_threshold
    })

job = {
    'job_id': 'test_hybrid',
    'dataset_path': 'test_hybrid_draws.json',
    'seed_start': 12345,
    'seed_end': 12346,
    'window_size': len(draws),
    'skip_range': [0, 10],
    'phase1_threshold': 0.0,
    'phase2_threshold': 0.50,
    'prng_families': ['mt19937'],
    'hybrid': True,
    'strategies': strategies_data,
    'sessions': ['midday'],
    'offset': 0
}

with open('test_job.json', 'w') as f:
    json.dump(job, f, indent=2)

print("\nRunning sieve_filter.py...")
result = subprocess.run(
    ['python3', 'sieve_filter.py', '--job-file', 'test_job.json', '--gpu-id', '0'],
    capture_output=True,
    text=True
)

print("\n" + "="*70)
print("STDERR:")
print("="*70)
print(result.stderr)

if result.returncode == 0:
    data = json.loads(result.stdout)
    survivors = data.get('survivors', [])
    print(f"\n✅ Found {len(survivors)} survivors")
    if survivors:
        for s in survivors[:3]:
            print(f"   Seed {s['seed']}: rate={s['match_rate']:.3f}, pattern={s.get('skip_pattern', [])[:10]}")
    
    # Check if our known seed was found
    found = any(s['seed'] == known_seed for s in survivors)
    if found:
        print(f"\n🎉 FOUND seed {known_seed}!")
    else:
        print(f"\n⚠️  Seed {known_seed} NOT in survivors")
else:
    print(f"\n❌ Failed: {result.returncode}")

