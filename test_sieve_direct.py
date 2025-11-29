#!/usr/bin/env python3
"""
Direct test of sieve_filter.py with seed 12345
"""
import subprocess
import json
import sys

# Create a simple job for sieve_filter.py
job_data = {
    "analysis_type": "sieve",
    "job_id": "test_direct",
    "dataset_path": "test_26gpu_large.json",
    "seed_start": 12345,
    "seed_end": 12346,  # Test ONLY seed 12345
    "window_size": 512,
    "min_match_threshold": 0.01,
    "skip_range": [0, 10],
    "prng_families": ["mt19937"],
    "sessions": ["midday"],
    "offset": 0,
    "hybrid": False,
    "phase1_threshold": None,
    "phase2_threshold": None,
}

with open('test_direct_job.json', 'w') as f:
    json.dump(job_data, f, indent=2)

print("Testing seed 12345 directly with sieve_filter.py...")
print(f"Job: {job_data}")

# Run sieve_filter.py directly
result = subprocess.run(
    ['python3', 'sieve_filter.py', '--job-file', 'test_direct_job.json', '--gpu-id', '0'],
    capture_output=True,
    text=True,
    timeout=30
)

print(f"\nReturn code: {result.returncode}")
print(f"\nSTDOUT:\n{result.stdout}")
if result.stderr:
    print(f"\nSTDERR:\n{result.stderr}")

# Parse result
try:
    result_data = json.loads(result.stdout)
    if result_data.get('success'):
        survivors = result_data.get('survivors', [])
        print(f"\n✅ Sieve succeeded: {len(survivors)} survivors")
        
        if survivors:
            for s in survivors:
                print(f"   Seed {s['seed']}: rate={s['match_rate']:.4f}, skip={s['best_skip']}")
                if s['seed'] == 12345:
                    print(f"   🎯 FOUND IT!")
        else:
            print(f"   ❌ No survivors (seed 12345 didn't pass threshold)")
    else:
        print(f"❌ Sieve failed: {result_data.get('error')}")
except:
    print(f"❌ Could not parse output")

