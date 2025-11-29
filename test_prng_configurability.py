#!/usr/bin/env python3
"""
Test that sieve_filter.py now uses configurable PRNG from job spec
"""
import json

# Create a test job with xorshift32
test_job = {
    "job_id": "test_prng_config",
    "dataset_path": "test_hybrid_draws.json",  # We have this from yesterday
    "seed_start": 0,
    "seed_end": 100000,
    "window_size": 100,
    "min_match_threshold": 0.5,
    "skip_range": [0, 10],
    "prng_families": ["xorshift32"],  # NOT mt19937!
    "sessions": ["midday"]
}

with open('test_prng_config_job.json', 'w') as f:
    json.dump(test_job, f, indent=2)

print("✅ Created test job with prng_families: ['xorshift32']")
print("   Job file: test_prng_config_job.json")
print("")
print("Now run:")
print("  python3 sieve_filter.py --job-file test_prng_config_job.json --gpu-id 0")
print("")
print("It should use xorshift32, NOT mt19937!")

