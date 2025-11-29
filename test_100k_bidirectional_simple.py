#!/usr/bin/env python3
"""
Simple 100K bidirectional sieve test - bypasses coordinator
"""

import sys
import json
import time
import subprocess

print("="*80)
print("100K BIDIRECTIONAL SIEVE TEST (SIMPLIFIED)")
print("="*80)
print()

# Create forward sieve job
forward_job = {
    "job_id": "test_100k_forward",
    "dataset_path": "daily3.json",
    "seed_start": 0,
    "seed_end": 100000,
    "window_size": 244,
    "min_match_threshold": 0.012,
    "skip_range": [3, 29],
    "prng_families": ["java_lcg"],
    "sessions": ["evening"]
}

# Save forward job
with open('test_100k_forward.json', 'w') as f:
    json.dump(forward_job, f, indent=2)

print("Step 1: Running FORWARD sieve on 100K seeds...")
print()

# Run forward sieve
start_time = time.time()

result = subprocess.run(
    ['python3', 'sieve_filter.py', '--job-file', 'test_100k_forward.json', '--gpu-id', '0'],
    capture_output=True,
    text=True
)

forward_time = time.time() - start_time

if result.returncode != 0:
    print(f"ERROR: Forward sieve failed!")
    print(result.stderr)
    sys.exit(1)

# Read result from FILE (not stdout)
with open('result_test_100k_forward.json', 'r') as f:
    forward_result = json.load(f)

forward_survivors = forward_result.get('survivors', [])

print(f"✅ Forward complete in {forward_time:.1f}s")
print(f"   Found {len(forward_survivors):,} forward survivors")
print()

if len(forward_survivors) == 0:
    print("No forward survivors found - test complete")
    sys.exit(0)

# Prepare reverse sieve with forward survivors (limit to 1000 for speed)
reverse_job = {
    "job_id": "test_100k_reverse",
    "dataset_path": "daily3.json",
    "candidate_seeds": [s['seed'] for s in forward_survivors[:1000]],
    "window_size": 244,
    "min_match_threshold": 0.012,
    "prng_families": ["java_lcg"],
    "sessions": ["evening"]
}

with open('test_100k_reverse.json', 'w') as f:
    json.dump(reverse_job, f, indent=2)

print(f"Step 2: Running REVERSE sieve on {len(reverse_job['candidate_seeds']):,} candidates...")
print()

start_time = time.time()

result = subprocess.run(
    ['python3', 'reverse_sieve_filter.py', '--job-file', 'test_100k_reverse.json', '--gpu-id', '0'],
    capture_output=True,
    text=True
)

reverse_time = time.time() - start_time

if result.returncode != 0:
    print(f"ERROR: Reverse sieve failed!")
    print(result.stderr)
    sys.exit(1)

# Read result from FILE (not stdout)
with open('result_test_100k_reverse.json', 'r') as f:
    reverse_result = json.load(f)

reverse_survivors = reverse_result.get('survivors', [])

print(f"✅ Reverse complete in {reverse_time:.1f}s")
print(f"   Found {len(reverse_survivors):,} reverse survivors")
print()

# Find intersection (bidirectional survivors)
reverse_seeds = {s['seed'] for s in reverse_survivors}
bidirectional = [s for s in forward_survivors if s['seed'] in reverse_seeds]

print("="*80)
print("BIDIRECTIONAL RESULTS")
print("="*80)
print(f"Forward survivors:       {len(forward_survivors):,}")
print(f"Reverse survivors:       {len(reverse_survivors):,}")
print(f"Bidirectional survivors: {len(bidirectional):,}")
print(f"Total time:              {forward_time + reverse_time:.1f}s")
print()

print("✅ Check your new results files:")
print("   results/summaries/")
print("   results/csv/")
print("   results/json/")
print()

# Show latest result files
import subprocess
subprocess.run(['ls', '-lht', 'results/summaries/'], check=False)
print()
subprocess.run(['ls', '-lht', 'results/csv/'], check=False)

print()

# Show top 5 bidirectional survivors
if bidirectional:
    print("="*80)
    print("Top 5 bidirectional survivors:")
    print("="*80)
    for i, s in enumerate(sorted(bidirectional, key=lambda x: x.get('matches', 0), reverse=True)[:5], 1):
        print(f"  {i}. Seed {s['seed']:,} - {s.get('matches', 0)} matches ({s.get('match_rate', 0)*100:.2f}%)")

print()
print("="*80)
print("🎉 TEST COMPLETE!")
print("="*80)
