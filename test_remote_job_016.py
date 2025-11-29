#!/usr/bin/env python3
"""
Test the specific remote job that should contain seed 12345
"""
import subprocess
import sys

# SSH into the remote node and run the job directly
print("Testing job sieve_016 directly on 192.168.3.154...")
print("This job should find seed 12345 in range [12320-13090)")

# First, verify the test file exists on remote
print("\n1. Checking if test file exists on remote...")
result = subprocess.run(
    ['ssh', '192.168.3.154', 'ls -lh /home/michael/distributed_prng_analysis/test_26gpu_large.json'],
    capture_output=True,
    text=True
)
print(result.stdout)
if result.returncode != 0:
    print(f"❌ File not found: {result.stderr}")
    sys.exit(1)

# Check file size
result = subprocess.run(
    ['ssh', '192.168.3.154', 'wc -l /home/michael/distributed_prng_analysis/test_26gpu_large.json'],
    capture_output=True,
    text=True
)
print(f"File lines: {result.stdout.strip()}")

# Now test seed 12345 directly on the remote node
print("\n2. Testing seed 12345 directly on remote node...")

job_cmd = """cd /home/michael/distributed_prng_analysis && python3 << 'PYEND'
import json
from prng_registry import mt19937_cpu

# Load the test file
with open('test_26gpu_large.json', 'r') as f:
    draws = [d['draw'] for d in json.load(f)]

print(f"Loaded {len(draws)} draws from file")
print(f"First 10: {draws[:10]}")

# Test seed 12345 with skip=5
outputs = mt19937_cpu(12345, 600, skip=0)
test_draws = []
idx = 0
for i in range(100):
    idx += 5
    if idx < len(outputs):
        test_draws.append(outputs[idx] % 1000)
    idx += 1

print(f"\\nGenerated {len(test_draws)} draws from seed 12345")
print(f"First 10: {test_draws[:10]}")

# Check matches
matches = sum(1 for i in range(min(100, len(draws), len(test_draws))) if test_draws[i] == draws[i])
print(f"\\nMatches: {matches}/100 = {matches/100:.1%}")

# Now run actual sieve on seed 12345
import subprocess
import json

job_data = {
    "analysis_type": "sieve",
    "job_id": "remote_test",
    "dataset_path": "test_26gpu_large.json",
    "seed_start": 12345,
    "seed_end": 12346,
    "window_size": 512,
    "min_match_threshold": 0.01,
    "skip_range": [0, 10],
    "prng_families": ["mt19937"],
    "sessions": ["midday"],
    "offset": 0,
    "hybrid": False,
}

with open('test_remote_direct.json', 'w') as f:
    json.dump(job_data, f)

result = subprocess.run(
    ['python3', 'sieve_filter.py', '--job-file', 'test_remote_direct.json', '--gpu-id', '0'],
    capture_output=True,
    text=True,
    timeout=30
)

print(f"\\nSieve result:")
if result.returncode == 0:
    try:
        sieve_result = json.loads(result.stdout)
        survivors = sieve_result.get('survivors', [])
        print(f"Success: {len(survivors)} survivors")
        if survivors:
            for s in survivors:
                print(f"  Seed {s['seed']}: rate={s['match_rate']:.4f}, skip={s['best_skip']}")
        else:
            print("  No survivors!")
    except:
        print(f"Could not parse: {result.stdout}")
else:
    print(f"Failed: {result.stderr}")
PYEND
"""

result = subprocess.run(
    ['ssh', '192.168.3.154', 'bash', '-c', job_cmd],
    capture_output=True,
    text=True,
    timeout=60
)

print(result.stdout)
if result.stderr:
    print(f"STDERR: {result.stderr}")

