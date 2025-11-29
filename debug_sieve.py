#!/usr/bin/env python3
"""
Standalone Sieve Debugger
This script runs a single sieve job directly on a worker node
to test the sieve_filter.py script for hardware errors.
"""

import json
import sys
import time
import os
import traceback

print("--- Starting Sieve Debug Test ---")
print(f"Python executable: {sys.executable}")

try:
    # 1. Import the sieve functions we want to test
    from sieve_filter import GPUSieve, execute_sieve_job
    import cupy as cp
    print("[SUCCESS] All modules imported correctly.")

except Exception as e:
    print("\n[FAIL] CRITICAL IMPORT ERROR")
    print(f"Error: {e}")
    print("This likely means a dependency is missing or the environment is not active.")
    sys.exit(1)


# 2. Define a simple test job (based on your crash log)
#    We use a small seed count for a fast test.
test_job = {
    "job_id": "debug_sieve_test",
    "dataset_path": "synthetic_lottery.json",
    "prng_families": ["java_lcg"],
    "seed_start": 0,
    "seed_end": 50000,  # Small count for a fast test
    "window_size": 1736,
    "offset": 0,
    "skip_range": [35, 192],
    "sessions": ["midday", "evening"],
    "min_match_threshold": 0.5
}

print(f"Created test job: {test_job['job_id']}")


# 3. Run the test
try:
    print("\nAttempting to initialize GPUSieve(gpu_id=0)...")
    # This is the line that fails if the fix is not applied
    sieve = GPUSieve(gpu_id=0)
    print(f"[SUCCESS] GPUSieve initialized. Using device: {sieve.device.id}")
    print("Device initialization appears successful.")

    print("\nAttempting to run the full job (execute_sieve_job)...")
    start_time = time.time()
    result = execute_sieve_job(test_job, gpu_id=0)
    duration = time.time() - start_time
    print(f"[SUCCESS] Job completed in {duration:.2f} seconds.")

    print("\n--- JOB RESULT ---")
    print(json.dumps(result, indent=2))

except Exception as e:
    print("\n--- SCRIPT FAILED ---")
    print(f"An error occurred: {e}")
    print("\n--- TRACEBACK ---")
    traceback.print_exc()
    print("\nIf you see 'cudaErrorDevicesUnavailable' or 'AttributeError: hip'", end="")
    print(", the fix in sieve_filter.py was NOT applied correctly.")
    sys.exit(1)
