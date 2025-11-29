#!/usr/bin/env python3

total_seeds = 100000
num_jobs = 26
seeds_per_job = (total_seeds + num_jobs - 1) // num_jobs

print(f"Total seeds: {total_seeds}")
print(f"Jobs: {num_jobs}")
print(f"Seeds per job: {seeds_per_job}")
print()

# Find which job contains seed 12345
for i in range(num_jobs):
    seed_start = i * seeds_per_job
    seed_end = min(total_seeds, seed_start + seeds_per_job)
    
    if seed_start <= 12345 < seed_end:
        print(f"✅ Seed 12345 is in job {i}: range [{seed_start}, {seed_end})")
        break
    
    if i < 5:
        print(f"Job {i}: [{seed_start}, {seed_end})")

print()
print("Seed 12345 SHOULD be found by forward sieve.")
print("If it's not being found, the issue is:")
print("  1. Window size too small (50 draws may not be enough)")
print("  2. Threshold too high (0.01 = 1% match)")
print("  3. The test data doesn't actually match seed 12345")
