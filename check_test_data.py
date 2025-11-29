#!/usr/bin/env python3
"""
Check what data the test is actually passing
"""
from prng_registry import xorshift32_cpu

known_seed = 54321
base_pattern = [5,5,3,7,5,5,8,4,5,5]
skip_pattern = base_pattern * 67  # What test_xorshift32_one_worker uses
k = len(skip_pattern)

total_needed = sum(skip_pattern) + k
all_outputs = xorshift32_cpu(known_seed, total_needed, skip=0)

draws = []
idx = 0
for skip in skip_pattern:
    idx += skip
    draws.append(all_outputs[idx] % 1000)
    idx += 1

print(f"Test data:")
print(f"  Number of draws (k): {len(draws)}")
print(f"  Skip pattern length: {len(skip_pattern)}")
print(f"  Total outputs generated: {total_needed}")
print(f"  First 10 draws: {draws[:10]}")

# Manual test used 100 draws and it worked
# This test uses 670 draws
# Is 670 > 512 (the max in the kernel)?

print(f"\n⚠️  WARNING: k={len(draws)} but kernel has max 512!")
print(f"   skip_sequences_gpu = cp.zeros(n_seeds * 512, dtype=cp.uint32)")
print(f"   for (int draw_idx = 0; draw_idx < k && draw_idx < 512; draw_idx++)")

