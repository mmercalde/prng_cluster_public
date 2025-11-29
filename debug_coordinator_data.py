#!/usr/bin/env python3
"""
Check what data the coordinator is actually passing to the worker
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from prng_registry import xorshift32_cpu
import json

known_seed = 54321
base_pattern = [5,5,3,7,5,5,8,4,5,5]
skip_pattern = base_pattern * 67  # 670 skips
k = len(skip_pattern)

total_needed = sum(skip_pattern) + k
all_outputs = xorshift32_cpu(known_seed, total_needed, skip=0)

draws = []
idx = 0
for skip in skip_pattern:
    idx += skip
    draws.append(all_outputs[idx] % 1000)
    idx += 1

print(f"Coordinator is passing:")
print(f"  Number of draws: {len(draws)}")
print(f"  First 10: {draws[:10]}")

print(f"\nDirect test used:")
print(f"  Number of draws: 100")
print(f"  First 10: [994, 610, 167, 442, 294, 777, 598, 51, 599, 598]")

print(f"\n⚠️  THE DIFFERENCE:")
print(f"  Coordinator: {len(draws)} draws")
print(f"  Direct test: 100 draws")
print(f"\n  Coordinator draws[2] = {draws[2]}")
print(f"  Direct test draws[2] = 167")
print(f"\n  THEY'RE DIFFERENT DRAWS!")

