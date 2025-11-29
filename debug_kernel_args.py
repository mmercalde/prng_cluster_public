#!/usr/bin/env python3
"""
Debug what parameters are actually being passed
"""
import cupy as cp
from prng_registry import get_kernel_info

config = get_kernel_info('xorshift32_hybrid')

print("Config default_params:")
print(config.get('default_params'))

print("\nUnpacking test:")
params_list = [cp.int32(v) for v in config.get('default_params', {}).values()]
print(f"Number of params: {len(params_list)}")
print(f"Values: {params_list}")

print("\nExpected kernel signature:")
print("  ..., threshold, shift_a, shift_b, shift_c, offset")

print("\nWhat we're passing:")
print("  ..., threshold,")
print(f"  *[{params_list}],  <- Should be 3 values")
print("  offset")

print("\nTotal parameters for xorshift32_hybrid kernel:")
sig = """
1. seeds
2. residues
3. survivors
4. match_rates
5. skip_sequences
6. strategy_ids
7. survivor_count
8. n_seeds
9. k
10. strategy_max_misses
11. strategy_tolerances
12. n_strategies
13. threshold
14. shift_a
15. shift_b
16. shift_c
17. offset
"""
print(sig)
print("Total: 17 parameters")

