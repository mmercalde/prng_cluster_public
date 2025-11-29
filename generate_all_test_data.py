#!/usr/bin/env python3
"""
Generate ALL test data for 3 new PRNGs:
- Forward constant skip
- Forward variable skip (hybrid)
"""
import json

# ============================================================================
# PRNG Step Functions
# ============================================================================

def java_lcg_step(state, a=25214903917, c=11, m=(1<<48)):
    state = (a * state + c) & (m - 1)
    output = (state >> 16) & 0xFFFFFFFF
    return state, output

def minstd_step(state, a=48271, m=2147483647):
    state = (a * state) % m
    return state, state

def xorshift128_step(x, y, z, w):
    t = x ^ (x << 11)
    t &= 0xFFFFFFFF
    new_x, new_y, new_z = y, z, w
    new_w = (w ^ (w >> 19) ^ (t ^ (t >> 8))) & 0xFFFFFFFF
    return new_x, new_y, new_z, new_w, new_w

# ============================================================================
# Test Data Generators
# ============================================================================

def generate_forward_constant(prng_name, step_func, seed, skip, num_draws):
    """Generate forward sieve with constant skip"""
    print(f"Generating test_multi_prng_{prng_name}.json (forward, constant skip={skip})...")
    
    if prng_name == 'xorshift128':
        x, y, z, w = seed & 0xFFFFFFFF, 362436069, 521288629, 88675123
        draws = []
        for i in range(num_draws):
            for _ in range(skip):
                x, y, z, w, _ = step_func(x, y, z, w)
            x, y, z, w, output = step_func(x, y, z, w)
            draws.append({'draw': output % 1000, 'session': 'midday', 'timestamp': 3000000 + i})
    elif prng_name == 'minstd':
        state = seed % 2147483647
        if state == 0:
            state = 1
        draws = []
        for i in range(num_draws):
            for _ in range(skip):
                state, _ = step_func(state)
            state, output = step_func(state)
            draws.append({'draw': output % 1000, 'session': 'midday', 'timestamp': 3000000 + i})
    else:  # java_lcg
        state = seed
        draws = []
        for i in range(num_draws):
            for _ in range(skip):
                state, _ = step_func(state)
            state, output = step_func(state)
            draws.append({'draw': output % 1000, 'session': 'midday', 'timestamp': 3000000 + i})
    
    with open(f'test_multi_prng_{prng_name}.json', 'w') as f:
        json.dump(draws, f, indent=2)
    print(f"  ✅ Created {len(draws)} draws (seed={seed}, skip={skip})")
    return draws


def generate_forward_variable(prng_name, step_func, seed, skip_pattern, num_draws):
    """Generate forward sieve with variable skip (hybrid)"""
    print(f"Generating test_{prng_name}_hybrid.json (forward, variable skip)...")
    
    # Repeat pattern to cover all draws
    full_pattern = (skip_pattern * ((num_draws // len(skip_pattern)) + 1))[:num_draws]
    
    if prng_name == 'xorshift128':
        x, y, z, w = seed & 0xFFFFFFFF, 362436069, 521288629, 88675123
        draws = []
        for i, skip in enumerate(full_pattern):
            for _ in range(skip):
                x, y, z, w, _ = step_func(x, y, z, w)
            x, y, z, w, output = step_func(x, y, z, w)
            draws.append({'draw': output % 1000, 'session': 'midday', 'timestamp': 3000000 + i})
    elif prng_name == 'minstd':
        state = seed % 2147483647
        if state == 0:
            state = 1
        draws = []
        for i, skip in enumerate(full_pattern):
            for _ in range(skip):
                state, _ = step_func(state)
            state, output = step_func(state)
            draws.append({'draw': output % 1000, 'session': 'midday', 'timestamp': 3000000 + i})
    else:  # java_lcg
        state = seed
        draws = []
        for i, skip in enumerate(full_pattern):
            for _ in range(skip):
                state, _ = step_func(state)
            state, output = step_func(state)
            draws.append({'draw': output % 1000, 'session': 'midday', 'timestamp': 3000000 + i})
    
    with open(f'test_{prng_name}_hybrid.json', 'w') as f:
        json.dump(draws, f, indent=2)
    print(f"  ✅ Created {len(draws)} draws (seed={seed}, pattern={skip_pattern})")
    return draws


# ============================================================================
# Main Generation
# ============================================================================

SEED = 12345
CONSTANT_SKIP = 5
VARIABLE_PATTERN = [5, 5, 3, 7, 5, 5, 8, 4, 5, 5]
NUM_DRAWS = 512

print("=" * 80)
print("GENERATING TEST DATA FOR 3 NEW PRNGs")
print("=" * 80)
print(f"Seed: {SEED}")
print(f"Constant skip: {CONSTANT_SKIP}")
print(f"Variable pattern: {VARIABLE_PATTERN}")
print(f"Number of draws: {NUM_DRAWS}")
print("=" * 80)
print()

# JAVA LCG
print("=" * 80)
print("JAVA LCG (java.util.Random)")
print("=" * 80)
generate_forward_constant('java_lcg', java_lcg_step, SEED, CONSTANT_SKIP, NUM_DRAWS)
generate_forward_variable('java_lcg', java_lcg_step, SEED, VARIABLE_PATTERN, NUM_DRAWS)
print()

# MINSTD
print("=" * 80)
print("MINSTD (Park & Miller)")
print("=" * 80)
generate_forward_constant('minstd', minstd_step, SEED, CONSTANT_SKIP, NUM_DRAWS)
generate_forward_variable('minstd', minstd_step, SEED, VARIABLE_PATTERN, NUM_DRAWS)
print()

# XORSHIFT128
print("=" * 80)
print("XORSHIFT128")
print("=" * 80)
generate_forward_constant('xorshift128', xorshift128_step, SEED, CONSTANT_SKIP, NUM_DRAWS)
generate_forward_variable('xorshift128', xorshift128_step, SEED, VARIABLE_PATTERN, NUM_DRAWS)
print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Generated 6 test files (2 per PRNG × 3 PRNGs):")
print()
print("Java LCG:")
print("  ✅ test_multi_prng_java_lcg.json (forward, constant)")
print("  ✅ test_java_lcg_hybrid.json (forward, variable)")
print()
print("MINSTD:")
print("  ✅ test_multi_prng_minstd.json (forward, constant)")
print("  ✅ test_minstd_hybrid.json (forward, variable)")
print()
print("Xorshift128:")
print("  ✅ test_multi_prng_xorshift128.json (forward, constant)")
print("  ✅ test_xorshift128_hybrid.json (forward, variable)")
print()
print("=" * 80)

# Verify
import os
expected = [
    'test_multi_prng_java_lcg.json', 'test_java_lcg_hybrid.json',
    'test_multi_prng_minstd.json', 'test_minstd_hybrid.json',
    'test_multi_prng_xorshift128.json', 'test_xorshift128_hybrid.json',
]

if all(os.path.exists(f) for f in expected):
    print("✅ ALL 6 TEST FILES CREATED SUCCESSFULLY!")
else:
    print("⚠️  Some files missing")
