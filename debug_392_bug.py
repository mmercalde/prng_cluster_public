#!/usr/bin/env python3
"""
Systematic debugging of the +392 offset bug in xoshiro256pp_reverse GPU kernel
Based on analysis of deterministic error pattern: every 3rd output is +392 higher
"""

import cupy as cp
import numpy as np
from prng_registry import get_cpu_reference

print("="*80)
print("SYSTEMATIC DEBUG: +392 Offset Bug in xoshiro256pp_reverse")
print("="*80)

# Get expected outputs
cpu_ref = get_cpu_reference('xoshiro256pp_reverse')
cpu_outputs = cpu_ref(1234, 10, skip=5, offset=0)
cpu_expected = [int(x % 1000) for x in cpu_outputs]

print(f"\nExpected outputs: {cpu_expected}")
print("Error pattern: Positions 0, 3, 6 are +392 higher")
print()

# =============================================================================
# TEST 1: Single-threaded execution
# =============================================================================
print("="*80)
print("TEST 1: Single-threaded Execution (Rule out threading issues)")
print("="*80)

single_thread_kernel = r'''
extern "C" __global__
void single_thread_xoshiro(unsigned long long seed, unsigned long long* outputs, int n, int skip_val) {
    // Only thread 0 executes
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    unsigned long long s0 = seed;
    unsigned long long s1 = 0x9E3779B97F4A7C15ULL;
    unsigned long long s2 = 0x6A09E667F3BCC908ULL;
    unsigned long long s3 = 0xBB67AE8584CAA73BULL;
    
    // Skip
    for (int s = 0; s < skip_val; s++) {
        unsigned long long temp = s0 + s3;
        unsigned long long result = ((temp << 23) | (temp >> 41)) + s0;
        unsigned long long t = s1 << 17;
        s2 ^= s0;
        s3 ^= s1;
        s1 ^= s2;
        s0 ^= s3;
        s2 ^= t;
        s3 = ((s3 << 45) | (s3 >> 19));
    }
    
    // Generate
    for (int i = 0; i < n; i++) {
        unsigned long long temp = s0 + s3;
        unsigned long long result = ((temp << 23) | (temp >> 41)) + s0;
        unsigned long long t = s1 << 17;
        s2 ^= s0;
        s3 ^= s1;
        s1 ^= s2;
        s0 ^= s3;
        s2 ^= t;
        s3 = ((s3 << 45) | (s3 >> 19));
        outputs[i] = result;
    }
}
'''

module = cp.RawModule(code=single_thread_kernel)
kernel = module.get_function('single_thread_xoshiro')
outputs_gpu = cp.zeros(10, dtype=cp.uint64)
kernel((1,), (1,), (cp.uint64(1234), outputs_gpu, 10, 5))
gpu_outputs = [int(x % 1000) for x in outputs_gpu.get()]

print(f"GPU (single-thread): {gpu_outputs}")
print(f"Match: {gpu_outputs == cpu_expected}")
if gpu_outputs != cpu_expected:
    print("❌ Bug persists even in single-threaded mode!")
    print("   This rules out thread indexing or synchronization issues.")
else:
    print("✅ Bug FIXED in single-threaded mode!")
    print("   This indicates a threading/indexing problem.")

# =============================================================================
# TEST 2: Aligned memory for state variables
# =============================================================================
print("\n" + "="*80)
print("TEST 2: Memory-Aligned State Variables")
print("="*80)

aligned_kernel = r'''
extern "C" __global__
void aligned_xoshiro(unsigned long long seed, unsigned long long* outputs, int n, int skip_val) {
    // Aligned 256-bit state (4 x 64-bit)
    __align__(32) unsigned long long s[4];
    s[0] = seed;
    s[1] = 0x9E3779B97F4A7C15ULL;
    s[2] = 0x6A09E667F3BCC908ULL;
    s[3] = 0xBB67AE8584CAA73BULL;
    
    // Skip
    for (int skip = 0; skip < skip_val; skip++) {
        unsigned long long temp = s[0] + s[3];
        unsigned long long result = ((temp << 23) | (temp >> 41)) + s[0];
        unsigned long long t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = ((s[3] << 45) | (s[3] >> 19));
    }
    
    // Generate
    for (int i = 0; i < n; i++) {
        unsigned long long temp = s[0] + s[3];
        unsigned long long result = ((temp << 23) | (temp >> 41)) + s[0];
        unsigned long long t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = ((s[3] << 45) | (s[3] >> 19));
        outputs[i] = result;
    }
}
'''

module = cp.RawModule(code=aligned_kernel)
kernel = module.get_function('aligned_xoshiro')
outputs_gpu = cp.zeros(10, dtype=cp.uint64)
kernel((1,), (1,), (cp.uint64(1234), outputs_gpu, 10, 5))
gpu_outputs = [int(x % 1000) for x in outputs_gpu.get()]

print(f"GPU (aligned):       {gpu_outputs}")
print(f"Match: {gpu_outputs == cpu_expected}")
if gpu_outputs != cpu_expected:
    print("❌ Bug persists with aligned memory!")
else:
    print("✅ Bug FIXED with aligned memory!")

# =============================================================================
# TEST 3: Bit-level analysis of the +392 error
# =============================================================================
print("\n" + "="*80)
print("TEST 3: Bit-Level Analysis of +392 Error")
print("="*80)

print(f"\n392 decimal = 0x{392:03x} = 0b{392:010b}")
print(f"Binary: {bin(392)}")

# Check if 392 relates to any state variables
print(f"\nAnalyzing first wrong output (position 0):")
print(f"  Expected: {cpu_expected[0]} (0x{cpu_expected[0]:03x})")
print(f"  Got:      {gpu_outputs[0]} (0x{gpu_outputs[0]:03x})")
print(f"  Diff:     {gpu_outputs[0] - cpu_expected[0]} (0x{(gpu_outputs[0] - cpu_expected[0]):03x})")

# =============================================================================
# TEST 4: Step-by-step state tracking
# =============================================================================
print("\n" + "="*80)
print("TEST 4: State Variable Tracking")
print("="*80)

debug_kernel = r'''
extern "C" __global__
void debug_state_xoshiro(unsigned long long seed, unsigned long long* outputs, 
                          unsigned long long* state_log, int n, int skip_val) {
    unsigned long long s0 = seed;
    unsigned long long s1 = 0x9E3779B97F4A7C15ULL;
    unsigned long long s2 = 0x6A09E667F3BCC908ULL;
    unsigned long long s3 = 0xBB67AE8584CAA73BULL;
    
    int log_idx = 0;
    
    // Log initial state
    state_log[log_idx++] = s0;
    state_log[log_idx++] = s1;
    state_log[log_idx++] = s2;
    state_log[log_idx++] = s3;
    
    // Skip
    for (int s = 0; s < skip_val; s++) {
        unsigned long long temp = s0 + s3;
        unsigned long long result = ((temp << 23) | (temp >> 41)) + s0;
        unsigned long long t = s1 << 17;
        s2 ^= s0;
        s3 ^= s1;
        s1 ^= s2;
        s0 ^= s3;
        s2 ^= t;
        s3 = ((s3 << 45) | (s3 >> 19));
    }
    
    // Log state after skip
    state_log[log_idx++] = s0;
    state_log[log_idx++] = s1;
    state_log[log_idx++] = s2;
    state_log[log_idx++] = s3;
    
    // Generate first 3 outputs and log states
    for (int i = 0; i < 3; i++) {
        unsigned long long temp = s0 + s3;
        unsigned long long result = ((temp << 23) | (temp >> 41)) + s0;
        unsigned long long t = s1 << 17;
        s2 ^= s0;
        s3 ^= s1;
        s1 ^= s2;
        s0 ^= s3;
        s2 ^= t;
        s3 = ((s3 << 45) | (s3 >> 19));
        outputs[i] = result;
        
        // Log state after each output
        state_log[log_idx++] = s0;
        state_log[log_idx++] = s1;
        state_log[log_idx++] = s2;
        state_log[log_idx++] = s3;
    }
}
'''

module = cp.RawModule(code=debug_kernel)
kernel = module.get_function('debug_state_xoshiro')
outputs_gpu = cp.zeros(3, dtype=cp.uint64)
state_log = cp.zeros(100, dtype=cp.uint64)
kernel((1,), (1,), (cp.uint64(1234), outputs_gpu, state_log, 3, 5))

outputs = outputs_gpu.get()
states = state_log.get()

print("State progression:")
print(f"Initial:       s0={states[0]:016x} s1={states[1]:016x} s2={states[2]:016x} s3={states[3]:016x}")
print(f"After skip(5): s0={states[4]:016x} s1={states[5]:016x} s2={states[6]:016x} s3={states[7]:016x}")
print()
print("Outputs and states:")
for i in range(3):
    idx = 8 + i*4
    print(f"Output {i}: {int(outputs[i] % 1000):3d} (expected {cpu_expected[i]:3d})")
    print(f"  State: s0={states[idx]:016x} s1={states[idx+1]:016x}")
    print(f"         s2={states[idx+2]:016x} s3={states[idx+3]:016x}")
    print()

print("\n" + "="*80)
print("SUMMARY OF FINDINGS")
print("="*80)
print()
print("Run this script to identify the source of the +392 bug.")
print("Next steps based on results:")
print("  • If single-threaded fixes it → threading/indexing issue")
print("  • If aligned memory fixes it → memory alignment issue")
print("  • If neither fixes it → fundamental CUDA compiler or arithmetic bug")
print()
print("Recommended action: Compare state_log output above with CPU reference")
print("to identify EXACTLY when the divergence occurs.")
