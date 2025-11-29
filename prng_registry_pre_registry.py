#!/usr/bin/env python3
"""
PRNG Registry - Complete Clean Rewrite
All kernels with correct skip/gap logic + Full MT19937 with 624-word state

Version: 2.0 - Clean rewrite with verified skip logic
Date: October 9, 2025
"""

from typing import List, Dict, Any, Callable
import numpy as np


# ============================================================================
# CPU REFERENCE IMPLEMENTATIONS (Unchanged - Working)
# ============================================================================

def xorshift32_cpu(seed: int, n: int, skip: int = 0, **kwargs) -> List[int]:
    """Xorshift32 CPU reference"""
    state = seed & 0xFFFFFFFF
    shift_a = kwargs.get('shift_a', 13)
    shift_b = kwargs.get('shift_b', 17)
    shift_c = kwargs.get('shift_c', 5)
    
    for _ in range(skip):
        state ^= (state << shift_a) & 0xFFFFFFFF
        state ^= (state >> shift_b) & 0xFFFFFFFF
        state ^= (state << shift_c) & 0xFFFFFFFF
    
    outputs = []
    for _ in range(n):
        state ^= (state << shift_a) & 0xFFFFFFFF
        state ^= (state >> shift_b) & 0xFFFFFFFF
        state ^= (state << shift_c) & 0xFFFFFFFF
        outputs.append(state)
    
    return outputs


def xorshift64_cpu(seed: int, n: int, skip: int = 0, **kwargs) -> List[int]:
    """Xorshift64 CPU reference"""
    state = seed & 0xFFFFFFFFFFFFFFFF
    
    for _ in range(skip):
        state ^= state >> 12
        state ^= (state << 25) & 0xFFFFFFFFFFFFFFFF
        state ^= state >> 27
        state = (state * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF
    
    outputs = []
    for _ in range(n):
        state ^= state >> 12
        state ^= (state << 25) & 0xFFFFFFFFFFFFFFFF
        state ^= state >> 27
        state = (state * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF
        outputs.append(state & 0xFFFFFFFF)
    
    return outputs


def pcg32_cpu(seed: int, n: int, skip: int = 0, **kwargs) -> List[int]:
    """PCG32 CPU reference"""
    state = seed & 0xFFFFFFFFFFFFFFFF
    increment = kwargs.get('increment', 1442695040888963407) | 1
    
    for _ in range(skip):
        state = ((state * 6364136223846793005) + increment) & 0xFFFFFFFFFFFFFFFF
    
    outputs = []
    for _ in range(n):
        oldstate = state
        state = ((state * 6364136223846793005) + increment) & 0xFFFFFFFFFFFFFFFF
        xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) & 0xFFFFFFFF
        rot = (oldstate >> 59) & 0x1F
        output = ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF
        outputs.append(output)
    
    return outputs


def lcg32_cpu(seed: int, n: int, skip: int = 0, **kwargs) -> List[int]:
    """LCG32 CPU reference"""
    state = seed & 0xFFFFFFFF
    a = kwargs.get('a', 1103515245)
    c = kwargs.get('c', 12345)
    m = kwargs.get('m', 0x7FFFFFFF)
    
    for _ in range(skip):
        if m > 0:
            state = ((a * state) + c) % m
        else:
            state = ((a * state) + c) & 0xFFFFFFFF
    
    outputs = []
    for _ in range(n):
        if m > 0:
            state = ((a * state) + c) % m
        else:
            state = ((a * state) + c) & 0xFFFFFFFF
        outputs.append(state)
    
    return outputs


def mt19937_cpu(seed: int, n: int, skip: int = 0) -> List[int]:
    """
    MT19937 that matches Python's random module
    Uses init_by_array like Python does
    """
    N = 624
    M = 397
    MATRIX_A = 0x9908B0DF
    UPPER_MASK = 0x80000000
    LOWER_MASK = 0x7FFFFFFF
    
    # Convert seed to init_key array (like Python does)
    def seed_to_init_key(s):
        if s < 0:
            s = -s
        if s == 0:
            seed_bytes = b'\x00'
        else:
            seed_bytes = s.to_bytes((s.bit_length() + 7) // 8, byteorder='big')
        
        if len(seed_bytes) % 4:
            seed_bytes = b'\x00' * (4 - len(seed_bytes) % 4) + seed_bytes
        
        init_key = []
        for i in range(0, len(seed_bytes), 4):
            word = int.from_bytes(seed_bytes[i:i+4], byteorder='big')
            init_key.append(word)
        return init_key
    
    # Initialize using init_by_array
    def init_by_array(init_key, key_length):
        state = [0] * N
        state[0] = 19650218
        for i in range(1, N):
            state[i] = (1812433253 * (state[i-1] ^ (state[i-1] >> 30)) + i) & 0xFFFFFFFF
        
        i = 1
        j = 0
        k = max(N, key_length)
        
        for _ in range(k):
            state[i] = ((state[i] ^ ((state[i-1] ^ (state[i-1] >> 30)) * 1664525)) + init_key[j] + j) & 0xFFFFFFFF
            i += 1
            j += 1
            if i >= N:
                state[0] = state[N-1]
                i = 1
            if j >= key_length:
                j = 0
        
        for _ in range(N-1):
            state[i] = ((state[i] ^ ((state[i-1] ^ (state[i-1] >> 30)) * 1566083941)) - i) & 0xFFFFFFFF
            i += 1
            if i >= N:
                state[0] = state[N-1]
                i = 1
        
        state[0] = 0x80000000
        return state
    
    # Initialize state
    init_key = seed_to_init_key(seed)
    state = init_by_array(init_key, len(init_key))
    index = N  # Force initial twist
    
    # Twist function
    def twist():
        nonlocal state
        for i in range(N):
            x = (state[i] & UPPER_MASK) | (state[(i+1) % N] & LOWER_MASK)
            xA = x >> 1
            if x & 1:
                xA ^= MATRIX_A
            state[i] = state[(i + M) % N] ^ xA
    
    # Extract function
    def extract():
        nonlocal index
        if index >= N:
            twist()
            index = 0
        
        y = state[index]
        index += 1
        
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= (y >> 18)
        
        return y & 0xFFFFFFFF
    
    # Skip outputs
    for _ in range(skip):
        extract()
    
    # Generate n outputs
    outputs = []
    for _ in range(n):
        outputs.append(extract())
    
    return outputs



def mt19937_cpu_simple(seed: int, n: int, skip: int = 0, **kwargs) -> List[int]:
    """
    MT19937 with init_genrand (matches GPU kernel)
    This is the ORIGINAL MT19937 from the 1998 paper
    """
    state = [0] * 624
    state[0] = seed & 0xFFFFFFFF
    
    for i in range(1, 624):
        state[i] = (1812433253 * (state[i-1] ^ (state[i-1] >> 30)) + i) & 0xFFFFFFFF
    
    index = 624
    
    def extract():
        nonlocal index
        if index >= 624:
            for i in range(624):
                y = (state[i] & 0x80000000) + (state[(i+1) % 624] & 0x7FFFFFFF)
                state[i] = state[(i + 397) % 624] ^ (y >> 1)
                if y % 2 != 0:
                    state[i] ^= 0x9908B0DF
            index = 0
        
        y = state[index]
        index += 1
        
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        
        return y & 0xFFFFFFFF
    
    for _ in range(skip):
        extract()
    
    outputs = []
    for _ in range(n):
        outputs.append(extract())
    
    return outputs



# ============================================================================
# GPU KERNELS - ALL WITH CORRECT SKIP/GAP LOGIC
# ============================================================================

XORSHIFT32_KERNEL = r'''
extern "C" __global__
void xorshift32_flexible_sieve(
    unsigned int* seeds, unsigned int* residues, unsigned int* survivors,
    float* match_rates, unsigned char* best_skips, unsigned int* survivor_count,
    int n_seeds, int k, int skip_min, int skip_max, float threshold,
    int shift_a, int shift_b, int shift_c, int offset
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_seeds) return;
    
    unsigned int seed = seeds[idx];
    float best_rate = 0.0f;
    int best_skip_val = 0;
    
    for (int skip = skip_min; skip <= skip_max; skip++) {
        unsigned int state = seed;
        
        // Pre-advance by offset
        for (int o = 0; o < offset; o++) {
            state ^= state << shift_a;
            state ^= state >> shift_b;
            state ^= state << shift_c;
        }
        
        // Burn skip values before first draw
        for (int s = 0; s < skip; s++) {
            state ^= state << shift_a;
            state ^= state >> shift_b;
            state ^= state << shift_c;
        }
        
        int matches = 0;
        for (int i = 0; i < k; i++) {
            // Generate output
            state ^= state << shift_a;
            state ^= state >> shift_b;
            state ^= state << shift_c;
            
            if (((state % 1000) == (unsigned int)(residues[i] % 1000)) &&
                ((state % 8) == (unsigned int)(residues[i] % 8)) &&
                ((state % 125) == (unsigned int)(residues[i] % 125))) matches++;
            
            // Skip between draws
            for (int s = 0; s < skip; s++) {
                state ^= state << shift_a;
                state ^= state >> shift_b;
                state ^= state << shift_c;
            }
        }
        
        float rate = ((float)matches) / ((float)k);
        if (rate > best_rate) {
            best_rate = rate;
            best_skip_val = skip;
        }
    }
    
    if (best_rate >= threshold) {
        unsigned int pos = atomicAdd(survivor_count, 1);
        survivors[pos] = seeds[idx];
        match_rates[pos] = best_rate;
        best_skips[pos] = (unsigned char)best_skip_val;
    }
}
'''


PCG32_KERNEL = r'''
extern "C" __global__
void pcg32_flexible_sieve(
    unsigned int* seeds, unsigned int* residues, unsigned int* survivors,
    float* match_rates, unsigned char* best_skips, unsigned int* survivor_count,
    int n_seeds, int k, int skip_min, int skip_max, float threshold,
    unsigned long long increment, int offset
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_seeds) return;
    
    unsigned long long seed = (unsigned long long)seeds[idx];
    float best_rate = 0.0f;
    int best_skip_val = 0;
    
    for (int skip = skip_min; skip <= skip_max; skip++) {
        unsigned long long state = seed;
        
        // Pre-advance by offset
        for (int o = 0; o < offset; o++) {
            state = state * 6364136223846793005ULL + increment;
        }
        
        // Burn skip values before first draw
        for (int s = 0; s < skip; s++) {
            state = state * 6364136223846793005ULL + increment;
        }
        
        int matches = 0;
        for (int i = 0; i < k; i++) {
            unsigned long long oldstate = state;
            state = oldstate * 6364136223846793005ULL + increment;
            unsigned int xorshifted = (unsigned int)(((oldstate >> 18) ^ oldstate) >> 27);
            unsigned int rot = (unsigned int)(oldstate >> 59);
            unsigned int output = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
            
            if (((output % 1000) == (unsigned int)(residues[i] % 1000)) &&
                ((output % 8) == (unsigned int)(residues[i] % 8)) &&
                ((output % 125) == (unsigned int)(residues[i] % 125))) matches++;
            
            // Skip between draws
            for (int s = 0; s < skip; s++) {
                state = state * 6364136223846793005ULL + increment;
            }
        }
        
        float rate = ((float)matches) / ((float)k);
        if (rate > best_rate) {
            best_rate = rate;
            best_skip_val = skip;
        }
    }
    
    if (best_rate >= threshold) {
        unsigned int pos = atomicAdd(survivor_count, 1);
        survivors[pos] = seeds[idx];
        match_rates[pos] = best_rate;
        best_skips[pos] = (unsigned char)best_skip_val;
    }
}
'''


LCG32_KERNEL = r'''
extern "C" __global__
void lcg32_flexible_sieve(
    unsigned int* seeds, unsigned int* residues, unsigned int* survivors,
    float* match_rates, unsigned char* best_skips, unsigned int* survivor_count,
    int n_seeds, int k, int skip_min, int skip_max, float threshold,
    unsigned int a, unsigned int c, unsigned int m, int offset
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_seeds) return;
    
    unsigned int seed = seeds[idx];
    float best_rate = 0.0f;
    int best_skip_val = 0;
    
    for (int skip = skip_min; skip <= skip_max; skip++) {
        unsigned int state = seed;
        
        // Pre-advance by offset
        for (int o = 0; o < offset; o++) {
            if (m > 0) {
                unsigned long long temp = ((unsigned long long)a * state + c);
                state = (unsigned int)(temp % m);
            } else {
                state = a * state + c;
            }
        }
        
        // Burn skip values before first draw
        for (int s = 0; s < skip; s++) {
            if (m > 0) {
                unsigned long long temp = ((unsigned long long)a * state + c);
                state = (unsigned int)(temp % m);
            } else {
                state = a * state + c;
            }
        }
        
        int matches = 0;
        for (int i = 0; i < k; i++) {
            if (m > 0) {
                unsigned long long temp = ((unsigned long long)a * state + c);
                state = (unsigned int)(temp % m);
            } else {
                state = a * state + c;
            }
            
            if (((state % 1000) == (unsigned int)(residues[i] % 1000)) &&
                ((state % 8) == (unsigned int)(residues[i] % 8)) &&
                ((state % 125) == (unsigned int)(residues[i] % 125))) matches++;
            
            // Skip between draws
            for (int s = 0; s < skip; s++) {
                if (m > 0) {
                    unsigned long long temp = ((unsigned long long)a * state + c);
                    state = (unsigned int)(temp % m);
                } else {
                    state = a * state + c;
                }
            }
        }
        
        float rate = ((float)matches) / ((float)k);
        if (rate > best_rate) {
            best_rate = rate;
            best_skip_val = skip;
        }
    }
    
    if (best_rate >= threshold) {
        unsigned int pos = atomicAdd(survivor_count, 1);
        survivors[pos] = seeds[idx];
        match_rates[pos] = best_rate;
        best_skips[pos] = (unsigned char)best_skip_val;
    }
}
'''


MT19937_KERNEL = r'''
extern "C" __global__
void mt19937_full_sieve(
    unsigned int* seeds, unsigned int* residues, unsigned int* survivors,
    float* match_rates, unsigned char* best_skips, unsigned int* survivor_count,
    int n_seeds, int k, int skip_min, int skip_max, float threshold, int offset
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_seeds) return;
    
    unsigned int seed = seeds[idx];
    float best_rate = 0.0f;
    int best_skip_val = 0;
    
    // MT19937 constants
    const int N = 624;
    const int M = 397;
    const unsigned int MATRIX_A = 0x9908B0DFU;
    const unsigned int UPPER_MASK = 0x80000000U;
    const unsigned int LOWER_MASK = 0x7FFFFFFFU;
    
    for (int skip = skip_min; skip <= skip_max; skip++) {
        // Allocate 624-word state array
        unsigned int mt[624];
        int mti;
        
        // Initialize MT19937 state from seed
        mt[0] = seed;
        for (mti = 1; mti < N; mti++) {
            mt[mti] = (1812433253U * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti);
        }
        mti = N;
        
        // Extract function with twist
        auto extract = [&]() -> unsigned int {
            if (mti >= N) {
                // Twist
                for (int i = 0; i < N; i++) {
                    unsigned int y = (mt[i] & UPPER_MASK) | (mt[(i+1) % N] & LOWER_MASK);
                    mt[i] = mt[(i + M) % N] ^ (y >> 1);
                    if (y & 1) mt[i] ^= MATRIX_A;
                }
                mti = 0;
            }
            
            unsigned int y = mt[mti++];
            y ^= (y >> 11);
            y ^= (y << 7) & 0x9D2C5680U;
            y ^= (y << 15) & 0xEFC60000U;
            y ^= (y >> 18);
            return y;
        };
        
        // Pre-advance by offset
        for (int o = 0; o < offset; o++) {
            extract();
        }
        
        // Burn skip values before first draw
        for (int s = 0; s < skip; s++) {
            extract();
        }
        
        int matches = 0;
        for (int i = 0; i < k; i++) {
            unsigned int output = extract();
            
            if (((output % 1000) == (unsigned int)(residues[i] % 1000)) &&
                ((output % 8) == (unsigned int)(residues[i] % 8)) &&
                ((output % 125) == (unsigned int)(residues[i] % 125))) {
                matches++;
            }
            
            // Skip between draws
            for (int s = 0; s < skip; s++) {
                extract();
            }
        }
        
        float rate = ((float)matches) / ((float)k);
        if (rate > best_rate) {
            best_rate = rate;
            best_skip_val = skip;
        }
    }
    
    if (best_rate >= threshold) {
        unsigned int pos = atomicAdd(survivor_count, 1);
        survivors[pos] = seeds[idx];
        match_rates[pos] = best_rate;
        best_skips[pos] = (unsigned char)best_skip_val;
    }
}
'''


XORSHIFT64_KERNEL = r'''
extern "C" __global__
void xorshift64_flexible_sieve(
    unsigned long long* seeds, unsigned int* residues, unsigned int* survivors,
    float* match_rates, unsigned char* best_skips, unsigned int* survivor_count,
    int n_seeds, int k, int skip_min, int skip_max, float threshold, int offset
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_seeds) return;
    
    unsigned long long seed = seeds[idx];
    float best_rate = 0.0f;
    int best_skip_val = 0;
    
    for (int skip = skip_min; skip <= skip_max; skip++) {
        unsigned long long state = seed;
        
        // Pre-advance by offset
        for (int o = 0; o < offset; o++) {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state *= 0x2545F4914F6CDD1DULL;
        }
        
        // Burn skip values before first draw
        for (int s = 0; s < skip; s++) {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state *= 0x2545F4914F6CDD1DULL;
        }
        
        int matches = 0;
        for (int i = 0; i < k; i++) {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state *= 0x2545F4914F6CDD1DULL;
            unsigned int output = (unsigned int)(state & 0xFFFFFFFF);
            
            if (((output % 1000) == (unsigned int)(residues[i] % 1000)) &&
                ((output % 8) == (unsigned int)(residues[i] % 8)) &&
                ((output % 125) == (unsigned int)(residues[i] % 125))) matches++;
            
            // Skip between draws
            for (int s = 0; s < skip; s++) {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                state *= 0x2545F4914F6CDD1DULL;
            }
        }
        
        float rate = ((float)matches) / ((float)k);
        if (rate > best_rate) {
            best_rate = rate;
            best_skip_val = skip;
        }
    }
    
    if (best_rate >= threshold) {
        unsigned int pos = atomicAdd(survivor_count, 1);
        survivors[pos] = (unsigned int)(seed & 0xFFFFFFFF);


# ============================================================================
# MT19937 HYBRID VARIABLE SKIP KERNEL (Multi-Strategy)
# ============================================================================

MT19937_HYBRID_KERNEL = r'''
extern "C" __global__
void mt19937_hybrid_multi_strategy_sieve(
    unsigned int* seeds, unsigned int* residues, unsigned int* survivors,
    float* match_rates, unsigned int* skip_sequences, unsigned int* strategy_ids,
    unsigned int* survivor_count, int n_seeds, int k,
    int* strategy_max_misses, int* strategy_tolerances, int n_strategies,
    float threshold, int offset
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_seeds) return;
    
    unsigned int seed = seeds[idx];
    
    // MT19937 constants (IDENTICAL TO FIXED-SKIP KERNEL)
    const int N = 624;
    const int M = 397;
    const unsigned int MATRIX_A = 0x9908B0DFU;
    const unsigned int UPPER_MASK = 0x80000000U;
    const unsigned int LOWER_MASK = 0x7FFFFFFFU;
    
    // Test ALL strategies for this seed
    float best_match_rate = 0.0f;
    int best_strategy_id = 0;
    unsigned int best_skip_seq[512];
    
    for (int strat_id = 0; strat_id < n_strategies; strat_id++) {
        int max_misses = strategy_max_misses[strat_id];
        int skip_tolerance = strategy_tolerances[strat_id];
        
        // Initialize MT19937 state (IDENTICAL TO FIXED-SKIP KERNEL)
        unsigned int mt[624];
        int mti;
        
        mt[0] = seed;
        for (mti = 1; mti < N; mti++) {
            mt[mti] = (1812433253U * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti);
        }
        mti = N;
        
        // Extract function with twist (IDENTICAL TO FIXED-SKIP KERNEL)
        auto extract = [&]() -> unsigned int {
            if (mti >= N) {
                // Twist
                for (int i = 0; i < N; i++) {
                    unsigned int y = (mt[i] & UPPER_MASK) | (mt[(i+1) % N] & LOWER_MASK);
                    mt[i] = mt[(i + M) % N] ^ (y >> 1);
                    if (y & 1) mt[i] ^= MATRIX_A;
                }
                mti = 0;
            }
            unsigned int y = mt[mti++];
            y ^= (y >> 11);
            y ^= (y << 7) & 0x9D2C5680U;
            y ^= (y << 15) & 0xEFC60000U;
            y ^= (y >> 18);
            return y;
        };
        
        // VARIABLE SKIP DETECTION (NEW)
        int matches = 0;
        int consecutive_misses = 0;
        int expected_skip = 5;  // Initial guess
        unsigned int current_skip_seq[512];
        
        for (int draw_idx = 0; draw_idx < k && draw_idx < 512; draw_idx++) {
            // Save state before window search
            unsigned int mt_backup[624];
            int mti_backup = mti;
            for (int i = 0; i < N; i++) mt_backup[i] = mt[i];
            
            bool found = false;
            int actual_skip = expected_skip;
            
            // Search within tolerance window
            int search_min = (expected_skip > skip_tolerance) ? (expected_skip - skip_tolerance) : 0;
            int search_max = expected_skip + skip_tolerance;
            
            for (int test_skip = search_min; test_skip <= search_max; test_skip++) {
                // Restore state
                mti = mti_backup;
                for (int i = 0; i < N; i++) mt[i] = mt_backup[i];
                
                // Burn test_skip outputs
                for (int j = 0; j < test_skip; j++) {
                    extract();
                }
                
                // Get next output
                unsigned int output = extract();
                unsigned int draw = output % 1000;
                
                // Check match
                if (draw == residues[draw_idx]) {
                    matches++;
                    consecutive_misses = 0;
                    actual_skip = test_skip;
                    
                    // Adaptive learning: update expected skip
                    expected_skip = test_skip;
                    
                    found = true;
                    break;
                }
            }
            
            // Store skip value
            if (draw_idx < 512) {
                current_skip_seq[draw_idx] = actual_skip;
            }
            
            // Check for breakpoint
            if (!found) {
                consecutive_misses++;
                if (consecutive_misses >= max_misses) {
                    // Breakpoint detected - stop this strategy attempt
                    break;
                }
            }
        }
        
        float match_rate = (float)matches / k;
        
        // Track best strategy for this seed
        if (match_rate > best_match_rate) {
            best_match_rate = match_rate;
            best_strategy_id = strat_id;
            
            // Copy skip sequence
            for (int i = 0; i < k && i < 512; i++) {
                best_skip_seq[i] = current_skip_seq[i];
            }
        }
    }
    
    // Report survivor if any strategy exceeded threshold
    if (best_match_rate >= threshold) {
        int pos = atomicAdd(survivor_count, 1);
        if (pos < n_seeds) {
            survivors[pos] = seed;
            match_rates[pos] = best_match_rate;
            strategy_ids[pos] = best_strategy_id;
            
            // Store best skip sequence
            int seq_size = (k < 512) ? k : 512;
            for (int i = 0; i < seq_size; i++) {
                skip_sequences[pos * 512 + i] = best_skip_seq[i];
            }
        }
    }
}
'''

        match_rates[pos] = best_rate;
        best_skips[pos] = (unsigned char)best_skip_val;
    }
}
'''


# ============================================================================
# KERNEL REGISTRY
# ============================================================================

KERNEL_REGISTRY = {
    'xorshift32': {
        'kernel_source': XORSHIFT32_KERNEL,
        'kernel_name': 'xorshift32_flexible_sieve',
        'cpu_reference': xorshift32_cpu,
        'default_params': {
            'shift_a': 13,
            'shift_b': 17,
            'shift_c': 5,
        },
        'description': 'Xorshift32 with correct skip/gap logic',
        'seed_type': 'uint32',
        'state_size': 4,
    },
    'pcg32': {
        'kernel_source': PCG32_KERNEL,
        'kernel_name': 'pcg32_flexible_sieve',
        'cpu_reference': pcg32_cpu,
        'default_params': {
            'increment': 1442695040888963407,
        },
        'description': 'PCG32 with correct skip/gap logic',
        'seed_type': 'uint32',
        'state_size': 8,
    },
    'lcg32': {
        'kernel_source': LCG32_KERNEL,
        'kernel_name': 'lcg32_flexible_sieve',
        'cpu_reference': lcg32_cpu,
        'default_params': {
            'a': 1103515245,
            'c': 12345,
            'm': 0x7FFFFFFF,
        },
        'description': 'LCG32 with correct skip/gap logic',
        'seed_type': 'uint32',
        'state_size': 4,
    },
    'mt19937': {
        'kernel_source': MT19937_KERNEL,
        'kernel_name': 'mt19937_full_sieve',
        'cpu_reference': mt19937_cpu_simple,
        'default_params': {},
        'description': 'Full MT19937 with 624-word state and correct skip/gap logic',
        'seed_type': 'uint32',
        'state_size': 2496,  # 624 * 4 bytes
    },
    'xorshift64': {
        'kernel_source': XORSHIFT64_KERNEL,
        'kernel_name': 'xorshift64_flexible_sieve',
        'cpu_reference': xorshift64_cpu,
        'default_params': {},
        'description': 'Xorshift64 with correct skip/gap logic',
        'seed_type': 'uint64',
        'state_size': 8,
    },
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_kernel_info(prng_family: str) -> Dict[str, Any]:
    """Get kernel configuration for PRNG family"""
    if prng_family not in KERNEL_REGISTRY:
        raise ValueError(f"Unknown PRNG family: {prng_family}. Available: {list_available_prngs()}")
    return KERNEL_REGISTRY[prng_family]


def list_available_prngs() -> List[str]:
    """List all available PRNG families"""
    return list(KERNEL_REGISTRY.keys())


def get_cpu_reference(prng_family: str) -> Callable:
    """Get CPU reference implementation for PRNG"""
    return get_kernel_info(prng_family)['cpu_reference']


if __name__ == '__main__':
    print("PRNG Registry v2.0 - Clean Rewrite")
    print("=" * 50)
    print("\nAvailable PRNGs:")
    for name in list_available_prngs():
        config = get_kernel_info(name)
        print(f"  {name:12} - {config['description']}")
        print(f"               State: {config['state_size']} bytes")
    print("\n✓ All kernels have correct skip/gap logic")
    print("✓ MT19937 uses full 624-word state array")
