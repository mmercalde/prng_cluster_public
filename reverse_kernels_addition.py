"""
REVERSE SIEVE KERNELS - Using SAME MT19937 as forward sieve
"""

# ============================================================================
# MT19937 REVERSE FIXED SKIP KERNEL
# Uses IDENTICAL mt19937_extract() function from forward sieve
# ============================================================================

MT19937_REVERSE_KERNEL = r'''
typedef unsigned int uint32_t;

// MT19937 extract - IDENTICAL to forward sieve
static __device__ __forceinline__ uint32_t mt19937_extract(
    uint32_t* mt, int& mti, const int N, const int M,
    const uint32_t UPPER_MASK, const uint32_t LOWER_MASK,
    const uint32_t MATRIX_A
) {
    if (mti >= N) {
        int kk = 0;
        uint32_t y;
        for (; kk < N - M; ++kk) {
            y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + M] ^ (y >> 1) ^ ((y & 1U) ? MATRIX_A : 0U);
        }
        for (; kk < N - 1; ++kk) {
            y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ ((y & 1U) ? MATRIX_A : 0U);
        }
        y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
        mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ ((y & 1U) ? MATRIX_A : 0U);
        mti = 0;
    }
    uint32_t y = mt[mti++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9D2C5680U;
    y ^= (y << 15) & 0xEFC60000U;
    y ^= (y >> 18);
    return y;
}

extern "C" __global__
void mt19937_reverse_sieve(
    unsigned int* candidate_seeds, unsigned int* residues, unsigned int* survivors,
    float* match_rates, unsigned char* best_skips, unsigned int* survivor_count,
    int n_candidates, int k, int skip_min, int skip_max, float threshold, int offset
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_candidates) return;

    unsigned int seed = candidate_seeds[idx];
    float best_rate = 0.0f;
    int best_skip_val = 0;

    // MT19937 constants - IDENTICAL to forward sieve
    const int N = 624;
    const int M = 397;
    const unsigned int MATRIX_A = 0x9908B0DFU;
    const unsigned int UPPER_MASK = 0x80000000U;
    const unsigned int LOWER_MASK = 0x7FFFFFFFU;

    for (int skip = skip_min; skip <= skip_max; skip++) {
        // Allocate 624-word state array
        uint32_t mt[624];
        int mti;

        // Initialize MT19937 state - IDENTICAL to forward sieve
        mt[0] = seed;
        for (mti = 1; mti < N; mti++) {
            mt[mti] = (1812433253U * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti);
        }
        mti = N;

        // Pre-advance by offset - IDENTICAL to forward sieve
        for (int o = 0; o < offset; o++) {
            mt19937_extract(mt, mti, N, M, UPPER_MASK, LOWER_MASK, MATRIX_A);
        }

        // Burn skip values before first draw - IDENTICAL to forward sieve
        for (int s = 0; s < skip; s++) {
            mt19937_extract(mt, mti, N, M, UPPER_MASK, LOWER_MASK, MATRIX_A);
        }

        int matches = 0;
        for (int i = 0; i < k; i++) {
            unsigned int output = mt19937_extract(mt, mti, N, M, UPPER_MASK, LOWER_MASK, MATRIX_A);

            // 3-lane matching - IDENTICAL to forward sieve
            if (((output % 1000) == (unsigned int)(residues[i] % 1000)) &&
                ((output % 8) == (unsigned int)(residues[i] % 8)) &&
                ((output % 125) == (unsigned int)(residues[i] % 125))) {
                matches++;
            }

            // Skip between draws - IDENTICAL to forward sieve
            for (int s = 0; s < skip; s++) {
                mt19937_extract(mt, mti, N, M, UPPER_MASK, LOWER_MASK, MATRIX_A);
            }
        }

        float rate = ((float)matches) / ((float)k);
        if (rate > best_rate) {
            best_rate = rate;
            best_skip_val = skip;
        }
    }

    if (best_rate >= threshold) {
        int pos = atomicAdd(survivor_count, 1);
        survivors[pos] = seed;
        match_rates[pos] = best_rate;
        best_skips[pos] = (unsigned char)best_skip_val;
    }
}
'''

# ============================================================================
# MT19937 REVERSE HYBRID KERNEL (Variable Skip)
# Uses IDENTICAL mt19937_extract() function from forward sieve
# ============================================================================

MT19937_HYBRID_REVERSE_KERNEL = r'''
typedef unsigned int uint32_t;

// MT19937 extract - IDENTICAL to forward sieve
static __device__ __forceinline__ uint32_t mt19937_extract(
    uint32_t* mt, int& mti, const int N, const int M,
    const uint32_t UPPER_MASK, const uint32_t LOWER_MASK,
    const uint32_t MATRIX_A
) {
    if (mti >= N) {
        int kk = 0;
        uint32_t y;
        for (; kk < N - M; ++kk) {
            y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + M] ^ (y >> 1) ^ ((y & 1U) ? MATRIX_A : 0U);
        }
        for (; kk < N - 1; ++kk) {
            y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ ((y & 1U) ? MATRIX_A : 0U);
        }
        y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
        mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ ((y & 1U) ? MATRIX_A : 0U);
        mti = 0;
    }
    uint32_t y = mt[mti++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9D2C5680U;
    y ^= (y << 15) & 0xEFC60000U;
    y ^= (y >> 18);
    return y;
}

extern "C" __global__
void mt19937_hybrid_reverse_sieve(
    unsigned int* candidate_seeds, unsigned int* residues, unsigned int* survivors,
    float* match_rates, unsigned int* skip_sequences, unsigned int* strategy_ids,
    unsigned int* survivor_count, int n_candidates, int k,
    int* strategy_max_misses, int* strategy_tolerances, int n_strategies,
    float threshold, int offset
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_candidates) return;

    unsigned int seed = candidate_seeds[idx];

    // MT19937 constants - IDENTICAL to forward sieve
    const int N = 624;
    const int M = 397;
    const unsigned int MATRIX_A = 0x9908B0DFU;
    const unsigned int UPPER_MASK = 0x80000000U;
    const unsigned int LOWER_MASK = 0x7FFFFFFFU;

    // Test each strategy
    for (int strat_id = 0; strat_id < n_strategies; strat_id++) {
        int max_consecutive_misses = strategy_max_misses[strat_id];
        int skip_tolerance = strategy_tolerances[strat_id];

        // Initialize MT19937 state - IDENTICAL to forward sieve
        uint32_t mt[624];
        int mti;
        mt[0] = seed;
        for (mti = 1; mti < N; mti++) {
            mt[mti] = (1812433253U * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti);
        }
        mti = N;

        // Pre-advance by offset - IDENTICAL to forward sieve
        for (int o = 0; o < offset; o++) {
            mt19937_extract(mt, mti, N, M, UPPER_MASK, LOWER_MASK, MATRIX_A);
        }

        int matches = 0;
        int consecutive_misses = 0;
        unsigned int skip_seq[512];
        bool failed = false;

        for (int i = 0; i < k && !failed; i++) {
            bool found = false;

            // Try skip values within tolerance
            for (int try_skip = 0; try_skip <= skip_tolerance && !found; try_skip++) {
                // Save state
                uint32_t mt_save[624];
                for (int j = 0; j < N; j++) mt_save[j] = mt[j];
                int mti_save = mti;

                // Apply skip
                for (int s = 0; s < try_skip; s++) {
                    mt19937_extract(mt, mti, N, M, UPPER_MASK, LOWER_MASK, MATRIX_A);
                }

                // Generate output
                unsigned int output = mt19937_extract(mt, mti, N, M, UPPER_MASK, LOWER_MASK, MATRIX_A);

                // 3-lane matching - IDENTICAL to forward sieve
                if (((output % 1000) == (unsigned int)(residues[i] % 1000)) &&
                    ((output % 8) == (unsigned int)(residues[i] % 8)) &&
                    ((output % 125) == (unsigned int)(residues[i] % 125))) {
                    found = true;
                    matches++;
                    consecutive_misses = 0;
                    skip_seq[i] = try_skip;
                } else {
                    // Restore state and try next skip
                    for (int j = 0; j < N; j++) mt[j] = mt_save[j];
                    mti = mti_save;
                }
            }

            if (!found) {
                consecutive_misses++;
                if (consecutive_misses > max_consecutive_misses) {
                    failed = true;
                }
                skip_seq[i] = 0;
            }
        }

        if (!failed) {
            float rate = ((float)matches) / ((float)k);

            if (rate >= threshold) {
                int pos = atomicAdd(survivor_count, 1);
                survivors[pos] = seed;
                match_rates[pos] = rate;
                strategy_ids[pos] = strat_id;

                // Copy skip sequence
                for (int i = 0; i < k; i++) {
                    skip_sequences[pos * 512 + i] = skip_seq[i];
                }

                return; // Found with this strategy
            }
        }
    }
}
'''

# Registry entries to ADD to KERNEL_REGISTRY
REVERSE_KERNEL_ADDITIONS = {
    'mt19937_reverse': {
        'kernel_source': MT19937_REVERSE_KERNEL,
        'kernel_name': 'mt19937_reverse_sieve',
        'description': 'MT19937 reverse sieve - IDENTICAL PRNG to forward sieve'
    },
    'mt19937_hybrid_reverse': {
        'kernel_source': MT19937_HYBRID_REVERSE_KERNEL,
        'kernel_name': 'mt19937_hybrid_reverse_sieve',
        'description': 'MT19937 hybrid reverse - IDENTICAL PRNG to forward sieve'
    }
}

print("✅ Reverse kernels using SAME MT19937 as forward sieve")
