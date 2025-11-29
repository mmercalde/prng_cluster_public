#!/usr/bin/env python3
"""Add MT19937_HYBRID_KERNEL after line 632"""

with open('prng_registry.py', 'r') as f:
    lines = f.readlines()

# Insert after line 632 (where MT19937_KERNEL closes)
insert_at = 632

print(f"✅ Inserting at line {insert_at + 1}")

hybrid_code = """

# ============================================================================
# MT19937 HYBRID VARIABLE SKIP KERNEL (Multi-Strategy)
# Uses IDENTICAL MT19937 algorithm to fixed-skip kernel above
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
    
    const int N = 624;
    const int M = 397;
    const unsigned int MATRIX_A = 0x9908B0DFU;
    const unsigned int UPPER_MASK = 0x80000000U;
    const unsigned int LOWER_MASK = 0x7FFFFFFFU;
    
    float best_match_rate = 0.0f;
    int best_strategy_id = 0;
    unsigned int best_skip_seq[512];
    
    for (int strat_id = 0; strat_id < n_strategies; strat_id++) {
        int max_misses = strategy_max_misses[strat_id];
        int skip_tolerance = strategy_tolerances[strat_id];
        
        unsigned int mt[624];
        int mti;
        mt[0] = seed;
        for (mti = 1; mti < N; mti++) {
            mt[mti] = (1812433253U * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti);
        }
        mti = N;
        
        auto extract = [&]() -> unsigned int {
            if (mti >= N) {
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
        
        int matches = 0;
        int consecutive_misses = 0;
        int expected_skip = 5;
        unsigned int current_skip_seq[512];
        
        for (int draw_idx = 0; draw_idx < k && draw_idx < 512; draw_idx++) {
            unsigned int mt_backup[624];
            int mti_backup = mti;
            for (int i = 0; i < N; i++) mt_backup[i] = mt[i];
            
            bool found = false;
            int actual_skip = expected_skip;
            int search_min = (expected_skip > skip_tolerance) ? (expected_skip - skip_tolerance) : 0;
            int search_max = expected_skip + skip_tolerance;
            
            for (int test_skip = search_min; test_skip <= search_max; test_skip++) {
                mti = mti_backup;
                for (int i = 0; i < N; i++) mt[i] = mt_backup[i];
                for (int j = 0; j < test_skip; j++) extract();
                
                unsigned int output = extract();
                unsigned int draw = output % 1000;
                
                if (draw == residues[draw_idx]) {
                    matches++;
                    consecutive_misses = 0;
                    actual_skip = test_skip;
                    expected_skip = test_skip;
                    found = true;
                    break;
                }
            }
            
            if (draw_idx < 512) current_skip_seq[draw_idx] = actual_skip;
            if (!found) {
                consecutive_misses++;
                if (consecutive_misses >= max_misses) break;
            }
        }
        
        float match_rate = (float)matches / k;
        if (match_rate > best_match_rate) {
            best_match_rate = match_rate;
            best_strategy_id = strat_id;
            for (int i = 0; i < k && i < 512; i++) {
                best_skip_seq[i] = current_skip_seq[i];
            }
        }
    }
    
    if (best_match_rate >= threshold) {
        int pos = atomicAdd(survivor_count, 1);
        if (pos < n_seeds) {
            survivors[pos] = seed;
            match_rates[pos] = best_match_rate;
            strategy_ids[pos] = best_strategy_id;
            int seq_size = (k < 512) ? k : 512;
            for (int i = 0; i < seq_size; i++) {
                skip_sequences[pos * 512 + i] = best_skip_seq[i];
            }
        }
    }
}
'''
"""

new_lines = lines[:insert_at] + [hybrid_code] + lines[insert_at:]

with open('prng_registry.py', 'w') as f:
    f.writelines(new_lines)

print("✅ Added hybrid kernel")

try:
    exec(open('prng_registry.py').read())
    print("✅ File compiles!")
except Exception as e:
    print(f"❌ Error: {e}")
