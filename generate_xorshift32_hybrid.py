#!/usr/bin/env python3
"""
Generate xorshift32_hybrid kernel by replacing PRNG-specific parts
"""

# The template - same structure as mt19937_hybrid but with xorshift32 state
XORSHIFT32_HYBRID_KERNEL = r'''
extern "C" __global__
void xorshift32_hybrid_multi_strategy_sieve(
    unsigned int* seeds, unsigned int* residues, unsigned int* survivors,
    float* match_rates, unsigned int* skip_sequences, unsigned int* strategy_ids,
    unsigned int* survivor_count, int n_seeds, int k,
    int* strategy_max_misses, int* strategy_tolerances, int n_strategies,
    float threshold, int shift_a, int shift_b, int shift_c, int offset
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_seeds) return;

    unsigned int seed = seeds[idx];
    
    float best_match_rate = 0.0f;
    int best_strategy_id = 0;
    unsigned int best_skip_seq[512];

    // Test each strategy
    for (int strat_id = 0; strat_id < n_strategies; strat_id++) {
        int max_misses = strategy_max_misses[strat_id];
        int skip_tolerance = strategy_tolerances[strat_id];

        // Initialize xorshift32 state (PRNG-SPECIFIC)
        unsigned int state = seed;

        int matches = 0;
        int consecutive_misses = 0;
        int expected_skip = 5;
        unsigned int current_skip_seq[512];

        for (int draw_idx = 0; draw_idx < k && draw_idx < 512; draw_idx++) {
            // Backup state
            unsigned int state_backup = state;
            
            bool found = false;
            int actual_skip = expected_skip;
            
            int search_min = (expected_skip > skip_tolerance) ? (expected_skip - skip_tolerance) : 0;
            int search_max = expected_skip + skip_tolerance;
            
            // Try different skip values
            for (int test_skip = search_min; test_skip <= search_max; test_skip++) {
                // Restore state
                state = state_backup;
                
                // Advance state test_skip times (PRNG-SPECIFIC)
                for (int j = 0; j < test_skip; j++) {
                    state ^= state << shift_a;
                    state ^= state >> shift_b;
                    state ^= state << shift_c;
                }
                
                // Generate output (PRNG-SPECIFIC)
                unsigned int temp_state = state;
                temp_state ^= temp_state << shift_a;
                temp_state ^= temp_state >> shift_b;
                temp_state ^= temp_state << shift_c;
                unsigned int output = temp_state;
                
                // Check 3-lane match (GENERIC)
                if (((output % 1000) == (unsigned int)(residues[draw_idx] % 1000)) &&
                    ((output % 8) == (unsigned int)(residues[draw_idx] % 8)) &&
                    ((output % 125) == (unsigned int)(residues[draw_idx] % 125))) {
                    matches++;
                    consecutive_misses = 0;
                    actual_skip = test_skip;
                    expected_skip = test_skip;
                    found = true;
                    state = temp_state;  // Keep the state that matched
                    break;
                }
            }
            
            if (draw_idx < 512) current_skip_seq[draw_idx] = actual_skip;
            
            if (!found) {
                consecutive_misses++;
                if (consecutive_misses >= max_misses) break;
            }
        }
        
        // Calculate match rate
        float match_rate = (float)matches / k;
        
        // Update best strategy
        if (match_rate > best_match_rate) {
            best_match_rate = match_rate;
            best_strategy_id = strat_id;
            for (int i = 0; i < k && i < 512; i++) {
                best_skip_seq[i] = current_skip_seq[i];
            }
        }
    }
    
    // Store survivor if above threshold
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

print("="*70)
print("XORSHIFT32_HYBRID KERNEL GENERATED")
print("="*70)
print("\nKernel size:", len(XORSHIFT32_HYBRID_KERNEL), "characters")
print("\nKey features:")
print("  ✅ Same strategy testing logic as mt19937")
print("  ✅ xorshift32-specific state initialization")
print("  ✅ xorshift32-specific state advancement")
print("  ✅ Same 3-lane matching")
print("  ✅ Same output format (skip_sequences array)")
print("\nReady to add to prng_registry.py")

# Save to file for inspection
with open('xorshift32_hybrid_kernel.txt', 'w') as f:
    f.write(XORSHIFT32_HYBRID_KERNEL)

print("\n✅ Saved to xorshift32_hybrid_kernel.txt for review")

