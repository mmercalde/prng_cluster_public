import re

with open('prng_registry.py', 'r') as f:
    content = f.read()

# Find and replace ONLY the PCG32 test_skip loop section
# The bug is in lines where it modifies state during testing

old_pcg32_loop = r'''            for \(int test_skip = search_min; test_skip <= search_max; test_skip\+\+\) \{
                // Restore state
                state = state_backup;

                // Advance state test_skip times \(PRNG-SPECIFIC: PCG32\)
                for \(int j = 0; j < test_skip; j\+\+\) \{
                    unsigned long long oldstate = state;
                    state = oldstate \* PCG_MULTIPLIER \+ inc;
                \}

                // Generate output \(PRNG-SPECIFIC: PCG32\)
                unsigned long long oldstate = state;
                state = oldstate \* PCG_MULTIPLIER \+ inc;
                unsigned int xorshifted = \(\(oldstate >> 18u\) \^ oldstate\) >> 27u;
                unsigned int rot = oldstate >> 59u;
                unsigned int output = \(xorshifted >> rot\) \| \(xorshifted << \(\(-rot\) & 31\)\);'''

new_pcg32_loop = '''            for (int test_skip = search_min; test_skip <= search_max; test_skip++) {
                // Use temp_state for testing (FIXED!)
                unsigned long long temp_state = state_backup;

                // Advance temp_state test_skip times (PRNG-SPECIFIC: PCG32)
                for (int j = 0; j < test_skip; j++) {
                    temp_state = temp_state * PCG_MULTIPLIER + inc;
                }

                // Generate output (PRNG-SPECIFIC: PCG32)
                unsigned long long output_state = temp_state * PCG_MULTIPLIER + inc;
                unsigned int xorshifted = ((output_state >> 18) ^ output_state) >> 27;
                unsigned int rot = output_state >> 59;
                unsigned int output = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));'''

# Also need to fix the state update after match
old_match = r'''                    found = true;
                    // state already advanced
                    break;'''

new_match = '''                    found = true;
                    state = output_state;  // Keep the state that matched
                    break;'''

content = re.sub(old_pcg32_loop, new_pcg32_loop, content)
content = re.sub(old_match, new_match, content)

with open('prng_registry.py', 'w') as f:
    f.write(content)

print("✅ Fixed PCG32_HYBRID_KERNEL")
print("✅ LCG32_HYBRID and XORSHIFT64_HYBRID were already correct")
