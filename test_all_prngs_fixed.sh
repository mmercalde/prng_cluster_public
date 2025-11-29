#!/bin/bash

echo "=================================================="
echo "COMPLETE PRNG TEST SUITE - Generating Test Data"
echo "=================================================="

# Generate missing test data first
python3 << 'GENDATA'
import json

def lcg32_step(state, a=1664525, c=1013904223, m=2**32):
    return (a * state + c) % m

def xorshift32_step(state, shift_a=13, shift_b=17, shift_c=5):
    state ^= (state << shift_a) & 0xFFFFFFFF
    state ^= (state >> shift_b) & 0xFFFFFFFF
    state ^= (state << shift_c) & 0xFFFFFFFF
    return state & 0xFFFFFFFF

def pcg32_step(state, inc=1442695040888963407):
    oldstate = state
    state = (state * 6364136223846793005 + inc) & 0xFFFFFFFFFFFFFFFF
    xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) & 0xFFFFFFFF
    rot = (oldstate >> 59) & 0x1F
    output = ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF
    return state, output

def xorshift64_step(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= (state >> 7) & 0xFFFFFFFFFFFFFFFF
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF

def mt19937_step(state):
    # Simplified MT19937 - just using python's random for demo
    import random
    random.seed(state)
    return random.getrandbits(32)

SEED = 12345
NUM_DRAWS = 512
SKIP = 5

# Generate fixed-skip test data
generators = {
    'lcg32': (lcg32_step, False),
    'xorshift32': (xorshift32_step, False),
    'xorshift64': (xorshift64_step, False),
}

for name, (gen_func, needs_output) in generators.items():
    print(f"Generating test_multi_prng_{name}.json...")
    state = SEED
    draws = []
    
    for i in range(NUM_DRAWS):
        # Skip forward
        for _ in range(SKIP):
            if needs_output:
                state, _ = gen_func(state)
            else:
                state = gen_func(state)
        
        # Get output
        if needs_output:
            state, output = gen_func(state)
        else:
            state = gen_func(state)
            output = state
        
        draw = output % 1000
        draws.append({'draw': draw, 'session': 'midday', 'timestamp': 3000000 + i})
    
    with open(f'test_multi_prng_{name}.json', 'w') as f:
        json.dump(draws, f, indent=2)
    print(f"  âœ… Created {len(draws)} draws")

# Generate PCG32 (needs special handling)
print("Generating test_multi_prng_pcg32.json...")
state = SEED
draws = []
for i in range(NUM_DRAWS):
    for _ in range(SKIP):
        state, _ = pcg32_step(state)
    state, output = pcg32_step(state)
    draw = output % 1000
    draws.append({'draw': draw, 'session': 'midday', 'timestamp': 3000000 + i})

with open('test_multi_prng_pcg32.json', 'w') as f:
    json.dump(draws, f, indent=2)
print(f"  âœ… Created {len(draws)} draws")

# Generate MT19937 fixed skip
print("Generating test_multi_prng_mt19937.json...")
import random
random.seed(SEED)
draws = []
for i in range(NUM_DRAWS):
    for _ in range(SKIP):
        random.getrandbits(32)
    output = random.getrandbits(32)
    draw = output % 1000
    draws.append({'draw': draw, 'session': 'midday', 'timestamp': 3000000 + i})

with open('test_multi_prng_mt19937.json', 'w') as f:
    json.dump(draws, f, indent=2)
print(f"  âœ… Created {len(draws)} draws")

# Generate MT19937 hybrid (variable skip)
print("Generating test_26gpu_hybrid.json...")
random.seed(SEED)
skip_pattern = [5, 5, 7, 7, 3, 3] * (NUM_DRAWS // 6 + 1)
skip_pattern = skip_pattern[:NUM_DRAWS]
draws = []
for i, skip in enumerate(skip_pattern):
    for _ in range(skip):
        random.getrandbits(32)
    output = random.getrandbits(32)
    draw = output % 1000
    draws.append({'draw': draw, 'session': 'midday', 'timestamp': 3000000 + i})

with open('test_26gpu_hybrid.json', 'w') as f:
    json.dump(draws, f, indent=2)
print(f"  âœ… Created {len(draws)} draws")

print("\nâœ… All test data generated!")
GENDATA

echo ""
echo "=================================================="
echo "COMPLETE PRNG TEST SUITE - ALL 10 PRNGs"
echo "Testing: Fixed Skip (5) + Hybrid (5)"
echo "=================================================="
echo ""

# Clean up old progress
rm -rf results/analysis_* .analysis_progress_*

SEED_COUNT=10000
THRESHOLD=0.50

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0
declare -a FAILED_TESTS

run_test() {
    local test_name=$1
    local test_file=$2
    local prng_type=$3
    local method=$4
    local extra_args=$5
    
    echo ""
    echo "========================================"
    echo "Testing: $test_name"
    echo "========================================"
    
    # Clean progress before each test
    rm -rf results/analysis_* .analysis_progress_* 2>/dev/null
    
    timeout 300 python3 coordinator.py "$test_file" \
        --method "$method" \
        --prng-type "$prng_type" \
        --seeds $SEED_COUNT \
        --window-size 512 \
        --threshold $THRESHOLD \
        --resume-policy restart \
        $extra_args 2>&1 | tee /tmp/test_output.txt
    
    if grep -q "Analysis completed successfully" /tmp/test_output.txt && \
       grep -q "Failed: 0" /tmp/test_output.txt; then
        echo -e "${GREEN}âœ… PASSED: $test_name${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}âŒ FAILED: $test_name${NC}"
        FAILED_TESTS+=("$test_name")
        ((FAILED++))
        return 1
    fi
}

echo ""
echo "=================================================="
echo "SECTION 1: FIXED SKIP (Constant Gap) - 5 PRNGs"
echo "=================================================="

run_test "xorshift32 (fixed skip)" "test_multi_prng_xorshift32.json" "xorshift32" "residue_sieve" "--skip 5"
run_test "pcg32 (fixed skip)" "test_multi_prng_pcg32.json" "pcg32" "residue_sieve" "--skip 5"
run_test "lcg32 (fixed skip)" "test_multi_prng_lcg32.json" "lcg32" "residue_sieve" "--skip 5"
run_test "xorshift64 (fixed skip)" "test_multi_prng_xorshift64.json" "xorshift64" "residue_sieve" "--skip 5"
run_test "mt19937 (fixed skip)" "test_multi_prng_mt19937.json" "mt19937" "residue_sieve" "--skip 5"

echo ""
echo "=================================================="
echo "SECTION 2: HYBRID (Variable Skip) - 5 PRNGs"
echo "=================================================="

run_test "xorshift32_hybrid" "test_xorshift32_hybrid.json" "xorshift32_hybrid" "residue_sieve" "--hybrid"
run_test "pcg32_hybrid" "test_pcg32_hybrid.json" "pcg32_hybrid" "residue_sieve" "--hybrid"
run_test "lcg32_hybrid" "test_lcg32_hybrid.json" "lcg32_hybrid" "residue_sieve" "--hybrid"
run_test "xorshift64_hybrid" "test_xorshift64_hybrid.json" "xorshift64_hybrid" "residue_sieve" "--hybrid"
run_test "mt19937_hybrid" "test_26gpu_hybrid.json" "mt19937_hybrid" "residue_sieve" "--hybrid"

echo ""
echo "=================================================="
echo "FINAL TEST SUMMARY - ALL 10 PRNGs"
echo "=================================================="
echo -e "Total Tests: $((PASSED + FAILED))"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed Tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo -e "  ${RED}âŒ $test${NC}"
    done
    exit 1
else
    echo ""
    echo -e "${GREEN}ðŸŽ‰ ALL 10 PRNG TESTS PASSED! ðŸŽ‰${NC}"
    echo ""
    echo "Verified PRNGs:"
    echo "  âœ… 5 Fixed Skip (Constant Gap)"
    echo "  âœ… 5 Hybrid (Variable Skip)"
    echo ""
    echo "Total: 10/10 PRNGs Operational on 26-GPU Cluster"
    exit 0
fi
