#!/bin/bash

echo "=================================================="
echo "COMPREHENSIVE PRNG TEST SUITE v2"
echo "Testing ALL PRNGs: Fixed Skip, Hybrid, and Reverse"
echo "=================================================="
echo ""

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
        echo -e "${GREEN}✅ PASSED: $test_name${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}❌ FAILED: $test_name${NC}"
        FAILED_TESTS+=("$test_name")
        ((FAILED++))
        return 1
    fi
}

echo ""
echo "=================================================="
echo "SECTION 1: FIXED SKIP (Constant Gap) - Forward"
echo "=================================================="

run_test "xorshift32 (fixed skip)" "test_multi_prng_xorshift32.json" "xorshift32" "residue_sieve" "--skip 5"
run_test "pcg32 (fixed skip)" "test_multi_prng_pcg32.json" "pcg32" "residue_sieve" "--skip 5"
run_test "lcg32 (fixed skip)" "test_multi_prng_lcg32.json" "lcg32" "residue_sieve" "--skip 5"
run_test "xorshift64 (fixed skip)" "test_multi_prng_xorshift64.json" "xorshift64" "residue_sieve" "--skip 5"
run_test "mt19937 (fixed skip)" "test_multi_prng_mt19937.json" "mt19937" "residue_sieve" "--skip 5"

echo ""
echo "=================================================="
echo "SECTION 2: HYBRID (Variable Skip) - Forward"
echo "=================================================="

run_test "xorshift32_hybrid" "test_xorshift32_hybrid.json" "xorshift32_hybrid" "residue_sieve" "--hybrid"
run_test "pcg32_hybrid" "test_pcg32_hybrid.json" "pcg32_hybrid" "residue_sieve" "--hybrid"
run_test "lcg32_hybrid" "test_lcg32_hybrid.json" "lcg32_hybrid" "residue_sieve" "--hybrid"
run_test "xorshift64_hybrid" "test_xorshift64_hybrid.json" "xorshift64_hybrid" "residue_sieve" "--hybrid"
run_test "mt19937_hybrid" "test_26gpu_hybrid.json" "mt19937_hybrid" "residue_sieve" "--hybrid"

echo ""
echo "=================================================="
echo "TEST SUMMARY"
echo "=================================================="
echo -e "Total Tests: $((PASSED + FAILED))"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed Tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo -e "  ${RED}❌ $test${NC}"
    done
    exit 1
else
    echo ""
    echo -e "${GREEN}🎉 ALL TESTS PASSED! 🎉${NC}"
    exit 0
fi
