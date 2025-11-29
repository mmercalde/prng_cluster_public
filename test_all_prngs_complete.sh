#!/bin/bash

echo "=================================================="
echo "COMPLETE PRNG TEST SUITE - ALL 10 PRNGs"
echo "Testing: Fixed Skip (5) + Hybrid (5)"
echo "=================================================="

# Clean up
rm -rf results/analysis_* .analysis_progress_*

SEED_COUNT=10000
THRESHOLD=0.50

GREEN='\033[0;32m'
RED='\033[0;31m'
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
echo "SECTION 1: FIXED SKIP (5 PRNGs)"
run_test "xorshift32 (fixed)" "test_multi_prng_xorshift32.json" "xorshift32" "residue_sieve" "--skip 5"
run_test "pcg32 (fixed)" "test_multi_prng_pcg32.json" "pcg32" "residue_sieve" "--skip 5"
run_test "lcg32 (fixed)" "test_multi_prng_lcg32.json" "lcg32" "residue_sieve" "--skip 5"
run_test "xorshift64 (fixed)" "test_multi_prng_xorshift64.json" "xorshift64" "residue_sieve" "--skip 5"
run_test "mt19937 (fixed)" "test_multi_prng_mt19937.json" "mt19937" "residue_sieve" "--skip 5"

echo ""
echo "SECTION 2: HYBRID (5 PRNGs)"
run_test "xorshift32_hybrid" "test_xorshift32_hybrid.json" "xorshift32_hybrid" "residue_sieve" "--hybrid"
run_test "pcg32_hybrid" "test_pcg32_hybrid.json" "pcg32_hybrid" "residue_sieve" "--hybrid"
run_test "lcg32_hybrid" "test_lcg32_hybrid.json" "lcg32_hybrid" "residue_sieve" "--hybrid"
run_test "xorshift64_hybrid" "test_xorshift64_hybrid.json" "xorshift64_hybrid" "residue_sieve" "--hybrid"
run_test "mt19937_hybrid" "test_26gpu_hybrid.json" "mt19937_hybrid" "residue_sieve" "--hybrid"

echo ""
echo "FINAL SUMMARY"
echo "=============="
echo -e "Total: $((PASSED + FAILED))"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"

if [ $FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 ALL 10 PRNG TESTS PASSED! 🎉${NC}"
    exit 0
else
    echo ""
    for test in "${FAILED_TESTS[@]}"; do
        echo -e "  ${RED}❌ $test${NC}"
    done
    exit 1
fi
