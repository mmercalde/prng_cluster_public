#!/bin/bash
# Test all 18 PRNG variants using EXISTING test files

echo "=========================================================================="
echo "COMPREHENSIVE TEST: ALL 18 PRNG VARIANTS"
echo "=========================================================================="
echo "Started: $(date)"
echo "=========================================================================="
echo ""

TOTAL=0
PASSED=0
FAILED=0

# Test function
test_prng() {
    local NAME=$1
    local PRNG_TYPE=$2
    local FLAGS=$3
    
    TOTAL=$((TOTAL + 1))
    echo -n "[$TOTAL/18] $NAME... "
    
    OUTPUT=$(echo "s" | timeout 120s python3 coordinator.py \
        test_multi_prng_${NAME}.json \
        --method residue_sieve \
        --prng-type $PRNG_TYPE \
        --seeds 5000 \
        --window-size 512 \
        --threshold 0.50 \
        $FLAGS \
        2>&1)
    
    if [ $? -eq 0 ] && echo "$OUTPUT" | grep -q "Successful: 26"; then
        echo "✅ PASS"
        PASSED=$((PASSED + 1))
    else
        echo "❌ FAIL"
        FAILED=$((FAILED + 1))
    fi
}

echo "PHASE 1: BASE PRNGs (Fixed-Skip)"
echo "----------------------------------------------------------------------"
test_prng "xorshift32" "xorshift32" "--skip 5"
test_prng "xorshift64" "xorshift64" "--skip 5"
test_prng "xorshift128" "xorshift128" "--skip 5"
test_prng "pcg32" "pcg32" "--skip 5"
test_prng "lcg32" "lcg32" "--skip 5"
test_prng "java_lcg" "java_lcg" "--skip 5"
test_prng "minstd" "minstd" "--skip 5"
test_prng "mt19937" "mt19937" "--skip 5"
echo ""

echo "PHASE 2: HYBRID PRNGs (Variable Skip)"
echo "----------------------------------------------------------------------"
test_prng "xorshift32" "xorshift32_hybrid" "--hybrid"
test_prng "xorshift64" "xorshift64_hybrid" "--hybrid"
test_prng "xorshift128" "xorshift128_hybrid" "--hybrid"
test_prng "pcg32" "pcg32_hybrid" "--hybrid"
test_prng "lcg32" "lcg32_hybrid" "--hybrid"
test_prng "java_lcg" "java_lcg_hybrid" "--hybrid"
test_prng "minstd" "minstd_hybrid" "--hybrid"
test_prng "mt19937" "mt19937_hybrid" "--hybrid"
echo ""

echo "PHASE 3: REVERSE PRNGs"
echo "----------------------------------------------------------------------"
test_prng "mt19937" "mt19937_reverse" "--skip 5 --reverse"
test_prng "mt19937" "mt19937_hybrid_reverse" "--hybrid --reverse"
echo ""

echo "=========================================================================="
echo "FINAL RESULTS"
echo "=========================================================================="
echo "✅ Passed: $PASSED/$TOTAL"
echo "❌ Failed: $FAILED/$TOTAL"

if [ $FAILED -eq 0 ]; then
    echo ""
    echo "🎉 ALL 18 PRNG VARIANTS WORKING!"
    exit 0
else
    echo ""
    echo "⚠️  Some tests failed"
    exit 1
fi
