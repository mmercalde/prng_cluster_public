#!/bin/bash

PASS=0
FAIL=0

echo "=========================================="
echo "LCG32 COMPREHENSIVE TEST - ALL 4 VARIANTS"
echo "=========================================="
echo ""

# Test 1: Forward Constant Skip
echo "TEST 1/4: lcg32 (Forward Constant Skip)"
python3 coordinator.py \
  test_multi_prng_lcg32.json \
  --method residue_sieve \
  --prng-type lcg32 \
  --seeds 5000 \
  --window-size 512 \
  --threshold 0.50 \
  --skip 5 2>&1 | tee /tmp/lcg32_test1.log

if grep -q "Successful: 26" /tmp/lcg32_test1.log && grep -q "Failed: 0" /tmp/lcg32_test1.log; then
    echo "✅ TEST 1 PASSED"
    ((PASS++))
else
    echo "❌ TEST 1 FAILED"
    ((FAIL++))
fi
echo ""

# Test 2: Forward Variable Skip
echo "TEST 2/4: lcg32_hybrid (Forward Variable Skip)"
python3 coordinator.py \
  test_lcg32_hybrid.json \
  --method residue_sieve \
  --prng-type lcg32_hybrid \
  --seeds 5000 \
  --window-size 512 \
  --threshold 0.50 \
  --hybrid 2>&1 | tee /tmp/lcg32_test2.log

if grep -q "Successful: 26" /tmp/lcg32_test2.log && grep -q "Failed: 0" /tmp/lcg32_test2.log; then
    echo "✅ TEST 2 PASSED"
    ((PASS++))
else
    echo "❌ TEST 2 FAILED"
    ((FAIL++))
fi
echo ""

# Test 3: Reverse Constant Skip
echo "TEST 3/4: lcg32_reverse (Reverse Constant Skip)"
python3 coordinator.py \
  test_multi_prng_lcg32.json \
  --method residue_sieve \
  --prng-type lcg32_reverse \
  --seeds 5000 \
  --window-size 512 \
  --threshold 0.50 \
  --skip 5 2>&1 | tee /tmp/lcg32_test3.log

if grep -q "Successful: 26" /tmp/lcg32_test3.log && grep -q "Failed: 0" /tmp/lcg32_test3.log; then
    echo "✅ TEST 3 PASSED"
    ((PASS++))
else
    echo "❌ TEST 3 FAILED"
    ((FAIL++))
fi
echo ""

# Test 4: Reverse Variable Skip
echo "TEST 4/4: lcg32_hybrid_reverse (Reverse Variable Skip)"
python3 coordinator.py \
  test_lcg32_hybrid.json \
  --method residue_sieve \
  --prng-type lcg32_hybrid_reverse \
  --seeds 5000 \
  --window-size 512 \
  --threshold 0.50 \
  --hybrid 2>&1 | tee /tmp/lcg32_test4.log

if grep -q "Successful: 26" /tmp/lcg32_test4.log && grep -q "Failed: 0" /tmp/lcg32_test4.log; then
    echo "✅ TEST 4 PASSED"
    ((PASS++))
else
    echo "❌ TEST 4 FAILED"
    ((FAIL++))
fi
echo ""

echo "=========================================="
echo "FINAL RESULTS"
echo "=========================================="
echo "Passed: $PASS/4"
echo "Failed: $FAIL/4"
echo ""

if [ $PASS -eq 4 ]; then
    echo "🎉 ALL 4 LCG32 VARIANTS WORKING! 🎉"
else
    echo "⚠️  Some tests failed - check logs in /tmp/lcg32_test*.log"
fi
