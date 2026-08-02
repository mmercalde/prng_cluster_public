#!/bin/bash
# Quick test of ALL 22 reverse kernels (forward + reverse)

PRNGS=(
    "java_lcg"
    "java_lcg_hybrid"
    "mt19937"
    "mt19937_hybrid"
    "xorshift32"
    "xorshift32_hybrid"
    "xorshift64"
    "xorshift64_hybrid"
    "xorshift128"
    "xorshift128_hybrid"
    "pcg32"
    "pcg32_hybrid"
    "lcg32"
    "lcg32_hybrid"
    "minstd"
    "minstd_hybrid"
    "philox4x32"
    "philox4x32_hybrid"
    "xoshiro256pp"
    "sfc64"
)

echo "================================================================================"
echo "QUICK TEST: ALL 22 REVERSE KERNELS"
echo "Test: 25k seeds, window 512, skip 0-10"
echo "================================================================================"

# [S184 bounded Phase 6 §6] This script used to write `reverse_kernel_test_
# results.txt` and TRUNCATE it (`> $RESULTS_FILE`). Team Beta ordered that file
# marked SUPERSEDED IN PLACE with its original rows preserved; a generator that
# truncates it would silently undo that order on the next run. So the output now
# goes to a timestamped file and the superseded record is never touched.
#
# Note also what this script actually asserts: only that the forward and reverse
# survivor COUNTS DIFFER. It is a differential check, not a known-answer test,
# and its "BOTH ZERO" branch is explicitly not counted as either pass or fail —
# which is how a 20/20 all-zero run came to be committed as if it were a result.
# For known-answer evidence on the four java_lcg variants see
# tests/phase6/known_answer_gate.py.
RESULTS_FILE="reverse_kernel_test_results_$(date -u +%Y%m%dT%H%M%SZ).txt"
if [ -e "reverse_kernel_test_results.txt" ]; then
    echo "NOTE: reverse_kernel_test_results.txt is SUPERSEDED and will not be"
    echo "      overwritten. Writing this run to: $RESULTS_FILE"
fi
> $RESULTS_FILE

passed=0
failed=0
total=0

for prng in "${PRNGS[@]}"; do
    total=$((total + 1))
    echo ""
    echo "[$total/${#PRNGS[@]}] Testing: $prng vs ${prng}_reverse"
    echo "----------------------------------------"
    
    # Forward
    echo -n "  Forward...  "
    fwd_output=$(python3 coordinator.py daily3.json \
        --method residue_sieve \
        --prng-type "$prng" \
        --seeds 25000 \
        --window-size 512 \
        --skip-min 0 \
        --skip-max 10 \
        --session-filter both \
        --max-concurrent 26 \
        2>&1 | grep "Successful jobs:")
    
    fwd_count=$(echo "$fwd_output" | grep -oP 'Successful jobs: \K\d+' || echo "0")
    echo "$fwd_count survivors"
    
    # Reverse
    echo -n "  Reverse...  "
    rev_output=$(python3 coordinator.py daily3.json \
        --method residue_sieve \
        --prng-type "${prng}_reverse" \
        --seeds 25000 \
        --window-size 512 \
        --skip-min 0 \
        --skip-max 10 \
        --session-filter both \
        --max-concurrent 26 \
        2>&1 | grep "Successful jobs:")
    
    rev_count=$(echo "$rev_output" | grep -oP 'Successful jobs: \K\d+' || echo "0")
    echo "$rev_count survivors"
    
    # Compare
    if [ "$fwd_count" == "0" ] && [ "$rev_count" == "0" ]; then
        status="⚠️  BOTH ZERO"
        echo "  Result: $status (might be valid - no survivors found)"
    elif [ "$fwd_count" == "$rev_count" ] && [ "$fwd_count" != "0" ]; then
        status="❌ IDENTICAL"
        failed=$((failed + 1))
        echo "  Result: $status (BROKEN - same survivor count)"
    else
        status="✅ DIFFERENT"
        passed=$((passed + 1))
        diff=$((fwd_count > rev_count ? fwd_count - rev_count : rev_count - fwd_count))
        echo "  Result: $status (Δ=$diff)"
    fi
    
    # Log results
    echo "$prng: Forward=$fwd_count, Reverse=$rev_count, $status" >> $RESULTS_FILE
done

echo ""
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo "Tested: $total PRNG pairs"
echo "Passed: $passed (forward ≠ reverse)"
echo "Failed: $failed (forward = reverse, non-zero)"
echo ""
echo "Detailed results saved to: $RESULTS_FILE"
echo "================================================================================"

if [ $failed -eq 0 ]; then
    echo "✅ ALL TESTS PASSED!"
    exit 0
else
    echo "❌ $failed TESTS FAILED - Some kernels still broken"
    exit 1
fi

