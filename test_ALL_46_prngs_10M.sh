#!/bin/bash
echo "Testing ALL 46 PRNGs with 10M seeds each"
echo "=========================================="

# ALL 46 PRNGs
PRNGS=(
    # Forward Constant (11)
    "lcg32" "xorshift32" "pcg32" "mt19937" "xorshift64" "java_lcg" "minstd" "xorshift128" "xoshiro256pp" "philox4x32" "sfc64"
    # Forward Variable (11)
    "xorshift32_hybrid" "pcg32_hybrid" "lcg32_hybrid" "xorshift64_hybrid" "mt19937_hybrid" "java_lcg_hybrid" "minstd_hybrid" "xorshift128_hybrid" "xoshiro256pp_hybrid" "philox4x32_hybrid" "sfc64_hybrid"
    # Reverse Constant (12)
    "mt19937_reverse" "lcg32_reverse" "xorshift32_reverse" "xorshift64_reverse" "xorshift128_reverse" "pcg32_reverse" "java_lcg_reverse" "minstd_reverse" "philox4x32_reverse" "xoshiro256pp_reverse" "sfc64_reverse"
    # Reverse Variable (12)
    "mt19937_hybrid_reverse" "lcg32_hybrid_reverse" "xorshift32_hybrid_reverse" "xorshift64_hybrid_reverse" "xorshift128_hybrid_reverse" "pcg32_hybrid_reverse" "java_lcg_hybrid_reverse" "minstd_hybrid_reverse" "philox4x32_hybrid_reverse" "xoshiro256pp_hybrid_reverse" "sfc64_hybrid_reverse"
)

START_TIME=$(date +%s)
SUCCESS=0
FAIL=0

for prng in "${PRNGS[@]}"; do
    echo ""
    echo "==================== $prng (10M seeds) ===================="
    OUTPUT=$(python3 coordinator.py \
        --resume-policy restart \
        --max-concurrent 26 \
        daily3.json \
        --method residue_sieve \
        --prng-type $prng \
        --skip-min 0 \
        --skip-max 20 \
        --threshold 0.01 \
        --window-size 768 \
        --session-filter both \
        --seed-start 0 \
        --seeds 10000000 2>&1)
    
    echo "$OUTPUT" | grep -E "Total runtime|Successful|Failed|COMPLETED"
    
    if echo "$OUTPUT" | grep -q "COMPLETED"; then
        ((SUCCESS++))
        echo "✅ $prng: SUCCESS"
    else
        ((FAIL++))
        echo "❌ $prng: FAILED"
    fi
done

END_TIME=$(date +%s)
TOTAL=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "All 46 PRNGs tested!"
echo "Success: $SUCCESS/46"
echo "Failed: $FAIL/46"
echo "Total time: ${TOTAL} seconds (~$((TOTAL/60)) minutes)"
