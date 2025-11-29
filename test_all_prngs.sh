#!/bin/bash
echo "Testing all 16 PRNG configurations with 100K seeds each"
echo "========================================================"

PRNGS=("lcg32" "xorshift32" "pcg32" "mt19937" "xorshift64" "java_lcg" "minstd" "xorshift128" "lcg32_hybrid" "xorshift32_hybrid" "pcg32_hybrid" "mt19937_hybrid" "xorshift64_hybrid" "java_lcg_hybrid" "minstd_hybrid" "xorshift128_hybrid")

SUCCESS=0
FAIL=0

for prng in "${PRNGS[@]}"; do
    echo ""
    echo "==================== Testing $prng ===================="
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
        --seeds 100000 2>&1)
    
    echo "$OUTPUT" | grep -E "Total runtime|Successful|Failed|survivors"
    
    if echo "$OUTPUT" | grep -q "COMPLETED"; then
        ((SUCCESS++))
        echo "✅ $prng: SUCCESS"
    else
        ((FAIL++))
        echo "❌ $prng: FAILED"
    fi
done

echo ""
echo "========================================================"
echo "All tests complete!"
echo "Success: $SUCCESS/16"
echo "Failed: $FAIL/16"
