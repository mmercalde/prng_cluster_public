#!/bin/bash
echo "Testing ALL 16 PRNGs with 10M seeds each"
echo "=========================================="

PRNGS=("lcg32" "xorshift32" "pcg32" "mt19937" "xorshift64" "java_lcg" "minstd" "xorshift128" "lcg32_hybrid" "xorshift32_hybrid" "pcg32_hybrid" "mt19937_hybrid" "xorshift64_hybrid" "java_lcg_hybrid" "minstd_hybrid" "xorshift128_hybrid")

START_TIME=$(date +%s)

for prng in "${PRNGS[@]}"; do
    echo ""
    echo "==================== $prng (10M seeds) ===================="
    python3 coordinator.py \
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
        --seeds 10000000 2>&1 | grep -E "Total runtime|Successful|Failed|COMPLETED"
done

END_TIME=$(date +%s)
TOTAL=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "All 16 PRNGs tested!"
echo "Total time: ${TOTAL} seconds (~$((TOTAL/60)) minutes)"
