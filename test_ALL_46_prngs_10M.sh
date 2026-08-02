#!/bin/bash
# ============================================================================
# [S184 bounded Phase 6 §6 — COUNT CORRECTION, Team Beta ordered]
#
# THE COUNTS IN THIS FILE WERE WRONG. Verified this session against the live
# `prng_registry.KERNEL_REGISTRY` on VM 101:
#
#   * the PRNGS array contains 44 entries, not 46;
#   * they are 11 + 11 + 11 + 11, not 11 + 11 + 12 + 12 — the two "Reverse"
#     category comments both said 12 and both had 11;
#   * all 44 are unique, all 44 ARE valid registry names, and they cover
#     KERNEL_REGISTRY exactly (len(KERNEL_REGISTRY) == 44, set difference empty
#     in both directions). Beta checked this and was right: the earlier claim
#     that this array contains two invalid registry names is FALSE and must not
#     be repeated.
#
# The FILENAME still says 46 and is left alone deliberately: renaming it would
# break every existing reference to it for no gain. Read the filename as a
# historical label, not as a count.
#
# WHAT THIS SCRIPT IS. A coverage/throughput exerciser: it runs the coordinator
# once per registry family at threshold 0.01 and records whether the run
# COMPLETED. It has no expected answer, no oracle and no per-seed comparison.
# It is therefore NOT a known-answer test and must not be cited as correctness
# evidence. The known-answer evidence for the four java_lcg variants is
# `tests/phase6/known_answer_gate.py` (bounded Phase 6 §3).
# ============================================================================
echo "Testing ALL 44 PRNGs with 10M seeds each"
echo "=========================================="

# ALL 44 registry families — 11 in each of four categories (verified S184)
PRNGS=(
    # Forward Constant (11)
    "lcg32" "xorshift32" "pcg32" "mt19937" "xorshift64" "java_lcg" "minstd" "xorshift128" "xoshiro256pp" "philox4x32" "sfc64"
    # Forward Variable (11)
    "xorshift32_hybrid" "pcg32_hybrid" "lcg32_hybrid" "xorshift64_hybrid" "mt19937_hybrid" "java_lcg_hybrid" "minstd_hybrid" "xorshift128_hybrid" "xoshiro256pp_hybrid" "philox4x32_hybrid" "sfc64_hybrid"
    # Reverse Constant (11)
    "mt19937_reverse" "lcg32_reverse" "xorshift32_reverse" "xorshift64_reverse" "xorshift128_reverse" "pcg32_reverse" "java_lcg_reverse" "minstd_reverse" "philox4x32_reverse" "xoshiro256pp_reverse" "sfc64_reverse"
    # Reverse Variable (11)
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
echo "All 44 PRNGs tested!"
echo "Success: $SUCCESS/44"
echo "Failed: $FAIL/44"
echo "Total time: ${TOTAL} seconds (~$((TOTAL/60)) minutes)"
