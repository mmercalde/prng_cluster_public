#!/bin/bash

# Delete old saved progress to avoid interactive prompts
rm -f analysis_*.json sieve_*.json

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Testing Last 4 PRNGs on 26-GPU Cluster                   ║"
echo "╚════════════════════════════════════════════════════════════╝"

PASSED=0
FAILED=0

test_prng() {
    local prng=$1
    local test_file=$2
    local skip_args=$3
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing: $prng"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    python3 coordinator.py \
        $test_file \
        --method residue_sieve \
        --prng-type $prng \
        --window-size 512 \
        --threshold 0.50 \
        $skip_args \
        --seeds 5000 \
        --resume-policy restart \
        > /tmp/test_${prng}.log 2>&1
    
    if [ $? -eq 0 ]; then
        GPU_COUNT=$(grep -oP '\d+/\d+ GPUs' /tmp/test_${prng}.log | tail -1)
        TIME=$(grep -oP '\d+\.\d+s' /tmp/test_${prng}.log | tail -1)
        echo "✅ $prng: $GPU_COUNT - $TIME"
        ((PASSED++))
    else
        echo "❌ $prng: FAILED"
        echo "Last 10 lines of log:"
        tail -10 /tmp/test_${prng}.log
        ((FAILED++))
    fi
}

# Fixed skip: use --skip-min 5 --skip-max 5
test_prng "xoshiro256pp_reverse" "test_multi_prng_xoshiro256pp.json" "--skip-min 5 --skip-max 5"

# Hybrid: search skip range 0-16
test_prng "xoshiro256pp_hybrid_reverse" "test_multi_prng_xoshiro256pp.json" "--skip-min 0 --skip-max 16"

# Fixed skip: use --skip-min 5 --skip-max 5
test_prng "sfc64_reverse" "test_multi_prng_sfc64.json" "--skip-min 5 --skip-max 5"

# Hybrid: search skip range 0-16
test_prng "sfc64_hybrid_reverse" "test_multi_prng_sfc64.json" "--skip-min 0 --skip-max 16"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    FINAL RESULTS                           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo "PASSED: $PASSED/4"
echo "FAILED: $FAILED/4"

if [ $FAILED -gt 0 ]; then
    echo "⚠️  $FAILED test(s) failed."
else
    echo "🎉 ALL 4 PRNGS PASSED! 26/26 GPUs operational!"
fi
