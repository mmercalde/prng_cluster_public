#!/bin/bash

echo "=== SYSTEMATIC RETEST: All uint64 PRNGs with Output Fix ==="
echo ""

PASSED=0
FAILED=0

# Function to test and verify
test_prng() {
    local prng=$1
    local test_file=$2
    
    echo "Testing $prng..."
    python3 coordinator.py \
        "$test_file" \
        --method residue_sieve \
        --prng-type "$prng" \
        --seeds 5000 \
        --window-size 512 \
        --threshold 0.50 \
        --skip 5 > /tmp/test_${prng}.log 2>&1
    
    # Check if found seed 1234
    python3 << EOF
import json
import glob
result_files = sorted(glob.glob('results/multi_gpu_analysis_*.json'))
with open(result_files[-1], 'r') as f:
    data = json.load(f)
found = False
for result in data.get('results', []):
    for surv in result.get('survivors', []):
        if surv['seed'] == 1234:
            found = True
            break
    if found:
        break
exit(0 if found else 1)
EOF
    
    if [ $? -eq 0 ]; then
        echo "  ✅ $prng: PASSED (found seed 1234)"
        ((PASSED++))
    else
        echo "  ❌ $prng: FAILED (seed 1234 not found)"
        ((FAILED++))
    fi
    echo ""
}

# Test all uint64 PRNGs
echo "FORWARD VARIANTS:"
test_prng "xorshift64" "test_multi_prng_xorshift64.json"
test_prng "java_lcg" "test_multi_prng_java_lcg.json"
test_prng "xoshiro256pp" "test_multi_prng_xoshiro256pp.json"
test_prng "sfc64" "test_multi_prng_sfc64.json"

echo ""
echo "REVERSE VARIANTS:"
test_prng "xorshift64_reverse" "test_multi_prng_xorshift64.json"
test_prng "java_lcg_reverse" "test_multi_prng_java_lcg.json"

echo ""
echo "=========================================="
echo "RESULTS: $PASSED passed, $FAILED failed"
echo "=========================================="
