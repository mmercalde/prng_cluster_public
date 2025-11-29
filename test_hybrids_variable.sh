#!/bin/bash

echo "=== Testing Hybrid Variants with Variable Skip Data ==="
echo ""

test_hybrid() {
    local prng=$1
    local test_file=$2
    
    echo "Testing $prng with variable skip..."
    python3 coordinator.py \
        "$test_file" \
        --method residue_sieve \
        --prng-type "$prng" \
        --seeds 5000 \
        --window-size 512 \
        --threshold 0.50 > /tmp/${prng}_variable.log 2>&1
    
    # Check for survivors
    python3 << EOF
import json
import glob
result_files = sorted(glob.glob('results/multi_gpu_analysis_*.json'))
with open(result_files[-1], 'r') as f:
    data = json.load(f)
total = sum(len(r.get('survivors', [])) for r in data.get('results', []))
found_1234 = any(s['seed'] == 1234 for r in data.get('results', []) for s in r.get('survivors', []))
if found_1234:
    print(f"  ✅ $prng: FOUND seed 1234 (total survivors: {total})")
elif total > 0:
    print(f"  ⚠️  $prng: Found {total} survivors (but not seed 1234)")
else:
    print(f"  ❌ $prng: 0 survivors")
EOF
    echo ""
}

echo "FORWARD HYBRID (variable skip):"
test_hybrid "xorshift64_hybrid" "test_multi_prng_xorshift64_variable.json"
test_hybrid "java_lcg_hybrid" "test_multi_prng_java_lcg_variable.json"
test_hybrid "xoshiro256pp_hybrid" "test_multi_prng_xoshiro256pp_variable.json"
test_hybrid "sfc64_hybrid" "test_multi_prng_sfc64_variable.json"

echo ""
echo "REVERSE HYBRID (variable skip):"
test_hybrid "xorshift64_hybrid_reverse" "test_multi_prng_xorshift64_variable.json"
test_hybrid "java_lcg_hybrid_reverse" "test_multi_prng_java_lcg_variable.json"

echo ""
echo "✅ ALL HYBRID VARIABLE SKIP TESTS COMPLETE!"
