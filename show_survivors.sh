#!/bin/bash
# Show actual survivor details from results files

echo "════════════════════════════════════════════════════════════"
echo "SURVIVOR DETAILS FROM RESULTS FILES"
echo "════════════════════════════════════════════════════════════"
echo ""

# Get the 4 most recent result files
RESULT_FILES=(
    "results/multi_gpu_analysis_1761177062.json"
    "results/multi_gpu_analysis_1761177074.json"
    "results/multi_gpu_analysis_1761177085.json"
    "results/multi_gpu_analysis_1761177096.json"
)

NAMES=(
    "xoshiro256pp_reverse"
    "xoshiro256pp_hybrid_reverse"
    "sfc64_reverse"
    "sfc64_hybrid_reverse"
)

for i in ${!RESULT_FILES[@]}; do
    file="${RESULT_FILES[$i]}"
    name="${NAMES[$i]}"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "TEST: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [ -f "$file" ]; then
        echo "FILE: $file"
        echo ""
        
        # Show summary
        echo "SUMMARY:"
        python3 << PYEOF
import json
with open('$file', 'r') as f:
    data = json.load(f)
    
total_survivors = 0
all_survivors = []

if 'results' in data:
    for result in data['results']:
        if 'survivors' in result:
            survivors = result['survivors']
            total_survivors += len(survivors)
            all_survivors.extend(survivors)

print(f"Total Survivors: {total_survivors}")
print(f"Total Seeds Tested: {data.get('total_seeds_tested', 'N/A')}")

if all_survivors:
    print(f"\nFirst 10 survivors:")
    for i, survivor in enumerate(all_survivors[:10]):
        seed = survivor.get('seed', 'N/A')
        match_rate = survivor.get('match_rate', 0)
        skip = survivor.get('skip', 'N/A')
        print(f"  {i+1}. Seed: {seed}, Match: {match_rate:.2%}, Skip: {skip}")
    
    if len(all_survivors) > 10:
        print(f"  ... and {len(all_survivors) - 10} more")
else:
    print("\n⚠️ NO SURVIVORS FOUND")
PYEOF
        
        echo ""
    else
        echo "❌ FILE NOT FOUND!"
        echo ""
    fi
done

echo "════════════════════════════════════════════════════════════"
