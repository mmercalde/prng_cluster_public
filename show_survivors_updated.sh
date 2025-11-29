#!/bin/bash
# Show actual survivor details from the MOST RECENT results files

echo "════════════════════════════════════════════════════════════"
echo "SURVIVOR DETAILS FROM MOST RECENT RESULTS"
echo "════════════════════════════════════════════════════════════"
echo ""

# Get the 4 most recent result files
RECENT_FILES=($(ls -t results/multi_gpu_analysis_*.json | head -4))

NAMES=(
    "TEST 1 (Most Recent)"
    "TEST 2"
    "TEST 3"
    "TEST 4 (Oldest)"
)

echo "Found ${#RECENT_FILES[@]} recent result files"
echo ""

for i in ${!RECENT_FILES[@]}; do
    file="${RECENT_FILES[$i]}"
    name="${NAMES[$i]}"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "$name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "FILE: $file"
    echo ""
    
    if [ -f "$file" ]; then
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
print(f"Analysis ID: {data.get('analysis_id', 'N/A')}")

# Try to extract PRNG type from job config
if 'results' in data and len(data['results']) > 0:
    first_job = data['results'][0]
    if 'job_config' in first_job:
        config = first_job['job_config']
        prng_families = config.get('prng_families', ['unknown'])
        print(f"PRNG Type: {prng_families[0]}")
        print(f"Seeds Tested: {first_job.get('seeds_analyzed', 'N/A')}")

if all_survivors:
    print(f"\nSurvivors found:")
    for idx, survivor in enumerate(all_survivors[:10]):
        seed = survivor.get('seed', 'N/A')
        match_rate = survivor.get('match_rate', 0)
        skip = survivor.get('skip', 'N/A')
        print(f"  {idx+1}. Seed: {seed}, Match: {match_rate*100:.2f}%, Skip: {skip}")
    
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
