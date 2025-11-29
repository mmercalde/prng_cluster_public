#!/bin/bash
# Automated Java LCG Bidirectional Survivor Test
# Runs forward + reverse sieves and finds intersection

set -e  # Exit on any error

SEEDS=1000000000  # 1 billion seeds (change to 10000000 for quick test)
DATASET="daily3.json"

echo "======================================"
echo "JAVA LCG BIDIRECTIONAL SURVIVOR TEST"
echo "======================================"
echo "Seeds: ${SEEDS}"
echo "Dataset: ${DATASET}"
echo ""

# Step 1: Forward Sieve
echo "[1/3] Running FORWARD sieve (java_lcg)..."
echo "Expected: ~10-12 minutes for 1B seeds"
python3 coordinator.py \
    --resume-policy restart \
    --max-concurrent 26 \
    ${DATASET} \
    --method residue_sieve \
    --prng-type java_lcg \
    --window-size 512 \
    --threshold 0.01 \
    --skip-max 20 \
    --seeds ${SEEDS}

echo ""
echo "✅ Forward sieve complete!"
echo ""

# Step 2: Reverse Sieve
echo "[2/3] Running REVERSE sieve (java_lcg_reverse)..."
echo "Expected: ~10-12 minutes for 1B seeds"
python3 coordinator.py \
    --resume-policy restart \
    --max-concurrent 26 \
    ${DATASET} \
    --method residue_sieve \
    --prng-type java_lcg_reverse \
    --window-size 512 \
    --threshold 0.01 \
    --skip-max 20 \
    --seeds ${SEEDS}

echo ""
echo "✅ Reverse sieve complete!"
echo ""

# Step 3: Find Intersection
echo "[3/3] Finding bidirectional survivors (intersection)..."
python3 << 'INTERSECT'
import json, glob, os

# Get the two latest result files
files = sorted(
    glob.glob('results/multi_gpu_analysis_*.json'), 
    key=lambda x: os.path.getmtime(x), 
    reverse=True
)

if len(files) < 2:
    print("❌ ERROR: Need at least 2 result files!")
    exit(1)

# Most recent should be reverse, second should be forward
rev_file = files[0]
fwd_file = files[1]

print(f"Forward file: {fwd_file}")
print(f"Reverse file: {rev_file}")
print("")

# Load survivors
with open(fwd_file) as f:
    fwd_data = json.load(f)
with open(rev_file) as f:
    rev_data = json.load(f)

fwd = {s['seed'] for res in fwd_data['results'] for s in res.get('survivors', [])}
rev = {s['seed'] for res in rev_data['results'] for s in res.get('survivors', [])}

# Find intersection
common = sorted(fwd & rev)

print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"FORWARD SURVIVORS:       {len(fwd):,}")
print(f"REVERSE SURVIVORS:       {len(rev):,}")
print(f"BIDIRECTIONAL SURVIVORS: {len(common)}")
print("")

if common:
    print("THE ETERNAL SEEDS (validated by both sieves):")
    for seed in common:
        print(f"  → {seed}")
    
    # Save to file
    output = {
        'forward_count': len(fwd),
        'reverse_count': len(rev),
        'bidirectional_count': len(common),
        'bidirectional_seeds': common,
        'forward_file': fwd_file,
        'reverse_file': rev_file
    }
    
    with open('bidirectional_survivors.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("")
    print("✅ Results saved to: bidirectional_survivors.json")
else:
    print("⚠️  No bidirectional survivors found!")
    print("   (This might be expected with fewer than 1B seeds)")

print("")
print("=" * 60)
INTERSECT

echo ""
echo "======================================"
echo "TEST COMPLETE!"
echo "======================================"
