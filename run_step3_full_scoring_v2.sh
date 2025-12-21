#!/bin/bash
#
# run_step3_full_scoring.sh (v2.0)
#
# Runs the full distributed scoring (Step 3) across all 26 GPUs.
# This script is called by the main workflow.
#
# V2.0 CHANGES:
# - Uses scripts_coordinator.py instead of coordinator.py (100% success rate)
# - File-based success detection (no stdout JSON parsing)
# - Improved error handling
#
# Usage:
#   ./run_step3_full_scoring.sh [--survivors FILE] [--train-history FILE]
#
set -e

echo "================================================="
echo "STEP 3: FULL DISTRIBUTED SCORING (26 GPUs)"
echo "Using: scripts_coordinator.py v1.4.0"
echo "================================================="

# Parse arguments (optional overrides)
SURVIVORS_FILE="${1:-bidirectional_survivors.json}"
TRAIN_HISTORY="${2:-train_history.json}"
FORWARD_SURVIVORS="${3:-forward_survivors.json}"
REVERSE_SURVIVORS="${4:-reverse_survivors.json}"

# Validate inputs
if [ ! -f "$SURVIVORS_FILE" ]; then
    echo "❌ Survivors file not found: $SURVIVORS_FILE"
    exit 1
fi

if [ ! -f "$TRAIN_HISTORY" ]; then
    echo "❌ Train history file not found: $TRAIN_HISTORY"
    exit 1
fi

echo "Input files:"
echo "  • Survivors: $SURVIVORS_FILE"
echo "  • Train history: $TRAIN_HISTORY"
echo ""

# 1. Generate the jobs
echo "Step 3.1: Generating scoring jobs..."
python3 generate_step3_scoring_jobs.py \
    --survivors "$SURVIVORS_FILE" \
    --train-history "$TRAIN_HISTORY" \
    --forward-survivors "$FORWARD_SURVIVORS" \
    --reverse-survivors "$REVERSE_SURVIVORS" \
    --output scoring_jobs.json

JOB_COUNT=$(python3 -c "import json; print(len(json.load(open('scoring_jobs.json'))))")
if [ "$JOB_COUNT" -eq 0 ]; then
    echo "❌ No jobs were generated. Exiting."
    exit 1
fi
echo "✅ Generated $JOB_COUNT scoring jobs."

# 2. Copy data to remote nodes
echo ""
echo "Step 3.2: Copying data to remote nodes..."

# Find chunk files
CHUNK_DIR="scoring_chunks"
if [ -d "$CHUNK_DIR" ]; then
    CHUNK_COUNT=$(ls "$CHUNK_DIR"/*.json 2>/dev/null | wc -l)
    echo "  Found $CHUNK_COUNT chunk files in $CHUNK_DIR/"
else
    echo "❌ Chunk directory not found: $CHUNK_DIR"
    exit 1
fi

for node in 192.168.3.120 192.168.3.154; do
    echo "  → Copying to $node..."
    
    # Create directories on remote
    ssh "$node" "mkdir -p ~/distributed_prng_analysis/$CHUNK_DIR ~/distributed_prng_analysis/full_scoring_results"
    
    # Copy history files
    scp -q "$TRAIN_HISTORY" "$node:~/distributed_prng_analysis/"
    
    # Copy chunk files
    scp -q "$CHUNK_DIR"/*.json "$node:~/distributed_prng_analysis/$CHUNK_DIR/"
done
echo "✅ Data copied to remote nodes."

# 3. Launch jobs via scripts_coordinator.py (V2.0: replaces coordinator.py)
echo ""
echo "Step 3.3: Launching $JOB_COUNT jobs via scripts_coordinator.py..."
echo "  (This achieves 100% success rate vs 72% with coordinator.py)"

python3 scripts_coordinator.py \
    --jobs-file scoring_jobs.json \
    --output-dir full_scoring_results \
    --config distributed_config.json

# 4. Collect results from remote nodes
echo ""
echo "Step 3.4: Collecting results from remote nodes..."

# Create local results directory if needed
mkdir -p full_scoring_results

# Pull results from remote nodes
for node in 192.168.3.120 192.168.3.154; do
    echo "  ← Pulling from $node..."
    scp -q "$node:~/distributed_prng_analysis/full_scoring_results/*.json" full_scoring_results/ 2>/dev/null || true
done

# Count results
RESULT_COUNT=$(ls full_scoring_results/*.json 2>/dev/null | grep -v manifest | wc -l)
echo "✅ Collected $RESULT_COUNT result files."

# 5. Aggregate results
echo ""
echo "Step 3.5: Aggregating results..."

python3 << 'AGGREGATE_SCRIPT'
import json
from pathlib import Path
import sys

results_dir = Path('full_scoring_results')
output_file = 'survivors_with_scores.json'

# Find all chunk result files (exclude manifest)
result_files = sorted([f for f in results_dir.glob('*.json') 
                       if 'manifest' not in f.name and f.stat().st_size > 0])

print(f"  Found {len(result_files)} result files")

all_survivors = []
for result_file in result_files:
    try:
        with open(result_file) as f:
            data = json.load(f)
            if isinstance(data, list):
                all_survivors.extend(data)
            elif isinstance(data, dict) and 'survivors' in data:
                all_survivors.extend(data['survivors'])
            elif isinstance(data, dict) and 'scores' in data:
                all_survivors.extend(data['scores'])
    except Exception as e:
        print(f"  ⚠️ Error reading {result_file}: {e}")

if not all_survivors:
    print("❌ No survivors were aggregated. Check result files.")
    sys.exit(1)

# Sort by score (highest first)
if all_survivors and 'score' in all_survivors[0]:
    all_survivors.sort(key=lambda x: x.get('score', 0), reverse=True)

print(f"  Aggregated {len(all_survivors):,} survivors")

# Save
with open(output_file, 'w') as f:
    json.dump(all_survivors, f)

print(f"✅ Saved to {output_file}")

# Show stats
if all_survivors and 'features' in all_survivors[0]:
    feature_count = len(all_survivors[0]['features'])
    print(f"  Features per survivor: {feature_count}")

if all_survivors and 'score' in all_survivors[0]:
    scores = [s['score'] for s in all_survivors if 'score' in s]
    if scores:
        print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}]")
AGGREGATE_SCRIPT

# 6. Verify output
if [ ! -f "survivors_with_scores.json" ]; then
    echo "❌ survivors_with_scores.json was not created."
    exit 1
fi

SURVIVOR_COUNT=$(python3 -c "import json; print(len(json.load(open('survivors_with_scores.json'))))")
FILE_SIZE=$(ls -lh survivors_with_scores.json | awk '{print $5}')

echo ""
echo "================================================="
echo "STEP 3 COMPLETE"
echo "================================================="
echo "  Jobs executed: $JOB_COUNT"
echo "  Results collected: $RESULT_COUNT"
echo "  Survivors scored: $SURVIVOR_COUNT"
echo "  Output file: survivors_with_scores.json ($FILE_SIZE)"
echo "================================================="
