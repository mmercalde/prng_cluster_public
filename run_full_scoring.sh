#!/bin/bash
#
# run_full_scoring.sh (v1.1)
#
# Runs the full distributed scoring (Step 3.5) across all 26 GPUs.
# This script is called by the main workflow.
# It takes the 'optimal_scorer_config.json' and scores ALL survivors.
# v1.1: Adds final aggregation step.

set -e

echo "================================================="
echo "STEP 3.5: FULL DISTRIBUTED SCORING (26 GPUs)"
echo "================================================="

# 1. Generate the jobs
echo "Generating scoring jobs from optimal config..."
python3 generate_full_scoring_jobs.py \
    --survivors bidirectional_survivors.json \
    --config optimal_scorer_config.json \
    --train-history train_history.json \
    --holdout-history holdout_history.json \
    --jobs-file scoring_jobs.json

JOB_COUNT=$(jq length scoring_jobs.json)
if [ "$JOB_COUNT" -eq 0 ]; then
    echo "❌ No jobs were generated. Exiting."
    exit 1
fi
echo "✅ Generated $JOB_COUNT scoring jobs."

# 2. Copy data (chunks + history) to remote nodes
echo "Copying input data and new seed chunks to remote nodes..."
CHUNK_FILES=$(ls chunk_scoring_seeds_*.json)

for node in 192.168.3.120 192.168.3.154; do
    echo "  → $node"
    scp -q train_history.json holdout_history.json $CHUNK_FILES $node:~/distributed_prng_analysis/
done
echo "✅ Data copied to remote nodes."

# 3. Launch jobs via coordinator
echo "Launching $JOB_COUNT jobs via coordinator.py..."
python3 coordinator.py --jobs-file scoring_jobs.json --max-concurrent 26

# 4. Collect and aggregate results
echo "Collecting and aggregating $JOB_COUNT results from all nodes..."
python3 -c "
import json
from coordinator import MultiGPUCoordinator
from pathlib import Path
import sys

print('Initializing coordinator to collect results...')
coord = MultiGPUCoordinator('ml_coordinator_config.json')
results_list = coord.collect_scorer_results(total_trials=$JOB_COUNT)

if len(results_list) < $JOB_COUNT:
    print(f'⚠️ WARNING: Only collected {len(results_list)}/{$JOB_COUNT} results.')
else:
    print(f'✅ Collected {len(results_list)}/{$JOB_COUNT} results.')

# Aggregate all 'scores' fields from the collected JSON data
all_scores = []
for result in results_list:
    if result.get('status') == 'success' and 'scores' in result:
        all_scores.extend(result['scores'])
    else:
        print(f\"⚠️ Skipping result (ID: {result.get('trial_id')}) due to status '{result.get('status')}' or missing 'scores' key.\")

if not all_scores:
    print('❌ ERROR: No scores were aggregated. Check worker logs.')
    sys.exit(1)

# The 'all_scores' list now contains all the survivor scores.
# The Adaptive Optimizer (Step 4) expects a file containing these.
AGGREGATE_FILE = 'survivors_with_scores.json'
print(f'✅ Aggregated {len(all_scores)} total scores.')
print(f'Saving to {AGGREGATE_FILE}...')
with open(AGGREGATE_FILE, 'w') as f:
    json.dump(all_scores, f)

print(f'✅ Full scoring data saved to {AGGREGATE_FILE}')
"
echo "================================================="
echo "STEP 3.5 COMPLETE"
echo "================================================="
