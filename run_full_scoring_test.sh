#!/bin/bash
#
# run_full_scoring_test.sh - TEST VERSION
# Tests Step 3 with 25 jobs across all 26 GPUs

set -e

echo "================================================="
echo "STEP 3: FULL DISTRIBUTED SCORING TEST (26 GPUs)"
echo "================================================="

# Jobs already generated (25 jobs in scoring_jobs_test.json)
JOB_COUNT=$(jq length scoring_jobs_test.json)
echo "✅ Testing with $JOB_COUNT jobs"

# Copy data (chunks + history) to remote nodes
echo "Copying input data and seed chunks to remote nodes..."
CHUNK_FILES=$(ls chunk_scoring_seeds_*.json)

for node in 192.168.3.120 192.168.3.154; do
    echo "  → $node"
    ssh $node "mkdir -p ~/distributed_prng_analysis/scorer_trial_results" 2>/dev/null || true
    scp -q train_history.json holdout_history.json $CHUNK_FILES $node:~/distributed_prng_analysis/
done
echo "✅ Data copied to remote nodes"

# Clean up old results
echo "Cleaning up old results..."
rm -rf scorer_trial_results/*
mkdir -p scorer_trial_results
for node in 192.168.3.120 192.168.3.154; do
    ssh $node "rm -rf ~/distributed_prng_analysis/scorer_trial_results/*" 2>/dev/null || true
done

# Launch jobs via coordinator
echo ""
echo "Launching $JOB_COUNT jobs via coordinator.py..."
python3 coordinator.py \
    --jobs-file scoring_jobs_test.json \
    --config ml_coordinator_config.json \
    --max-concurrent 26 \
    --resume-policy restart

echo ""
echo "================================================="
echo "STEP 3 TEST COMPLETE"
echo "================================================="

# Show summary
echo ""
echo "=== Results Summary ==="
echo "Local results: $(ls scorer_trial_results/*.json 2>/dev/null | wc -l)"
echo "Remote results (192.168.3.120): $(ssh 192.168.3.120 'ls ~/distributed_prng_analysis/scorer_trial_results/*.json 2>/dev/null | wc -l')"
echo "Remote results (192.168.3.154): $(ssh 192.168.3.154 'ls ~/distributed_prng_analysis/scorer_trial_results/*.json 2>/dev/null | wc -l')"
echo ""
echo "Next: Check if you want to collect and aggregate results"
