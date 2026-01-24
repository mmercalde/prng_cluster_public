#!/bin/bash
# Test: Do jobs finishing together cause GPU issues?
# Uses minimal data, single rig, captures timing

set -e
REMOTE="192.168.3.120"
TEST_SURVIVORS=500
TEST_CHUNKS=6  # 6 jobs on 12 GPUs = light load

echo "=== FINISH STORM HYPOTHESIS TEST ==="
echo "Survivors: $TEST_SURVIVORS"
echo "Chunks: $TEST_CHUNKS"
echo "Target: $REMOTE only"
echo ""

# Check GPU state BEFORE
echo "=== GPU STATE BEFORE ==="
ssh $REMOTE "rocm-smi" | grep -E "^[0-9]|Perf"
echo ""

# Generate tiny job set
echo "=== GENERATING TEST JOBS ==="
python3 generate_step3_scoring_jobs.py \
    --survivors test_survivors_500.npz \
    --train-history train_history.json \
    --holdout-history holdout_history.json \
    --chunk-size $((TEST_SURVIVORS / TEST_CHUNKS)) \
    --output-file test_scoring_jobs.json

echo ""
echo "=== RUNNING WITH TIMING ==="
echo "Start time: $(date '+%H:%M:%S')"

# Run with verbose timing
python3 scripts_coordinator.py --config test_config_rig6600_only.json \
    --jobs-file test_scoring_jobs.json \
    --output-dir test_scoring_results \
     \
    --verbose 2>&1 | tee test_run.log

echo "End time: $(date '+%H:%M:%S')"
echo ""

# Check GPU state IMMEDIATELY AFTER
echo "=== GPU STATE AFTER (immediate) ==="
ssh $REMOTE "rocm-smi" | grep -E "^[0-9]|Perf"

# Wait 5 seconds, check again
sleep 5
echo ""
echo "=== GPU STATE AFTER (+5s) ==="
ssh $REMOTE "rocm-smi" | grep -E "^[0-9]|Perf"

echo ""
echo "=== TIMING ANALYSIS ==="
grep -E "START|FINISH|complete" test_run.log | tail -20
