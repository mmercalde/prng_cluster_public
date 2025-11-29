#!/bin/bash
# quick_cluster_test.sh - Fast verification of distributed GPU system

echo "========================================================================"
echo "QUICK CLUSTER TEST - Version 2.3"
echo "========================================================================"
echo ""

echo "Test Configuration:"
echo "  Seeds per iteration: 10000"
echo "  Iterations: 3"
echo ""

echo "Running window optimizer..."
python3 coordinator.py daily3.json \
  --optimize-window \
  --prng-type java_lcg_hybrid \
  --opt-strategy random \
  --opt-iterations 3 \
  --opt-seed-count 10000 \
  2>&1 | tee cluster_test_output.log

echo ""
echo "Analyzing results..."

TOTAL_JOBS=$(grep "Total jobs executed:" cluster_test_output.log | tail -1 | awk '{print $4}')
SUCCESSFUL_JOBS=$(grep "Successful jobs:" cluster_test_output.log | tail -1 | awk '{print $3}')
FAILED_JOBS=$(grep "Failed jobs:" cluster_test_output.log | tail -1 | awk '{print $3}')

echo ""
echo "Results: Total=$TOTAL_JOBS Success=$SUCCESSFUL_JOBS Failed=$FAILED_JOBS"

if [ "$FAILED_JOBS" -eq 0 ]; then
    echo "🎉 ✅ ALL TESTS PASSED!"
else
    echo "⚠️  Some failures detected"
fi
