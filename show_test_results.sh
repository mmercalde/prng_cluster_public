#!/bin/bash
# Show detailed results from the test logs

echo "════════════════════════════════════════════════════════════"
echo "DETAILED RESULTS FROM TEST LOGS"
echo "════════════════════════════════════════════════════════════"
echo ""

for log in /tmp/test_xoshiro256pp_reverse.log /tmp/test_xoshiro256pp_hybrid_reverse.log /tmp/test_sfc64_reverse.log /tmp/test_sfc64_hybrid_reverse.log; do
    if [ -f "$log" ]; then
        basename=$(basename "$log" .log)
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "FILE: $log"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "FULL CONTENTS:"
        cat "$log"
        echo ""
        echo ""
    fi
done
