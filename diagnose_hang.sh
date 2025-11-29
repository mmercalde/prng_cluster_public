#!/bin/bash

echo "======================================================================"
echo "DIAGNOSING COORDINATOR HANG"
echo "======================================================================"

echo ""
echo "1. Checking if coordinator is running:"
ps aux | grep coordinator.py | grep -v grep

echo ""
echo "2. Checking if workers are running:"
ps aux | grep worker.py | grep -v grep

echo ""
echo "3. Checking network connections:"
netstat -tnp 2>/dev/null | grep python || ss -tnp | grep python

echo ""
echo "4. Checking recent log output:"
if [ -f /tmp/test_xoshiro256pp_reverse.log ]; then
    echo "Last 20 lines of test log:"
    tail -20 /tmp/test_xoshiro256pp_reverse.log
fi

echo ""
echo "5. Testing if coordinator can be imported:"
timeout 5 python3 -c "from prng_registry import get_cpu_reference; print('✅ Import works')" || echo "❌ Import failed or timed out"

echo ""
echo "6. Testing basic coordinator startup:"
timeout 10 python3 coordinator.py --help > /dev/null 2>&1 && echo "✅ Coordinator starts" || echo "❌ Coordinator hangs on startup"

echo ""
echo "======================================================================"
