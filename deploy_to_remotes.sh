#!/bin/bash

echo "=========================================="
echo "DEPLOYING TO REMOTE NODES"
echo "=========================================="

# Deploy to node 1
echo ""
echo "→ Deploying to 192.168.3.120..."
scp prng_registry.py sieve_filter.py michael@192.168.3.120:~/distributed_prng_analysis/
if [ $? -eq 0 ]; then
    echo "✅ 192.168.3.120 SUCCESS"
else
    echo "❌ 192.168.3.120 FAILED"
    exit 1
fi

# Deploy to node 2
echo ""
echo "→ Deploying to 192.168.3.154..."
scp prng_registry.py sieve_filter.py michael@192.168.3.154:~/distributed_prng_analysis/
if [ $? -eq 0 ]; then
    echo "✅ 192.168.3.154 SUCCESS"
else
    echo "❌ 192.168.3.154 FAILED"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ DEPLOYMENT COMPLETE!"
echo "=========================================="
echo "Deployed files:"
echo "  • prng_registry.py"
echo "  • sieve_filter.py"
echo ""
echo "All 26 GPUs now have the correct reverse sieve code!"
echo "=========================================="

