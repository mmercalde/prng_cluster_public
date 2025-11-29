#!/bin/bash

echo "=== Regenerating ALL test data with seed 1234 ==="

# Update all test generation scripts
for script in create_*_test.py; do
    if [ -f "$script" ]; then
        echo "Updating $script..."
        sed -i 's/SEED = 12345/SEED = 1234/g' "$script"
    fi
done

# Regenerate all test files
for script in create_*_test.py; do
    if [ -f "$script" ]; then
        echo "Running $script..."
        python3 "$script"
    fi
done

# Deploy to remote nodes
echo ""
echo "Deploying test files to remote nodes..."
for host in 192.168.3.120 192.168.3.154; do
    scp test_multi_prng_*.json $host:~/distributed_prng_analysis/
done

echo ""
echo "✅ All test files regenerated with seed 1234 and deployed!"
