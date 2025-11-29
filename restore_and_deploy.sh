#!/bin/bash

TIMESTAMP="20251014_204724"
echo "Restoring from backup: ${TIMESTAMP}"

# Files to restore
FILES=(
    "coordinator.py"
    "sieve_filter.py"
    "reverse_sieve_filter.py"
    "prng_registry.py"
    "hybrid_strategy.py"
    "distributed_config.json"
)

echo ""
echo "RESTORING LOCAL FILES:"
for file in "${FILES[@]}"; do
    if [ -f "${file}.bak_${TIMESTAMP}" ]; then
        cp "${file}.bak_${TIMESTAMP}" "$file"
        echo "  ✅ Restored $file"
    else
        echo "  ⚠️  Backup ${file}.bak_${TIMESTAMP} not found"
    fi
done

echo ""
echo "DEPLOYING TO REMOTE NODES:"
REMOTE_NODES=("192.168.3.120" "192.168.3.154")
DEPLOY_FILES=("sieve_filter.py" "reverse_sieve_filter.py" "prng_registry.py" "hybrid_strategy.py" "coordinator.py")

for node in "${REMOTE_NODES[@]}"; do
    echo "  Deploying to $node..."
    for file in "${DEPLOY_FILES[@]}"; do
        scp "$file" "${node}:/home/michael/distributed_prng_analysis/"
    done
    echo "    ✅ $node complete"
done

echo ""
echo "✅ RESTORE AND DEPLOYMENT COMPLETE!"
echo "All files restored to working state from ${TIMESTAMP}"

