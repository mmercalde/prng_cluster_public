#!/bin/bash

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Creating in-place backups with timestamp: ${TIMESTAMP}"

# Key files to backup
FILES=(
    "coordinator.py"
    "sieve_filter.py"
    "reverse_sieve_filter.py"
    "prng_registry.py"
    "hybrid_strategy.py"
    "distributed_config.json"
)

echo ""
echo "LOCAL BACKUPS:"
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "${file}.bak_${TIMESTAMP}"
        echo "  ✅ $file → ${file}.bak_${TIMESTAMP}"
    fi
done

echo ""
echo "REMOTE BACKUPS:"
REMOTE_NODES=("192.168.3.120" "192.168.3.154")

for node in "${REMOTE_NODES[@]}"; do
    echo "  $node..."
    ssh $node "cd /home/michael/distributed_prng_analysis && \
        cp sieve_filter.py sieve_filter.py.bak_${TIMESTAMP} 2>/dev/null; \
        cp reverse_sieve_filter.py reverse_sieve_filter.py.bak_${TIMESTAMP} 2>/dev/null; \
        cp prng_registry.py prng_registry.py.bak_${TIMESTAMP} 2>/dev/null; \
        cp hybrid_strategy.py hybrid_strategy.py.bak_${TIMESTAMP} 2>/dev/null; \
        echo '    ✅ Backups created with .bak_${TIMESTAMP}'"
done

echo ""
echo "✅ ALL BACKUPS COMPLETE"
echo "Files backed up in place with .bak_${TIMESTAMP} extension"

