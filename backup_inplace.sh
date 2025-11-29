#!/bin/bash
# In-place backups with timestamp suffix

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Creating in-place backups with suffix _${TIMESTAMP}..."

# Local backups
for file in prng_registry.py sieve_filter.py coordinator.py hybrid_strategy.py reverse_sieve_filter.py; do
    if [ -f "$file" ]; then
        cp "$file" "${file}.bak_${TIMESTAMP}"
        echo "✅ Backed up: ${file} -> ${file}.bak_${TIMESTAMP}"
    fi
done

# Remote backups
for host in 192.168.3.120 192.168.3.154; do
    echo ""
    echo "Backing up on $host..."
    ssh $host "cd ~/distributed_prng_analysis && \
        for file in prng_registry.py sieve_filter.py reverse_sieve_filter.py; do \
            [ -f \"\$file\" ] && cp \"\$file\" \"\${file}.bak_${TIMESTAMP}\" && echo \"✅ \$file\"; \
        done"
done

echo ""
echo "✅ All backups complete with suffix: .bak_${TIMESTAMP}"
