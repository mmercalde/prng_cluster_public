#!/bin/bash
# clear_ramdisk.sh - Force refresh ramdisk data on next run
# Use after Step 1 regenerates survivor data

echo "Clearing ramdisk sentinel on remote nodes..."

REMOTE_NODES=$(python3 -c "
import json
with open('distributed_config.json') as f:
    cfg = json.load(f)
for node in cfg['nodes']:
    if node['hostname'] != 'localhost':
        print(node['hostname'])
")

for REMOTE in $REMOTE_NODES; do
    echo "  → $REMOTE"
    ssh "$REMOTE" "rm -f /dev/shm/prng/.ready && echo '    ✓ Sentinel cleared'"
done

echo "Done. Next job run will refresh ramdisk data."
