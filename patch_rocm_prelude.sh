#!/bin/bash
# patch_rocm_prelude.sh
# Adds "rig-6600c" to the ROCm hostname list in all required files
# Run from ~/distributed_prng_analysis on Zeus

set -e

echo "=== ROCm Prelude Patch for rig-6600c ==="
echo "Backing up files first..."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Files requiring ROCm prelude update
FILES=(
    "distributed_worker.py"
    "sieve_filter.py"
    "enhanced_gpu_model_id.py"
    "survivor_scorer.py"
    "reinforcement_engine.py"
    "scorer_trial_worker.py"
    "anti_overfit_trial_worker.py"
)

# Backup and patch each file
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "Processing $file..."
        
        # Backup
        cp "$file" "${file}.bak_${TIMESTAMP}"
        
        # Patch: Add rig-6600c to the hostname list
        # Handles both quote styles: "rig-6600b" and 'rig-6600b'
        sed -i 's/\["rig-6600", "rig-6600b"\]/["rig-6600", "rig-6600b", "rig-6600c"]/g' "$file"
        sed -i "s/\['rig-6600', 'rig-6600b'\]/['rig-6600', 'rig-6600b', 'rig-6600c']/g" "$file"
        
        # Also handle tuple/parentheses style if present
        sed -i 's/("rig-6600", "rig-6600b")/("rig-6600", "rig-6600b", "rig-6600c")/g' "$file"
        sed -i "s/('rig-6600', 'rig-6600b')/('rig-6600', 'rig-6600b', 'rig-6600c')/g" "$file"
        
        # Verify the change
        if grep -q "rig-6600c" "$file"; then
            echo "  ✅ $file patched successfully"
        else
            echo "  ⚠️  $file - pattern not found (may already be patched or different format)"
        fi
    else
        echo "  ⏭️  $file not found, skipping"
    fi
done

echo ""
echo "=== Verification ==="
echo "Checking ROCm prelude in patched files:"
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -n "$file: "
        grep -o 'rig-6600[abc"'"'"']*' "$file" | head -1 || echo "not found"
    fi
done

echo ""
echo "=== Done ==="
echo "Backups saved with suffix: .bak_${TIMESTAMP}"
echo ""
echo "Next steps:"
echo "1. Review changes: diff distributed_worker.py distributed_worker.py.bak_${TIMESTAMP}"
echo "2. Deploy to rigs:"
echo "   for host in 192.168.3.120 192.168.3.154 192.168.3.162; do"
echo "       scp ${FILES[*]} \$host:~/distributed_prng_analysis/"
echo "   done"
