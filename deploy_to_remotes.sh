#!/bin/bash
echo "=========================================="
echo "DEPLOYING TO REMOTE NODES"
echo "=========================================="

REMOTES="192.168.3.120 192.168.3.154"

# Critical Python files
FILES=(
    "prng_registry.py"
    "sieve_filter.py"
    "cluster_models.py"
    "distributed_worker.py"
    "scorer_trial_worker.py"
    "anti_overfit_trial_worker.py"
    "adaptive_thresholds.py"
    "survivor_scorer.py"
    "reinforcement_engine.py"
)

# Critical directories
DIRS=(
    "integration"
    "core"
    "schemas"
)

for remote in $REMOTES; do
    echo ""
    echo "→ Deploying to $remote..."
    
    # Deploy files
    for f in "${FILES[@]}"; do
        if [ -f "$f" ]; then
            scp -q "$f" "michael@${remote}:~/distributed_prng_analysis/" && echo "  ✓ $f"
        fi
    done
    
    # Deploy directories
    for d in "${DIRS[@]}"; do
        if [ -d "$d" ]; then
            rsync -aq "$d/" "michael@${remote}:~/distributed_prng_analysis/${d}/" && echo "  ✓ $d/"
        fi
    done
    
    echo "✅ $remote complete"
done

echo ""
echo "=========================================="
echo "✅ DEPLOYMENT COMPLETE!"
echo "=========================================="
