#!/bin/bash
# Emergency Rollback Script - Reverts ALL Step 2.5 GPU changes
# Run this if anything goes wrong

set -e

echo "======================================================================="
echo "Step 2.5 GPU Implementation - EMERGENCY ROLLBACK"
echo "======================================================================="
echo ""
echo "⚠️  WARNING: This will revert all changes made by the deployment"
echo ""

# Find backup files
PRNG_BACKUPS=$(ls -t prng_registry.py.backup_* 2>/dev/null | head -5)
SCORER_BACKUPS=$(ls -t survivor_scorer.py.backup_* 2>/dev/null | head -5)

if [ -z "$PRNG_BACKUPS" ] && [ -z "$SCORER_BACKUPS" ]; then
    echo "❌ ERROR: No backup files found!"
    echo "   Nothing to restore."
    exit 1
fi

echo "Found backups:"
echo ""

if [ -n "$PRNG_BACKUPS" ]; then
    echo "prng_registry.py backups:"
    ls -lh prng_registry.py.backup_* 2>/dev/null | tail -5 | awk '{print "  " $9 " (" $6 " " $7 " " $8 ")"}'
    echo ""
fi

if [ -n "$SCORER_BACKUPS" ]; then
    echo "survivor_scorer.py backups:"
    ls -lh survivor_scorer.py.backup_* 2>/dev/null | tail -5 | awk '{print "  " $9 " (" $6 " " $7 " " $8 ")"}'
    echo ""
fi

read -p "Continue with rollback? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Rollback cancelled."
    exit 0
fi

# Get most recent backups
LATEST_PRNG=$(ls -t prng_registry.py.backup_* 2>/dev/null | head -1)
LATEST_SCORER=$(ls -t survivor_scorer.py.backup_* 2>/dev/null | head -1)

# Rollback prng_registry.py
if [ -n "$LATEST_PRNG" ]; then
    echo ""
    echo "Rolling back prng_registry.py..."
    echo "  From: $LATEST_PRNG"
    
    # Create safety backup of current modified version
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    cp prng_registry.py "prng_registry.py.modified_${TIMESTAMP}"
    echo "  Safety backup: prng_registry.py.modified_${TIMESTAMP}"
    
    # Restore
    cp "$LATEST_PRNG" prng_registry.py
    echo "  ✅ prng_registry.py restored"
    
    # Verify
    if python3 -c "import prng_registry" 2>/dev/null; then
        echo "  ✅ Syntax verified"
    else
        echo "  ⚠️  Syntax check failed (but file restored)"
    fi
fi

# Rollback survivor_scorer.py
if [ -n "$LATEST_SCORER" ]; then
    echo ""
    echo "Rolling back survivor_scorer.py..."
    echo "  From: $LATEST_SCORER"
    
    # Create safety backup of current modified version
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    cp survivor_scorer.py "survivor_scorer.py.modified_${TIMESTAMP}"
    echo "  Safety backup: survivor_scorer.py.modified_${TIMESTAMP}"
    
    # Restore
    cp "$LATEST_SCORER" survivor_scorer.py
    echo "  ✅ survivor_scorer.py restored"
    
    # Verify
    if python3 -c "from survivor_scorer import SurvivorScorer" 2>/dev/null; then
        echo "  ✅ Syntax verified"
    else
        echo "  ⚠️  Syntax check failed (but file restored)"
    fi
fi

echo ""
echo "Rollback to remote nodes?"
echo "======================================================================="
read -p "Rollback to 192.168.3.120 and 192.168.3.154? (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    for host in 192.168.3.120 192.168.3.154; do
        echo ""
        echo "Rolling back $host..."
        
        if [ -n "$LATEST_PRNG" ]; then
            scp prng_registry.py $host:~/distributed_prng_analysis/
            echo "  ✅ prng_registry.py deployed"
        fi
        
        if [ -n "$LATEST_SCORER" ]; then
            scp survivor_scorer.py $host:~/distributed_prng_analysis/
            echo "  ✅ survivor_scorer.py deployed"
        fi
        
        # Verify remote
        if ssh $host "cd ~/distributed_prng_analysis && python3 -c 'import prng_registry; from survivor_scorer import SurvivorScorer'" 2>/dev/null; then
            echo "  ✅ $host verified"
        else
            echo "  ⚠️  $host verification failed"
        fi
    done
fi

echo ""
echo "======================================================================="
echo "✅ ROLLBACK COMPLETE"
echo "======================================================================="
echo ""
echo "Restored files:"
if [ -n "$LATEST_PRNG" ]; then
    echo "  ✅ prng_registry.py (from $LATEST_PRNG)"
fi
if [ -n "$LATEST_SCORER" ]; then
    echo "  ✅ survivor_scorer.py (from $LATEST_SCORER)"
fi
echo ""
echo "Safety backups of modified versions:"
echo "  prng_registry.py.modified_*"
echo "  survivor_scorer.py.modified_*"
echo ""
echo "Test that everything works:"
echo "  bash run_scorer_meta_optimizer.sh 6"
echo ""
echo "All changes reverted - system back to pre-GPU state."
echo ""
