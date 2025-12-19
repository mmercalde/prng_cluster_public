#!/bin/bash
# Step 3 Clean Test Script
# Run this on Zeus to test Step 3 end-to-end

echo "=============================================="
echo "STEP 3: CLEAN END-TO-END TEST"
echo "=============================================="
echo ""

cd ~/distributed_prng_analysis

# Phase 0: Clean up ALL old results
echo "Phase 0: Cleaning up old results..."
echo "----------------------------------------------"

# Local cleanup
echo "  Removing local TEMPORARY files..."
rm -rf scoring_chunks/ 2>/dev/null && echo "    - scoring_chunks/ (temp chunks)"
rm -rf full_scoring_results/ 2>/dev/null && echo "    - full_scoring_results/ (temp results)"
rm -f scoring_jobs.json 2>/dev/null && echo "    - scoring_jobs.json (job manifest)"

# Backup survivors_with_scores.json if it exists (this is the FINAL output)
if [ -f "survivors_with_scores.json" ]; then
    BACKUP_NAME="survivors_with_scores.json.backup_$(date +%Y%m%d_%H%M%S)"
    mv survivors_with_scores.json "$BACKUP_NAME"
    echo "    - survivors_with_scores.json → BACKED UP to $BACKUP_NAME"
fi

# Remote cleanup
for node in 192.168.3.120 192.168.3.154; do
    echo "  Cleaning remote node: $node"
    ssh michael@$node "cd ~/distributed_prng_analysis && rm -rf scoring_chunks/ full_scoring_results/ scoring_jobs.json 2>/dev/null" 2>/dev/null
done

echo "✓ Cleanup complete"
echo ""

# Phase 1: Verify prerequisites
echo "Phase 1: Verifying prerequisites..."
echo "----------------------------------------------"

PREREQS_OK=true

# Check required input files
for file in bidirectional_survivors.json train_history.json optimal_scorer_config.json; do
    if [ -f "$file" ]; then
        SIZE=$(ls -lh "$file" | awk '{print $5}')
        echo "  ✓ $file ($SIZE)"
    else
        echo "  ✗ MISSING: $file"
        PREREQS_OK=false
    fi
done

# Check optional but recommended files
for file in forward_survivors.json reverse_survivors.json; do
    if [ -f "$file" ]; then
        SIZE=$(ls -lh "$file" | awk '{print $5}')
        echo "  ✓ $file ($SIZE) [optional]"
    else
        echo "  ⚠ $file not found (metadata merge will use defaults)"
    fi
done

# Check scripts exist
for script in generate_step3_scoring_jobs.py full_scoring_worker.py; do
    if [ -f "$script" ]; then
        echo "  ✓ $script exists"
    else
        echo "  ✗ MISSING SCRIPT: $script"
        PREREQS_OK=false
    fi
done

if [ "$PREREQS_OK" = false ]; then
    echo ""
    echo "❌ Prerequisites check failed. Fix missing files first."
    exit 1
fi

echo "✓ All prerequisites verified"
echo ""

# Phase 2: Show survivor count
echo "Phase 2: Counting survivors..."
echo "----------------------------------------------"
SURVIVOR_COUNT=$(python3 -c "import json; print(len(json.load(open('bidirectional_survivors.json'))))" 2>/dev/null)
echo "  Survivors to process: $SURVIVOR_COUNT"
echo ""

# Phase 3: Check cluster connectivity
echo "Phase 3: Testing cluster connectivity..."
echo "----------------------------------------------"
for node in localhost 192.168.3.120 192.168.3.154; do
    if [ "$node" = "localhost" ]; then
        echo "  ✓ localhost (Zeus) - OK"
    else
        if ssh -o ConnectTimeout=5 michael@$node "echo OK" >/dev/null 2>&1; then
            GPU_COUNT=$(ssh michael@$node "ls /sys/class/drm/ | grep -c 'card[0-9]*$'" 2>/dev/null || echo "?")
            echo "  ✓ $node - OK (GPUs detected: $GPU_COUNT)"
        else
            echo "  ✗ $node - Connection failed"
        fi
    fi
done
echo ""

echo "=============================================="
echo "Ready to run Step 3!"
echo "=============================================="
echo ""
echo "To execute, run:"
echo "  ./run_step3_full_scoring.sh --survivors bidirectional_survivors.json --train-history train_history.json"
echo ""
echo "Or with forward/reverse for full metadata:"
echo "  ./run_step3_full_scoring.sh \\"
echo "    --survivors bidirectional_survivors.json \\"
echo "    --train-history train_history.json \\"
echo "    --forward-survivors forward_survivors.json \\"
echo "    --reverse-survivors reverse_survivors.json"
