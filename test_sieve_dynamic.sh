#!/bin/bash
# Test script for sieve dynamic optimization

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     Sieve Dynamic Optimization - Test & Apply             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Backup current coordinator.py
echo "📦 Creating backup..."
cp coordinator.py coordinator.py.backup_$(date +%s)
echo "✅ Backup created"
echo ""

# Step 2: Apply the fix
echo "🔧 Applying sieve dynamic optimization..."
python3 enable_sieve_dynamic.py
echo ""

# Step 3: Check if fix was successful
if [ -f coordinator_sieve_dynamic.py ]; then
    echo "✅ Fix script completed"
    echo ""
    
    # Show the differences
    echo "📊 Changes preview (first 50 lines of diff):"
    diff -u coordinator.py coordinator_sieve_dynamic.py | head -50
    echo ""
    
    # Ask user to confirm
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Apply these changes? (y/n)"
    read -p "> " confirm
    
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        cp coordinator_sieve_dynamic.py coordinator.py
        echo "✅ Changes applied!"
        echo ""
        
        # Step 4: Run quick test
        echo "🧪 Running quick test (10K seeds)..."
        echo ""
        python3 coordinator.py \
            --resume-policy restart \
            --max-concurrent 26 \
            daily3.json \
            --method residue_sieve \
            --prng-type lcg32 \
            --skip-min 0 \
            --skip-max 20 \
            --threshold 0.01 \
            --window-size 768 \
            --session-filter both \
            --seed-start 0 \
            --seeds 10000
        
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "✅ Test complete!"
        echo ""
        echo "If test passed, run full analysis:"
        echo "  python3 coordinator.py --resume-policy restart --max-concurrent 26 \\"
        echo "    daily3.json --method residue_sieve --prng-type lcg32 \\"
        echo "    --skip-min 0 --skip-max 20 --threshold 0.01 \\"
        echo "    --window-size 768 --session-filter both \\"
        echo "    --seed-start 0 --seeds 10000000"
        echo ""
        echo "Expected time: ~15-20 seconds (vs 321 seconds before!)"
    else
        echo "❌ Changes NOT applied"
        echo "To restore backup: cp coordinator.py.backup_* coordinator.py"
    fi
else
    echo "❌ Fix script failed - check enable_sieve_dynamic.py output"
fi
