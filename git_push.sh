#!/bin/bash
# Quick git push script for distributed_prng_analysis
cd ~/distributed_prng_analysis

# Show status
echo "=== Current Status ==="
git status --short

# Prompt for commit message
echo ""
read -p "Commit message: " msg

if [ -z "$msg" ]; then
    echo "❌ No commit message provided. Aborting."
    exit 1
fi

# Add all changes, commit, and push
git add -A
git commit -m "$msg"
git push

echo ""
echo "✅ Push complete"
