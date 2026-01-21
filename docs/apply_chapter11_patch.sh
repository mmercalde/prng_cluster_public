#!/bin/bash
# ============================================================================
# PATCH: Chapter 11 LLM Update - Qwen to DeepSeek + Claude
# Date: 2026-01-19
# ============================================================================

set -e

CHAPTER_FILE="CHAPTER_11_FEATURE_IMPORTANCE_VISUALIZATION.md"
BACKUP_FILE="CHAPTER_11_FEATURE_IMPORTANCE_VISUALIZATION.md.bak_qwen_$(date +%Y%m%d)"

cd ~/distributed_prng_analysis/docs

# Backup first
echo "Creating backup: $BACKUP_FILE"
cp "$CHAPTER_FILE" "$BACKUP_FILE"

# Patch 1: Line 56 - Key Features table
echo "Patch 1: Updating Key Features table..."
sed -i 's/LLM-powered analysis via Qwen2.5-Math/LLM-powered analysis via DeepSeek-R1-14B + Claude backup/g' "$CHAPTER_FILE"

# Patch 2: Line 763 - Port 8081 to 8080
echo "Patch 2: Updating LLM endpoint port..."
sed -i 's|http://localhost:8081/v1/completions|http://localhost:8080/completion|g' "$CHAPTER_FILE"

# Patch 3: Line 766 - Docstring
echo "Patch 3: Updating docstring..."
sed -i 's/Generate AI-powered interpretation using Qwen2.5-Math/Generate AI-powered interpretation using DeepSeek-R1-14B/g' "$CHAPTER_FILE"

# Patch 4: Line 768 - Comment
echo "Patch 4: Updating comment..."
sed -i 's/Uses the Math LLM for statistical reasoning/Uses DeepSeek-R1-14B (primary) with Claude backup for statistical reasoning/g' "$CHAPTER_FILE"

# Patch 5: Line 809 - Example header
echo "Patch 5: Updating example header..."
sed -i 's/AI Interpretation (Qwen2.5-Math-7B):/AI Interpretation (DeepSeek-R1-14B):/g' "$CHAPTER_FILE"

# Patch 6: Line 1084 - Summary
echo "Patch 6: Updating summary..."
sed -i 's/AI interpretation via Qwen2.5-Math/AI interpretation via DeepSeek-R1-14B + Claude backup/g' "$CHAPTER_FILE"

echo ""
echo "=============================================="
echo "Verifying patches..."
echo "=============================================="

# Verify no Qwen references remain
QWEN_COUNT=$(grep -ci "qwen" "$CHAPTER_FILE" || echo "0")
if [ "$QWEN_COUNT" -gt "0" ]; then
    echo "⚠️  WARNING: $QWEN_COUNT Qwen references still found:"
    grep -in "qwen" "$CHAPTER_FILE"
else
    echo "✅ No Qwen references remain"
fi

# Verify DeepSeek references exist
DEEPSEEK_COUNT=$(grep -ci "deepseek" "$CHAPTER_FILE" || echo "0")
echo "✅ Found $DEEPSEEK_COUNT DeepSeek references"

# Verify old port removed
OLD_PORT=$(grep -c "8081" "$CHAPTER_FILE" || echo "0")
if [ "$OLD_PORT" -gt "0" ]; then
    echo "⚠️  WARNING: Old port 8081 still found"
else
    echo "✅ Old port 8081 removed"
fi

echo ""
echo "=============================================="
echo "✅ Chapter 11 patched successfully!"
echo "=============================================="
echo ""
echo "Backup: $BACKUP_FILE"
echo ""
echo "To review changes:"
echo "  diff $BACKUP_FILE $CHAPTER_FILE | head -50"
echo ""
echo "To revert:"
echo "  cp $BACKUP_FILE $CHAPTER_FILE"
