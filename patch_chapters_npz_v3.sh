#!/bin/bash
# patch_chapters_npz_v3.sh - Update Chapter 3 and 4 with NPZ v3.0 info
#
# USAGE:
#   cd ~/distributed_prng_analysis
#   bash patch_chapters_npz_v3.sh

set -e
cd ~/distributed_prng_analysis/docs

echo "=============================================="
echo "Updating Chapters 3 & 4 for NPZ v3.0"
echo "=============================================="

# ============================================
# CHAPTER 3: Update NPZ table
# ============================================
echo ""
echo "[1/2] Updating CHAPTER_3_SCORER_META_OPTIMIZER.md..."

# Backup
cp CHAPTER_3_SCORER_META_OPTIMIZER.md CHAPTER_3_SCORER_META_OPTIMIZER.md.bak

# Replace the NPZ table (lines 743-746)
sed -i 's/| JSON | 258 MB | 4.2s |/| JSON | 57.9 MB | 4.2s |/' CHAPTER_3_SCORER_META_OPTIMIZER.md
sed -i 's/| NPZ | 0.6 MB | 0.05s |/| NPZ v3.0 | 733 KB | 0.05s |/' CHAPTER_3_SCORER_META_OPTIMIZER.md

# Add note about v3.0 after the conversion line
sed -i '/python3 convert_survivors_to_binary.py bidirectional_survivors.json/a\
\
**Note (Jan 23, 2026):** NPZ v3.0 preserves all 22 metadata fields. Earlier versions (v1/v2) only saved 3 arrays, causing 14/47 ML features to be zeroed in Step 3. Always use v3.0 for full feature extraction.' CHAPTER_3_SCORER_META_OPTIMIZER.md

echo "✓ CHAPTER_3 updated"

# ============================================
# CHAPTER 4: Add NPZ v3.0 data source note
# ============================================
echo ""
echo "[2/2] Updating CHAPTER_4_FULL_SCORING.md..."

# Backup
cp CHAPTER_4_FULL_SCORING.md CHAPTER_4_FULL_SCORING.md.bak

# Add NPZ data source note after "### 1.3 Key Features" section (after line ~59)
# Find the line with "Holdout Integration" and add after its table row
sed -i '/| \*\*Holdout Integration\*\* | Computes y-label for Step 5 ML training |/a\
| **NPZ v3.0 Input** | Loads 22 metadata fields from binary format |' CHAPTER_4_FULL_SCORING.md

# Add a new subsection about data loading after BUG FIX 5
sed -i '/BUG FIX 5: Explicit two-step NumPy to GPU tensor transfer (ROCm stability)/a\
BUG FIX 6: NPZ v3.0 metadata preservation (Jan 23, 2026) - all 22 fields now loaded' CHAPTER_4_FULL_SCORING.md

echo "✓ CHAPTER_4 updated"

# ============================================
# Verify
# ============================================
echo ""
echo "=============================================="
echo "Verification"
echo "=============================================="

echo ""
echo "CHAPTER_3 NPZ section:"
grep -A5 "Binary Survivor Loading" CHAPTER_3_SCORER_META_OPTIMIZER.md | head -10

echo ""
echo "CHAPTER_4 Key Features table:"
grep -A2 "NPZ v3.0" CHAPTER_4_FULL_SCORING.md | head -5

echo ""
echo "=============================================="
echo "PATCH COMPLETE"
echo "=============================================="
echo ""
echo "Next: git add and commit the updated chapters"
