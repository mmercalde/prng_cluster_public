#!/bin/bash
# ============================================================
# Documentation Paradigm Correction Script - CORRECTED PATHS
# Date: 2026-01-19
# Applies functional mimicry language fixes
# ============================================================

set -e
cd ~/distributed_prng_analysis

echo "=== Documentation Paradigm Correction ==="
echo ""

# ============================================================
# PATCH 1: CHAPTER_13_LIVE_FEEDBACK_LOOP.md (main directory)
# ============================================================
TARGET1="CHAPTER_13_LIVE_FEEDBACK_LOOP.md"

if [ -f "$TARGET1" ]; then
    echo "Patching $TARGET1..."
    cp "$TARGET1" "${TARGET1}.bak"
    
    python3 << 'PATCH1'
target = 'CHAPTER_13_LIVE_FEEDBACK_LOOP.md'

with open(target, 'r') as f:
    content = f.read()

# Check if already patched
if 'functional mimicry quality' in content:
    print("  ✅ Already patched")
    exit(0)

old_section = '''### Validation Purpose

With known `true_seed`:
- System should "find" the correct pattern
- True seed should rise in survivor rankings over iterations
- Validates learning loop actually learns

### Convergence Expectation

> **With a correct PRNG hypothesis, the true seed should enter the top-K survivors within N synthetic draws.**

| Metric | Target | Failure Indicates |
|--------|--------|-------------------|
| True seed in top-100 | ≤ 20 draws | Feature construction issue |
| True seed in top-20 | ≤ 50 draws | Learning loop issue |
| Confidence on true seed | Increasing trend | Calibration issue |

This provides a quantitative pass/fail test for the learning loop.'''

new_section = '''### Validation Purpose

With known `true_seed`:
- System generates **consistent, reproducible test sequences**
- Allows measurement of whether pattern learning improves over iterations
- Validates that the feedback loop actually learns surface patterns

> **Note:** The `true_seed` is used for reproducible test data generation, NOT as a discovery target. The system learns output patterns, not seed values.

### Convergence Expectation

> **With a correct PRNG hypothesis, learned patterns should produce measurable prediction lift over random baseline.**

| Metric | Target | Failure Indicates |
|--------|--------|-------------------|
| Hit Rate (Top-20) | > 5% (vs 0.1% random) | Pattern extraction not working |
| Confidence Calibration | Correlation > 0.3 | Confidence scores meaningless |
| Hit Rate Trend | Non-decreasing over N draws | Learning loop not improving |

This provides a quantitative pass/fail test for functional mimicry quality.'''

if old_section in content:
    content = content.replace(old_section, new_section)
    with open(target, 'w') as f:
        f.write(content)
    print("  ✅ Patched successfully")
else:
    # Try to find the section with possible encoding variations
    if 'True seed in top-100' in content or 'true seed should enter the top-K' in content:
        # Do line-by-line replacement
        lines = content.split('\n')
        new_lines = []
        skip_mode = False
        inserted = False
        
        for i, line in enumerate(lines):
            if '### Validation Purpose' in line and not inserted:
                new_lines.append(new_section)
                skip_mode = True
                inserted = True
            elif skip_mode and ('### Safety: Test Mode Gating' in line or '### Safety:' in line):
                skip_mode = False
                new_lines.append('')
                new_lines.append(line)
            elif not skip_mode:
                new_lines.append(line)
        
        if inserted:
            content = '\n'.join(new_lines)
            with open(target, 'w') as f:
                f.write(content)
            print("  ✅ Patched (line-by-line mode)")
        else:
            print("  ⚠️  Could not locate section markers")
    else:
        print("  ⚠️  Section not found - may need manual review")
PATCH1
else
    echo "  ⚠️  $TARGET1 not found"
fi

# ============================================================
# PATCH 2: CHAPTER_13_IMPLEMENTATION_PROGRESS_v1_1.md 
# ============================================================
TARGET2="CHAPTER_13_IMPLEMENTATION_PROGRESS_v1_1.md"

if [ -f "$TARGET2" ]; then
    echo ""
    echo "Checking $TARGET2..."
    
    # Check if it has the old language
    if grep -q "True seed in top-100\|True seed in top-20" "$TARGET2" 2>/dev/null; then
        echo "  Found old language, patching..."
        cp "$TARGET2" "${TARGET2}.bak"
        
        python3 << 'PATCH2'
target = 'CHAPTER_13_IMPLEMENTATION_PROGRESS_v1_1.md'

with open(target, 'r') as f:
    content = f.read()

# Replace old convergence metrics table
old_table = '''| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| True seed in top-100 | ≤20 draws | - | 🔲 |
| True seed in top-20 | ≤50 draws | - | 🔲 |
| Confidence trend | Increasing | - | 🔲 |
| Hit rate | >0.05 | - | 🔲 |'''

new_table = '''| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| **Hit Rate** | >5% (better than random) | - | 🔲 |
| **Confidence Calibration** | Correlation >0.3 | - | 🔲 |
| **Hit Rate Improvement** | Increasing over N draws | - | 🔲 |
| **Pattern Stability** | Consistent across PRNG types | - | 🔲 |'''

if old_table in content:
    content = content.replace(old_table, new_table)
    with open(target, 'w') as f:
        f.write(content)
    print("  ✅ Patched successfully")
else:
    # Check for variations
    if 'True seed in top-100' in content:
        content = content.replace('True seed in top-100', 'Hit Rate (Top-20)')
        content = content.replace('≤20 draws', '>5%')
        content = content.replace('True seed in top-20', 'Confidence Calibration')
        content = content.replace('≤50 draws', 'Correlation >0.3')
        with open(target, 'w') as f:
            f.write(content)
        print("  ✅ Patched (string replacement)")
    else:
        print("  ⚠️  Could not find exact match")
PATCH2
    else
        echo "  ✅ Already uses correct language"
    fi
else
    echo "  ⚠️  $TARGET2 not found"
fi

# ============================================================
# PATCH 3: docs/CHAPTER_13_IMPLEMENTATION_PROGRESS_v1_2.md
# ============================================================
TARGET3="docs/CHAPTER_13_IMPLEMENTATION_PROGRESS_v1_2.md"

if [ -f "$TARGET3" ]; then
    echo ""
    echo "Checking $TARGET3..."
    
    if grep -q "True seed in top-100\|True seed in top-20" "$TARGET3" 2>/dev/null; then
        echo "  Found old language - needs patching"
    else
        echo "  ✅ Already uses correct language"
    fi
else
    echo "  ⚠️  $TARGET3 not found"
fi

# ============================================================
# VERIFICATION
# ============================================================
echo ""
echo "=== Verification ==="

echo "Checking for old 'seed ranking' language:"
grep -rn "True seed in top-100\|True seed in top-20\|true.*seed.*ranking" *.md docs/*.md 2>/dev/null || echo "  ✅ No old language found"

echo ""
echo "Checking for new 'functional mimicry' language:"
grep -l "functional mimicry\|Hit Rate.*Top-20\|pattern.learning" *.md docs/*.md 2>/dev/null | while read f; do
    echo "  ✅ $f"
done

# ============================================================
# GIT COMMIT
# ============================================================
echo ""
echo "=== Files Modified ==="
git status --short

echo ""
read -p "Commit and push? (y/n): " confirm
if [ "$confirm" = "y" ]; then
    git add -A
    git commit -m "docs: Correct functional mimicry paradigm language

Files updated:
- CHAPTER_13_LIVE_FEEDBACK_LOOP.md (Section 6.1)
- CHAPTER_13_IMPLEMENTATION_PROGRESS_v1_1.md (if applicable)

Changes:
- Removed 'true seed in top-K' convergence targets
- Added proper metrics: hit rate, confidence calibration
- Clarified true_seed is for reproducible test data, not discovery
- Added paradigm notes explaining seeds as vehicles for heuristics

Code is unchanged - already measures correct metrics."
    
    git push
    echo "✅ Committed and pushed"
else
    echo "Skipped commit. Run manually when ready."
fi

echo ""
echo "=== Done ==="
