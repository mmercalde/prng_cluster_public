#!/bin/bash
# ============================================================
# Documentation Paradigm Correction Script
# Date: 2026-01-19
# Applies functional mimicry language fixes to 2 files
# ============================================================

set -e
cd ~/distributed_prng_analysis

echo "=== Documentation Paradigm Correction ==="
echo ""

# ============================================================
# PATCH 1: CANONICAL_PIPELINE_AND_CH13_WITH_STARTUP_COMPLETE.md
# ============================================================
echo "Patching CANONICAL_PIPELINE_AND_CH13_WITH_STARTUP_COMPLETE.md..."

# Create backup
cp CANONICAL_PIPELINE_AND_CH13_WITH_STARTUP_COMPLETE.md CANONICAL_PIPELINE_AND_CH13_WITH_STARTUP_COMPLETE.md.bak

# Use Python for reliable multi-line replacement
python3 << 'PATCH1'
import re

with open('CANONICAL_PIPELINE_AND_CH13_WITH_STARTUP_COMPLETE.md', 'r') as f:
    content = f.read()

old_section = '''## 9. Convergence Expectations (Test Mode)

| Metric | Target |
|------|-------|
| True seed in top‑100 | ≤ 20 draws |
| True seed in top‑20 | ≤ 50 draws |
| Confidence trend | Increasing |
| Diagnostics | Stable or improving |

Failure to converge is **information**, not error.'''

new_section = '''## 9. Convergence Expectations (Test Mode)

> **Paradigm:** This system performs functional mimicry — learning output patterns to predict future draws. Seeds generate candidate sequences for heuristic extraction; they are not discovery targets.

| Metric | Target | Meaning |
|--------|--------|---------|
| Hit Rate (Top-20) | > 5% | Predictions outperform random (0.1%) |
| Confidence Calibration | Correlation > 0.3 | High confidence → higher hit probability |
| Hit Rate Trend | Non-decreasing | Learning loop is improving |
| Diagnostics | Stable or improving | No feature drift or degradation |

Failure to converge is **information**, not error — it indicates the PRNG hypothesis may need adjustment, not that the system is broken.'''

if old_section in content:
    content = content.replace(old_section, new_section)
    with open('CANONICAL_PIPELINE_AND_CH13_WITH_STARTUP_COMPLETE.md', 'w') as f:
        f.write(content)
    print("  ✅ CANONICAL_PIPELINE patched successfully")
else:
    # Try with different dash encoding
    content_normalized = content.replace('–', '-').replace('‑', '-')
    old_normalized = old_section.replace('–', '-').replace('‑', '-')
    if old_normalized in content_normalized:
        # Find and replace accounting for encoding
        content = content.replace('True seed in top‑100', 'MARKER_TOP100')
        content = content.replace('True seed in top-100', 'MARKER_TOP100')
        if 'MARKER_TOP100' in content:
            # Section exists, do full replacement via line-by-line
            lines = content.split('\n')
            new_lines = []
            skip_until_next_section = False
            for i, line in enumerate(lines):
                if '## 9. Convergence Expectations' in line:
                    new_lines.append(new_section)
                    skip_until_next_section = True
                elif skip_until_next_section and line.startswith('## 10.'):
                    skip_until_next_section = False
                    new_lines.append('')
                    new_lines.append(line)
                elif not skip_until_next_section:
                    new_lines.append(line)
            content = '\n'.join(new_lines)
            with open('CANONICAL_PIPELINE_AND_CH13_WITH_STARTUP_COMPLETE.md', 'w') as f:
                f.write(content)
            print("  ✅ CANONICAL_PIPELINE patched (alternate encoding)")
        else:
            print("  ⚠️  Could not find section to patch - may already be updated")
    else:
        print("  ⚠️  Section not found - file may already be updated")
PATCH1

# ============================================================
# PATCH 2: CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md  
# ============================================================
echo ""
echo "Patching CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md..."

# Create backup
cp CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md.bak

python3 << 'PATCH2'
with open('CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md', 'r') as f:
    content = f.read()

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
    with open('CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md', 'w') as f:
        f.write(content)
    print("  ✅ CHAPTER_13_LIVE_FEEDBACK_LOOP patched successfully")
else:
    # Check if already patched
    if 'functional mimicry quality' in content:
        print("  ✅ CHAPTER_13_LIVE_FEEDBACK_LOOP already patched")
    else:
        print("  ⚠️  Section not found exactly - attempting fuzzy match...")
        # Try line-by-line approach
        lines = content.split('\n')
        new_lines = []
        skip_mode = False
        inserted = False
        
        for i, line in enumerate(lines):
            if '### Validation Purpose' in line and not inserted:
                new_lines.append(new_section)
                skip_mode = True
                inserted = True
            elif skip_mode and '### Safety: Test Mode Gating' in line:
                skip_mode = False
                new_lines.append('')
                new_lines.append(line)
            elif not skip_mode:
                new_lines.append(line)
        
        if inserted:
            content = '\n'.join(new_lines)
            with open('CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md', 'w') as f:
                f.write(content)
            print("  ✅ CHAPTER_13_LIVE_FEEDBACK_LOOP patched (fuzzy match)")
        else:
            print("  ❌ Could not patch - manual intervention needed")
PATCH2

# ============================================================
# VERIFICATION
# ============================================================
echo ""
echo "=== Verification ==="

# Check for old language (should return nothing)
echo "Checking for old 'seed ranking' language (should be empty):"
grep -n "seed.*top-100\|seed.*top-20\|true.*seed.*ranking" *.md 2>/dev/null || echo "  ✅ No old language found"

echo ""
echo "Checking for new 'functional mimicry' language:"
grep -c "functional mimicry\|Hit Rate.*Top-20\|pattern.learning" CANONICAL_PIPELINE_AND_CH13_WITH_STARTUP_COMPLETE.md CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md 2>/dev/null && echo "  ✅ New language present"

# ============================================================
# GIT COMMIT
# ============================================================
echo ""
echo "=== Git Commit ==="
git add -A
git status

echo ""
read -p "Commit and push? (y/n): " confirm
if [ "$confirm" = "y" ]; then
    git commit -m "docs: Correct functional mimicry paradigm language

Files updated:
- CANONICAL_PIPELINE_AND_CH13_WITH_STARTUP_COMPLETE.md (Section 9)
- CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md (Section 6.1)

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
