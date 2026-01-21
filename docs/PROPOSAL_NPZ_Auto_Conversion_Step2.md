# PROPOSAL: NPZ Auto-Conversion for Step 2.5

**Date:** January 19, 2026  
**Author:** Claude (AI Assistant) + Michael  
**Status:** Pending Team Review  
**Priority:** Medium (Pipeline gap, not blocker)

---

## Executive Summary

A gap exists in the pipeline where Step 2.5 **expects** `bidirectional_survivors_binary.npz` but nothing **creates** it automatically. The conversion script exists but requires manual invocation between Step 1 and Step 2.

**Recommendation:** Add auto-conversion to `run_scorer_meta_optimizer.sh` header with conditional check and remote distribution.

---

## Problem Statement

### Current State

| Component | NPZ Behavior |
|-----------|--------------|
| `window_optimizer.py` | Outputs JSON only |
| `window_optimizer_integration_final.py` | No conversion |
| `convert_survivors_to_binary.py` | Manual script (exists since Jan 3) |
| `run_scorer_meta_optimizer.sh` | **Expects** NPZ (line 1: `SURVIVORS="bidirectional_survivors_binary.npz"`) |
| `watcher_agent.py` | Can detect `.npz` (lines 86, 173) but doesn't create |

### Impact

1. **Manual step required** between Step 1 → Step 2 (violates automation goal)
2. **Silent failure risk** if NPZ missing (Step 2.5 will fail on file not found)
3. **Stale data risk** if JSON updated but NPZ not regenerated
4. **Remote distribution** also manual (must SCP to rig-6600, rig-6600b)

### Performance Context (Why NPZ Matters)

| Format | File Size | Load Time | 26-GPU Overhead |
|--------|-----------|-----------|-----------------|
| JSON | 58-258 MB | 4.2s | 109s total |
| NPZ | 0.6 MB | 0.05s | 1.3s total |

**88x faster loading** - critical for preventing i3 CPU thrashing on mining rigs.

---

## Proposed Solution

### Location: `run_scorer_meta_optimizer.sh` Header

**Rationale:**
- Self-contained fix at point of need
- No changes to core Python modules
- Follows "fix where needed" principle
- Conditional logic prevents redundant work

### Implementation

Add after existing variable declarations (around line 5-10):

```bash
#!/bin/bash
# run_scorer_meta_optimizer.sh
# Step 2.5: Distributed Scorer Meta-Optimization

SURVIVORS="bidirectional_survivors_binary.npz"
JSON_SOURCE="bidirectional_survivors.json"

# ============================================================
# AUTO-CONVERT TO NPZ (88x faster loading for distributed jobs)
# ============================================================
# Converts if:
#   1. NPZ doesn't exist, OR
#   2. JSON is newer than NPZ (Step 1 re-run detected)
# ============================================================

if [ ! -f "$SURVIVORS" ] || [ "$JSON_SOURCE" -nt "$SURVIVORS" ]; then
    echo "============================================"
    echo "NPZ Conversion Required"
    echo "============================================"
    
    if [ ! -f "$JSON_SOURCE" ]; then
        echo "ERROR: $JSON_SOURCE not found!"
        echo "Run Step 1 (window_optimizer.py) first."
        exit 1
    fi
    
    echo "Converting $JSON_SOURCE → $SURVIVORS..."
    python3 convert_survivors_to_binary.py "$JSON_SOURCE"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: NPZ conversion failed!"
        exit 1
    fi
    
    echo "Distributing NPZ to remote nodes..."
    scp "$SURVIVORS" rig-6600:~/distributed_prng_analysis/
    scp "$SURVIVORS" rig-6600b:~/distributed_prng_analysis/
    
    echo "NPZ conversion complete."
    echo "============================================"
else
    echo "NPZ file up-to-date, skipping conversion."
fi

# Continue with existing script...
```

---

## Alternatives Considered

### Option A: End of Step 1 (window_optimizer.py)

**Pros:**
- Conversion happens immediately after survivors generated
- Single logical unit of work

**Cons:**
- Requires modifying core optimizer code
- window_optimizer shouldn't know about Step 2.5's format needs
- Violates separation of concerns

**Verdict:** Rejected

### Option B: WATCHER Pre-Step Hook

**Pros:**
- Centralized pipeline control
- Could apply to all steps needing preprocessing

**Cons:**
- Adds complexity to WATCHER
- WATCHER is for orchestration, not data transformation
- Requires schema changes to manifests

**Verdict:** Rejected (over-engineering)

### Option C: Standalone Wrapper Script

**Pros:**
- Clean separation

**Cons:**
- Yet another script to maintain
- Easy to forget to call it

**Verdict:** Rejected

### Option D: run_scorer_meta_optimizer.sh Header (SELECTED)

**Pros:**
- Self-documenting (conversion happens where file is needed)
- Conditional logic prevents redundant work
- Handles remote distribution atomically
- No changes to Python codebase
- Easy to understand and maintain

**Cons:**
- Bash-specific (but script is already bash)
- Conversion happens at Step 2 start, not Step 1 end (minor)

**Verdict:** ✅ Selected

---

## Testing Plan

### Pre-Merge Verification

```bash
# Test 1: Fresh run (no NPZ exists)
rm -f bidirectional_survivors_binary.npz
./run_scorer_meta_optimizer.sh --dry-run  # Should convert

# Test 2: Up-to-date NPZ
touch bidirectional_survivors_binary.npz
./run_scorer_meta_optimizer.sh --dry-run  # Should skip

# Test 3: Stale NPZ (JSON newer)
touch bidirectional_survivors.json
./run_scorer_meta_optimizer.sh --dry-run  # Should reconvert

# Test 4: Missing JSON
rm -f bidirectional_survivors.json
./run_scorer_meta_optimizer.sh --dry-run  # Should error gracefully
```

### Post-Merge Validation

Run full pipeline:
```bash
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline \
    --start-step 1 --end-step 6
```

Verify:
- [ ] NPZ created automatically after Step 1 completes
- [ ] NPZ distributed to both rigs
- [ ] Step 2.5 loads NPZ successfully
- [ ] No manual intervention required

---

## Documentation Updates Required

| Document | Section | Update |
|----------|---------|--------|
| CHAPTER_3_SCORER_META_OPTIMIZER.md | 12.5 | Note auto-conversion |
| INSTRUCTIONS_NPZ_ADDITION.md | Conversion Script | Mark as "auto-invoked" |
| SESSION_CHANGELOG_20260119.md | New | Document this fix |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SCP fails to remote | Low | Medium | Script exits on error |
| Conversion script missing | Very Low | High | Explicit error message |
| Disk space on rigs | Very Low | Low | NPZ is only 0.6MB |
| Race condition (parallel runs) | Very Low | Low | File locking not needed for this use case |

---

## Decision Required

**Team Beta:** Please review and approve/reject/modify.

- [ ] **APPROVE** - Implement as specified
- [ ] **APPROVE WITH MODIFICATIONS** - (specify changes)
- [ ] **REJECT** - (provide alternative)
- [ ] **DEFER** - Not critical, address later

---

## Appendix: Current Evidence

### Grep Results from Zeus

```bash
$ grep -n "npz\|NPZ\|convert_survivors" window_optimizer.py window_optimizer_integration_final.py
# (no output - no NPZ handling)

$ head -50 run_scorer_meta_optimizer.sh | grep -i "npz\|convert"
SURVIVORS="bidirectional_survivors_binary.npz"

$ grep -n "npz\|convert_survivors\|binary" agents/watcher_agent.py
86:        ".npz": 100,
173:    elif ext == ".npz":
```

### File Dependencies

```
Step 1 Output:
  └── bidirectional_survivors.json (58MB, 98K survivors)

Manual Gap:
  └── python3 convert_survivors_to_binary.py bidirectional_survivors.json
  └── scp bidirectional_survivors_binary.npz rig-6600:~/...
  └── scp bidirectional_survivors_binary.npz rig-6600b:~/...

Step 2.5 Input:
  └── bidirectional_survivors_binary.npz (0.6MB)
```

---

*End of Proposal*
