# PROPOSAL: Unified Ramdisk Extension to Steps 3 and 5

**To:** Team Beta  
**From:** Claude (AI Assistant)  
**Date:** 2026-01-21  
**Version:** 1.1 (added Lifecycle Management section)  
**Status:** PENDING REVIEW  
**Proposal ID:** PROP-2026-01-21-RAMDISK-UNIFIED  
**Depends On:** Step 2 Ramdisk (APPROVED & VALIDATED - 30/30 trials, 16 trials/min)

---

## 1. Executive Summary

This proposal extends the `/dev/shm` ramdisk preloading infrastructure (validated for Step 2) to cover **all distributed pipeline steps**: Steps 3 and 5.

**New in v1.1:** Ramdisk lifecycle management options for cleanup strategy.

| Aspect | Detail |
|--------|--------|
| **Scope** | Steps 2, 3, 5 (all distributed steps) |
| **Effort** | ~2 hours total implementation |
| **Risk Level** | Low (proven pattern from Step 2) |
| **Memory Impact** | ~1 GB combined (25% of 4GB `/dev/shm`) |

---

## 2. Current State (Step 2 Validated)

Step 2 ramdisk is **working and tested**:

```
✅ 30/30 trials completed
✅ 16.0 trials/min throughput
✅ Ramdisk preload on all 3 nodes (Zeus + rigs)
✅ /dev/shm/prng/ path unified
✅ .ready sentinel prevents redundant copies
```

Current ramdisk contents (Step 2):
```
/dev/shm/prng/ = 280KB
├── bidirectional_survivors_binary.npz (229KB)
├── train_history.json (27KB)
├── holdout_history.json (6.8KB)
└── scorer_jobs.json (9KB)
```

---

## 3. Proposed Extension

### 3.1 Steps Requiring Ramdisk

| Step | Script | Files | Est. Size | Complexity |
|------|--------|-------|-----------|------------|
| 2 ✅ | `run_scorer_meta_optimizer.sh` | NPZ + 3 JSON | 280 KB | Done |
| 3 | `run_step3_full_scoring.sh` | 4 JSON | ~520 MB | Shell (same pattern) |
| 5 | `meta_prediction_optimizer_anti_overfit.py` | 2 JSON | ~400 MB | Python orchestrator |

### 3.2 Step 3 Files

```bash
# Step 3 input files
bidirectional_survivors.json    # ~58 MB (from Step 1)
train_history.json              # ~27 KB
forward_survivors.json          # ~200+ MB
reverse_survivors.json          # ~200+ MB
```

### 3.3 Step 5 Files

```bash
# Step 5 input files
survivors_with_scores.json      # ~400 MB (from Step 3)
holdout_history.json            # ~7 KB
```

---

## 4. Memory Budget Analysis

### 4.1 Node Specifications

| Node | Total RAM | /dev/shm (50%) | Available |
|------|-----------|----------------|-----------|
| Zeus | 64 GB | 32 GB | Abundant |
| rig-6600 | 8 GB | 4 GB | Constrained |
| rig-6600b | 8 GB | 4 GB | Constrained |

### 4.2 Per-Step Memory Usage

| Step | Files Size | % of 4GB |
|------|------------|----------|
| Step 2 | 280 KB | 0.007% |
| Step 3 | ~520 MB | 13% |
| Step 5 | ~400 MB | 10% |

### 4.3 Scenarios

| Scenario | Memory Used | % of 4GB | Status |
|----------|-------------|----------|--------|
| Step 2 only | 280 KB | 0.007% | ✅ Safe |
| Step 3 only | 520 MB | 13% | ✅ Safe |
| Step 5 only | 400 MB | 10% | ✅ Safe |
| Steps 2 + 3 | 520 MB | 13% | ✅ Safe |
| Steps 3 + 5 | 920 MB | 23% | ✅ Safe |
| **All steps (no cleanup)** | **~1 GB** | **25%** | ⚠️ Works but tight |

**Conclusion:** Even worst case (all steps loaded) uses only 25% of available `/dev/shm`. However, cleanup between steps is **recommended** for headroom.

---

## 5. Ramdisk Lifecycle Management (NEW in v1.1)

### 5.1 The Question

When should ramdisk data be cleared?

### 5.2 Options

| Option | Trigger | Pros | Cons |
|--------|---------|------|------|
| **A) After each job** | Job completes → clear | Immediate memory reclaim | ❌ Defeats purpose - re-copy 12+ times per step |
| **B) After step completes** | Step ends → clear | Data persists for retries within step | Memory held for entire step duration |
| **C) Watchdog monitor** | `/dev/shm` > 75% used → clear oldest | Adaptive, only cleans when needed | Complex, potential race conditions, might clear mid-job |
| **D) Before next step** | Step N+1 starts → clear Step N files | Simple, predictable | Previous step data unavailable for debugging |
| **E) WATCHER-managed** | WATCHER clears between step transitions | Single control point, orchestrator owns lifecycle | Requires WATCHER dependency |
| **F) Manual only** | Operator runs `clear_ramdisk.sh` | Full control | Easy to forget, stale data risk |

### 5.3 Detailed Analysis

#### Option A: After Each Job ❌ NOT RECOMMENDED

```
Job 1 completes → clear ramdisk → Job 2 starts → reload data → ...
```

- **Problem:** 12 concurrent jobs on rig-6600 = 12 reloads per batch
- **Impact:** Negates all ramdisk benefits
- **Verdict:** Defeats the purpose

#### Option B: After Step Completes ⚠️ ACCEPTABLE

```
Step 2 runs (all jobs) → Step 2 completes → clear ramdisk → Step 3 starts
```

- **Implementation:** Add cleanup to end of each step's shell launcher
- **Pros:** Data available for retries within step
- **Cons:** Where exactly does "step complete" trigger? Success only? Failure too?

```bash
# At end of run_scorer_meta_optimizer.sh
cleanup_ramdisk() {
    echo "[INFO] Clearing Step 2 ramdisk data..."
    rm -rf /dev/shm/prng/*
    for NODE in $(get_remote_nodes); do
        ssh "$NODE" "rm -rf /dev/shm/prng/*"
    done
}
trap cleanup_ramdisk EXIT
```

#### Option C: Watchdog Monitor ⚠️ COMPLEX

```python
# ramdisk_watchdog.py (would run as daemon)
def check_usage():
    usage = get_shm_usage_percent()
    if usage > 75:
        clear_oldest_step_data()
```

- **Pros:** Adaptive, only cleans when necessary
- **Cons:** 
  - Race condition: might clear while job is reading
  - Complexity: needs daemon process
  - Debugging: hard to predict behavior
- **Verdict:** Over-engineered for current needs

#### Option D: Before Next Step ✅ SIMPLE & RECOMMENDED

```
Step 2 completes → Step 3 starts → clear Step 2 data → load Step 3 data
```

- **Implementation:** Each step's preload clears previous step's data first
- **Pros:** Simple, predictable, always fresh data
- **Cons:** Can't debug previous step's ramdisk state (minor - files still on disk)

```bash
# In ramdisk_preload.sh
preload_ramdisk() {
    local step_id=$1
    shift
    local files=("$@")
    
    # Clear ALL previous ramdisk data before loading new step
    echo "[INFO] Clearing ramdisk for fresh Step $step_id load..."
    rm -rf /dev/shm/prng/*
    
    # Now load this step's files
    for f in "${files[@]}"; do
        cp "$f" /dev/shm/prng/
    done
    touch "/dev/shm/prng/.ready_step${step_id}"
}
```

#### Option E: WATCHER-Managed ✅ CLEANEST ARCHITECTURE

```
WATCHER: run_step(2) → complete → clear_ramdisk() → run_step(3) → ...
```

- **Implementation:** WATCHER owns the cleanup lifecycle
- **Pros:**
  - Single point of control
  - Already orchestrates step transitions
  - Can make intelligent decisions (skip cleanup if same files needed)
  - Audit trail in WATCHER logs
- **Cons:** Only works when running through WATCHER (not standalone step runs)

```python
# In watcher_agent.py
def run_step(self, step: int):
    # Clear ramdisk before starting new step
    if step > 1:
        self.clear_ramdisk_for_previous_step(step - 1)
    
    # Load this step's data to ramdisk
    self.preload_ramdisk_for_step(step)
    
    # Execute the step
    result = self.execute_step_script(step)
    
    return result

def clear_ramdisk_for_previous_step(self, prev_step: int):
    """Clear ramdisk data from previous step on all nodes."""
    logger.info(f"Clearing ramdisk data from Step {prev_step}")
    
    for node in self.get_nodes():
        if node == 'localhost':
            subprocess.run(['rm', '-rf', '/dev/shm/prng/*'])
        else:
            subprocess.run(['ssh', node, 'rm -rf /dev/shm/prng/*'])
```

#### Option F: Manual Only ⚠️ CURRENT STATE

```
Operator: ./clear_ramdisk.sh (when they remember)
```

- **Pros:** Full control, no automation surprises
- **Cons:** Easy to forget, stale data accumulates
- **Verdict:** Acceptable for testing, not for production autonomy

### 5.4 Recommendation

**Primary: Option E (WATCHER-managed)** for autonomous pipeline runs

**Fallback: Option D (Before next step)** for standalone step execution

**Rationale:**
1. WATCHER already owns step orchestration
2. Cleanup is a natural part of step transitions
3. Single point of control = easier debugging
4. Preserves manual `clear_ramdisk.sh` for operator override

### 5.5 Hybrid Implementation

```bash
# ramdisk_preload.sh - Used by all step launchers

RAMDISK_DIR="/dev/shm/prng"

preload_ramdisk() {
    local step_id="${RAMDISK_STEP_ID:-unknown}"
    local files=("$@")
    local sentinel="${RAMDISK_DIR}/.ready_step${step_id}"
    
    # Check if WATCHER is managing cleanup (env var set by WATCHER)
    if [ -z "$WATCHER_MANAGED_RAMDISK" ]; then
        # Standalone execution: clear previous data ourselves
        echo "[INFO] Standalone mode: clearing ramdisk before Step $step_id"
        rm -rf ${RAMDISK_DIR}/*
    fi
    
    # Preload files if not already loaded for this step
    if [ ! -f "$sentinel" ]; then
        echo "[INFO] Preloading ${#files[@]} files for Step $step_id..."
        mkdir -p $RAMDISK_DIR
        
        for f in "${files[@]}"; do
            [ -f "$f" ] && cp "$f" $RAMDISK_DIR/
        done
        
        touch "$sentinel"
        echo "[INFO] Ramdisk preload complete"
    else
        echo "[INFO] Ramdisk already loaded for Step $step_id (skipped)"
    fi
}
```

```python
# In watcher_agent.py

def run_pipeline(self, start_step, end_step):
    for step in range(start_step, end_step + 1):
        # Set env var to tell launcher that WATCHER owns cleanup
        os.environ['WATCHER_MANAGED_RAMDISK'] = '1'
        
        # Clear previous step's ramdisk (WATCHER-managed)
        if step > start_step:
            self.clear_ramdisk()
        
        # Run the step (launcher will preload but not clear)
        self.run_step(step)
    
    # Final cleanup after pipeline completes
    self.clear_ramdisk()
```

---

## 6. Shared Infrastructure

### 6.1 Shell Module (`ramdisk_preload.sh`)

```bash
#!/bin/bash
# ramdisk_preload.sh - Source from any step launcher
# Usage: 
#   export RAMDISK_STEP_ID=2
#   source ramdisk_preload.sh
#   preload_ramdisk file1.json file2.npz file3.json

RAMDISK_DIR="/dev/shm/prng"

get_cluster_nodes() {
    python3 -c "
import json
with open('distributed_config.json') as f:
    cfg = json.load(f)
for node in cfg['nodes']:
    print(node['hostname'])
"
}

preload_ramdisk() {
    local step_id="${RAMDISK_STEP_ID:-unknown}"
    local files=("$@")
    local sentinel="${RAMDISK_DIR}/.ready_step${step_id}"
    
    echo "[INFO] Ramdisk preload for Step $step_id (${#files[@]} files)..."
    
    # Clear if standalone (WATCHER sets this env var when it manages cleanup)
    if [ -z "$WATCHER_MANAGED_RAMDISK" ]; then
        echo "[INFO] Standalone mode: clearing previous ramdisk data"
        rm -rf ${RAMDISK_DIR}/* 2>/dev/null
        for NODE in $(get_cluster_nodes); do
            if [ "$NODE" != "localhost" ]; then
                ssh "$NODE" "rm -rf ${RAMDISK_DIR}/*" 2>/dev/null
            fi
        done
    fi
    
    for NODE in $(get_cluster_nodes); do
        echo "  → $NODE"
        
        if [ "$NODE" = "localhost" ]; then
            mkdir -p $RAMDISK_DIR
            if [ ! -f "$sentinel" ]; then
                for f in "${files[@]}"; do
                    [ -f "$f" ] && cp "$f" $RAMDISK_DIR/
                done
                touch "$sentinel"
                echo "    ✓ Preloaded"
            else
                echo "    ✓ Already loaded (skipped)"
            fi
        else
            ssh "$NODE" "mkdir -p $RAMDISK_DIR" 2>/dev/null
            if ! ssh "$NODE" "[ -f $sentinel ]" 2>/dev/null; then
                for f in "${files[@]}"; do
                    [ -f "$f" ] && scp -q "$f" "$NODE:$RAMDISK_DIR/"
                done
                ssh "$NODE" "touch $sentinel"
                echo "    ✓ Preloaded"
            else
                echo "    ✓ Already loaded (skipped)"
            fi
        fi
    done
    
    echo "[INFO] Ramdisk preload complete"
}

clear_ramdisk() {
    echo "[INFO] Clearing ramdisk on all nodes..."
    
    rm -rf ${RAMDISK_DIR}/* 2>/dev/null
    echo "  → localhost: cleared"
    
    for NODE in $(get_cluster_nodes); do
        if [ "$NODE" != "localhost" ]; then
            ssh "$NODE" "rm -rf ${RAMDISK_DIR}/*" 2>/dev/null
            echo "  → $NODE: cleared"
        fi
    done
}
```

### 6.2 Python Module (`ramdisk_config.py`)

```python
#!/usr/bin/env python3
"""
ramdisk_config.py - Shared ramdisk configuration for all pipeline steps
Import this in any job generator to get consistent paths.
"""

import os
import json
import subprocess
from pathlib import Path

# Global configuration
USE_RAMDISK = True
RAMDISK_DIR = "/dev/shm/prng"
SSD_DIR = "/home/michael/distributed_prng_analysis"

def get_data_root():
    """Return data root path based on ramdisk setting."""
    return RAMDISK_DIR if USE_RAMDISK else SSD_DIR

def get_data_path(filename):
    """Return full path to data file."""
    return f"{get_data_root()}/{filename}"

def get_cluster_nodes():
    """Get list of cluster nodes from config."""
    config_path = Path(__file__).parent / "distributed_config.json"
    with open(config_path) as f:
        cfg = json.load(f)
    return [node['hostname'] for node in cfg['nodes']]

def clear_ramdisk():
    """Clear ramdisk on all nodes."""
    for node in get_cluster_nodes():
        if node == 'localhost':
            subprocess.run(['rm', '-rf', f'{RAMDISK_DIR}/*'], shell=True)
        else:
            subprocess.run(['ssh', node, f'rm -rf {RAMDISK_DIR}/*'])

def preload_ramdisk(step_id: int, files: list):
    """
    Preload files to ramdisk on all nodes.
    
    Args:
        step_id: Pipeline step number (for sentinel naming)
        files: List of local file paths to copy
    """
    sentinel = f"{RAMDISK_DIR}/.ready_step{step_id}"
    
    # Check if WATCHER is managing cleanup
    watcher_managed = os.environ.get('WATCHER_MANAGED_RAMDISK')
    
    for node in get_cluster_nodes():
        if node == 'localhost':
            os.makedirs(RAMDISK_DIR, exist_ok=True)
            if not os.path.exists(sentinel):
                if not watcher_managed:
                    # Standalone: clear first
                    subprocess.run(f'rm -rf {RAMDISK_DIR}/*', shell=True)
                for f in files:
                    if os.path.exists(f):
                        subprocess.run(['cp', f, RAMDISK_DIR])
                Path(sentinel).touch()
        else:
            subprocess.run(['ssh', node, f'mkdir -p {RAMDISK_DIR}'])
            result = subprocess.run(['ssh', node, f'[ -f {sentinel} ]'])
            if result.returncode != 0:
                if not watcher_managed:
                    subprocess.run(['ssh', node, f'rm -rf {RAMDISK_DIR}/*'])
                for f in files:
                    if os.path.exists(f):
                        subprocess.run(['scp', '-q', f, f'{node}:{RAMDISK_DIR}/'])
                subprocess.run(['ssh', node, f'touch {sentinel}'])
```

---

## 7. Per-Step Implementation

### 7.1 Step 2 (Already Done)

```bash
# run_scorer_meta_optimizer.sh (current implementation)
export RAMDISK_STEP_ID=2
source ramdisk_preload.sh

preload_ramdisk \
    bidirectional_survivors_binary.npz \
    train_history.json \
    holdout_history.json
```

### 7.2 Step 3

```bash
# run_step3_full_scoring.sh (to be modified)
export RAMDISK_STEP_ID=3
source ramdisk_preload.sh

preload_ramdisk \
    bidirectional_survivors.json \
    train_history.json \
    forward_survivors.json \
    reverse_survivors.json
```

### 7.3 Step 5

Step 5 uses Python orchestrator, not shell launcher:

```python
# In meta_prediction_optimizer_anti_overfit.py (add near top)
from ramdisk_config import preload_ramdisk, get_data_path

# Before job dispatch
preload_ramdisk(step_id=5, files=[
    'survivors_with_scores.json',
    'holdout_history.json'
])

# In job args, use:
survivor_path = get_data_path('survivors_with_scores.json')
holdout_path = get_data_path('holdout_history.json')
```

---

## 8. NPZ Conversion Consideration

### 8.1 Current State

| Step | Input File | Format | Size |
|------|------------|--------|------|
| 2 | `bidirectional_survivors_binary.npz` | NPZ | 229 KB |
| 3 | `bidirectional_survivors.json` | JSON | ~58 MB |
| 5 | `survivors_with_scores.json` | JSON | ~400 MB |

### 8.2 Potential NPZ Savings

Step 2's NPZ conversion achieved **430x size reduction** (258 MB → 0.6 MB).

If applied to Steps 3 and 5:

| Step | Current | With NPZ | Reduction |
|------|---------|----------|-----------|
| 3 | 520 MB | ~1-2 MB | ~300x |
| 5 | 400 MB | ~1 MB | ~400x |

### 8.3 Recommendation

**Defer NPZ for Steps 3/5.** Ramdisk alone is sufficient:
- 1 GB total is only 25% of 4GB `/dev/shm`
- NPZ conversion adds complexity (schema changes, worker updates)
- Can revisit if memory becomes constraint

---

## 9. Implementation Phases

| Phase | Scope | Effort | Deliverable |
|-------|-------|--------|-------------|
| 1 | Shared infrastructure | 30 min | `ramdisk_preload.sh`, `ramdisk_config.py` |
| 2 | Step 3 extension | 30 min | Modified `run_step3_full_scoring.sh` |
| 3 | Step 5 extension | 45 min | Modified `meta_prediction_optimizer_anti_overfit.py` |
| 4 | WATCHER integration | 30 min | Cleanup hooks in `watcher_agent.py` |
| **Total** | | **~2.5 hours** | |

---

## 10. Rollback Plan

Each step has independent ramdisk logic:

```bash
# Disable ramdisk globally
# In ramdisk_config.py:
USE_RAMDISK = False

# Disable per-step (in shell launchers):
# Comment out: source ramdisk_preload.sh

# Restore from backups:
ls -la *.backup_*
cp run_step3_full_scoring.sh.backup_TIMESTAMP run_step3_full_scoring.sh

# Emergency clear:
./clear_ramdisk.sh
```

---

## 11. Questions for Team Beta

1. **Cleanup Strategy:** Do you concur with **Option E (WATCHER-managed)** as primary, **Option D (before next step)** as fallback?

2. **Step 5 Architecture:** Should we create a shell wrapper (`run_step5_anti_overfit.sh`) for consistency with Steps 2/3, or keep Python-native preload?

3. **NPZ Conversion:** Defer to later, or implement now for Steps 3/5?

4. **Sentinel Design:** Keep boolean `.ready_stepN` or upgrade to content-hash sentinels?

5. **Memory Threshold:** Should we add a pre-flight check that warns if `/dev/shm` usage > 50% before preload?

---

## 12. Recommendation

**APPROVE for phased implementation** with:
- **Option E (WATCHER-managed cleanup)** as primary strategy
- **Option D (before next step)** as standalone fallback
- Defer NPZ conversion for Steps 3/5
- Add pre-flight `/dev/shm` usage check

---

**Submitted for Team Beta review.**

*Claude (AI Assistant)*  
*2026-01-21*
