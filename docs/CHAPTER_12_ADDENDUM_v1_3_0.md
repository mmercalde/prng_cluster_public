# CHAPTER 12 ADDENDUM: Preflight & Cleanup Integration

**Version:** 1.3.0  
**Date:** January 25, 2026  
**New Files:** `preflight_check.py`, `gpu_cleanup.py`

---

## INSERT AFTER SECTION 3 (Watcher Agent)

---

## 3.5 Preflight & Cleanup Integration (v1.3.0)

### 3.5.1 Overview

As of v1.3.0, the WATCHER agent includes two integrated safety mechanisms:

| Component | Purpose | Blocking? |
|-----------|---------|-----------|
| **Preflight Check** | Validates cluster before step execution | Yes (on critical failures) |
| **GPU Cleanup** | Clears GPU memory after distributed steps | No (warnings only) |

### 3.5.2 Preflight Check Integration

**File:** `preflight_check.py`

The preflight check runs automatically at the start of each step's `run_step()` method:

```python
# In run_step() - automatically called
preflight_passed, preflight_msg = self._run_preflight_check(step)
if not preflight_passed:
    return {
        "success": False,
        "error": preflight_msg,
        "blocked_by": "preflight_check"
    }
```

**Checks Performed:**

| Check | Method | Hard Block? |
|-------|--------|-------------|
| SSH connectivity | Ping each node | ✅ Yes |
| Ramdisk files exist | Check `/dev/shm/prng/stepN/` | ✅ Yes |
| Input files exist | Check required inputs | ✅ Yes |
| GPU count matches config | Compare `rocm-smi` vs `distributed_config.json` | ⚠️ Warning only |

**Failure Categories:**

```python
# Hard failures (block execution)
hard_fail_keywords = ["ssh", "unreachable", "ramdisk", "input", "not found"]

# Soft failures (warnings only)
# - GPU count mismatch
# - Non-critical validation errors
```

### 3.5.3 GPU Cleanup Integration

**File:** `gpu_cleanup.py`

GPU cleanup runs automatically after distributed steps (Steps 1, 2, 3):

```python
# In run_step() - automatically called after step completes
DISTRIBUTED_STEPS = {1, 2, 3}

def _run_post_step_cleanup(self, step: int) -> None:
    if step not in DISTRIBUTED_STEPS:
        return
    # ... cleanup logic (never blocks)
```

**Behavior:**
- Clears PyTorch/HIP allocator caches on ROCm nodes
- Best-effort only - failures never block pipeline
- Logs warnings if cleanup fails

### 3.5.4 Module Availability

Both integrations are optional - if modules aren't available, WATCHER proceeds normally:

```python
# Preflight
try:
    from preflight_check import PreflightChecker
    PREFLIGHT_AVAILABLE = True
except ImportError:
    PREFLIGHT_AVAILABLE = False

# GPU Cleanup
try:
    from gpu_cleanup import post_batch_cleanup, cleanup_all_nodes
    GPU_CLEANUP_AVAILABLE = True
except ImportError:
    GPU_CLEANUP_AVAILABLE = False
```

### 3.5.5 CLI Behavior

The preflight and cleanup integration is automatic - no CLI flags required:

```bash
# Preflight runs automatically
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 3 --end-step 3

# Output includes preflight status
[INFO] Running Step 3: run_step3_full_scoring.sh
[DEBUG] Preflight check: SSH OK, Ramdisk OK, Inputs OK
[INFO] EXEC CMD: bash run_step3_full_scoring.sh ...
...
[INFO] [CLEANUP] Running post-step cleanup for Step 3
[INFO] [CLEANUP] Complete: 2 nodes cleaned
```

### 3.5.6 Error Handling Examples

**Example 1: SSH Failure (Hard Block)**
```
[ERROR] Preflight BLOCKED: SSH unreachable: 192.168.3.120
Step 3 will NOT execute.
```

**Example 2: Ramdisk Missing (Hard Block)**
```
[ERROR] Preflight BLOCKED: Ramdisk not found: /dev/shm/prng/step3/train_history.json
Step 3 will NOT execute.
```

**Example 3: GPU Count Mismatch (Warning Only)**
```
[WARNING] Preflight warnings: ['GPU_COUNT_MISMATCH: rig-6600 expected 12, found 10']
Step 3 will execute (degraded capacity).
```

**Example 4: Cleanup Failure (Warning Only)**
```
[WARNING] [CLEANUP] Warning (non-blocking): Connection refused
Pipeline continues normally.
```

---

## Version History Update

| Version | Date | Changes |
|---------|------|---------|
| 1.3.0 | 2026-01-25 | Added preflight check + GPU cleanup integration |
| 1.2.0 | 2026-01-23 | Step 3 script routing fix |
| 1.1.0 | 2026-01-04 | Grammar-constrained LLM integration |
| 1.0.0 | 2025-12-15 | Initial WATCHER implementation |
