#!/usr/bin/env python3
"""
Chapter Documentation Patcher
Merges addendum content into main chapter files.

Usage:
    python3 patch_chapters.py
    python3 patch_chapters.py --dry-run
"""

import os
import sys
import shutil
from datetime import datetime

# ============================================================================
# CHAPTER 12 PATCH CONTENT
# ============================================================================

CHAPTER_12_SECTION_3_5 = '''
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

### 3.5.5 Error Handling Examples

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

'''

CHAPTER_12_CHANGELOG_ENTRY = '''### v1.3.0 (January 25, 2026)
- Added Section 3.5: Preflight & Cleanup Integration
- Integrated `preflight_check.py` - validates cluster before step execution
- Integrated `gpu_cleanup.py` - clears GPU memory after distributed steps
- Hard blocks on SSH/ramdisk/input failures; warnings only for GPU count mismatches

'''

# ============================================================================
# CHAPTER 9 PATCH CONTENT
# ============================================================================

CHAPTER_9_SECTION_8_6_8_7 = '''
### 8.6 GPU Diagnostic Battery

**File:** `debug_gpu_failures.sh` (v1.0.0, January 25, 2026)

Comprehensive cluster diagnostics for troubleshooting GPU failures.

**Usage:**
```bash
cd ~/distributed_prng_analysis
bash debug_gpu_failures.sh
# Output: debug_gpu_failures_YYYYMMDD_HHMMSS.log
```

**Checks Performed:**

| Check | Command | Looking For |
|-------|---------|-------------|
| Memory status | `free -h` | Available RAM < 4GB = OOM risk |
| GPU health | `rocm-smi` | N/A, "unknown", or missing GPUs |
| OOM killer | `dmesg \\| grep oom` | "Killed" or "Out of memory" messages |
| ROCm/HIP errors | `dmesg \\| grep amdgpu` | Driver or HIP initialization failures |
| Ramdisk status | `ls /dev/shm/prng/step3/` | Missing files |
| HIP cache size | `du -sh ~/.cache/hip_*` | Large cache = memory pressure |

**Interpreting Results:**

| Pattern in Output | Meaning | Action |
|-------------------|---------|--------|
| `Killed` in dmesg | OOM killer terminated process | Reduce `chunk_size` or concurrent workers |
| GPU shows `N/A` | GPU not responding | Reboot node |
| GPU shows `unknown` | SMU communication failure | Reboot node or reseat GPU |
| Ramdisk missing | Files not preloaded | Run `ramdisk_preload.sh` |
| HIP cache > 1GB | Memory pressure | Clear cache with `rm -rf ~/.cache/hip_*` |

### 8.7 Ramdisk Preload v2.1.0

**File:** `ramdisk_preload.sh` (v2.1.0, January 25, 2026)

**Bug Fixed:** v2.0.0 created `.ready` sentinel marker even when file copy failed, causing Step 3 to believe ramdisk was populated when files were missing.

**Changes from v2.0.0:**

| Issue | v2.0.0 | v2.1.0 |
|-------|--------|--------|
| Incomplete copy | `.ready` created anyway | Only if ALL files copied |
| Missing source files | Silent failure | Fails loudly before copying |
| Stale sentinel | Left `.ready` on failure | Removed on incomplete copy |

**Key Fix:**
```bash
# v2.1.0 FIX: Only create .ready if ALL files copied
if [ $copied -eq $expected_count ]; then
    ssh "$NODE" "touch $RAMDISK_SENTINEL"
    echo "    ✓ Preloaded ($copied files)"
else
    echo "    ❌ INCOMPLETE: Only $copied/$expected_count files copied"
    ssh "$NODE" "rm -f $RAMDISK_SENTINEL"  # Remove stale sentinel
fi
```

**Standalone Preload Script:**

For manual ramdisk population after node reboot:

```bash
# Usage
bash ramdisk_preload_fixed.sh 3  # Preload for Step 3

# Output
[11:16:32] Final verification:
[11:16:32]   ✅ localhost: All files present
[11:16:33]   ✅ 192.168.3.120: All files present
[11:16:33]   ✅ 192.168.3.154: All files present
```

'''

CHAPTER_9_VERSION_ENTRY = "| 2.2.0 | 2026-01-25 | GPU diagnostic battery (8.6), ramdisk v2.1.0 fix (8.7) |\n"


def patch_chapter_12(dry_run=False):
    """Patch Chapter 12 with Section 3.5 and changelog entry."""
    path = "docs/CHAPTER_12_WATCHER_AGENT.md"
    
    if not os.path.exists(path):
        print(f"  ❌ {path} not found")
        return False
    
    with open(path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "## 3.5 Preflight" in content:
        print("  ⚠️  Chapter 12 already has Section 3.5")
        return True
    
    original = content
    
    # Insert Section 3.5 before Section 4
    marker1 = "---\n\n## 4. Fingerprint Registry"
    if marker1 not in content:
        print(f"  ❌ Cannot find marker: {marker1[:40]}...")
        return False
    
    content = content.replace(marker1, CHAPTER_12_SECTION_3_5 + marker1)
    print("  ✓ Inserted Section 3.5")
    
    # Insert changelog entry
    marker2 = "## 12. Changelog\n\n### v1.2.0"
    if marker2 not in content:
        print(f"  ❌ Cannot find changelog marker")
        return False
    
    content = content.replace(marker2, "## 12. Changelog\n\n" + CHAPTER_12_CHANGELOG_ENTRY + "### v1.2.0")
    print("  ✓ Added changelog entry v1.3.0")
    
    # Update version in header
    content = content.replace("**Version:** 1.2.0", "**Version:** 1.3.0")
    content = content.replace("**Date:** January 23, 2026", "**Date:** January 25, 2026")
    print("  ✓ Updated version to 1.3.0")
    
    if dry_run:
        print(f"  [DRY RUN] Would write {len(content)} bytes")
        return True
    
    # Backup and write
    backup = f"{path}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(path, backup)
    
    with open(path, 'w') as f:
        f.write(content)
    
    print(f"  ✓ Written: {path}")
    print(f"  ✓ Backup: {backup}")
    return True


def patch_chapter_9(dry_run=False):
    """Patch Chapter 9 with Sections 8.6, 8.7 and version entry."""
    path = "docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md"
    
    if not os.path.exists(path):
        print(f"  ❌ {path} not found")
        return False
    
    with open(path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "### 8.6 GPU Diagnostic" in content:
        print("  ⚠️  Chapter 9 already has Section 8.6")
        return True
    
    original = content
    
    # Insert Sections 8.6 and 8.7 before Section 9
    marker1 = "---\n\n## 9. Job Types and Routing"
    if marker1 not in content:
        print(f"  ❌ Cannot find marker: {marker1[:40]}...")
        return False
    
    content = content.replace(marker1, CHAPTER_9_SECTION_8_6_8_7 + "\n" + marker1)
    print("  ✓ Inserted Sections 8.6 and 8.7")
    
    # Insert version entry
    marker2 = "| 2.1.0 | 2026-01-24 |"
    if marker2 not in content:
        print(f"  ❌ Cannot find version history marker")
        return False
    
    content = content.replace(marker2, CHAPTER_9_VERSION_ENTRY + marker2)
    print("  ✓ Added version entry v2.2.0")
    
    if dry_run:
        print(f"  [DRY RUN] Would write {len(content)} bytes")
        return True
    
    # Backup and write
    backup = f"{path}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(path, backup)
    
    with open(path, 'w') as f:
        f.write(content)
    
    print(f"  ✓ Written: {path}")
    print(f"  ✓ Backup: {backup}")
    return True


def main():
    dry_run = "--dry-run" in sys.argv
    
    print("=" * 60)
    print("Chapter Documentation Patcher")
    print("=" * 60)
    if dry_run:
        print("[DRY RUN MODE]")
    print()
    
    print("Patching Chapter 12 (WATCHER Agent)...")
    ch12_ok = patch_chapter_12(dry_run)
    
    print()
    print("Patching Chapter 9 (GPU Cluster Infrastructure)...")
    ch9_ok = patch_chapter_9(dry_run)
    
    print()
    print("=" * 60)
    if ch12_ok and ch9_ok:
        print("✅ All patches applied successfully")
    else:
        print("❌ Some patches failed")
    print("=" * 60)
    
    return 0 if (ch12_ok and ch9_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
