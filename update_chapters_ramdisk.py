#!/usr/bin/env python3
"""
Update Chapters 4, 9, and 12 with Ramdisk Documentation
========================================================
Session: January 22, 2026

Adds documentation for /dev/shm/prng/step3/ ramdisk requirement.

Usage:
  # On SER8 (local docs)
  python3 update_chapters_ramdisk.py --path ~/Downloads/CONCISE_OPERATING_GUIDE_v1.0
  
  # On Zeus
  python3 update_chapters_ramdisk.py --path ~/distributed_prng_analysis
  
  # Dry run (preview only)
  python3 update_chapters_ramdisk.py --path ~/Downloads/CONCISE_OPERATING_GUIDE_v1.0 --dry-run
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path
import re

# =============================================================================
# PATCH CONTENT
# =============================================================================

CHAPTER_4_ADDITION = '''
---

## 2.5 Ramdisk Prerequisites (CRITICAL)

**Added: 2026-01-22**

Step 3 jobs expect training data at `/dev/shm/prng/step3/` on **ALL nodes**.

### Required Files

| File | Purpose |
|------|---------|
| `train_history.json` | Training draws for scoring |
| `holdout_history.json` | Holdout draws for validation |

### Why Ramdisk?

Jobs reference paths like `/dev/shm/prng/step3/train_history.json` to avoid:
- Disk I/O contention with 12 concurrent GPU jobs
- NFS/network latency on distributed nodes
- JSON parsing overhead on repeated loads

### Verification (Before Running Step 3)

```bash
# Check all nodes
ssh 192.168.3.120 "ls -la /dev/shm/prng/step3/"
ssh 192.168.3.154 "ls -la /dev/shm/prng/step3/"
ls -la /dev/shm/prng/step3/  # Zeus
```

**Expected:** Both files present on all three nodes.

### Manual Population (If Missing)

```bash
# On Zeus
mkdir -p /dev/shm/prng/step3
cp train_history.json holdout_history.json /dev/shm/prng/step3/

# On rig-6600
ssh 192.168.3.120 "mkdir -p /dev/shm/prng/step3"
scp train_history.json holdout_history.json 192.168.3.120:/dev/shm/prng/step3/

# On rig-6600b
ssh 192.168.3.154 "mkdir -p /dev/shm/prng/step3"
scp train_history.json holdout_history.json 192.168.3.154:/dev/shm/prng/step3/
```

### Automatic Preload

The WATCHER agent runs ramdisk preload but in **"Standalone mode"** — it only populates the local node. For distributed execution, you must manually populate remote nodes or run the preload script on each node.

### Common Failure Pattern

If Step 3 jobs fail instantly (~3 seconds) with no output file:
1. Check ramdisk exists on failing node
2. Worker expects `--train-history /dev/shm/prng/step3/train_history.json`
3. Missing file = immediate argument parsing failure

'''

CHAPTER_9_ADDITION = '''
---

## 8.5 Ramdisk Deployment for Step 3

**Added: 2026-01-22**

Step 3 (Full Scoring) requires ramdisk files on all nodes before distributed execution.

### Path Structure

```
/dev/shm/prng/step3/
├── train_history.json      # ~28KB
└── holdout_history.json    # ~7KB
```

### Deployment Commands

```bash
# From Zeus - deploy to all nodes
for node in localhost 192.168.3.120 192.168.3.154; do
    if [ "$node" = "localhost" ]; then
        mkdir -p /dev/shm/prng/step3
        cp train_history.json holdout_history.json /dev/shm/prng/step3/
    else
        ssh $node "mkdir -p /dev/shm/prng/step3"
        scp train_history.json holdout_history.json $node:/dev/shm/prng/step3/
    fi
done
```

### Persistence Warning

**Ramdisk (`/dev/shm`) does NOT survive reboot.**

After any node reboot, you must repopulate the ramdisk before running Step 3.

### Verification Script

```bash
#!/bin/bash
# verify_step3_ramdisk.sh
echo "=== Ramdisk Status ==="
for node in localhost 192.168.3.120 192.168.3.154; do
    echo -n "$node: "
    if [ "$node" = "localhost" ]; then
        ls /dev/shm/prng/step3/*.json 2>/dev/null | wc -l | xargs -I{} echo "{} files"
    else
        ssh $node "ls /dev/shm/prng/step3/*.json 2>/dev/null | wc -l" | xargs -I{} echo "{} files"
    fi
done
```

**Expected output:**
```
localhost: 2 files
192.168.3.120: 2 files
192.168.3.154: 2 files
```

'''

CHAPTER_12_ADDITION = '''
---

## 10.5 Ramdisk Preload Limitation

**Added: 2026-01-22**

### Standalone Mode Behavior

When WATCHER runs Step 3, it performs ramdisk preload:

```
[INFO] Ramdisk preload for Step 3 (4 files)...
[INFO] Target: /dev/shm/prng/step3
[INFO] Standalone mode
[INFO] Ramdisk preload complete
```

**"Standalone mode" means:** Only the local node (Zeus) is populated.

Remote nodes (rig-6600, rig-6600b) are **NOT** automatically populated.

### Impact on Distributed Execution

| Node | Ramdisk Populated | Jobs Will... |
|------|-------------------|--------------|
| Zeus (local) | ✅ Yes | Work |
| rig-6600 | ❌ No | Fail immediately |
| rig-6600b | ❌ No | Fail immediately |

### Workaround

Before running Step 3 via WATCHER, manually populate remote ramdisks:

```bash
# Run on Zeus before WATCHER pipeline
for node in 192.168.3.120 192.168.3.154; do
    ssh $node "mkdir -p /dev/shm/prng/step3"
    scp train_history.json holdout_history.json $node:/dev/shm/prng/step3/
done
```

### Future Enhancement

TODO: Modify `run_step3_full_scoring.sh` ramdisk preload to:
1. Detect distributed mode
2. SCP files to all configured nodes in `distributed_config.json`
3. Verify files exist before proceeding

'''

# =============================================================================
# INSERTION POINTS
# =============================================================================

PATCHES = {
    'CHAPTER_4_FULL_SCORING.md': {
        'marker': '## 2.5 Ramdisk Prerequisites',
        'insert_after': '## 2. Environment Setup',
        'content': CHAPTER_4_ADDITION,
        'fallback_after': '---\n\n## 2.'  # Backup pattern
    },
    'CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md': {
        'marker': '## 8.5 Ramdisk Deployment',
        'insert_after': '## 8.4 ROCm Stability Envelope',
        'content': CHAPTER_9_ADDITION,
        'fallback_after': '### 8.4'
    },
    'CHAPTER_12_WATCHER_AGENT.md': {
        'marker': '## 10.5 Ramdisk Preload Limitation',
        'insert_after': '## 10. Troubleshooting',
        'content': CHAPTER_12_ADDITION,
        'fallback_after': '## 10.'
    }
}

# =============================================================================
# FUNCTIONS
# =============================================================================

def create_backup(filepath: Path) -> Path:
    """Create timestamped backup."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = filepath.parent / f"{filepath.name}.bak_{timestamp}"
    shutil.copy2(filepath, backup)
    return backup


def is_already_patched(content: str, marker: str) -> bool:
    """Check if patch already applied."""
    return marker in content


def find_insertion_point(content: str, insert_after: str, fallback_after: str) -> int:
    """Find where to insert new content."""
    # Try primary pattern
    idx = content.find(insert_after)
    if idx != -1:
        # Find end of that section (next ## or ---)
        section_end = content.find('\n## ', idx + len(insert_after))
        if section_end == -1:
            section_end = content.find('\n---', idx + len(insert_after))
        if section_end == -1:
            section_end = len(content)
        return section_end
    
    # Try fallback
    idx = content.find(fallback_after)
    if idx != -1:
        section_end = content.find('\n## ', idx + len(fallback_after))
        if section_end == -1:
            section_end = len(content)
        return section_end
    
    return -1


def patch_file(filepath: Path, patch_config: dict, dry_run: bool = False) -> tuple:
    """
    Apply patch to a chapter file.
    
    Returns: (success, message)
    """
    if not filepath.exists():
        return (False, f"File not found: {filepath}")
    
    content = filepath.read_text()
    
    # Check if already patched
    if is_already_patched(content, patch_config['marker']):
        return (True, "Already patched (skipped)")
    
    # Find insertion point
    insert_idx = find_insertion_point(
        content, 
        patch_config['insert_after'],
        patch_config['fallback_after']
    )
    
    if insert_idx == -1:
        return (False, f"Could not find insertion point after '{patch_config['insert_after']}'")
    
    # Build new content
    new_content = content[:insert_idx] + patch_config['content'] + content[insert_idx:]
    
    if dry_run:
        return (True, f"Would insert {len(patch_config['content'])} chars at position {insert_idx}")
    
    # Create backup and write
    backup = create_backup(filepath)
    filepath.write_text(new_content)
    
    return (True, f"Patched (backup: {backup.name})")


def main():
    parser = argparse.ArgumentParser(
        description='Update chapters 4, 9, 12 with ramdisk documentation'
    )
    parser.add_argument('--path', type=str, required=True,
                        help='Path to chapter files directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview changes without modifying files')
    
    args = parser.parse_args()
    base_path = Path(args.path).expanduser()
    
    print("=" * 70)
    print("Update Chapters with Ramdisk Documentation")
    print("=" * 70)
    print(f"Path: {base_path}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'APPLY'}")
    print()
    
    if not base_path.exists():
        print(f"❌ ERROR: Path not found: {base_path}")
        return 1
    
    results = []
    for filename, config in PATCHES.items():
        filepath = base_path / filename
        print(f"\n{filename}:")
        
        success, message = patch_file(filepath, config, args.dry_run)
        status = "✅" if success else "❌"
        print(f"  {status} {message}")
        results.append((filename, success, message))
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    success_count = sum(1 for _, s, _ in results if s)
    print(f"Patched: {success_count}/{len(results)}")
    
    if not args.dry_run and success_count > 0:
        print("\nNext steps:")
        print("  1. Review changes in each file")
        print("  2. Copy updated files to Zeus if editing on SER8")
        print("  3. git add / commit / push")
    
    return 0 if success_count == len(results) else 1


if __name__ == '__main__':
    exit(main())
