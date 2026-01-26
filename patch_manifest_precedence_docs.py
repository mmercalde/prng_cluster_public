#!/usr/bin/env python3
"""
Patch Script: Add Manifest Parameter Precedence Documentation
=============================================================
Version: 1.0.0
Date: 2026-01-25

Adds documentation about agent_manifests/*.json overriding script defaults
to three chapters:
- CHAPTER_12_WATCHER_AGENT.md
- CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md  
- CHAPTER_4_FULL_SCORING.md

Usage:
    python3 patch_manifest_precedence_docs.py

After running on Zeus, copy updated chapters to ser8 project:
    scp CHAPTER_*.md ser8:~/path/to/project/
"""

import re
from pathlib import Path
from datetime import datetime

# Content to add to each chapter
CHAPTER_12_ADDITION = '''
---

## 3.6 Manifest Parameter Precedence (CRITICAL)

**Added: 2026-01-25** — Lesson learned from OOM debugging.

### The Rule

`agent_manifests/*.json` `default_params` **override** script defaults!

| Priority | Source | Example |
|----------|--------|---------|
| 1 (highest) | CLI `--params '{...}'` | `--params '{"chunk_size": 500}'` |
| 2 | `agent_manifests/*.json` `default_params` | `"chunk_size": 1000` |
| 3 (lowest) | Script hardcoded default | `CHUNK_SIZE=5000` in .sh |

### Why This Matters

When WATCHER runs a step, it:
1. Loads `agent_manifests/{step}.json`
2. Reads `default_params` section
3. Passes ALL params to the script
4. Script's internal defaults are **never used**

### Real Example (2026-01-25 OOM Bug)

**Symptom:** Step 3 ran with `chunk_size=5000` despite script having `CHUNK_SIZE=1000`

**Root cause:**
```json
// agent_manifests/full_scoring.json
"default_params": {
    "chunk_size": 5000,  // ← THIS overrode script default
    ...
}
```

**Fix:** Changed manifest to `"chunk_size": 1000`

### Debugging Checklist

When a WATCHER-run step uses unexpected parameters:

1. ✅ Check `agent_manifests/{step}.json` `default_params` FIRST
2. ✅ Check script hardcoded defaults SECOND
3. ✅ Check CLI `--params` if passed

### Best Practice

Keep manifest `default_params` and script defaults **in sync**:

```bash
# Verify consistency
grep "chunk_size" agent_manifests/full_scoring.json
grep "CHUNK_SIZE" run_step3_full_scoring.sh
```

'''

CHAPTER_9_ADDITION = '''
---

### 5.8 Configuration Location for chunk_size (CRITICAL)

**Added: 2026-01-25** — Prevents OOM confusion.

The `chunk_size=1000` memory-safe setting must be in **TWO places**:

| File | Setting | Used By |
|------|---------|---------|
| `run_step3_full_scoring.sh` | `CHUNK_SIZE=1000` | Manual runs |
| `agent_manifests/full_scoring.json` | `"chunk_size": 1000` | WATCHER automated runs |

#### Why Two Places?

- **Manual run:** `bash run_step3_full_scoring.sh` uses script's `CHUNK_SIZE=` variable
- **WATCHER run:** Agent reads `default_params` from manifest, passes to script

**Warning:** Manifest overrides script default! If WATCHER uses wrong chunk_size, check `agent_manifests/full_scoring.json` FIRST.

#### Verification Commands

```bash
# Check both locations are in sync
echo "=== Script default ==="
grep "^CHUNK_SIZE=" run_step3_full_scoring.sh

echo "=== Manifest default ==="
grep '"chunk_size"' agent_manifests/full_scoring.json | head -1
```

**Expected output:**
```
=== Script default ===
CHUNK_SIZE=1000
=== Manifest default ===
    "chunk_size": 1000,
```

#### If They Don't Match

```bash
# Fix manifest (authoritative for WATCHER)
sed -i 's/"chunk_size": [0-9]*/"chunk_size": 1000/' agent_manifests/full_scoring.json

# Fix script (authoritative for manual runs)
sed -i 's/^CHUNK_SIZE=.*/CHUNK_SIZE=1000/' run_step3_full_scoring.sh
```

'''

CHAPTER_4_ADDITION = '''
---

## 1.5 Configuration Sources (CRITICAL)

**Added: 2026-01-25** — Understand where parameters come from.

### Parameter Source by Run Method

| Run Method | Config Source | chunk_size Location |
|------------|---------------|---------------------|
| Manual: `bash run_step3_full_scoring.sh` | Script default | Line 70: `CHUNK_SIZE=1000` |
| Manual with override: `bash run_step3_full_scoring.sh --chunk-size 500` | CLI argument | Passed directly |
| WATCHER: `--start-step 3 --end-step 3` | Manifest | `agent_manifests/full_scoring.json` |
| WATCHER with override: `--params '{"chunk_size": 500}'` | CLI params | Overrides manifest |

### Key Insight

**WATCHER ignores script defaults.** It reads `default_params` from the manifest and passes them explicitly to the script.

### To Change chunk_size Permanently

Update **BOTH** files:

```bash
# 1. Update script default (for manual runs)
sed -i 's/^CHUNK_SIZE=.*/CHUNK_SIZE=1000/' run_step3_full_scoring.sh

# 2. Update manifest default (for WATCHER runs)
sed -i 's/"chunk_size": [0-9]*/"chunk_size": 1000/' agent_manifests/full_scoring.json

# 3. Verify
grep "CHUNK_SIZE=" run_step3_full_scoring.sh
grep '"chunk_size"' agent_manifests/full_scoring.json
```

### OOM Prevention Reminder

Use `chunk_size=1000` (not 5000) to prevent OOM on 7.7GB mining rigs:
- 1000 seeds/chunk × ~500MB = safe for 12 concurrent workers
- 5000 seeds/chunk × ~1.5GB = OOM with 7+ concurrent workers

'''


def patch_chapter(filepath: Path, addition: str, insert_after_pattern: str, section_marker: str) -> bool:
    """
    Add content to a chapter file after a specific pattern.
    
    Args:
        filepath: Path to the markdown file
        addition: Content to add
        insert_after_pattern: Regex pattern to find insertion point
        section_marker: String to check if already patched
    
    Returns:
        True if patched, False if already patched or error
    """
    if not filepath.exists():
        print(f"  ⚠️  File not found: {filepath}")
        return False
    
    content = filepath.read_text()
    
    # Check if already patched
    if section_marker in content:
        print(f"  ⏭️  Already patched: {filepath.name}")
        return False
    
    # Find insertion point
    match = re.search(insert_after_pattern, content, re.MULTILINE | re.DOTALL)
    if not match:
        print(f"  ⚠️  Pattern not found in {filepath.name}, appending to end")
        new_content = content + "\n" + addition
    else:
        # Insert after the matched section
        insert_pos = match.end()
        new_content = content[:insert_pos] + addition + content[insert_pos:]
    
    # Write updated content
    filepath.write_text(new_content)
    print(f"  ✅ Patched: {filepath.name}")
    return True


def main():
    print("=" * 60)
    print("PATCH: Manifest Parameter Precedence Documentation")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    base_dir = Path(".")
    patched = []
    
    # Patch Chapter 12 - after Section 3.5 or at end of Section 3
    print("Patching CHAPTER_12_WATCHER_AGENT.md...")
    ch12 = base_dir / "CHAPTER_12_WATCHER_AGENT.md"
    if patch_chapter(
        ch12,
        CHAPTER_12_ADDITION,
        r"(## 3\.5[^\n]*\n(?:.*?\n)*?)(?=\n## [4-9]|\n---\n## [4-9]|\Z)",
        "## 3.6 Manifest Parameter Precedence"
    ):
        patched.append("CHAPTER_12_WATCHER_AGENT.md")
    
    # Patch Chapter 9 - after Section 5.7
    print("Patching CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md...")
    ch9 = base_dir / "CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md"
    if patch_chapter(
        ch9,
        CHAPTER_9_ADDITION,
        r"(### 5\.7[^\n]*\n(?:.*?\n)*?)(?=\n### 5\.[89]|\n## [6-9]|\n---\n## |\Z)",
        "### 5.8 Configuration Location for chunk_size"
    ):
        patched.append("CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md")
    
    # Patch Chapter 4 - after Section 1.4 or beginning
    print("Patching CHAPTER_4_FULL_SCORING.md...")
    ch4 = base_dir / "CHAPTER_4_FULL_SCORING.md"
    if patch_chapter(
        ch4,
        CHAPTER_4_ADDITION,
        r"(## 1\.4[^\n]*\n(?:.*?\n)*?)(?=\n## 1\.[56]|\n## 2\.|\Z)",
        "## 1.5 Configuration Sources"
    ):
        patched.append("CHAPTER_4_FULL_SCORING.md")
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if patched:
        print(f"✅ Patched {len(patched)} file(s):")
        for f in patched:
            print(f"   - {f}")
        print()
        print("NEXT STEPS:")
        print("1. Review changes: git diff CHAPTER_*.md")
        print("2. Commit on Zeus:")
        print("   git add CHAPTER_*.md")
        print('   git commit -m "docs: Add manifest parameter precedence documentation"')
        print("   git push")
        print()
        print("3. Update ser8 project (copy files):")
        print("   scp CHAPTER_12_WATCHER_AGENT.md CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md \\")
        print("       CHAPTER_4_FULL_SCORING.md ser8:~/claude_project/")
    else:
        print("No files needed patching (already up to date)")
    
    print()


if __name__ == "__main__":
    main()
