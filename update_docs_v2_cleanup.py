#!/usr/bin/env python3
"""
Documentation Updater v2 — Remaining Issues Cleanup
====================================================
Fixes patterns that v1 missed due to:
  1. replace(old, new, 1) only hitting first occurrence
  2. Formatting variants not in the pattern list
  3. Multi-line blocks needing special handling

Usage:
  python3 update_docs_v2_cleanup.py /path/to/docs          # Dry run
  python3 update_docs_v2_cleanup.py /path/to/docs --apply   # Apply
"""

import os
import sys
import re
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# ============================================================================
# PHASE 1: Global regex replacements — ALL occurrences, ALL target files
# ============================================================================
GLOBAL_REGEX_ALL = [
    # Catch every variant of "12x/×/Ã— RX 6600" → "8..."
    (r'12x RX 6600',   '8x RX 6600',   "12x → 8x RX 6600"),
    (r'12× RX 6600',   '8× RX 6600',   "12× → 8× RX 6600"),
    (r'12Ã— RX 6600',  '8Ã— RX 6600',  "12Ã— → 8Ã— RX 6600"),

    # gpu_count in JSON examples
    (r'"gpu_count":\s*12',  '"gpu_count": 8',  "JSON gpu_count 12 → 8"),

    # Prose references "12 for rigs"
    (r'2 for Zeus, 12 for\s+rigs',  '2 for Zeus, 8 for rigs',  "Prose gpu_count description"),

    # TFLOPS per rig (12×8.93≈108, 8×8.93≈71)
    (r'\~108\s*\|\s*Worker',  '~71 | Worker',  "TFLOPS ~108 → ~71"),
    (r'\~113\s*\|',           '~71 |',          "TFLOPS ~113 → ~71"),

    # VRAM per rig (12×8=96, 8×8=64)
    (r'96GB VRAM',  '64GB VRAM',  "VRAM 96GB → 64GB"),

    # "Expected: 12" in test commands (catch any remaining)
    (r'# Expected: 12',  '# Expected: 8',  "Expected GPU count 12 → 8"),

    # Zeus venv path (catch any remaining)
    (r'venvs/tf/bin',  'venvs/torch/bin',  "Zeus venv tf → torch"),
    (r'source ~/tf/bin/activate',  'source ~/rocm_env/bin/activate',  "ROCm activation path"),

    # "Both rig-6600" → "All rig-6600"  
    (r'Both rig-6600 systems',  'All rig-6600 systems',  "Both → All rig-6600"),
]

# ============================================================================
# PHASE 2: Per-file block replacements for structural changes
# ============================================================================

def fix_complete_guide(content: str) -> Tuple[str, List[str]]:
    """Fix COMPLETE_OPERATING_GUIDE remaining issues."""
    changes = []

    # Title/subtitle line
    old = 'Zeus (2× RTX 3080 Ti) + rig-6600 (12× RX 6600) + rig-6600b (12× RX 6600)'
    new = 'Zeus (2× RTX 3080 Ti) + rig-6600 (8× RX 6600) + rig-6600b (8× RX 6600) + rig-6600c (8× RX 6600)'
    if old in content:
        content = content.replace(old, new)
        changes.append("Title line: added rig-6600c, fixed GPU counts")

    # Same with Ã— encoding
    old = 'Zeus (2Ã— RTX 3080 Ti) + rig-6600 (12Ã— RX 6600) + rig-6600b (12Ã— RX 6600)'
    new = 'Zeus (2Ã— RTX 3080 Ti) + rig-6600 (8Ã— RX 6600) + rig-6600b (8Ã— RX 6600) + rig-6600c (8Ã— RX 6600)'
    if old in content:
        content = content.replace(old, new)
        changes.append("Title line (Ã— encoding): added rig-6600c, fixed GPU counts")

    # Hardware table — fix rig-6600 row (12→8)
    old_pattern = r'(rig-6600\s+RX 6600 \(8GB\)\s+)12(\s+ROCm / HIP)'
    new_sub = r'\g<1>8\2'
    content_new = re.sub(old_pattern, new_sub, content)
    if content_new != content:
        changes.append("Hardware table: rig-6600 GPU count 12 → 8")
        content = content_new

    # Hardware table — fix rig-6600b row and add rig-6600c
    # Match the rig-6600b row followed by the separator
    old_pattern = r'(rig-6600b\s+RX 6600 \(8GB\)\s+)12(\s+ROCm / HIP\s*\n\s*-+)'
    def add_rig6600c_row(m):
        prefix = m.group(1)
        suffix = m.group(2)
        # Build rig-6600c row with same formatting
        return f"{prefix}8{suffix.split(chr(10))[0]}\n\n  rig-6600c       RX 6600 (8GB)      8               ROCm / HIP\n  {'-' * 72}"
    content_new = re.sub(old_pattern, add_rig6600c_row, content)
    if content_new != content:
        changes.append("Hardware table: rig-6600b 12→8 + added rig-6600c row")
        content = content_new

    # IP addresses line
    old = 'rig-6600 (192.168.3.120), rig-6600b (192.168.3.154).'
    new = 'rig-6600 (192.168.3.120), rig-6600b (192.168.3.154), rig-6600c (192.168.3.162).'
    if old in content:
        content = content.replace(old, new)
        changes.append("IP addresses: added rig-6600c")

    # SSH keys prerequisite
    old = 'SSH keys configured for passwordless access to rig-6600 and\n    rig-6600b'
    new = 'SSH keys configured for passwordless access to rig-6600,\n    rig-6600b, and rig-6600c'
    if old in content:
        content = content.replace(old, new)
        changes.append("SSH prerequisite: added rig-6600c")

    old = 'SSH keys configured for passwordless access to rig-6600 and rig-6600b'
    new = 'SSH keys configured for passwordless access to rig-6600, rig-6600b, and rig-6600c'
    if old in content:
        content = content.replace(old, new)
        changes.append("SSH prerequisite (single line): added rig-6600c")

    return content, changes


def fix_cluster_manual(content: str) -> Tuple[str, List[str]]:
    """Fix Cluster_operating_manual.txt JSON example blocks."""
    changes = []

    # The JSON example has two rig entries with gpu_count: 12
    # We need to add a rig-6600c entry after rig-6600b
    # Match the rig-6600b JSON block ending with }
    rig6600b_block = '''"hostname": "192.168.3.154",
      "username": "michael",
      "gpu_count": 8,
      "gpu_type": "RX 6600",
      "script_path": "/home/michael/distributed_prng_analysis",
      "python_env": "/home/michael/rocm_env/bin/python",
      "password": "your_password"
    }
  ]'''

    rig6600c_addition = '''"hostname": "192.168.3.154",
      "username": "michael",
      "gpu_count": 8,
      "gpu_type": "RX 6600",
      "script_path": "/home/michael/distributed_prng_analysis",
      "python_env": "/home/michael/rocm_env/bin/python",
      "password": "your_password"
    },
    {
      "hostname": "192.168.3.162",
      "username": "michael",
      "gpu_count": 8,
      "gpu_type": "RX 6600",
      "script_path": "/home/michael/distributed_prng_analysis",
      "python_env": "/home/michael/rocm_env/bin/python",
      "password": "your_password"
    }
  ]'''

    if rig6600b_block in content and '192.168.3.162' not in content:
        content = content.replace(rig6600b_block, rig6600c_addition, 1)
        changes.append("JSON example: added rig-6600c node block")

    # Also handle \r\n variant
    rig6600b_block_crlf = rig6600b_block.replace('\n', '\r\n')
    rig6600c_addition_crlf = rig6600c_addition.replace('\n', '\r\n')
    if rig6600b_block_crlf in content and '192.168.3.162' not in content:
        content = content.replace(rig6600b_block_crlf, rig6600c_addition_crlf, 1)
        changes.append("JSON example: added rig-6600c node block (CRLF)")

    # Also fix zeus python_env in JSON example
    old = '"python_env": "/home/michael/venvs/tf/bin/python"'
    new = '"python_env": "/home/michael/venvs/torch/bin/python"'
    if old in content:
        content = content.replace(old, new)
        changes.append("JSON example: Zeus python_env tf → torch")

    return content, changes


def fix_instructions(content: str) -> Tuple[str, List[str]]:
    """Fix instructions.txt remaining issues."""
    changes = []

    # Second hardware list (same pattern, different location)
    # The v1 script only replaced the FIRST occurrence
    # Global regex phase already handles 12x → 8x, so we just need to add rig-6600c
    # after the second rig-6600b line

    # Find all occurrences of the rig-6600b hardware line
    pattern = r'(\- \*\*rig-6600b \(192\.168\.3\.154\)\*\*: 8x RX 6600 \(ROCm\))'
    matches = list(re.finditer(pattern, content))
    if len(matches) >= 2:
        # Add rig-6600c after the SECOND occurrence
        pos = matches[1].end()
        if 'rig-6600c' not in content[pos:pos+100]:
            insert = '\n- **rig-6600c (192.168.3.162)**: 8x RX 6600 (ROCm)'
            content = content[:pos] + insert + content[pos:]
            changes.append("Second hardware list: added rig-6600c entry")

    # JSON example blocks — add rig-6600c after rig-6600b
    # Similar approach as cluster manual
    rig6600b_json_pattern = r'("hostname": "192\.168\.3\.154"[^}]+?"password": "your_password"\s*\})\s*\n\s*\]'
    
    def add_rig6600c_json(m):
        return m.group(1) + ''',
    {
      "hostname": "192.168.3.162",
      "username": "michael",
      "gpu_count": 8,
      "gpu_type": "RX 6600",
      "script_path": "/home/michael/distributed_prng_analysis",
      "python_env": "/home/michael/rocm_env/bin/python",
      "password": "your_password"
    }
  ]'''
    
    content_new, n = re.subn(rig6600b_json_pattern, add_rig6600c_json, content, flags=re.DOTALL)
    if n > 0 and '192.168.3.162' not in content:
        content = content_new
        changes.append(f"JSON example blocks: added rig-6600c ({n} blocks)")

    return content, changes


def fix_readme(content: str) -> Tuple[str, List[str]]:
    """Fix README.md remaining table patterns."""
    changes = []

    # The node table on Zeus might have different formatting than expected
    # Use regex to match any table row with rig-6600b and 12×
    pattern = r'(rig-6600b\s+)12[×x](\s+RX 6600\s+ROCm\s+Worker Node 2)'
    replacement = r'\g<1>8×\2'
    content_new = re.sub(pattern, replacement, content)
    if content_new != content:
        changes.append("Node table: rig-6600b 12× → 8×")
        content = content_new

    # Same for rig-6600
    pattern = r'(rig-6600\s+)12[×x](\s+RX 6600\s+ROCm\s+Worker Node 1)'
    replacement = r'\g<1>8×\2'
    content_new = re.sub(pattern, replacement, content)
    if content_new != content:
        changes.append("Node table: rig-6600 12× → 8×")
        content = content_new

    # Replace "rig-6600xt (planned)" row with rig-6600c
    pattern = r'rig-6600xt \(planned\)\s+RX 6600 XT\s+ROCm\s+Worker Node 3'
    replacement = 'rig-6600c       8× RX 6600      ROCm    Worker Node 3'
    content_new = re.sub(pattern, replacement, content)
    if content_new != content:
        changes.append("Node table: replaced rig-6600xt (planned) with rig-6600c")
        content = content_new

    # If rig-6600c not in table but rig-6600b is, add it
    if 'rig-6600b' in content and 'rig-6600c' not in content:
        # Find the rig-6600b table row and add rig-6600c after
        pattern = r'(rig-6600b\s+8[×x] RX 6600\s+ROCm\s+Worker Node 2[^\n]*)'
        def add_row(m):
            return m.group(1) + '\nrig-6600c       8× RX 6600      ROCm    Worker Node 3'
        content_new = re.sub(pattern, add_row, content)
        if content_new != content:
            changes.append("Node table: added rig-6600c row after rig-6600b")
            content = content_new

    return content, changes


def fix_chapter9(content: str) -> Tuple[str, List[str]]:
    """Fix CHAPTER_9 remaining topology diagram references."""
    changes = []

    # The topology box diagram has "12Ã— RX 6600" in the ASCII art
    # The global regex should catch this, but the box art may need
    # the (planned) node replaced with rig-6600c info
    
    # Replace "(planned)" label in topology with "rig-6600c"
    if '(planned)' in content:
        content = content.replace('(planned)', 'rig-6600c ')
        changes.append("Topology diagram: (planned) → rig-6600c")

    return content, changes


# ============================================================================
# FILE DISPATCH
# ============================================================================
FIXERS = {
    'COMPLETE_OPERATING_GUIDE_v1_1.md': fix_complete_guide,
    'COMPLETE_OPERATING_GUIDE_v1.1.md': fix_complete_guide,
    'Cluster_operating_manual.txt': fix_cluster_manual,
    'instructions.txt': fix_instructions,
    'README.md': fix_readme,
    'CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md': fix_chapter9,
}

# Skip docs/proposals/README.md — it's not the main README
SKIP_PATHS = ['docs/proposals/README.md']

TARGET_FILES = [
    "instructions.txt",
    "Cluster_operating_manual.txt",
    "COMPLETE_OPERATING_GUIDE_v1_1.md",
    "COMPLETE_OPERATING_GUIDE_v1.1.md",
    "CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md",
    "CHAPTER_2_BIDIRECTIONAL_SIEVE.md",
    "CHAPTER_4_FULL_SCORING.md",
    "CHAPTER_12_WATCHER_AGENT.md",
    "SYSTEM_ARCHITECTURE_REFERENCE.md",
    "README.md",
]


class DocUpdaterV2:
    def __init__(self, root_dir: str, apply: bool = False):
        self.root_dir = Path(root_dir)
        self.apply = apply
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.files_changed = 0
        self.total_fixes = 0

    def log(self, msg: str):
        print(msg)

    def find_files(self) -> List[Path]:
        found = []
        for target in TARGET_FILES:
            for path in self.root_dir.rglob(target):
                rel = str(path.relative_to(self.root_dir))
                if any(skip in rel for skip in SKIP_PATHS):
                    continue
                if 'backup' in rel.lower() or 'old_results' in rel.lower():
                    continue
                found.append(path)
        return sorted(set(found))

    def process_file(self, filepath: Path):
        filename = filepath.name
        rel = str(filepath.relative_to(self.root_dir))
        self.log(f"\n{'='*60}")
        self.log(f"📄 {rel}")
        self.log(f"{'='*60}")

        try:
            content = filepath.read_text(encoding='utf-8', errors='replace')
        except Exception as e:
            self.log(f"  ❌ Could not read: {e}")
            return

        original = content
        changes = []

        # Phase 1: Global regex (ALL occurrences)
        for pattern, replacement, desc in GLOBAL_REGEX_ALL:
            new_content, n = re.subn(pattern, replacement, content)
            if n > 0:
                changes.append(f"[global] {desc} ({n}x)")
                content = new_content

        # Phase 2: Per-file structural fixes
        for key, fixer in FIXERS.items():
            if filename == key:
                content, file_changes = fixer(content)
                changes.extend(f"[struct] {c}" for c in file_changes)

        if not changes:
            self.log("  ℹ️  No remaining issues")
            return

        for c in changes:
            self.log(f"  ✅ {c}")

        self.total_fixes += len(changes)
        self.files_changed += 1

        if self.apply:
            filepath.write_text(content, encoding='utf-8')
            self.log(f"  💾 WRITTEN — {len(changes)} fixes applied")
        else:
            self.log(f"  🔍 DRY RUN — {len(changes)} fixes would be applied")

    def verify(self, filepath: Path):
        """Post-update verification."""
        try:
            content = filepath.read_text(encoding='utf-8', errors='replace')
        except:
            return

        issues = []
        for pattern in [r'12[x×Ã—]\s*RX\s*6600', r'"gpu_count":\s*12', r'Expected: 12']:
            matches = re.findall(pattern, content)
            if matches:
                issues.append(f"Still has: {matches[0]}")

        if issues:
            self.log(f"\n  ⚠️  STILL REMAINING in {filepath.name}:")
            for i in issues:
                self.log(f"     → {i}")

    def run(self):
        self.log("=" * 70)
        self.log(f"📋 DOCUMENTATION UPDATER v2 — Remaining Issues Cleanup")
        self.log(f"   Root: {self.root_dir}")
        self.log(f"   Mode: {'APPLY' if self.apply else 'DRY RUN'}")
        self.log("=" * 70)

        if not self.root_dir.exists():
            self.log(f"❌ Directory not found: {self.root_dir}")
            return 1

        files = self.find_files()
        if not files:
            self.log(f"❌ No target files found")
            return 1

        self.log(f"\n📂 Found {len(files)} files")

        for f in files:
            self.process_file(f)

        if self.apply:
            self.log(f"\n{'='*60}")
            self.log("🔍 POST-UPDATE VERIFICATION")
            self.log(f"{'='*60}")
            for f in files:
                self.verify(f)

        self.log(f"\n{'='*70}")
        self.log(f"📊 SUMMARY: {self.files_changed} files, {self.total_fixes} fixes")
        if not self.apply and self.total_fixes > 0:
            self.log(f"   ⚡ Run with --apply to write changes")
        self.log("=" * 70)
        return 0


def main():
    parser = argparse.ArgumentParser(description="Doc updater v2 — cleanup remaining issues")
    parser.add_argument("directory", help="Root directory")
    parser.add_argument("--apply", action="store_true", help="Apply changes")
    args = parser.parse_args()

    updater = DocUpdaterV2(root_dir=args.directory, apply=args.apply)
    sys.exit(updater.run())


if __name__ == "__main__":
    main()
