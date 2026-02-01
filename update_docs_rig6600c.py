#!/usr/bin/env python3
"""
Documentation Updater — rig-6600c Integration + GPU Count Correction
=====================================================================
Scans documentation files and applies targeted replacements to bring
all docs in sync with the current cluster state:

  Zeus: 2× RTX 3080 Ti (CUDA)
  rig-6600:  8× RX 6600 (ROCm 6.4.3) — 192.168.3.120
  rig-6600b: 8× RX 6600 (ROCm 6.4.3) — 192.168.3.154
  rig-6600c: 8× RX 6600 (ROCm 6.4.3) — 192.168.3.162
  Total: 26 GPUs

Usage:
  python3 update_docs_rig6600c.py /path/to/docs          # Preview (dry run)
  python3 update_docs_rig6600c.py /path/to/docs --apply   # Apply changes
  python3 update_docs_rig6600c.py /path/to/docs --apply --no-backup  # Skip backup

Runs on Zeus: python3 update_docs_rig6600c.py ~/distributed_prng_analysis
Runs on ser8: python3 update_docs_rig6600c.py ~/Downloads/CONCISE_OPERATING_GUIDE_v1.0
"""

import os
import sys
import re
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict

# ============================================================================
# TARGET FILES — only touch documentation, never code
# ============================================================================
TARGET_FILES = [
    "instructions.txt",
    "Cluster_operating_manual.txt",
    "COMPLETE_OPERATING_GUIDE_v1_1.md",
    "COMPLETE_OPERATING_GUIDE_v1.1.md",        # alternate naming
    "CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md",
    "CHAPTER_2_BIDIRECTIONAL_SIEVE.md",
    "CHAPTER_4_FULL_SCORING.md",
    "CHAPTER_12_WATCHER_AGENT.md",
    "SYSTEM_ARCHITECTURE_REFERENCE.md",
    "README.md",
]

# ============================================================================
# GLOBAL REPLACEMENTS — applied to ALL target files
# ============================================================================
GLOBAL_REPLACEMENTS: List[Tuple[str, str, str]] = [
    # --- ROCm prelude hostname checks (Python list syntax) ---
    (
        r'if HOST in \["rig-6600", "rig-6600b"\]',
        'if HOST in ["rig-6600", "rig-6600b", "rig-6600c"]',
        "ROCm prelude hostname check (list, double-quote)"
    ),
    (
        r"if HOST in \['rig-6600', 'rig-6600b'\]",
        "if HOST in ['rig-6600', 'rig-6600b', 'rig-6600c']",
        "ROCm prelude hostname check (list, single-quote)"
    ),
    (
        r'if HOST in \("rig-6600", "rig-6600b"\)',
        'if HOST in ("rig-6600", "rig-6600b", "rig-6600c")',
        "ROCm prelude hostname check (tuple, double-quote)"
    ),
    (
        r"if HOST in \('rig-6600', 'rig-6600b'\)",
        "if HOST in ('rig-6600', 'rig-6600b', 'rig-6600c')",
        "ROCm prelude hostname check (tuple, single-quote)"
    ),
    # Escaped quotes in markdown (docx rendering)
    (
        r"if HOST in \\\['rig-6600', 'rig-6600b'\\\]",
        "if HOST in \\'rig-6600\\', \\'rig-6600b\\', \\'rig-6600c\\'\\]",
        "ROCm prelude (escaped markdown)"
    ),
    (
        r"""if HOST in \\\\?\[\\\\?'rig-6600\\\\?', \\\\?'rig-6600b\\\\?'\\\\?\]""",
        """if HOST in \\'rig-6600\\', \\'rig-6600b\\', \\'rig-6600c\\'\\]""",
        "ROCm prelude (double-escaped markdown)"
    ),

    # --- ROCm version ---
    (
        r"ROCm 5\.7\+?",
        "ROCm 6.4.3",
        "ROCm version update"
    ),

    # --- Expected GPU count in test commands ---
    (
        r"# Expected: 12",
        "# Expected: 8",
        "Expected GPU count in test commands"
    ),

    # --- Zeus python_env ---
    (
        r"venvs/tf/bin/python",
        "venvs/torch/bin/python",
        "Zeus python_env path"
    ),

    # --- "Both rig-6600 systems" ---
    (
        r"AMD Nodes \(Both rig-6600 systems\)",
        "AMD Nodes (All rig-6600 systems)",
        "AMD nodes section header"
    ),
    (
        r"Both rig-6600 systems",
        "All rig-6600 systems",
        "Both → All rig-6600 reference"
    ),
]


# ============================================================================
# PER-FILE TARGETED REPLACEMENTS
# ============================================================================
PER_FILE_REPLACEMENTS: Dict[str, List[Tuple[str, str, str]]] = {

    # === instructions.txt ===
    "instructions.txt": [
        (
            '- **rig-6600 (192.168.3.120)**: 12x RX 6600 (ROCm)\r\n- **rig-6600b (192.168.3.154)**: 12x RX 6600 (ROCm)',
            '- **rig-6600 (192.168.3.120)**: 8x RX 6600 (ROCm)\n- **rig-6600b (192.168.3.154)**: 8x RX 6600 (ROCm)\n- **rig-6600c (192.168.3.162)**: 8x RX 6600 (ROCm)',
            "instructions.txt: Hardware list (with \\r\\n)"
        ),
        (
            '- **rig-6600 (192.168.3.120)**: 12x RX 6600 (ROCm)\n- **rig-6600b (192.168.3.154)**: 12x RX 6600 (ROCm)',
            '- **rig-6600 (192.168.3.120)**: 8x RX 6600 (ROCm)\n- **rig-6600b (192.168.3.154)**: 8x RX 6600 (ROCm)\n- **rig-6600c (192.168.3.162)**: 8x RX 6600 (ROCm)',
            "instructions.txt: Hardware list (unix newlines)"
        ),
        # Test commands — add rig-6600c test
        (
            '# Test rig-6600b\nssh 192.168.3.154',
            '# Test rig-6600b\nssh 192.168.3.154',
            "SKIP — handled by block replacement below"
        ),
        # distributed_config.json example gpu_count
        (
            '"gpu_count": 12,\r\n      "gpu_type',
            '"gpu_count": 8,\n      "gpu_type',
            "instructions.txt: config example gpu_count (\\r\\n)"
        ),
        (
            '"gpu_count": 12,\n      "gpu_type',
            '"gpu_count": 8,\n      "gpu_type',
            "instructions.txt: config example gpu_count (unix)"
        ),
    ],

    # === Cluster_operating_manual.txt ===
    "Cluster_operating_manual.txt": [
        (
            'rig-6600 (192.168.3.120): 12x RX 6600 (ROCm)\r\nrig-6600b (192.168.3.154): 12x RX 6600 (ROCm)',
            'rig-6600 (192.168.3.120): 8x RX 6600 (ROCm)\nrig-6600b (192.168.3.154): 8x RX 6600 (ROCm)\nrig-6600c (192.168.3.162): 8x RX 6600 (ROCm)',
            "Cluster manual: Hardware list (\\r\\n)"
        ),
        (
            'rig-6600 (192.168.3.120): 12x RX 6600 (ROCm)\nrig-6600b (192.168.3.154): 12x RX 6600 (ROCm)',
            'rig-6600 (192.168.3.120): 8x RX 6600 (ROCm)\nrig-6600b (192.168.3.154): 8x RX 6600 (ROCm)\nrig-6600c (192.168.3.162): 8x RX 6600 (ROCm)',
            "Cluster manual: Hardware list (unix)"
        ),
    ],

    # === COMPLETE_OPERATING_GUIDE ===
    "COMPLETE_OPERATING_GUIDE_v1_1.md": [
        # Hardware table rows
        (
            'rig-6600        RX 6600 (8GB)      12              ROCm / HIP',
            'rig-6600        RX 6600 (8GB)      8               ROCm / HIP',
            "Guide: rig-6600 GPU count in table"
        ),
        (
            'rig-6600b       RX 6600 (8GB)      12              ROCm / HIP',
            'rig-6600b       RX 6600 (8GB)      8               ROCm / HIP',
            "Guide: rig-6600b GPU count in table"
        ),
    ],
    "COMPLETE_OPERATING_GUIDE_v1.1.md": [
        (
            'rig-6600        RX 6600 (8GB)      12              ROCm / HIP',
            'rig-6600        RX 6600 (8GB)      8               ROCm / HIP',
            "Guide: rig-6600 GPU count in table"
        ),
        (
            'rig-6600b       RX 6600 (8GB)      12              ROCm / HIP',
            'rig-6600b       RX 6600 (8GB)      8               ROCm / HIP',
            "Guide: rig-6600b GPU count in table"
        ),
    ],

    # === CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md ===
    "CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md": [
        # Performance table
        (
            '| rig-6600 | 12× RX 6600 | ROCm | ~108 | Worker |',
            '| rig-6600 | 8× RX 6600 | ROCm | ~71 | Worker |',
            "Ch9: rig-6600 performance table"
        ),
        (
            '| rig-6600b | 12× RX 6600 | ROCm | ~108 | Worker |',
            '| rig-6600b | 8× RX 6600 | ROCm | ~71 | Worker |\n| rig-6600c | 8× RX 6600 | ROCm | ~71 | Worker |',
            "Ch9: rig-6600b performance table + add rig-6600c"
        ),
        # Topology diagram GPU counts (unicode ×)
        (
            '12× RX 6600',
            '8× RX 6600',
            "Ch9: GPU count with unicode ×"
        ),
        # Topology diagram GPU counts (UTF-8 Ã—)
        (
            '12Ã— RX 6600',
            '8Ã— RX 6600',
            "Ch9: GPU count with Ã—"
        ),
        # PER_NODE_CONCURRENCY dict
        (
            "'rig-6600': 1,    # Weak CPU, limit to 1\n    'rig-6600b': 1,   # Weak CPU, limit to 1",
            "'rig-6600': 1,    # Weak CPU, limit to 1\n    'rig-6600b': 1,   # Weak CPU, limit to 1\n    'rig-6600c': 1,   # Weak CPU, limit to 1",
            "Ch9: PER_NODE_CONCURRENCY add rig-6600c"
        ),
        # Ramdisk deployment loop
        (
            'for node in localhost 192.168.3.120 192.168.3.154; do',
            'for node in localhost 192.168.3.120 192.168.3.154 192.168.3.162; do',
            "Ch9: Ramdisk deployment loop add rig-6600c"
        ),
        # Validated configuration concurrency
        (
            'max_concurrent_script_jobs: 12     # Full GPU utilization',
            'max_concurrent_script_jobs: 8      # Full GPU utilization',
            "Ch9: Validated config concurrency"
        ),
        # Planned node in topology
        (
            '(planned)',
            'rig-6600c',
            "Ch9: Replace (planned) with rig-6600c"
        ),
    ],

    # === README.md ===
    "README.md": [
        (
            'rig-6600        12× RX 6600     ROCm    Worker Node 1\r\nrig-6600b       12× RX 6600     ROCm    Worker Node 2\r\nrig-6600xt (planned)    RX 6600 XT      ROCm    Worker Node 3',
            'rig-6600        8× RX 6600      ROCm    Worker Node 1\nrig-6600b       8× RX 6600      ROCm    Worker Node 2\nrig-6600c       8× RX 6600      ROCm    Worker Node 3',
            "README: Node table (\\r\\n)"
        ),
        (
            'rig-6600        12× RX 6600     ROCm    Worker Node 1\nrig-6600b       12× RX 6600     ROCm    Worker Node 2\nrig-6600xt (planned)    RX 6600 XT      ROCm    Worker Node 3',
            'rig-6600        8× RX 6600      ROCm    Worker Node 1\nrig-6600b       8× RX 6600      ROCm    Worker Node 2\nrig-6600c       8× RX 6600      ROCm    Worker Node 3',
            "README: Node table (unix)"
        ),
        (
            'source ~/tf/bin/activate',
            'source ~/rocm_env/bin/activate',
            "README: ROCm activation command"
        ),
    ],

    # === SYSTEM_ARCHITECTURE_REFERENCE.md ===
    "SYSTEM_ARCHITECTURE_REFERENCE.md": [
        (
            '| **rig-6600** | 12× RX 6600 | ROCm | ~113 | Distributed sieving workers |',
            '| **rig-6600** | 8× RX 6600 | ROCm | ~71 | Distributed sieving workers |',
            "SysArch: rig-6600 table row"
        ),
        (
            '| **rig-6600b** | 12× RX 6600 | ROCm | ~113 | Distributed sieving workers |',
            '| **rig-6600b** | 8× RX 6600 | ROCm | ~71 | Distributed sieving workers |\n| **rig-6600c** | 8× RX 6600 | ROCm | ~71 | Distributed sieving workers |',
            "SysArch: rig-6600b table row + add rig-6600c"
        ),
    ],

    # === CHAPTER_4_FULL_SCORING.md ===
    "CHAPTER_4_FULL_SCORING.md": [
        (
            '# On rig-6600\n',
            '# On rig-6600 / rig-6600c\n',
            "Ch4: rig-6600 comment"
        ),
    ],

    # === CHAPTER_12_WATCHER_AGENT.md ===
    "CHAPTER_12_WATCHER_AGENT.md": [
        (
            'Remote nodes (rig-6600, rig-6600b) are **NOT** automatically populated.',
            'Remote nodes (rig-6600, rig-6600b, rig-6600c) are **NOT** automatically populated.',
            "Ch12: Remote nodes ramdisk warning"
        ),
        (
            '| rig-6600 | ❌ No | Fail immediately |\n| rig-6600b | ❌ No | Fail immediately |',
            '| rig-6600 | ❌ No | Fail immediately |\n| rig-6600b | ❌ No | Fail immediately |\n| rig-6600c | ❌ No | Fail immediately |',
            "Ch12: Ramdisk status table add rig-6600c"
        ),
        # Try with encoded characters too
        (
            '| rig-6600 | âŒ No | Fail immediately |\n| rig-6600b | âŒ No | Fail immediately |',
            '| rig-6600 | âŒ No | Fail immediately |\n| rig-6600b | âŒ No | Fail immediately |\n| rig-6600c | âŒ No | Fail immediately |',
            "Ch12: Ramdisk status table add rig-6600c (encoded)"
        ),
    ],
}

# ============================================================================
# BLOCK INSERTIONS — add new content after a marker line
# ============================================================================
BLOCK_INSERTIONS: Dict[str, List[Tuple[str, str, str]]] = {
    # Add rig-6600c test commands in instructions.txt after rig-6600b test
    "instructions.txt": [
        (
            "# Expected: 8\n\n# NEW: Test sieve on AMD nodes",
            "# Expected: 8\n\n# Test rig-6600c\nssh 192.168.3.162 'source ~/rocm_env/bin/activate && cd distributed_prng_analysis && python3 -c \"import cupy; print(cupy.cuda.runtime.getDeviceCount())\"'\n# Expected: 8\n\n# NEW: Test sieve on AMD nodes",
            "instructions.txt: Add rig-6600c test command"
        ),
    ],
    "Cluster_operating_manual.txt": [
        (
            "# Expected: 8\r\n\r\n# NEW: Test sieve",
            "# Expected: 8\n\n# Test rig-6600c\nssh 192.168.3.162 'source ~/rocm_env/bin/activate && cd distributed_prng_analysis && python3 -c \"import cupy; print(cupy.cuda.runtime.getDeviceCount())\"'\n# Expected: 8\n\n# NEW: Test sieve",
            "Cluster manual: Add rig-6600c test command (\\r\\n)"
        ),
        (
            "# Expected: 8\n\n# NEW: Test sieve",
            "# Expected: 8\n\n# Test rig-6600c\nssh 192.168.3.162 'source ~/rocm_env/bin/activate && cd distributed_prng_analysis && python3 -c \"import cupy; print(cupy.cuda.runtime.getDeviceCount())\"'\n# Expected: 8\n\n# NEW: Test sieve",
            "Cluster manual: Add rig-6600c test command (unix)"
        ),
    ],
}


# ============================================================================
# ENGINE
# ============================================================================

class DocUpdater:
    def __init__(self, root_dir: str, apply: bool = False, backup: bool = True):
        self.root_dir = Path(root_dir)
        self.apply = apply
        self.backup = backup
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report: List[str] = []
        self.files_changed = 0
        self.total_replacements = 0

    def log(self, msg: str):
        self.report.append(msg)
        print(msg)

    def find_files(self) -> List[Path]:
        """Find all target documentation files in the root directory (recursive)."""
        found = []
        for target in TARGET_FILES:
            for path in self.root_dir.rglob(target):
                # Skip backup directories
                if 'backup' in str(path).lower() or 'old_results' in str(path).lower():
                    continue
                if '__pycache__' in str(path):
                    continue
                found.append(path)
        return sorted(set(found))

    def backup_file(self, filepath: Path):
        """Create timestamped backup."""
        if not self.backup:
            return
        backup_dir = self.root_dir / f"doc_backup_{self.timestamp}"
        backup_dir.mkdir(exist_ok=True)
        dest = backup_dir / filepath.name
        shutil.copy2(filepath, dest)
        self.log(f"  📦 Backup: {dest}")

    def apply_global_replacements(self, content: str, filename: str) -> Tuple[str, int]:
        """Apply regex-based global replacements."""
        count = 0
        for pattern, replacement, description in GLOBAL_REPLACEMENTS:
            new_content, n = re.subn(pattern, replacement, content)
            if n > 0:
                count += n
                self.log(f"  ✅ {description} ({n}x)")
                content = new_content
        return content, count

    def apply_per_file_replacements(self, content: str, filename: str) -> Tuple[str, int]:
        """Apply file-specific literal replacements."""
        count = 0
        # Check all possible filename keys
        for key in PER_FILE_REPLACEMENTS:
            if filename == key or filename.replace('.', '_') == key.replace('.', '_'):
                for old, new, description in PER_FILE_REPLACEMENTS[key]:
                    if "SKIP" in description:
                        continue
                    if old in content:
                        content = content.replace(old, new, 1)
                        count += 1
                        self.log(f"  ✅ {description}")
        return content, count

    def apply_block_insertions(self, content: str, filename: str) -> Tuple[str, int]:
        """Apply block insertions (add new content after markers)."""
        count = 0
        for key in BLOCK_INSERTIONS:
            if filename == key:
                for marker, replacement, description in BLOCK_INSERTIONS[key]:
                    if marker in content and replacement not in content:
                        # Only insert if the new content isn't already there
                        if "rig-6600c" not in content.split(marker)[0].split('\n')[-1]:
                            content = content.replace(marker, replacement, 1)
                            count += 1
                            self.log(f"  ✅ {description}")
        return content, count

    def process_file(self, filepath: Path):
        """Process a single documentation file."""
        filename = filepath.name
        self.log(f"\n{'='*60}")
        self.log(f"📄 {filepath}")
        self.log(f"{'='*60}")

        try:
            content = filepath.read_text(encoding='utf-8', errors='replace')
        except Exception as e:
            self.log(f"  ❌ Could not read: {e}")
            return

        original = content

        # Apply replacements in order
        content, n1 = self.apply_global_replacements(content, filename)
        content, n2 = self.apply_per_file_replacements(content, filename)
        content, n3 = self.apply_block_insertions(content, filename)

        total = n1 + n2 + n3

        if total == 0:
            self.log("  ℹ️  No changes needed")
            return

        self.total_replacements += total
        self.files_changed += 1

        if self.apply:
            self.backup_file(filepath)
            filepath.write_text(content, encoding='utf-8')
            self.log(f"  💾 WRITTEN — {total} replacements applied")
        else:
            self.log(f"  🔍 DRY RUN — {total} replacements would be applied")

    def verify_remaining_issues(self, filepath: Path):
        """Check for any remaining stale patterns after processing."""
        try:
            content = filepath.read_text(encoding='utf-8', errors='replace')
        except:
            return

        issues = []

        # Check for remaining "12x RX" or "12× RX" that we missed
        for pattern in [r'12[x×Ã—]\s*RX\s*6600', r'gpu_count.*12', r'"gpu_count": 12']:
            matches = re.findall(pattern, content)
            if matches:
                issues.append(f"Remaining '12 GPU' reference: {matches[0]}")

        # Check for rig-6600/rig-6600b without rig-6600c nearby
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'rig-6600b' in line and 'rig-6600c' not in line:
                # Check if rig-6600c is on the next line
                next_line = lines[i+1] if i+1 < len(lines) else ""
                if 'rig-6600c' not in next_line:
                    # Check context — is this a list/table that should have rig-6600c?
                    if any(kw in line for kw in ['192.168.3.154', 'Worker', 'ROCm', 'RX 6600']):
                        issues.append(f"Line {i+1}: rig-6600b without rig-6600c: {line.strip()[:80]}")

        if issues:
            self.log(f"\n  ⚠️  REMAINING ISSUES in {filepath.name}:")
            for issue in issues[:5]:
                self.log(f"     → {issue}")

    def run(self):
        """Run the full update process."""
        self.log("=" * 70)
        self.log(f"📋 DOCUMENTATION UPDATER — rig-6600c Integration")
        self.log(f"   Root: {self.root_dir}")
        self.log(f"   Mode: {'APPLY' if self.apply else 'DRY RUN (preview only)'}")
        self.log(f"   Backup: {'Yes' if self.backup else 'No'}")
        self.log("=" * 70)

        if not self.root_dir.exists():
            self.log(f"❌ Directory not found: {self.root_dir}")
            return 1

        files = self.find_files()
        if not files:
            self.log(f"❌ No target documentation files found in {self.root_dir}")
            self.log(f"   Looking for: {', '.join(TARGET_FILES[:5])}...")
            return 1

        self.log(f"\n📂 Found {len(files)} documentation files:")
        for f in files:
            self.log(f"   • {f.relative_to(self.root_dir)}")

        # Process each file
        for filepath in files:
            self.process_file(filepath)

        # Post-processing verification
        if self.apply:
            self.log(f"\n{'='*60}")
            self.log("🔍 POST-UPDATE VERIFICATION")
            self.log(f"{'='*60}")
            for filepath in files:
                self.verify_remaining_issues(filepath)

        # Summary
        self.log(f"\n{'='*70}")
        self.log(f"📊 SUMMARY")
        self.log(f"{'='*70}")
        self.log(f"   Files scanned:     {len(files)}")
        self.log(f"   Files changed:     {self.files_changed}")
        self.log(f"   Total replacements: {self.total_replacements}")
        if not self.apply and self.total_replacements > 0:
            self.log(f"\n   ⚡ Run with --apply to write changes")

        # Save report
        report_path = self.root_dir / f"doc_update_report_{self.timestamp}.txt"
        report_path.write_text('\n'.join(self.report), encoding='utf-8')
        self.log(f"\n   📝 Report saved: {report_path}")

        return 0


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Update documentation for rig-6600c integration + GPU count correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ~/distributed_prng_analysis                    # Dry run on Zeus
  %(prog)s ~/distributed_prng_analysis --apply            # Apply on Zeus
  %(prog)s ~/Downloads/CONCISE_OPERATING_GUIDE_v1.0       # Dry run on ser8
  %(prog)s ~/Downloads/CONCISE_OPERATING_GUIDE_v1.0 --apply  # Apply on ser8
        """
    )
    parser.add_argument("directory", help="Root directory containing documentation files")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default: dry run)")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backup files")

    args = parser.parse_args()

    updater = DocUpdater(
        root_dir=args.directory,
        apply=args.apply,
        backup=not args.no_backup
    )
    sys.exit(updater.run())


if __name__ == "__main__":
    main()
