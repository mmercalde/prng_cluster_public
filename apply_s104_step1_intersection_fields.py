#!/usr/bin/env python3
"""
apply_s104_step1_intersection_fields.py

Patch: window_optimizer_integration_final.py v3.0 -> v3.1

PROBLEM:
  S103 rewrote the accumulator to fix per-seed match rates. In doing so,
  7 trial-level intersection fields were accidentally omitted from metadata_base:
    - intersection_count
    - intersection_ratio
    - forward_only_count
    - reverse_only_count
    - survivor_overlap_ratio
    - bidirectional_selectivity
    - intersection_weight

  These fields were present in v2.0 (verified from bak_20260221_pre_s103).
  They were carried into NPZ and consumed by Step 3 as ML metadata features.
  Without them, 7 of 22 NPZ arrays are all zeros, degrading ML signal.

FIX:
  Add the 7 fields back to metadata_base (constant skip block) and
  metadata_base_hybrid (variable skip block). Formulas taken directly
  from v2.0 backup. Variable names updated to match v3.0 naming.

  Constant block uses:
    forward_set, reverse_set, bidirectional_constant, forward_records, reverse_records

  Hybrid block uses:
    forward_set_hybrid, reverse_set_hybrid, bidirectional_variable,
    forward_records_hybrid, reverse_records_hybrid

VERSION: v3.0 -> v3.1
"""

import shutil
import ast
from datetime import datetime

TARGET = 'window_optimizer_integration_final.py'
BACKUP = f'{TARGET}.bak_s104_pre'

# ============================================================
# PATCH 1 — Constant skip metadata_base
# ============================================================

OLD_CONSTANT = """        metadata_base = {
            'window_size': config.window_size,
            'offset': config.offset,
            'skip_min': config.skip_min,
            'skip_max': config.skip_max,
            'skip_range': config.skip_max - config.skip_min,
            'sessions': config.sessions,
            'trial_number': trial_number,
            'prng_base': prng_base,
            'skip_mode': 'constant',
            'prng_type': prng_base,
            # Trial-level counts retained as context
            'forward_count': len(forward_records),
            'reverse_count': len(reverse_records),
            'bidirectional_count': len(bidirectional_constant),
        }"""

NEW_CONSTANT = """        # v3.1: Compute trial-level intersection statistics
        _union_size = len(forward_set | reverse_set)
        metadata_base = {
            'window_size': config.window_size,
            'offset': config.offset,
            'skip_min': config.skip_min,
            'skip_max': config.skip_max,
            'skip_range': config.skip_max - config.skip_min,
            'sessions': config.sessions,
            'trial_number': trial_number,
            'prng_base': prng_base,
            'skip_mode': 'constant',
            'prng_type': prng_base,
            # Trial-level counts
            'forward_count': len(forward_records),
            'reverse_count': len(reverse_records),
            'bidirectional_count': len(bidirectional_constant),
            # v3.1: Restored intersection fields (were in v2.0, lost in S103 rewrite)
            'intersection_count': len(bidirectional_constant),
            'intersection_ratio': len(bidirectional_constant) / max(_union_size, 1),
            'forward_only_count': len(forward_set - reverse_set),
            'reverse_only_count': len(reverse_set - forward_set),
            'survivor_overlap_ratio': len(bidirectional_constant) / max(len(forward_set), 1),
            'bidirectional_selectivity': len(forward_set) / max(len(reverse_set), 1),
            'intersection_weight': len(bidirectional_constant) / max(len(forward_set) + len(reverse_set), 1),
        }"""

# ============================================================
# PATCH 2 — Hybrid/variable skip metadata_base_hybrid
# ============================================================

OLD_HYBRID = """            metadata_base_hybrid = {
                'window_size': config.window_size,
                'offset': config.offset,
                'skip_min': config.skip_min,
                'skip_max': config.skip_max,
                'skip_range': config.skip_max - config.skip_min,
                'sessions': config.sessions,
                'trial_number': trial_number,
                'prng_base': prng_base,
                'skip_mode': 'variable',
                'prng_type': prng_hybrid,
                'forward_count': len(forward_records_hybrid),
                'reverse_count': len(reverse_records_hybrid),
                'bidirectional_count': len(bidirectional_variable),
            }"""

NEW_HYBRID = """            # v3.1: Compute trial-level intersection statistics (variable skip)
            _union_size_hybrid = len(forward_set_hybrid | reverse_set_hybrid)
            metadata_base_hybrid = {
                'window_size': config.window_size,
                'offset': config.offset,
                'skip_min': config.skip_min,
                'skip_max': config.skip_max,
                'skip_range': config.skip_max - config.skip_min,
                'sessions': config.sessions,
                'trial_number': trial_number,
                'prng_base': prng_base,
                'skip_mode': 'variable',
                'prng_type': prng_hybrid,
                # Trial-level counts
                'forward_count': len(forward_records_hybrid),
                'reverse_count': len(reverse_records_hybrid),
                'bidirectional_count': len(bidirectional_variable),
                # v3.1: Restored intersection fields (were in v2.0, lost in S103 rewrite)
                'intersection_count': len(bidirectional_variable),
                'intersection_ratio': len(bidirectional_variable) / max(_union_size_hybrid, 1),
                'forward_only_count': len(forward_set_hybrid - reverse_set_hybrid),
                'reverse_only_count': len(reverse_set_hybrid - forward_set_hybrid),
                'survivor_overlap_ratio': len(bidirectional_variable) / max(len(forward_set_hybrid), 1),
                'bidirectional_selectivity': len(forward_set_hybrid) / max(len(reverse_set_hybrid), 1),
                'intersection_weight': len(bidirectional_variable) / max(len(forward_set_hybrid) + len(reverse_set_hybrid), 1),
            }"""

# ============================================================
# PATCH 3 — Version header bump
# ============================================================

OLD_VERSION = """Version: 3.0
Date: 2026-02-21"""

NEW_VERSION = """Version: 3.1
Date: 2026-02-22"""

OLD_CHANGELOG = """  v3.0 (2026-02-21) - S103 FIX: Preserve per-seed match rates from sieve"""

NEW_CHANGELOG = """  v3.1 (2026-02-22) - S104 FIX: Restore 7 missing intersection fields
    Fields lost during S103 rewrite: intersection_count, intersection_ratio,
    forward_only_count, reverse_only_count, survivor_overlap_ratio,
    bidirectional_selectivity, intersection_weight.
    Formulas restored from v2.0 backup (bak_20260221_pre_s103).
    Variable names updated to match v3.0 naming (forward_records not forward_survivors).
    Applied to both constant skip and variable skip (hybrid) blocks.

  v3.0 (2026-02-21) - S103 FIX: Preserve per-seed match rates from sieve"""


def apply_patch():
    print(f"Patching {TARGET}...")

    # Read
    with open(TARGET, 'r') as f:
        content = f.read()

    # Backup
    shutil.copy2(TARGET, BACKUP)
    print(f"  Backup: {BACKUP}")

    # Verify all OLD strings exist exactly once
    patches = [
        ("Constant metadata_base", OLD_CONSTANT, NEW_CONSTANT),
        ("Hybrid metadata_base_hybrid", OLD_HYBRID, NEW_HYBRID),
        ("Version header", OLD_VERSION, NEW_VERSION),
        ("Changelog entry", OLD_CHANGELOG, NEW_CHANGELOG),
    ]

    for name, old, new in patches:
        count = content.count(old)
        if count == 0:
            print(f"  ERROR: '{name}' — old string NOT FOUND. Aborting.")
            return False
        if count > 1:
            print(f"  ERROR: '{name}' — found {count} times (expected 1). Aborting.")
            return False
        print(f"  OK: '{name}' found exactly once")

    # Apply all patches
    for name, old, new in patches:
        content = content.replace(old, new)
        print(f"  Applied: {name}")

    # Write
    with open(TARGET, 'w') as f:
        f.write(content)

    # Syntax check
    print("\nSyntax check...")
    with open(TARGET, 'r') as f:
        source = f.read()
    try:
        ast.parse(source)
        print("  PASS — no syntax errors")
    except SyntaxError as e:
        print(f"  FAIL — syntax error: {e}")
        print(f"  Restoring backup...")
        shutil.copy2(BACKUP, TARGET)
        return False

    # Verify new fields present
    print("\nVerification checks...")
    checks = [
        'intersection_count',
        'intersection_ratio',
        'forward_only_count',
        'reverse_only_count',
        'survivor_overlap_ratio',
        'bidirectional_selectivity',
        'intersection_weight',
        '_union_size',
        '_union_size_hybrid',
        'Version: 3.1',
    ]
    for check in checks:
        found = check in content
        print(f"  {'OK' if found else 'MISSING'}: {check}")
        if not found:
            print(f"  ERROR: '{check}' not found after patch. Restoring backup.")
            shutil.copy2(BACKUP, TARGET)
            return False

    print(f"\n✅ Patch complete — window_optimizer_integration_final.py v3.1")
    return True


if __name__ == '__main__':
    success = apply_patch()
    exit(0 if success else 1)
