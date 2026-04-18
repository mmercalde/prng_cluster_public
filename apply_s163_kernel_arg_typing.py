#!/usr/bin/env python3
"""
apply_s163_kernel_arg_typing.py

Adds explicit cp.int32() wrapping to kernel_args in sieve_filter.py run_sieve()
non-hybrid path (lines ~270-273). The hybrid path at line ~435 already has this
wrapping — this patch brings the non-hybrid path in line.

HYPOTHESIS
----------
CuPy's RawKernel marshaling of bare Python ints is platform/version dependent.
When the C++ kernel signature declares `int n_seeds, int k, int skip_min, int skip_max`
(int32) but Python passes native Python ints (which may marshal as int64 on x86_64),
the arg buffer layout is off by 4 bytes per untyped int. Subsequent args read from
wrong offsets — `a`, `c` (LCG multiplier/increment) get misread values, `offset` may
pick up garbage. More critically, if `n_seeds` itself is misread, bounds check fails
and threads read past `seeds[]` buffer — out-of-bounds GPU reads producing page faults
with the exact signature we're seeing (GCVM_L2_PROTECTION_FAULT_STATUS:0xFFFFFFFF,
unknown client 0x1ff, WALKER_ERROR:0x7).

WHY THIS IS CONSISTENT WITH S163 FINAL WORKING EARLIER
------------------------------------------------------
Whether the overshoot threads (blocks*threads_per_block > n_seeds by up to 255) hit
valid or unmapped memory depends on CUDA allocator placement of the seeds_gpu buffer.
Some runs place it away from page boundaries — overshoot lands in valid neighboring
allocations, no fault. Other runs place it at page end — overshoot lands in unmapped
page, fault triggers. Latent bug with buffer-placement-dependent expression.

WHY S163 FINAL AND HEAD BOTH FAIL TODAY
---------------------------------------
The bug was latent in both versions. S163 FINAL was validated on April 12-13 with
whatever allocator state produced safe buffer placement. By April 17, allocator
state differs enough that unsafe placement is more common.

CHANGES
-------
File:  sieve_filter.py
Lines: ~270-273
Before:
    kernel_args = [
        seeds_gpu, residues_gpu, survivors_gpu,
        match_rates_gpu, best_skips_gpu, survivor_count_gpu,
        n_seeds, k, skip_min, skip_max, cp.float32(min_match_threshold)
    ]
After:
    kernel_args = [
        seeds_gpu, residues_gpu, survivors_gpu,
        match_rates_gpu, best_skips_gpu, survivor_count_gpu,
        cp.int32(n_seeds), cp.int32(k), cp.int32(skip_min), cp.int32(skip_max),
        cp.float32(min_match_threshold)
    ]

SAFETY
------
This change cannot make things worse. Explicit typing is strictly more defined
behavior than implicit typing. If the bug hypothesis is correct, crashes stop.
If the bug hypothesis is wrong, behavior is unchanged (CuPy may have been
marshaling correctly on this platform already).

Matches the existing pattern in run_hybrid_sieve (line ~435-437) which already
uses explicit cp.int32() wrapping — this patch brings non-hybrid path in line.

USAGE
-----
    python3 apply_s163_kernel_arg_typing.py --dry-run   # show diff, don't apply
    python3 apply_s163_kernel_arg_typing.py             # apply patch

Creates sieve_filter.py.bak_s163_kernel_arg_typing before modifying.
"""

import argparse
import shutil
import sys
from pathlib import Path

TARGET_FILE = "sieve_filter.py"
BACKUP_SUFFIX = ".bak_s163_kernel_arg_typing"

OLD_BLOCK = """                kernel_args = [
                    seeds_gpu, residues_gpu, survivors_gpu,
                    match_rates_gpu, best_skips_gpu, survivor_count_gpu,
                    n_seeds, k, skip_min, skip_max, cp.float32(min_match_threshold)
                ]"""

NEW_BLOCK = """                # [S163-KARG] Explicit int32 typing — C++ kernel signature is int32.
                # Bare Python ints marshal as int64 on x86_64 CuPy, shifting arg
                # buffer layout by 4 bytes per arg. Misaligned args past this point
                # read garbage (including n_seeds itself if misread earlier).
                # Mirrors run_hybrid_sieve pattern (line ~435) which already uses
                # explicit cp.int32() wrapping.
                kernel_args = [
                    seeds_gpu, residues_gpu, survivors_gpu,
                    match_rates_gpu, best_skips_gpu, survivor_count_gpu,
                    cp.int32(n_seeds), cp.int32(k),
                    cp.int32(skip_min), cp.int32(skip_max),
                    cp.float32(min_match_threshold),
                ]"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Show diff, do not apply")
    args = ap.parse_args()

    target = Path(TARGET_FILE)
    if not target.exists():
        sys.exit(f"ERROR: {TARGET_FILE} not found in CWD — run from project root")

    src = target.read_text()

    # Count occurrences — should be exactly 1 (the non-hybrid path)
    count = src.count(OLD_BLOCK)
    if count == 0:
        sys.exit(
            "ERROR: Target block not found. Either:\n"
            "  - Patch already applied (look for [S163-KARG] comment)\n"
            "  - File differs from expected S163 FINAL baseline\n"
            "  - Whitespace mismatch"
        )
    if count > 1:
        sys.exit(f"ERROR: Found {count} matches of target block — ambiguous, refusing to patch")

    new_src = src.replace(OLD_BLOCK, NEW_BLOCK)

    if args.dry_run:
        print("=" * 70)
        print("DRY RUN — would apply these changes:")
        print("=" * 70)
        print("\n--- OLD ---")
        print(OLD_BLOCK)
        print("\n--- NEW ---")
        print(NEW_BLOCK)
        print("\n" + "=" * 70)
        print(f"Would write {len(new_src)} bytes to {TARGET_FILE}")
        print(f"Would create backup at {TARGET_FILE}{BACKUP_SUFFIX}")
        print("=" * 70)
        return

    # Create backup
    backup = Path(f"{TARGET_FILE}{BACKUP_SUFFIX}")
    shutil.copy2(target, backup)
    print(f"✅ Backup created: {backup}")

    # Apply
    target.write_text(new_src)
    print(f"✅ Patch applied to {TARGET_FILE}")

    # Verify
    verify = target.read_text()
    if "[S163-KARG]" in verify and "cp.int32(n_seeds), cp.int32(k)" in verify:
        print("✅ Verification: [S163-KARG] marker and explicit int32 wrapping present")
    else:
        print("⚠️  Verification FAILED — restore from backup:")
        print(f"    cp {backup} {target}")
        sys.exit(1)

    print("\nNext steps:")
    print("  1. Review:  diff sieve_filter.py.bak_s163_kernel_arg_typing sieve_filter.py")
    print("  2. Deploy to rigs (zeus + 3 AMD rigs)")
    print("  3. Test with S163 FINAL params")


if __name__ == "__main__":
    main()
