#!/usr/bin/env python3
"""
apply_s163karg_dedup_timing_diag.py
====================================
Adds timing instrumentation around the three deduplicate_survivors() calls
in window_optimizer_integration_final.py to PROVE which step is consuming
inter-trial CPU time.

Instruments lines ~1432-1434:
    forward_deduped     = deduplicate_survivors(survivor_accumulator['forward'])
    reverse_deduped     = deduplicate_survivors(survivor_accumulator['reverse'])
    bidirectional_deduped = deduplicate_survivors(survivor_accumulator['bidirectional'])

Output added to coordinator log:
    [DEDUP-DIAG] forward    input=701,309  output=687,xxx  time=47.3s
    [DEDUP-DIAG] reverse    input=700,749  output=685,xxx  time=46.8s
    [DEDUP-DIAG] bidi       input=1,549    output=1,549    time=0.001s
    [DEDUP-DIAG] TOTAL dedup time: 94.1s  (fwd+rev=94.1s, bidi=0.001s)

This patch is DIAGNOSTIC ONLY — zero behavior change.
Remove after root cause confirmed.

Usage:
    python3 apply_s163karg_dedup_timing_diag.py --dry-run
    python3 apply_s163karg_dedup_timing_diag.py
"""

import argparse
import os
import shutil
import sys

TARGET = os.path.expanduser(
    "~/distributed_prng_analysis/window_optimizer_integration_final.py"
)

# ── Anchor — must match exactly ──────────────────────────────────────────────
OLD = """            forward_deduped = deduplicate_survivors(survivor_accumulator['forward'])
            reverse_deduped = deduplicate_survivors(survivor_accumulator['reverse'])
            bidirectional_deduped = deduplicate_survivors(survivor_accumulator['bidirectional'])"""

# ── Replacement — wraps each call with timing ────────────────────────────────
NEW = """            # [S163-DEDUP-DIAG] Timing instrumentation — prove fwd/rev dedup bottleneck
            import time as _dedup_time
            print(f"\\n[DEDUP-DIAG] Starting deduplication...")
            print(f"[DEDUP-DIAG] Input sizes: "
                  f"forward={len(survivor_accumulator['forward']):,}  "
                  f"reverse={len(survivor_accumulator['reverse']):,}  "
                  f"bidi={len(survivor_accumulator['bidirectional']):,}")

            _t0_fwd = _dedup_time.time()
            forward_deduped = deduplicate_survivors(survivor_accumulator['forward'])
            _t1_fwd = _dedup_time.time()
            print(f"[DEDUP-DIAG] forward    input={len(survivor_accumulator['forward']):,}  "
                  f"output={len(forward_deduped):,}  time={_t1_fwd - _t0_fwd:.3f}s")

            _t0_rev = _dedup_time.time()
            reverse_deduped = deduplicate_survivors(survivor_accumulator['reverse'])
            _t1_rev = _dedup_time.time()
            print(f"[DEDUP-DIAG] reverse    input={len(survivor_accumulator['reverse']):,}  "
                  f"output={len(reverse_deduped):,}  time={_t1_rev - _t0_rev:.3f}s")

            _t0_bid = _dedup_time.time()
            bidirectional_deduped = deduplicate_survivors(survivor_accumulator['bidirectional'])
            _t1_bid = _dedup_time.time()
            print(f"[DEDUP-DIAG] bidi       input={len(survivor_accumulator['bidirectional']):,}  "
                  f"output={len(bidirectional_deduped):,}  time={_t1_bid - _t0_bid:.3f}s")

            _total_dedup = (_t1_fwd - _t0_fwd) + (_t1_rev - _t0_rev) + (_t1_bid - _t0_bid)
            _fwd_rev_total = (_t1_fwd - _t0_fwd) + (_t1_rev - _t0_rev)
            print(f"[DEDUP-DIAG] TOTAL dedup time: {_total_dedup:.3f}s  "
                  f"(fwd+rev={_fwd_rev_total:.3f}s  bidi={_t1_bid - _t0_bid:.3f}s)")
            # [END S163-DEDUP-DIAG]"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview only — no file changes")
    args = ap.parse_args()

    # Read target
    if not os.path.exists(TARGET):
        print(f"ERROR: target not found: {TARGET}")
        sys.exit(1)

    content = open(TARGET).read()

    # Verify anchor
    if OLD not in content:
        print("ERROR: anchor not found in target file.")
        print("The file may have been patched already or has changed.")
        print("\nExpected anchor:")
        print(OLD)
        sys.exit(1)

    if "[S163-DEDUP-DIAG]" in content:
        print("WARNING: [S163-DEDUP-DIAG] marker already present — already patched?")
        sys.exit(1)

    count = content.count(OLD)
    print(f"Anchor found: {count} occurrence(s)")
    print(f"Target: {TARGET}")

    if args.dry_run:
        print("\n[DRY RUN] Would replace:")
        print(f"  OLD ({len(OLD)} chars):")
        for line in OLD.splitlines():
            print(f"    {line}")
        print(f"\n  NEW ({len(NEW)} chars) — adds timing around each dedup call")
        print("\n[DRY RUN] No changes made.")
        return

    # Backup
    bak = TARGET + ".bak_s163_dedup_diag"
    shutil.copy2(TARGET, bak)
    print(f"Backup: {bak}")

    # Apply
    new_content = content.replace(OLD, NEW, 1)

    # Verify
    if "[S163-DEDUP-DIAG]" not in new_content:
        print("ERROR: patch verification failed — marker not found after replace")
        sys.exit(1)

    open(TARGET, "w").write(new_content)
    print(f"Patch applied: {TARGET}")

    # AST check
    import ast
    try:
        ast.parse(new_content)
        print("AST check: PASSED")
    except SyntaxError as e:
        print(f"AST check FAILED: {e}")
        print("Restoring backup...")
        shutil.copy2(bak, TARGET)
        sys.exit(1)

    print("\nDone. Next run will emit [DEDUP-DIAG] timing lines to coordinator log.")
    print("Look for: grep 'DEDUP-DIAG' logs/<run>.log")


if __name__ == "__main__":
    main()
