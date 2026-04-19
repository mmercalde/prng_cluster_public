#!/usr/bin/env python3
"""
apply_s163karg_npz_sort_fix.py
================================
TB-approved fix for IndexError in S145-R1 v2 NPZ accumulator merge.

    IndexError: index 9277 is out of bounds for axis 0 with size 0

Two patches applied to window_optimizer_integration_final.py:

PATCH 1 — Sort fix with schema backfill (TB-approved):
  - Backfill missing/empty non-seed fields to zeros(seed_len)
  - Raise tagged [S163-KARG-NPZ] ValueError on wrong-length non-empty fields
  - Sort all fields normally

PATCH 2 — Tagged ValueError re-raise (TB-approved):
  - Split except clause: re-raise only [S163-KARG-NPZ] tagged ValueErrors
  - All other ValueError and Exception still fall through to fallback
  - Prevents silent data loss when bidirectional_survivors.json is summary-only

Usage:
    python3 apply_s163karg_npz_sort_fix.py --dry-run
    python3 apply_s163karg_npz_sort_fix.py
"""

import argparse
import ast
import os
import shutil
import sys

TARGET = os.path.expanduser(
    "~/distributed_prng_analysis/window_optimizer_integration_final.py"
)

# ── PATCH 1: Sort fix with schema backfill ───────────────────────────────────

OLD_1 = """                # Sort merged arrays by seed value
                _sort_idx = _np_s145.argsort(_merged_arrays['seeds'])
                for _fname in _merged_arrays:
                    _merged_arrays[_fname] = _merged_arrays[_fname][_sort_idx]"""

NEW_1 = """                # [S163-KARG-NPZ] TB-approved fix: backfill missing fields before sort.
                # Fields absent from older prior NPZ schemas produce size-0 arrays.
                # Backfill to zeros(seed_len) ensures rectangular NPZ — safe for all
                # downstream readers. Tagged ValueError on wrong-length non-empty fields
                # is caught by Patch 2 below and re-raised to prevent silent data loss.
                _seed_len = len(_merged_arrays['seeds'])

                def _dtype_for_field(_fn):
                    if _fn in _FIELDS_UINT32:  return _np_s145.uint32
                    if _fn in _FIELDS_INT32:   return _np_s145.int32
                    if _fn in _FIELDS_UINT8:   return _np_s145.uint8
                    return _np_s145.float32

                for _fn in _FIELDS_UINT32 + _FIELDS_INT32 + _FIELDS_FLOAT32 + _FIELDS_UINT8:
                    if _fn == 'seeds':
                        continue
                    if _fn not in _merged_arrays or len(_merged_arrays[_fn]) == 0:
                        # Missing or empty — backfill with zeros to keep schema rectangular
                        _merged_arrays[_fn] = _np_s145.zeros(
                            _seed_len, dtype=_dtype_for_field(_fn)
                        )
                    elif len(_merged_arrays[_fn]) != _seed_len:
                        raise ValueError(
                            f"[S163-KARG-NPZ] Field {_fn} length "
                            f"{len(_merged_arrays[_fn])} != seeds length {_seed_len}"
                        )
                # [END S163-KARG-NPZ backfill]

                # Sort merged arrays by seed value (all fields now seed_len — safe)
                _sort_idx = _np_s145.argsort(_merged_arrays['seeds'])
                for _fname in _merged_arrays:
                    _merged_arrays[_fname] = _merged_arrays[_fname][_sort_idx]"""

# ── PATCH 2: Tagged ValueError re-raise ─────────────────────────────────────

OLD_2 = """            except Exception as _accum_err:
                print(f"\\n⚠️  [S145-R1 v2][NPZ ACCUMULATOR] Failed: {_accum_err}")
                print(f"   Falling back to per-run convert_survivors_to_binary.py")
                import traceback as _tb_s145
                _tb_s145.print_exc()
                # Fallback: use original conversion path
                from subprocess import run as subprocess_run, CalledProcessError
                try:
                    subprocess_run(
                        ["python3", "convert_survivors_to_binary.py",
                         "bidirectional_survivors.json"],
                        check=True
                    )
                    print(f"✅ Fallback: converted bidirectional_survivors.json to NPZ")
                except CalledProcessError as _e:
                    print(f"❌ NPZ conversion failed: {_e}")
                    raise RuntimeError("Step 1 incomplete - NPZ conversion required for Step 2")"""

NEW_2 = """            except ValueError as _accum_err:
                # [S163-KARG-NPZ] Re-raise only tagged schema-mismatch ValueErrors.
                # Untagged ValueErrors (from numpy/conversion code) still fall through
                # to the fallback path — they may be reasonable fallback candidates.
                # Re-raising tagged errors prevents silent data loss when
                # bidirectional_survivors.json is summary-only (JSON guard active).
                if str(_accum_err).startswith("[S163-KARG-NPZ]"):
                    raise  # schema mismatch — do not silently fall back
                print(f"\\n⚠️  [S145-R1 v2][NPZ ACCUMULATOR] Failed: {_accum_err}")
                print(f"   Falling back to per-run convert_survivors_to_binary.py")
                import traceback as _tb_s145
                _tb_s145.print_exc()
                # Fallback: use original conversion path
                from subprocess import run as subprocess_run, CalledProcessError
                try:
                    subprocess_run(
                        ["python3", "convert_survivors_to_binary.py",
                         "bidirectional_survivors.json"],
                        check=True
                    )
                    print(f"✅ Fallback: converted bidirectional_survivors.json to NPZ")
                except CalledProcessError as _e:
                    print(f"❌ NPZ conversion failed: {_e}")
                    raise RuntimeError("Step 1 incomplete - NPZ conversion required for Step 2")
            except Exception as _accum_err:
                print(f"\\n⚠️  [S145-R1 v2][NPZ ACCUMULATOR] Failed: {_accum_err}")
                print(f"   Falling back to per-run convert_survivors_to_binary.py")
                import traceback as _tb_s145
                _tb_s145.print_exc()
                # Fallback: use original conversion path
                from subprocess import run as subprocess_run, CalledProcessError
                try:
                    subprocess_run(
                        ["python3", "convert_survivors_to_binary.py",
                         "bidirectional_survivors.json"],
                        check=True
                    )
                    print(f"✅ Fallback: converted bidirectional_survivors.json to NPZ")
                except CalledProcessError as _e:
                    print(f"❌ NPZ conversion failed: {_e}")
                    raise RuntimeError("Step 1 incomplete - NPZ conversion required for Step 2")"""

# ── TB-required post-patch markers ───────────────────────────────────────────

REQUIRED_MARKERS = [
    "_seed_len = len(_merged_arrays['seeds'])",
    "if _fn == 'seeds':",
    "_np_s145.zeros(",
    "!= _seed_len",
    'if str(_accum_err).startswith("[S163-KARG-NPZ]"):',
    "raise  # schema mismatch — do not silently fall back",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview only — no file changes")
    args = ap.parse_args()

    if not os.path.exists(TARGET):
        print(f"ERROR: target not found: {TARGET}")
        sys.exit(1)

    with open(TARGET, "r", encoding="utf-8") as f:
        content = f.read()

    print(f"Target: {TARGET}")

    # Pre-flight: check not already patched
    if "[S163-KARG-NPZ]" in content:
        print("WARNING: [S163-KARG-NPZ] marker already present — already patched?")
        sys.exit(1)

    # Validate both anchors before touching anything
    c1 = content.count(OLD_1)
    c2 = content.count(OLD_2)
    print(f"Patch 1 anchor: {c1} occurrence(s)")
    print(f"Patch 2 anchor: {c2} occurrence(s)")
    if c1 != 1 or c2 != 1:
        print(f"ERROR: expected exactly 1 match per anchor. Got P1={c1} P2={c2}")
        sys.exit(1)

    if args.dry_run:
        print("\n[DRY RUN] Patch 1 — sort fix with schema backfill")
        print(f"  OLD ({len(OLD_1)} chars) → NEW ({len(NEW_1)} chars)")
        print("\n[DRY RUN] Patch 2 — tagged ValueError re-raise")
        print(f"  OLD ({len(OLD_2)} chars) → NEW ({len(NEW_2)} chars)")
        print("\n[DRY RUN] No changes made.")
        return

    # Backup
    import datetime
    _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = TARGET + f".bak_s163_npz_sort_fix_{_ts}"
    shutil.copy2(TARGET, bak)
    print(f"Backup: {bak}")

    # Apply both patches
    new_content = content.replace(OLD_1, NEW_1, 1)
    new_content = new_content.replace(OLD_2, NEW_2, 1)

    # Verify all required markers
    for marker in REQUIRED_MARKERS:
        if marker not in new_content:
            print(f"ERROR: required marker missing after patch: {marker!r}")
            shutil.copy2(bak, TARGET)
            sys.exit(1)
        print(f"Marker OK: {marker[:60]}")

    # Write
    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"\nPatches applied: {TARGET}")

    # AST check
    try:
        ast.parse(new_content)
        print("AST check: PASSED")
    except SyntaxError as e:
        print(f"AST check FAILED: {e}")
        print("Restoring backup...")
        shutil.copy2(bak, TARGET)
        sys.exit(1)

    print("\nDone. NPZ accumulator is now schema-safe:")
    print("  - Missing fields backfilled to zeros(seed_len)")
    print("  - Tagged [S163-KARG-NPZ] ValueError re-raised (no silent fallback)")
    print("  - Untagged ValueError and Exception still use fallback path")
    print("  - All fields sorted normally")
    print("  - This specific NPZ merge/sort crash should no longer trigger fallback")


if __name__ == "__main__":
    main()
