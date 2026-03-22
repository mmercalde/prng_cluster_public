#!/usr/bin/env python3
"""
apply_s152_incremental_npz_flush.py
=====================================
Patch: Write survivors to disk as-found (not just at trial-end).

WHAT THIS FIXES
---------------
Previously, bidirectional survivors were held entirely in memory inside
`accumulator['bidirectional']` and only flushed to NPZ at the end of each
Optuna trial (via the save_best_so_far callback).

If the process was killed mid-trial — by crash, OOM, or manual stop —
all survivors found in that trial were lost.

WHAT THIS DOES
--------------
Adds a `_flush_npz_incremental()` helper that performs an atomic
merge-and-write of the current accumulator state to disk.

Calls this helper inside `_build_test_result_from_pw()` immediately
after every chunk result is merged into the accumulator, controlled by
a flush threshold: only flushes when at least INCREMENTAL_FLUSH_EVERY
new bidirectional survivors have been added since the last flush.

Default threshold: 10 survivors (configurable via env var
PRNG_FLUSH_EVERY — set to 1 to flush every single chunk).

Files patched
-------------
  window_optimizer_integration_final.py

Backup created at: window_optimizer_integration_final.py.bak_s152_flush
"""

import re
import shutil
from pathlib import Path

TARGET = Path("window_optimizer_integration_final.py")
BACKUP = Path("window_optimizer_integration_final.py.bak_s152_flush")

DRY_RUN = "--dry-run" in __import__("sys").argv

# ─────────────────────────────────────────────────────────────────────────────
# INSERTION 1 — flush helper inserted after the module-level imports block
# We insert just before `def _build_test_result_from_pw`
# ─────────────────────────────────────────────────────────────────────────────

FLUSH_HELPER = '''\
# ─────────────────────────────────────────────────────────────────────────────
# [S152] Incremental NPZ flush — write survivors to disk as-found
# ─────────────────────────────────────────────────────────────────────────────
import os as _os_flush
import numpy as _np_flush

# Flush threshold: flush NPZ after this many NEW bidi survivors accumulate.
# Override via env: PRNG_FLUSH_EVERY=1 to flush after every chunk.
_FLUSH_EVERY = int(_os_flush.environ.get("PRNG_FLUSH_EVERY", "10"))

# Tracks how many survivors were present at the last flush (module-level state,
# reset to 0 at process start — safe because each run is a fresh process).
_flush_last_count = 0


def _flush_npz_incremental(accumulator: dict, label: str = "") -> None:
    """
    Atomic merge-write of accumulator bidirectional survivors to NPZ.

    - Deduplicates by seed (highest score wins).
    - Merges with any pre-existing NPZ on disk.
    - Writes atomically via .tmp → rename.
    - Updates both bidirectional_survivors_all.npz  (with scores)
      and    bidirectional_survivors_binary.npz     (Steps 2-6 format).
    - Non-fatal: any write error is logged but does not raise.
    """
    global _flush_last_count

    bidi = accumulator.get("bidirectional", [])
    current_count = len(bidi)

    new_since_last = current_count - _flush_last_count
    if new_since_last < _FLUSH_EVERY:
        return  # not enough new survivors yet

    try:
        _ACCUM_NPZ  = "bidirectional_survivors_all.npz"
        _BINARY_NPZ = "bidirectional_survivors_binary.npz"

        # Deduplicate: highest score per seed wins
        seen: dict = {}
        for s in bidi:
            seed = int(s["seed"])
            if seed not in seen or s.get("score", 0.0) > seen[seed].get("score", 0.0):
                seen[seed] = s

        # Merge with prior NPZ if it exists
        if _os_flush.path.exists(_ACCUM_NPZ):
            try:
                prior = _np_flush.load(_ACCUM_NPZ)
                prior_seeds  = prior["seeds"]
                prior_scores = prior.get("score", _np_flush.zeros(len(prior_seeds)))
                for i, pseed in enumerate(prior_seeds):
                    pseed = int(pseed)
                    pscore = float(prior_scores[i])
                    if pseed not in seen or pscore > seen[pseed].get("score", 0.0):
                        seen[pseed] = {"seed": pseed, "score": pscore}
            except Exception as _me:
                print(f"[S152-FLUSH] Warning: could not read prior NPZ for merge: {_me}")

        all_survivors = list(seen.values())
        seeds  = _np_flush.array([s["seed"]  for s in all_survivors], dtype=_np_flush.uint64)
        scores = _np_flush.array([s.get("score", 0.0) for s in all_survivors], dtype=_np_flush.float32)
        fwd_mr = _np_flush.array([s.get("forward_match_rate", 0.0) for s in all_survivors], dtype=_np_flush.float32)
        rev_mr = _np_flush.array([s.get("reverse_match_rate", 0.0) for s in all_survivors], dtype=_np_flush.float32)

        # Atomic write — accumulator NPZ
        _tmp = _ACCUM_NPZ + ".flush.tmp"
        _np_flush.savez_compressed(_tmp, seeds=seeds, score=scores)
        _os_flush.replace(_tmp, _ACCUM_NPZ)

        # Atomic write — binary NPZ (Steps 2-6)
        _tmp_bin = _BINARY_NPZ + ".flush.tmp"
        _np_flush.savez_compressed(_tmp_bin, seeds=seeds,
                                   forward_match_rate=fwd_mr,
                                   reverse_match_rate=rev_mr,
                                   score=scores)
        _os_flush.replace(_tmp_bin, _BINARY_NPZ)

        _flush_last_count = current_count
        _tag = f" [{label}]" if label else ""
        print(
            f"[S152-FLUSH]{_tag} NPZ flushed: {len(seeds):,} total survivors "
            f"(+{new_since_last} new this flush, threshold={_FLUSH_EVERY})"
        )

    except Exception as _fe:
        print(f"[S152-FLUSH] Warning: incremental flush failed (non-fatal): {_fe}")


# ─────────────────────────────────────────────────────────────────────────────
# END [S152] incremental flush helper
# ─────────────────────────────────────────────────────────────────────────────

'''

# ─────────────────────────────────────────────────────────────────────────────
# INSERTION 2 — call flush at the end of _build_test_result_from_pw,
# right before the `return TestResult(...)` line.
# ─────────────────────────────────────────────────────────────────────────────

OLD_RETURN_BLOCK = '''\
        accumulator['forward'].extend(fwd_records + fwd_h_records)
        accumulator['reverse'].extend(rev_records + rev_h_records)

    return TestResult('''

NEW_RETURN_BLOCK = '''\
        accumulator['forward'].extend(fwd_records + fwd_h_records)
        accumulator['reverse'].extend(rev_records + rev_h_records)

        # [S152] Flush survivors to disk as-found (incremental, threshold-gated)
        _flush_npz_incremental(accumulator, label=f"chunk/trial-{trial_number}")

    return TestResult('''

# ─────────────────────────────────────────────────────────────────────────────
# Anchor for insertion 1 — insert flush helper immediately before the function
# ─────────────────────────────────────────────────────────────────────────────
ANCHOR_FOR_HELPER = "def _build_test_result_from_pw("


def apply():
    src = TARGET.read_text()

    # Validate anchors
    assert ANCHOR_FOR_HELPER in src, f"ANCHOR NOT FOUND: {ANCHOR_FOR_HELPER!r}"
    assert OLD_RETURN_BLOCK in src, f"OLD_RETURN_BLOCK not found in source. Check indentation."

    # Check idempotency
    if "[S152-FLUSH]" in src:
        print("⚠️  [S152-FLUSH] marker already present — patch appears already applied. Aborting.")
        return

    # Apply insertion 1: helper before _build_test_result_from_pw
    patched = src.replace(ANCHOR_FOR_HELPER, FLUSH_HELPER + ANCHOR_FOR_HELPER, 1)

    # Apply insertion 2: flush call inside accumulator update block
    patched = patched.replace(OLD_RETURN_BLOCK, NEW_RETURN_BLOCK, 1)

    if DRY_RUN:
        print("=== DRY RUN — no files written ===")
        # Show diff summary
        orig_lines = src.splitlines()
        new_lines  = patched.splitlines()
        added   = len(new_lines) - len(orig_lines)
        print(f"  Lines added: {added}")
        print(f"  [S152-FLUSH] present in patched: {'[S152-FLUSH]' in patched}")
        print(f"  _flush_npz_incremental call present: {'_flush_npz_incremental(accumulator' in patched}")
        return

    shutil.copy2(TARGET, BACKUP)
    print(f"✅ Backup: {BACKUP}")

    TARGET.write_text(patched)
    print(f"✅ Patched: {TARGET}")
    print()
    print("Verification:")
    print(f"  [S152-FLUSH] marker present: {'[S152-FLUSH]' in patched}")
    print(f"  _flush_npz_incremental defined: {'def _flush_npz_incremental(' in patched}")
    print(f"  flush called in _build_test_result_from_pw: {'_flush_npz_incremental(accumulator' in patched}")
    print()
    print("To tune flush threshold (default=10 new survivors):")
    print("  export PRNG_FLUSH_EVERY=1   # flush after every chunk")
    print("  export PRNG_FLUSH_EVERY=50  # flush less often")


if __name__ == "__main__":
    apply()
