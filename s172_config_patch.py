#!/usr/bin/env python3
"""
S172 distributed_config.json patch — REFINED v2.

Per Team Beta refined ruling 2026-04-30:

    "We shouldn't constrain Optuna — except where math proves a region is invalid."

    W=2 is mathematically invalid (~39% survivor rate by chance alone).
    threshold=0.30 is NOT invalid — Optuna should keep exploring it.

Therefore this patch touches ONLY window_size.min. Threshold bounds in the JSON
remain at the operator-set values (currently 0.30 → 0.75) so Optuna can
optimize across the full plausible threshold range with the threshold-drop
bug now fixed in the code.

What this patch does:
    window_size.min:  2 → 6   (configurable: --min-window 8 for more conservative)

What this patch does NOT do:
    Does NOT touch forward_threshold or reverse_threshold bounds.
    Does NOT touch defaults.
    Does NOT touch any other search bound.

Properties:
    - Reads JSON with object_pairs_hook=OrderedDict to preserve key order
    - Pretty-prints output to match existing formatting
    - Creates timestamped backup
    - Idempotent (no-op if window_size.min already at or above target)
    - Reports a unified diff of changes

Usage on Zeus:
    cd ~/distributed_prng_analysis
    python3 s172_config_patch.py --dry-run         # preview
    python3 s172_config_patch.py                   # min_window_size = 6 (default)
    python3 s172_config_patch.py --min-window 8    # more conservative
"""
from __future__ import annotations
import argparse
import datetime as _dt
import difflib
import json
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent
CFG_PATH   = REPO_ROOT / "distributed_config.json"
TIMESTAMP  = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    parser = argparse.ArgumentParser(description="S172 config patch (window_size only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show the diff without writing")
    parser.add_argument("--min-window", type=int, default=6, choices=[6, 8],
                        help="Minimum window_size (6 = TB default; 8 = conservative)")
    args = parser.parse_args()

    if not CFG_PATH.exists():
        print(f"❌ Config not found: {CFG_PATH}")
        return 2

    print("=" * 70)
    print(f"S172 CONFIG PATCH v2  ({'DRY RUN' if args.dry_run else 'APPLY'})")
    print(f"Config: {CFG_PATH}")
    print(f"Target: window_size.min = {args.min_window}")
    print(f"Threshold bounds: PRESERVED (per TB refined ruling)")
    print("=" * 70)
    print()

    original_text = CFG_PATH.read_text()
    cfg = json.loads(original_text, object_pairs_hook=OrderedDict)

    sb = cfg.get("search_bounds")
    if sb is None:
        print("❌ search_bounds block missing — refusing to edit unfamiliar config")
        return 2

    changes = []

    # window_size.min — the ONLY change
    ws = sb.get("window_size", OrderedDict())
    cur_min = ws.get("min")
    if cur_min is None:
        print("❌ window_size.min missing in config — refusing to edit (unexpected schema)")
        return 2

    if cur_min < args.min_window:
        ws["min"] = args.min_window
        changes.append(f"window_size.min: {cur_min} → {args.min_window}")

        # Add traceability note (does NOT replace existing _calibration_note)
        ws["_s172_note"] = (
            f"S172 (2026-04-30): min raised from {cur_min} to {args.min_window} "
            f"per TB ruling. W=2/3 produces ~39%/53% survivor rate by chance alone, "
            f"regardless of threshold. Threshold bounds intentionally PRESERVED so "
            f"Optuna can continue optimizing across [min, max]."
        )
    else:
        print(f"⏭  window_size.min already {cur_min} (>= {args.min_window}) — no change")

    sb["window_size"] = ws
    cfg["search_bounds"] = sb

    if not changes:
        print("✅ Config already at target — no changes needed")
        return 0

    # Render new JSON
    new_text = json.dumps(cfg, indent=2) + "\n"

    # Diff
    diff = difflib.unified_diff(
        original_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile=str(CFG_PATH) + " (original)",
        tofile=str(CFG_PATH) + " (patched)",
        n=2,
    )
    print("Changes:")
    for c in changes:
        print(f"  • {c}")
    print()
    print("Unified diff:")
    print("─" * 70)
    for line in diff:
        sys.stdout.write(line)
    print("─" * 70)
    print()

    if args.dry_run:
        print("DRY RUN — no file written")
        return 0

    # Backup and write
    bak = CFG_PATH.with_suffix(CFG_PATH.suffix + f".s172_bak_{TIMESTAMP}")
    shutil.copy2(CFG_PATH, bak)
    print(f"📦 Backup: {bak.name}")

    CFG_PATH.write_text(new_text)
    print(f"✅ Wrote {CFG_PATH.name}")

    # Re-verify
    cfg2 = json.loads(CFG_PATH.read_text())
    sb2 = cfg2["search_bounds"]
    print()
    print("Post-write verification:")
    print(f"  window_size.min        = {sb2['window_size']['min']}        (target >= {args.min_window})")
    print(f"  forward_threshold.min  = {sb2['forward_threshold']['min']}     (PRESERVED — Optuna will explore)")
    print(f"  forward_threshold.max  = {sb2['forward_threshold']['max']}    (PRESERVED — Optuna will explore)")
    print(f"  reverse_threshold.min  = {sb2['reverse_threshold']['min']}     (PRESERVED — Optuna will explore)")
    print(f"  reverse_threshold.max  = {sb2['reverse_threshold']['max']}    (PRESERVED — Optuna will explore)")

    if sb2['window_size']['min'] < args.min_window:
        print()
        print(f"❌ POST-WRITE CHECK FAILED: window_size.min still {sb2['window_size']['min']} (< {args.min_window})")
        return 1

    print()
    print("✅ Target satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
