#!/usr/bin/env python3
"""
apply_s149_npz_checkpoint_fix2.py
===================================
Fix the wiring bug in apply_s149_npz_checkpoint.py.

Bug: _survivor_accumulator was set on the WindowOptimizer instance but
BayesianOptimization.optimize() reads it from self (the BayesianOptimization
instance) — a different object. So getattr(self, '_survivor_accumulator', None)
always returns None and the checkpoint never fires.

Fix: Set _survivor_accumulator on the strategy object (BayesianOptimization)
instead of the optimizer object (WindowOptimizer).

Change: window_optimizer_integration_final.py
  - Remove: optimizer._survivor_accumulator = survivor_accumulator
  - Add:    strategy._survivor_accumulator = survivor_accumulator

Usage:
  python3 apply_s149_npz_checkpoint_fix2.py [--dry-run]
"""

import argparse
import shutil
import os
import sys

DRY_RUN = False


def log(msg):
    print(msg)


def backup(path, tag="s149_ckpt_fix2"):
    if DRY_RUN:
        log(f"  [DRY-RUN] would create backup {path}.bak_{tag}")
        return
    bak = f"{path}.bak_{tag}"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        log(f"  backup → {bak}")


def main():
    global DRY_RUN
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()
    DRY_RUN = args.dry_run

    print(f"{'[DRY-RUN] ' if DRY_RUN else ''}S149 NPZ Checkpoint Wiring Fix")
    print("Fix: set _survivor_accumulator on strategy not optimizer")
    print("=" * 60)

    path = os.path.join(args.base_dir, "window_optimizer_integration_final.py")
    backup(path)

    with open(path) as f:
        content = f.read()

    old = "        optimizer._survivor_accumulator = survivor_accumulator  # [S149] per-trial NPZ checkpoint"
    new = "        strategy._survivor_accumulator = survivor_accumulator   # [S149] set on BayesianOptimization instance"

    if old not in content:
        log("  ERROR: anchor not found — already fixed or drifted")
        # Check if new version is already there
        if "strategy._survivor_accumulator" in content:
            log("  New version already present — fix already applied")
            return True
        return False

    content = content.replace(old, new, 1)
    log("  patched: _survivor_accumulator set on strategy (BayesianOptimization)")

    if not DRY_RUN:
        with open(path, "w") as f:
            f.write(content)
        log(f"  wrote {path}")
    else:
        log(f"  [DRY-RUN] would write {path}")

    # Verify
    with open(path) as f:
        check = f.read()
    ok = "strategy._survivor_accumulator = survivor_accumulator" in check
    log(f"\n  verify: {'✓ PASS' if ok else '✗ FAIL'}")

    print("\n" + "=" * 60)
    if ok:
        print("✓ Wiring fix COMPLETE")
        print()
        print("Commit:")
        print("  git add window_optimizer_integration_final.py")
        print("  git commit -m 'fix(s149): NPZ checkpoint wiring — set on strategy not optimizer'")
        print("  git push origin main && git push public main")
        print()
        print("The running sweep will pick this up on the NEXT resume.")
        print("Current trial survivors are still accumulating in memory.")
    else:
        print("✗ Fix FAILED")

    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
