#!/usr/bin/env python3
"""
apply_s148_manifest_update.py
================================
Update informational threshold fields in agent_manifests/window_optimizer.json
to reflect S148 empirical calibration.

Only touches parameter_bounds[forward/reverse_threshold].default and _bounds_reference
(informational annotations — NOT runtime values, which come from distributed_config.json).

Usage:
  python3 apply_s148_manifest_update.py [--dry-run]
"""
import argparse, json, shutil, os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()

    path = os.path.join(args.base_dir, "agent_manifests", "window_optimizer.json")
    bak  = path + ".bak_s148"

    print(f"{'[DRY-RUN] ' if args.dry_run else ''}Manifest informational threshold update")
    print(f"  file: {path}")

    if not os.path.exists(path):
        print(f"  ERROR: {path} not found"); return

    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"  backup → {bak}")

    with open(path) as f:
        m = json.load(f)

    pb = m.setdefault("parameter_bounds", {})
    changes = 0

    for key in ("forward_threshold", "reverse_threshold"):
        entry = pb.get(key, {})
        old_def = entry.get("default")
        old_ref = entry.get("_bounds_reference")

        entry["default"] = 0.30
        entry["_bounds_reference"] = "[0.30, 0.75] - see distributed_config.json (S148 empirical calibration)"
        pb[key] = entry

        print(f"  {key}: default {old_def}→0.30, _bounds_reference updated")
        changes += 1

    print(f"  {changes}/2 entries updated")

    if not args.dry_run:
        with open(path, "w") as f:
            json.dump(m, f, indent=2)
        print(f"  wrote {path}")
    else:
        print(f"  [DRY-RUN] would write {path}")

    print("  DONE")

if __name__ == "__main__":
    main()
