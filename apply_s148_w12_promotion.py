#!/usr/bin/env python3
"""
apply_s148_w12_promotion.py
============================
S148 Run-1 ruling: promote window_size default from 8 → 12.

Session ruling:
  "Run 1 = W12 / threshold 0.30 / bounds 0.30–0.75.
   Rationale: orders-of-magnitude forward-noise reduction with
   empirically validated signal retention."

Noise comparison at 1B seeds / 50 trials:
  W8  + T0.30 → ~680,000 false fwd survivors/trial → ~34M total
  W12 + T0.30 → ~5 false fwd survivors/trial        → ~250 total

Files patched:
  1. distributed_config.json
       search_bounds.window_size.default  (new field) = 12
  2. agent_manifests/window_optimizer.json
       parameter_bounds.window_size.default  8 → 12 (informational)
  3. baselines/baseline_window_thresholds.json
       window_size already 12 (written by apply_s148_threshold_calibration.py)
       → confirm, add run1_ruling annotation

Usage:
  python3 apply_s148_w12_promotion.py [--dry-run]
"""

import argparse
import json
import os
import shutil

DRY_RUN = False


def log(msg):
    print(msg)


def backup(path):
    bak = path + ".bak_s148_w12"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        log(f"  backup → {bak}")


def write_json(path, obj):
    if DRY_RUN:
        log(f"  [DRY-RUN] would write {path}")
    else:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
        log(f"  wrote {path}")


# ---------------------------------------------------------------------------
# 1. distributed_config.json — add search_bounds.window_size.default = 12
# ---------------------------------------------------------------------------

def patch_distributed_config(path):
    log(f"\n[1] {path}")
    backup(path)

    with open(path) as f:
        cfg = json.load(f)

    sb = cfg.get("search_bounds", {})
    ws = sb.get("window_size", {})

    old_default = ws.get("default", "<absent>")
    ws["default"] = 12
    ws["_calibration_note"] = (
        "S148 Run-1 ruling: W12 is empirically preferred production baseline. "
        "Optuna still explores full [min,max] range. "
        "W12+T0.30 gives ~5 false fwd survivors/200k vs ~272 at W8/T0.25."
    )
    sb["window_size"] = ws
    cfg["search_bounds"] = sb

    log(f"  search_bounds.window_size.default: {old_default} → 12")
    log(f"  search_bounds.window_size._calibration_note: added")
    write_json(path, cfg)
    return True


# ---------------------------------------------------------------------------
# 2. agent_manifests/window_optimizer.json — informational default
# ---------------------------------------------------------------------------

def patch_manifest(path):
    log(f"\n[2] {path}")
    backup(path)

    with open(path) as f:
        m = json.load(f)

    pb = m.setdefault("parameter_bounds", {})
    ws = pb.get("window_size", {})

    old_default = ws.get("default", "<absent>")
    ws["default"] = 12
    ws["_calibration_note"] = (
        "S148 Run-1 ruling: W12 promoted as production default. "
        "Bounds [2,50] unchanged — Optuna explores freely."
    )
    pb["window_size"] = ws
    m["parameter_bounds"] = pb

    log(f"  parameter_bounds.window_size.default: {old_default} → 12")
    write_json(path, m)
    return True


# ---------------------------------------------------------------------------
# 3. baselines/baseline_window_thresholds.json — confirm + annotate
# ---------------------------------------------------------------------------

def patch_baseline(path):
    log(f"\n[3] {path}")

    if not os.path.exists(path):
        log(f"  WARNING: {path} not found — was apply_s148_threshold_calibration.py run first?")
        # Create it
        os.makedirs(os.path.dirname(path), exist_ok=True)
        baseline = {}
    else:
        backup(path)
        with open(path) as f:
            baseline = json.load(f)

    old_ws = baseline.get("window_size", "<absent>")

    baseline["window_size"] = 12
    baseline["forward_threshold"] = 0.30
    baseline["reverse_threshold"] = 0.30
    baseline["run1_ruling"] = (
        "S148 Run-1 ruling: W12 + T0.30 is first production sweep under "
        "empirical threshold governance. Promoted from synthetic-era W8/T0.25."
    )
    baseline["calibration_source"] = "THRESHOLD_CALIBRATION_FINDINGS_S148.md"
    baseline["calibration_date"] = "2026-03-19"

    log(f"  window_size: {old_ws} → 12 (confirmed)")
    log(f"  run1_ruling annotation: added")
    write_json(path, baseline)
    return True


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(base_dir):
    log("\n[VERIFY]")
    errors = []

    # distributed_config
    with open(os.path.join(base_dir, "distributed_config.json")) as f:
        cfg = json.load(f)
    val = cfg.get("search_bounds", {}).get("window_size", {}).get("default")
    ok = "✓" if val == 12 else "✗"
    log(f"  distributed_config search_bounds.window_size.default = {val} {ok}")
    if val != 12:
        errors.append(f"distributed_config window_size default: expected 12, got {val}")

    # manifest
    with open(os.path.join(base_dir, "agent_manifests", "window_optimizer.json")) as f:
        m = json.load(f)
    val = m.get("parameter_bounds", {}).get("window_size", {}).get("default")
    ok = "✓" if val == 12 else "✗"
    log(f"  manifest parameter_bounds.window_size.default = {val} {ok}")
    if val != 12:
        errors.append(f"manifest window_size default: expected 12, got {val}")

    # baseline
    bl_path = os.path.join(base_dir, "baselines", "baseline_window_thresholds.json")
    if os.path.exists(bl_path):
        with open(bl_path) as f:
            bl = json.load(f)
        val = bl.get("window_size")
        ok = "✓" if val == 12 else "✗"
        log(f"  baseline window_size = {val} {ok}")
        has_ruling = "run1_ruling" in bl
        log(f"  baseline run1_ruling present: {has_ruling} {'✓' if has_ruling else '✗'}")
        if val != 12:
            errors.append(f"baseline window_size: expected 12, got {val}")
    else:
        log("  baseline file: NOT FOUND ✗")
        errors.append("baseline file not found")

    # Confirm threshold bounds still correct (regression check)
    fwd = cfg.get("search_bounds", {}).get("forward_threshold", {})
    rev = cfg.get("search_bounds", {}).get("reverse_threshold", {})
    for label, entry in [("fwd", fwd), ("rev", rev)]:
        for sub, expected in [("min", 0.3), ("max", 0.75), ("default", 0.3)]:
            val = entry.get(sub)
            ok = "✓" if val == expected else "✗"
            log(f"  threshold_regression {label}.{sub} = {val} {ok}")
            if val != expected:
                errors.append(f"threshold regression {label}.{sub}: expected {expected}, got {val}")

    if errors:
        log(f"\n  FAIL — {len(errors)} error(s):")
        for e in errors:
            log(f"    • {e}")
        return False
    log("\n  PASS — all checks green")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global DRY_RUN
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()
    DRY_RUN = args.dry_run
    base = args.base_dir

    print(f"{'[DRY-RUN] ' if DRY_RUN else ''}S148 W12 Promotion Patch")
    print("Session ruling: Run 1 = W12 / T0.30 / bounds [0.30, 0.75]")
    print("=" * 60)

    results = []
    results.append(patch_distributed_config(
        os.path.join(base, "distributed_config.json")))
    results.append(patch_manifest(
        os.path.join(base, "agent_manifests", "window_optimizer.json")))
    results.append(patch_baseline(
        os.path.join(base, "baselines", "baseline_window_thresholds.json")))

    if not DRY_RUN:
        passed = verify(base)
    else:
        log("\n[VERIFY] skipped in dry-run mode")
        passed = all(results)

    print("\n" + "=" * 60)
    if passed and all(results):
        print(f"✓ W12 promotion COMPLETE — {sum(results)}/3 files patched")
        print()
        print("Run 1 is cleared for launch:")
        print("  W12 + T0.30 + Optuna bounds [0.30, 0.75]")
        print("  Expected false fwd survivors: ~5/trial vs ~680,000 (W8/T0.30)")
        print()
        print("Commit command:")
        print("  git add distributed_config.json \\")
        print("        agent_manifests/window_optimizer.json \\")
        print("        baselines/baseline_window_thresholds.json")
        print("  git commit -m 'fix(s148): W12 promotion — Run 1 under empirical threshold governance'")
        print("  git push origin main && git push public main")
        print()
        print("Then launch:")
        print("  bash sweep_run1.sh")
    else:
        print(f"✗ W12 promotion INCOMPLETE — review errors above")


if __name__ == "__main__":
    main()
