#!/usr/bin/env python3
"""
apply_s148_threshold_calibration.py
====================================
S148 threshold calibration patch — empirically-grounded defaults.

Based on THRESHOLD_CALIBRATION_FINDINGS_S148.md:
  - threshold 0.25 → 0.30 (empirical zero-noise floor, window=8+)
  - Optuna min  0.15 → 0.30 (eliminates noise-dominated search region)
  - Optuna max  0.60 → 0.75 (known seed survives to 0.75 in all experiments)

Files patched:
  1. persistent_worker_coordinator.py  — functional defaults (lines 73, 964)
  2. window_optimizer.py               — WindowSearchBounds (lines 123-129)
  3. distributed_config.json           — Optuna search_bounds
  4. baselines/baseline_window_thresholds.json  — NEW file (created)
  5. THRESHOLD_GOVERNANCE.md           — change history append (created if absent)

Usage:
  python3 apply_s148_threshold_calibration.py [--dry-run]
"""

import argparse
import json
import os
import re
import shutil
from datetime import datetime

DRY_RUN = False
CHANGES = []

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def backup(path):
    bak = path + ".bak_s148"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        log(f"  backup → {bak}")


def log(msg):
    CHANGES.append(msg)
    print(msg)


def read(path):
    with open(path) as f:
        return f.read()


def write(path, content):
    if DRY_RUN:
        log(f"  [DRY-RUN] would write {path}")
    else:
        with open(path, "w") as f:
            f.write(content)
        log(f"  wrote {path}")


def replace_exact(content, old, new, label):
    if old not in content:
        log(f"  ERROR: anchor not found — {label!r}")
        return content, False
    count = content.count(old)
    if count != 1:
        log(f"  WARNING: {count} occurrences of {label!r} — replacing first only")
    result = content.replace(old, new, 1)
    log(f"  patched: {label}")
    return result, True


# ---------------------------------------------------------------------------
# 1. persistent_worker_coordinator.py
# ---------------------------------------------------------------------------

def patch_pwc(path):
    log(f"\n[1] {path}")
    backup(path)
    content = read(path)
    ok_count = 0

    # Line 73 context: threshold=0.25, in example call block
    content, ok = replace_exact(
        content,
        "    threshold=0.25,\n    window_size=8,\n    output_file=\"results/window_opt_forward_8_43_t1.json\"",
        "    threshold=0.30,\n    window_size=8,\n    output_file=\"results/window_opt_forward_8_43_t1.json\"",
        "example call threshold=0.25→0.30"
    )
    ok_count += ok

    # Line 964 context: smoke test block
    content, ok = replace_exact(
        content,
        "        threshold   = 0.25,\n        window_size = 8,\n        output_file = \"/tmp/pwc_smoke_test.json\"",
        "        threshold   = 0.30,\n        window_size = 8,\n        output_file = \"/tmp/pwc_smoke_test.json\"",
        "smoke test threshold=0.25→0.30"
    )
    ok_count += ok

    log(f"  PWC: {ok_count}/2 substitutions applied")
    write(path, content)
    return ok_count == 2


# ---------------------------------------------------------------------------
# 2. window_optimizer.py — WindowSearchBounds
# ---------------------------------------------------------------------------

def patch_window_optimizer(path):
    log(f"\n[2] {path}")
    backup(path)
    content = read(path)
    ok_count = 0

    patches = [
        ("    min_forward_threshold: float = 0.15",
         "    min_forward_threshold: float = 0.30",
         "min_forward 0.15→0.30"),
        ("    max_forward_threshold: float = 0.60",
         "    max_forward_threshold: float = 0.75",
         "max_forward 0.60→0.75"),
        ("    min_reverse_threshold: float = 0.15",
         "    min_reverse_threshold: float = 0.30",
         "min_reverse 0.15→0.30"),
        ("    max_reverse_threshold: float = 0.60",
         "    max_reverse_threshold: float = 0.75",
         "max_reverse 0.60→0.75"),
        ("    default_forward_threshold: float = 0.25",
         "    default_forward_threshold: float = 0.30",
         "default_forward 0.25→0.30"),
        ("    default_reverse_threshold: float = 0.25",
         "    default_reverse_threshold: float = 0.30",
         "default_reverse 0.25→0.30"),
    ]

    for old, new, label in patches:
        content, ok = replace_exact(content, old, new, label)
        ok_count += ok

    log(f"  window_optimizer: {ok_count}/6 substitutions applied")
    write(path, content)
    return ok_count == 6


# ---------------------------------------------------------------------------
# 3. distributed_config.json — search_bounds
# ---------------------------------------------------------------------------

def patch_distributed_config(path):
    log(f"\n[3] {path}")
    backup(path)

    with open(path) as f:
        cfg = json.load(f)

    sb = cfg.get("search_bounds", {})
    errors = []

    for key in ("forward_threshold", "reverse_threshold"):
        if key not in sb:
            errors.append(f"search_bounds.{key} not found")
            continue
        old_min = sb[key].get("min")
        old_max = sb[key].get("max")
        old_def = sb[key].get("default")
        sb[key]["min"] = 0.30
        sb[key]["max"] = 0.75
        sb[key]["default"] = 0.30
        log(f"  {key}: min {old_min}→0.30, max {old_max}→0.75, default {old_def}→0.30")

    if errors:
        for e in errors:
            log(f"  ERROR: {e}")
        return False

    if not DRY_RUN:
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        log(f"  wrote {path}")
    else:
        log(f"  [DRY-RUN] would write {path}")
    return True


# ---------------------------------------------------------------------------
# 4. baselines/baseline_window_thresholds.json — CREATE
# ---------------------------------------------------------------------------

def create_baseline(base_dir):
    path = os.path.join(base_dir, "baselines", "baseline_window_thresholds.json")
    log(f"\n[4] {path}")

    baseline = {
        "forward_threshold": 0.30,
        "reverse_threshold": 0.30,
        "window_size": 12,
        "skip_max": 200,
        "expected_survivor_band": [1000, 10000],
        "calibration_source": "THRESHOLD_CALIBRATION_FINDINGS_S148.md",
        "calibration_date": "2026-03-19",
        "notes": (
            "Empirically calibrated S148. Window=12 + threshold=0.30 gives "
            "zero-noise at window=12. Window=8 + threshold=0.30 gives near-zero "
            "(permissive). Known seed survives to threshold=0.75 — ceiling safe."
        )
    }

    os.makedirs(os.path.join(base_dir, "baselines"), exist_ok=True)

    if not DRY_RUN:
        with open(path, "w") as f:
            json.dump(baseline, f, indent=2)
        log(f"  created {path}")
    else:
        log(f"  [DRY-RUN] would create {path}")
    return True


# ---------------------------------------------------------------------------
# 5. THRESHOLD_GOVERNANCE.md — append change history entry
# ---------------------------------------------------------------------------

def patch_governance(base_dir):
    path = os.path.join(base_dir, "THRESHOLD_GOVERNANCE.md")
    log(f"\n[5] {path}")

    entry = """
---

## S148 Change History Entry — 2026-03-19

**Session:** S148  
**Author:** Team Alpha  
**Change:** Empirical threshold calibration — synthetic-era defaults replaced.

### What changed
| Parameter | Old | New | Source |
|-----------|-----|-----|--------|
| PWC default threshold | 0.25 | 0.30 | S148 calibration, window=8 skip sweep |
| window_optimizer default_forward_threshold | 0.25 | 0.30 | same |
| window_optimizer default_reverse_threshold | 0.25 | 0.30 | same |
| window_optimizer min_forward_threshold | 0.15 | 0.30 | empirical zero-noise floor |
| window_optimizer min_reverse_threshold | 0.15 | 0.30 | same |
| window_optimizer max_forward_threshold | 0.60 | 0.75 | known seed survives to 0.75 |
| window_optimizer max_reverse_threshold | 0.60 | 0.75 | same |
| distributed_config forward_threshold.min | 0.15 | 0.30 | same as min bounds |
| distributed_config reverse_threshold.min | 0.15 | 0.30 | same |
| distributed_config forward_threshold.max | 0.60 | 0.75 | same as max bounds |
| distributed_config reverse_threshold.max | 0.60 | 0.75 | same |
| distributed_config *.default | 0.25 | 0.30 | same |

### Reference
See `THRESHOLD_CALIBRATION_FINDINGS_S148.md` for full methodology, raw data,
and rationale. The window=12 + threshold=0.30 preferred configuration is
recorded in `baselines/baseline_window_thresholds.json`.

### Invariant preserved
`baseline ∈ [search_min, search_max]`: 0.30 ∈ [0.30, 0.75] ✓
"""

    if os.path.exists(path):
        existing = read(path)
        if "S148 Change History" in existing:
            log("  S148 entry already present — skipping")
            return True
        content = existing + entry
        log("  appending S148 change history entry")
    else:
        content = f"# THRESHOLD_GOVERNANCE.md\n\nThreshold governance model for the distributed PRNG sieve.\n{entry}"
        log("  creating THRESHOLD_GOVERNANCE.md with S148 entry")

    write(path, content)
    return True


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(base_dir):
    log("\n[VERIFY]")
    errors = []

    # PWC: both threshold=0.30 present (smoke + example)
    pwc = read(os.path.join(base_dir, "persistent_worker_coordinator.py"))
    count_30 = len(re.findall(r"threshold\s*=\s*0\.30", pwc))
    count_25 = len(re.findall(r"threshold\s*=\s*0\.25", pwc))
    log(f"  PWC threshold=0.30 occurrences: {count_30} (expect 2)")
    log(f"  PWC threshold=0.25 occurrences: {count_25} (expect 0 in functional code)")
    if count_30 < 2:
        errors.append("PWC: fewer than 2 threshold=0.30 found")

    # window_optimizer: 6 fields correct
    wo = read(os.path.join(base_dir, "window_optimizer.py"))
    for field, expected in [
        ("min_forward_threshold", "0.30"),
        ("max_forward_threshold", "0.75"),
        ("min_reverse_threshold", "0.30"),
        ("max_reverse_threshold", "0.75"),
        ("default_forward_threshold", "0.30"),
        ("default_reverse_threshold", "0.30"),
    ]:
        m = re.search(rf"{field}: float = (\S+)", wo)
        val = m.group(1) if m else "NOT FOUND"
        ok = "✓" if val == expected else "✗"
        log(f"  WO {field} = {val} {ok}")
        if val != expected:
            errors.append(f"WO {field}: expected {expected}, got {val}")

    # distributed_config.json
    with open(os.path.join(base_dir, "distributed_config.json")) as f:
        cfg = json.load(f)
    sb = cfg.get("search_bounds", {})
    for key in ("forward_threshold", "reverse_threshold"):
        for sub, expected in [("min", 0.30), ("max", 0.75), ("default", 0.30)]:
            val = sb.get(key, {}).get(sub)
            ok = "✓" if val == expected else "✗"
            log(f"  cfg {key}.{sub} = {val} {ok}")
            if val != expected:
                errors.append(f"cfg {key}.{sub}: expected {expected}, got {val}")

    # baseline file
    bl_path = os.path.join(base_dir, "baselines", "baseline_window_thresholds.json")
    if os.path.exists(bl_path):
        with open(bl_path) as f:
            bl = json.load(f)
        log(f"  baseline: fwd={bl['forward_threshold']}, rev={bl['reverse_threshold']}, window={bl['window_size']} ✓")
    else:
        log("  baseline file: NOT FOUND ✗")
        errors.append("baseline file not created")

    if errors:
        log(f"\n  FAIL — {len(errors)} error(s):")
        for e in errors:
            log(f"    • {e}")
        return False
    else:
        log(f"\n  PASS — all checks green")
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

    print(f"{'[DRY-RUN] ' if DRY_RUN else ''}S148 Threshold Calibration Patch")
    print("=" * 60)

    results = []
    results.append(patch_pwc(os.path.join(base, "persistent_worker_coordinator.py")))
    results.append(patch_window_optimizer(os.path.join(base, "window_optimizer.py")))
    results.append(patch_distributed_config(os.path.join(base, "distributed_config.json")))
    results.append(create_baseline(base))
    results.append(patch_governance(base))

    if not DRY_RUN:
        passed = verify(base)
    else:
        log("\n[VERIFY] skipped in dry-run mode")
        passed = all(results)

    print("\n" + "=" * 60)
    if passed and all(results):
        print(f"✓ S148 patch COMPLETE — {sum(results)}/5 files patched")
        print()
        print("Next steps:")
        print("  git add persistent_worker_coordinator.py window_optimizer.py \\")
        print("        distributed_config.json baselines/ THRESHOLD_GOVERNANCE.md")
        print("  git commit -m 'fix(s148): empirical threshold calibration — 0.25→0.30, max 0.60→0.75'")
        print("  git push origin main && git push public main")
    else:
        print(f"✗ S148 patch INCOMPLETE — review errors above")


if __name__ == "__main__":
    main()
