#!/usr/bin/env python3
"""
apply_s120_manifest_fix.py
==========================
Fixes window_optimizer.json manifest so WATCHER builds correct CLI command.

Problems fixed:
1. window_trials → trials  (WATCHER does key.replace("_","-") → --window-trials, but CLI expects --trials)
2. enable_pruning missing from default_params → silently dropped by step-scoped filter
3. n_parallel missing from default_params → silently dropped by step-scoped filter

Also updates parameter_bounds to rename window_trials → trials entry.
"""

import json
import shutil
from datetime import datetime

MANIFEST = "agent_manifests/window_optimizer.json"
BACKUP = f"{MANIFEST}.bak_s120_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

print("=" * 60)
print("S120: window_optimizer.json manifest fix")
print("=" * 60)

# Read
with open(MANIFEST) as f:
    manifest = json.load(f)

# Backup
shutil.copy(MANIFEST, BACKUP)
print(f"  📦 Backup: {BACKUP}")

# --- Fix 1: rename window_trials → trials in default_params ---
dp = manifest["default_params"]
if "window_trials" in dp:
    dp["trials"] = dp.pop("window_trials")
    print("  ✅ Fix 1: default_params window_trials → trials")
elif "trials" in dp:
    print("  ℹ️  Fix 1: trials already present in default_params")
else:
    dp["trials"] = 100
    print("  ✅ Fix 1: added trials=100 to default_params")

# --- Fix 2: add enable_pruning to default_params ---
if "enable_pruning" not in dp:
    dp["enable_pruning"] = False
    print("  ✅ Fix 2: added enable_pruning=false to default_params")
else:
    print("  ℹ️  Fix 2: enable_pruning already in default_params")

# --- Fix 3: add n_parallel to default_params ---
if "n_parallel" not in dp:
    dp["n_parallel"] = 1
    print("  ✅ Fix 3: added n_parallel=1 to default_params")
else:
    print("  ℹ️  Fix 3: n_parallel already in default_params")

# --- Fix 4: rename window_trials → trials in parameter_bounds ---
pb = manifest.get("parameter_bounds", {})
if "window_trials" in pb:
    pb["trials"] = pb.pop("window_trials")
    print("  ✅ Fix 4: parameter_bounds window_trials → trials")
else:
    print("  ℹ️  Fix 4: parameter_bounds already uses trials or missing")

# --- Fix 5: update actions args_map to match ---
for action in manifest.get("actions", []):
    am = action.get("args_map", {})
    if "trials" in am and am["trials"] == "window_trials":
        am["trials"] = "trials"
        print("  ✅ Fix 5: actions[].args_map trials mapping updated")

# Write
with open(MANIFEST, "w") as f:
    json.dump(manifest, f, indent=2)

print()
print("── Verification ──")
with open(MANIFEST) as f:
    m = json.load(f)
dp = m["default_params"]
checks = [
    ("trials in default_params",        "trials" in dp),
    ("window_trials NOT in default_params", "window_trials" not in dp),
    ("enable_pruning in default_params", "enable_pruning" in dp),
    ("n_parallel in default_params",     "n_parallel" in dp),
    ("resume_study in default_params",   "resume_study" in dp),
]
all_pass = True
for label, result in checks:
    mark = "✅" if result else "❌"
    print(f"  {mark} {label}")
    if not result:
        all_pass = False

print()
if all_pass:
    print("✅ All fixes applied. WATCHER will now build:")
    print("   python3 window_optimizer.py --lottery-file daily3.json")
    print("   --strategy bayesian --max-seeds 10000000 --prng-type java_lcg")
    print("   --output optimal_window_config.json --test-both-modes")
    print("   --resume-study --study-name window_opt_1772507547")
    print("   --trials 50 --enable-pruning --n-parallel 2")
else:
    print("❌ Some fixes failed — check output above")
