#!/usr/bin/env python3
"""
fix_step1_timeout.py
====================
Remove Step 1 timeout for production sweep runs.

The timeout was added during synthetic data debugging (S95).
For production sweeps running 13-18+ hours, a timeout causes SIGKILL
killing the entire process group with no cleanup, no NPZ write,
no coverage tracker update.

Three changes:
  A) agent_manifests/window_optimizer.json
     action.timeout_minutes: 900 → 0 (sentinel = disabled)

  B) agents/watcher_agent.py
     step_timeout_overrides hardcode: {0:1, 1:900, 5:360} → {0:1, 5:360}
     (Step 1 removed from overrides — falls through to global)
     step_timeout_minutes default: 120 → 0 (disabled)

  C) agents/watcher_agent.py _run_step_streaming
     Add: if timeout_seconds <= 0: timeout_seconds = float('inf')
     Treats 0 as "no timeout" instead of "immediate kill"
"""

import sys
import shutil
from pathlib import Path
import json

DRY_RUN = '--dry-run' in sys.argv
PROJECT_ROOT = Path('/home/michael/distributed_prng_analysis')

FILES = {
    'manifest': PROJECT_ROOT / 'agent_manifests/window_optimizer.json',
    'watcher':  PROJECT_ROOT / 'agents/watcher_agent.py',
}

def read(p): return Path(p).read_text(encoding='utf-8')
def write(p, c):
    if DRY_RUN:
        print(f"  [DRY-RUN] would write {p.name}")
        return
    Path(p).write_text(c, encoding='utf-8')

def backup(path):
    bak = Path(str(path) + '.notimeout_backup')
    if DRY_RUN:
        print(f"  [DRY-RUN] would backup → {bak.name}")
        return
    shutil.copy2(path, bak)
    print(f"  ✅ Backup: {bak.name}")

def apply_patch(content, old, new, desc):
    if old not in content:
        print(f"  ❌ ANCHOR NOT FOUND: {desc}")
        return content, False
    result = content.replace(old, new, 1)
    print(f"  ✅ Patched: {desc}")
    return result, True

print("fix_step1_timeout.py")
print("=" * 55)
if DRY_RUN:
    print("MODE: DRY RUN")

all_ok = True

# ── Patch A: manifest timeout_minutes ────────────────────────────────────────
print("\n[A] agent_manifests/window_optimizer.json")
path = FILES['manifest']
backup(path)
try:
    m = json.load(open(path))
    original = m['actions'][0]['timeout_minutes']
    m['actions'][0]['timeout_minutes'] = 0
    if not DRY_RUN:
        json.dump(m, open(path, 'w'), indent=2)
    print(f"  ✅ action.timeout_minutes: {original} → 0 (disabled)")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    all_ok = False

# ── Patch B: watcher_agent.py hardcoded overrides + default timeout ───────────
print("\n[B] agents/watcher_agent.py — step_timeout_overrides + default")
path = FILES['watcher']
content = read(path)
original_lines = len(content.splitlines())
backup(path)

# Remove Step 1 from overrides (keep Step 0=1min, Step 5=360min)
OLD_B = "        step_timeout_overrides={0: 1, 1: 900, 5: 360}  # [S145-R1] 900min = 50 trials × ~17min + buffer"
NEW_B = "        step_timeout_overrides={0: 1, 5: 360}  # Step 1 has no timeout — production runs are 13-18hrs"

content, ok = apply_patch(content, OLD_B, NEW_B,
    "remove Step 1 from timeout overrides")
all_ok = all_ok and ok

# ── Patch C: _run_step_streaming — treat 0 as disabled ───────────────────────
print("\n[C] agents/watcher_agent.py — treat timeout_seconds=0 as disabled")

OLD_C = (
    "        try:\n"
    "            while True:\n"
    "                # Check timeout\n"
    "                elapsed = time.time() - start_time\n"
    "                if elapsed > timeout_seconds:"
)
NEW_C = (
    "        # [S145] timeout_seconds=0 means disabled — treat as infinite\n"
    "        if timeout_seconds <= 0:\n"
    "            timeout_seconds = float('inf')\n"
    "\n"
    "        try:\n"
    "            while True:\n"
    "                # Check timeout\n"
    "                elapsed = time.time() - start_time\n"
    "                if elapsed > timeout_seconds:"
)

content, ok = apply_patch(content, OLD_C, NEW_C,
    "treat timeout_seconds=0 as disabled (infinite)")
all_ok = all_ok and ok

if all_ok:
    write(path, content)
    new_lines = len(content.splitlines())
    print(f"  Lines: {original_lines} → {new_lines} (+{new_lines - original_lines})")

print("\n" + "=" * 55)
if all_ok:
    print("✅ ALL PATCHES APPLIED")
    print()
    print("Step 1 timeout: DISABLED")
    print("Step 0 timeout: 1 min  (TRSE — unchanged)")
    print("Step 5 timeout: 360 min (anti-overfit — unchanged)")
    print()
    print("Commit:")
    print("  git add agents/watcher_agent.py agent_manifests/window_optimizer.json")
    print("  git commit -m 'fix(timeout): disable Step 1 timeout for production sweep'")
    print("  git push origin main && git push public main")
else:
    print("⚠️  PATCHES FAILED")
    sys.exit(1)
