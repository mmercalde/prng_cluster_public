#!/usr/bin/env python3
"""
fix_s145r1_pruning_crash.py
===========================
Two fixes for S145-R1 smoke test failures:

Fix A — window_optimizer_bayesian.py:
  Guard save_best_so_far callback against ValueError when all trials
  are pruned and study has no completed trials (study.best_trial raises
  ValueError: Record does not exist rather than returning None).

Fix B — agent_manifests/window_optimizer.json:
  Disable enable_pruning for smoke tests — re-enable before production runs.
  Pruning is correct for production (saves ~17min per zero-survivor trial)
  but kills smoke tests at 5M seeds where most trials produce zero survivors.

Usage:
    python3 fix_s145r1_pruning_crash.py [--dry-run]
"""

import sys
import json
from pathlib import Path

DRY_RUN = '--dry-run' in sys.argv
PROJECT_ROOT = Path('/home/michael/distributed_prng_analysis')

def read(p): return Path(p).read_text(encoding='utf-8')
def write(p, c):
    if DRY_RUN:
        print(f"  [DRY-RUN] would write {p}")
        return
    Path(p).write_text(c, encoding='utf-8')

print("fix_s145r1_pruning_crash.py")
print("=" * 55)
if DRY_RUN:
    print("MODE: DRY RUN")

# ── FIX A — window_optimizer_bayesian.py ─────────────────
print("\n[Fix A] window_optimizer_bayesian.py — guard best_trial call")

BAYESIAN = PROJECT_ROOT / 'window_optimizer_bayesian.py'
content = read(BAYESIAN)
original_lines = len(content.splitlines())

OLD_A = '        if study.best_trial is not None:'
NEW_A = '''        # [S145-R1] Guard against ValueError when all trials are pruned
        # Optuna raises ValueError (not returns None) when no completed trials exist
        try:
            _best_trial_exists = study.best_trial is not None
        except ValueError:
            _best_trial_exists = False
        if _best_trial_exists:'''

if OLD_A not in content:
    print("  ❌ ANCHOR NOT FOUND — skipping Fix A")
    fix_a_ok = False
else:
    content = content.replace(OLD_A, NEW_A, 1)
    print("  ✅ Patched: best_trial ValueError guard")
    write(BAYESIAN, content)
    new_lines = len(read(BAYESIAN).splitlines()) if not DRY_RUN else original_lines + 4
    print(f"  Lines: {original_lines} → {new_lines} (+{new_lines - original_lines})")
    fix_a_ok = True

# ── FIX B — window_optimizer.json ────────────────────────
print("\n[Fix B] window_optimizer.json — disable pruning for smoke tests")
print("  NOTE: Re-enable (true) before production sweep runs")

MANIFEST = PROJECT_ROOT / 'agent_manifests/window_optimizer.json'
content = read(MANIFEST)

OLD_B = '"enable_pruning": true,'
NEW_B = '"enable_pruning": false,'

if OLD_B not in content:
    print("  ❌ ANCHOR NOT FOUND — pruning may already be false")
    fix_b_ok = False
else:
    content = content.replace(OLD_B, NEW_B, 1)
    # Validate JSON
    try:
        json.loads(content)
        print("  ✅ JSON valid")
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON invalid: {e}")
        sys.exit(1)
    write(MANIFEST, content)
    print("  ✅ Patched: enable_pruning true → false (smoke test mode)")
    fix_b_ok = True

# ── Summary ───────────────────────────────────────────────
print("\n" + "=" * 55)
if fix_a_ok and fix_b_ok:
    print("✅ Both fixes applied")
    print("\nNow re-run smoke tests:")
    print("  bash s145r1_smoke_tests.sh")
    print()
    print("After smoke tests pass — re-enable pruning for production:")
    print('  ssh rzeus "sed -i \'s/\"enable_pruning\": false,/\"enable_pruning\": true,/\' '
          '~/distributed_prng_analysis/agent_manifests/window_optimizer.json"')
else:
    print("⚠️  Some fixes failed — check anchors above")
    sys.exit(1)
