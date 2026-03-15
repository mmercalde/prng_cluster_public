#!/usr/bin/env python3
"""
fix_s145r1_pruning_scope.py
============================
Fix NameError: enable_pruning not in scope inside run_bidirectional_test().

The function is standalone — enable_pruning must be passed as a parameter.
Two anchors:
  A) Add enable_pruning=False to run_bidirectional_test() signature
  B) Pass enable_pruning=enable_pruning in the test_config call site
"""

import sys
import shutil
from pathlib import Path

DRY_RUN = '--dry-run' in sys.argv
PROJECT_ROOT = Path('/home/michael/distributed_prng_analysis')
TARGET = PROJECT_ROOT / 'window_optimizer_integration_final.py'

def read(p): return Path(p).read_text(encoding='utf-8')
def write(p, c):
    if DRY_RUN:
        print(f"  [DRY-RUN] would write {p.name}")
        return
    Path(p).write_text(c, encoding='utf-8')

def backup(path):
    bak = Path(str(path) + '.s145r1_scope_backup')
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

print("fix_s145r1_pruning_scope.py")
print("=" * 55)
if DRY_RUN:
    print("MODE: DRY RUN")

content = read(TARGET)
original_lines = len(content.splitlines())
print(f"\nTarget: {TARGET.name} ({original_lines} lines)")
backup(TARGET)
all_ok = True

# ── Patch A: Add enable_pruning to run_bidirectional_test signature ───────
OLD_A = (
    "                           trial_number: int = 0,\n"
    "                           accumulator: Dict[str, List] = None,\n"
    "                           optuna_trial=None) -> TestResult:  # S115 M2"
)
NEW_A = (
    "                           trial_number: int = 0,\n"
    "                           accumulator: Dict[str, List] = None,\n"
    "                           optuna_trial=None,\n"
    "                           enable_pruning: bool = False) -> TestResult:  # S115 M2, [S145-R1]"
)

content, ok = apply_patch(content, OLD_A, NEW_A,
    "add enable_pruning param to run_bidirectional_test signature")
all_ok = all_ok and ok

# ── Patch B: Pass enable_pruning at the call site in test_config ──────────
OLD_B = (
    "                accumulator=survivor_accumulator,\n"
    "                optuna_trial=optuna_trial          # S119 Gap5\n"
    "            )"
)
NEW_B = (
    "                accumulator=survivor_accumulator,\n"
    "                optuna_trial=optuna_trial,         # S119 Gap5\n"
    "                enable_pruning=enable_pruning      # [S145-R1] closure var\n"
    "            )"
)

content, ok = apply_patch(content, OLD_B, NEW_B,
    "pass enable_pruning from closure into run_bidirectional_test")
all_ok = all_ok and ok

if all_ok:
    write(TARGET, content)
    new_lines = len(content.splitlines())
    print(f"  Lines: {original_lines} → {new_lines} (+{new_lines - original_lines})")

print("\n" + "=" * 55)
if all_ok:
    print("✅ ALL PATCHES APPLIED")
    print()
    print("Clean and re-run smoke tests:")
    print("  ssh rzeus \"rm -f ~/distributed_prng_analysis/bidirectional_survivors*.json \\")
    print("      ~/distributed_prng_analysis/bidirectional_survivors*.npz \\")
    print("      ~/distributed_prng_analysis/optimal_window_config.json\"")
    print()
    print("  ssh rzeus \"cd ~/distributed_prng_analysis && \\")
    print("      source ~/venvs/torch/bin/activate && \\")
    print("      bash s145r1_smoke_tests.sh\"")
else:
    print("⚠️  PATCHES FAILED")
    sys.exit(1)
