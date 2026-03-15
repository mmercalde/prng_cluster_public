#!/usr/bin/env python3
"""
fix_s145r1_pruning_gate.py
==========================
Two fixes to window_optimizer_integration_final.py:

Fix A — Primary pruning block (line ~323):
  The TrialPruned raise fires whenever forward_count==0 regardless of
  enable_pruning flag. Gate it on enable_pruning parameter flowing
  through from the caller.

Fix B — KeyError 'window_size' (line ~1234):
  Guard against empty best_config dict when all trials pruned.

The enable_pruning parameter already flows through the call chain:
  window_optimizer.py args.enable_pruning
  → run_bayesian_optimization(enable_pruning=...)
  → coordinator.optimize_window(enable_pruning=...)
  → BayesianOptimization(enable_pruning=...)
  → objective function via closure

We just need to capture it in the objective closure and use it
to gate the primary prune block.
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
    bak = Path(str(path) + '.s145r1_prunegate_backup')
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

print("fix_s145r1_pruning_gate.py")
print("=" * 55)
if DRY_RUN:
    print("MODE: DRY RUN")

content = read(TARGET)
original_lines = len(content.splitlines())
print(f"\nTarget: {TARGET.name} ({original_lines} lines)")
backup(TARGET)

all_ok = True

# ── Fix A: Gate primary prune block on enable_pruning ─────────────────────
# The objective function is defined inside optimize_window which has
# enable_pruning in its signature — it's available via closure.
OLD_A = (
    "    # S115 M2: prune dead trials (forward==0) before expensive reverse sieve\n"
    "    if optuna_trial is not None:\n"
    "        if not _OPTUNA_AVAILABLE:\n"
    "            print(\"      ⚠️  optuna_trial passed but Optuna not installed — pruning disabled.\")\n"
    "        elif len(forward_records) == 0:\n"
    "            print(f\"      ✂️  PRUNED  trial={optuna_trial.number}  \"\n"
    "                  f\"window={config.window_size}  offset={config.offset}  \"\n"
    "                  f\"skip={config.skip_min}-{config.skip_max}  forward_count=0\")\n"
    "            raise _optuna_module.exceptions.TrialPruned()"
)

NEW_A = (
    "    # S115 M2: prune dead trials (forward==0) before expensive reverse sieve\n"
    "    # [S145-R1] Gate on enable_pruning — when False, always run reverse sieve\n"
    "    if optuna_trial is not None and enable_pruning:\n"
    "        if not _OPTUNA_AVAILABLE:\n"
    "            print(\"      ⚠️  optuna_trial passed but Optuna not installed — pruning disabled.\")\n"
    "        elif len(forward_records) == 0:\n"
    "            print(f\"      ✂️  PRUNED  trial={optuna_trial.number}  \"\n"
    "                  f\"window={config.window_size}  offset={config.offset}  \"\n"
    "                  f\"skip={config.skip_min}-{config.skip_max}  forward_count=0\")\n"
    "            raise _optuna_module.exceptions.TrialPruned()"
)

content, ok = apply_patch(content, OLD_A, NEW_A,
    "gate primary prune block on enable_pruning flag")
all_ok = all_ok and ok

# ── Fix B: Guard empty best_config KeyError ───────────────────────────────
OLD_B = (
    "        best = results['best_config']\n"
    "        print(f\"  Window size: {best['window_size']}\")\n"
    "        print(f\"  Offset: {best['offset']}\")\n"
    "        print(f\"  Sessions: {', '.join(best['sessions'])}\")\n"
    "        print(f\"  Skip range: [{best['skip_min']}, {best['skip_max']}]\")\n"
    "        print(f\"  Bidirectional survivors: {results['best_result']['bidirectional_count']:,}\")\n"
    "        print(f\"{'='*80}\\n\")"
)

NEW_B = (
    "        best = results.get('best_config', {})\n"
    "        # [S145-R1] Guard: best_config empty when all trials pruned\n"
    "        if best and 'window_size' in best:\n"
    "            print(f\"  Window size: {best['window_size']}\")\n"
    "            print(f\"  Offset: {best['offset']}\")\n"
    "            print(f\"  Sessions: {', '.join(best.get('sessions', []))}\")\n"
    "            print(f\"  Skip range: [{best['skip_min']}, {best['skip_max']}]\")\n"
    "            print(f\"  Bidirectional survivors: {results['best_result'].get('bidirectional_count', 0):,}\")\n"
    "        else:\n"
    "            print(f\"  ⚠️  All trials pruned — no survivors found in this seed range\")\n"
    "            print(f\"  Coverage tracker will advance seed_start on next run\")\n"
    "        print(f\"{'='*80}\\n\")"
)

content, ok = apply_patch(content, OLD_B, NEW_B,
    "guard empty best_config KeyError when all trials pruned")
all_ok = all_ok and ok

if all_ok:
    write(TARGET, content)
    new_lines = len(content.splitlines())
    print(f"  Lines: {original_lines} → {new_lines} (+{new_lines - original_lines})")

print("\n" + "=" * 55)
if all_ok:
    print("✅ ALL PATCHES APPLIED")
    print()
    print("enable_pruning=false in manifest → primary prune block skipped")
    print("enable_pruning=true in manifest  → primary prune block active")
    print()
    print("Re-run smoke tests:")
    print("  bash s145r1_smoke_tests.sh")
else:
    print("⚠️  PATCHES FAILED — check anchors above")
    sys.exit(1)
