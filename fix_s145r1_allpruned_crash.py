#!/usr/bin/env python3
"""
fix_s145r1_allpruned_crash.py
==============================
Fix AttributeError when all Optuna trials are pruned (best_result is None).

Two locations in window_optimizer_bayesian.py need guarding:
  Line ~605: best_result.config.description() — crashes when all pruned
  Line ~619: study.best_trial.number — crashes when no complete trials

Both locations already have the pattern — just need None guards added.
"""

import sys
import shutil
from pathlib import Path

DRY_RUN = '--dry-run' in sys.argv
PROJECT_ROOT = Path('/home/michael/distributed_prng_analysis')
TARGET = PROJECT_ROOT / 'window_optimizer_bayesian.py'

def read(p): return Path(p).read_text(encoding='utf-8')
def write(p, c):
    if DRY_RUN:
        print(f"  [DRY-RUN] would write {p.name}")
        return
    Path(p).write_text(c, encoding='utf-8')

def backup(path):
    bak = Path(str(path) + '.s145r1_allpruned_backup')
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

print("fix_s145r1_allpruned_crash.py")
print("=" * 55)
if DRY_RUN:
    print("MODE: DRY RUN")

content = read(TARGET)
original_lines = len(content.splitlines())
print(f"\nTarget: {TARGET.name} ({original_lines} lines)")
backup(TARGET)

all_ok = True

# ── Patch A: guard best_result None before printing summary ──────────────────
OLD_A = '''        # Print summary
        print(f"\\n{'='*80}")
        print(f"🏆 OPTIMIZATION COMPLETE")
        print(f"   Best score: {best_score:.2f}")
        print(f"   Best config: {best_result.config.description()}")
        print(f"   Bidirectional survivors: {best_result.bidirectional_count}")
        print(f"   📊 Optuna-optimized thresholds:")
        print(f"      Forward threshold: {best_result.config.forward_threshold}")
        print(f"      Reverse threshold: {best_result.config.reverse_threshold}")
        print(f"{'='*80}\\n")
        
        return {
            'strategy': 'optuna_bayesian',
            'best_config': best_result.config.to_dict(),
            'best_result': best_result.to_dict(),
            'best_score': best_score,
            'all_results': [r.to_dict() for r in all_results],
            'iterations': len(all_results),
            'optuna_study': {
                'best_trial': study.best_trial.number,
                'best_value': study.best_value,
                'best_params': study.best_params
            }
        }'''

NEW_A = '''        # Print summary
        print(f"\\n{'='*80}")
        print(f"🏆 OPTIMIZATION COMPLETE")
        print(f"   Best score: {best_score:.2f}")
        # [S145-R1] Guard: best_result is None when all trials pruned
        if best_result is not None:
            print(f"   Best config: {best_result.config.description()}")
            print(f"   Bidirectional survivors: {best_result.bidirectional_count}")
            print(f"   📊 Optuna-optimized thresholds:")
            print(f"      Forward threshold: {best_result.config.forward_threshold}")
            print(f"      Reverse threshold: {best_result.config.reverse_threshold}")
        else:
            print(f"   ⚠️  All trials pruned — no survivors found in this seed range")
            print(f"   Try wider thresholds, smaller window sizes, or a different seed range")
        print(f"{'='*80}\\n")

        # [S145-R1] Guard: return safely when all trials pruned
        if best_result is None:
            return {
                'strategy': 'optuna_bayesian',
                'best_config': {},
                'best_result': {'bidirectional_count': 0, 'forward_count': 0, 'reverse_count': 0},
                'best_score': best_score,
                'all_results': [],
                'iterations': len(all_results),
                'all_pruned': True
            }

        # [S145-R1] Guard: study.best_trial raises ValueError when all pruned
        try:
            _best_trial_num = study.best_trial.number
            _best_value = study.best_value
            _best_params = study.best_params
        except ValueError:
            _best_trial_num = -1
            _best_value = 0
            _best_params = {}

        return {
            'strategy': 'optuna_bayesian',
            'best_config': best_result.config.to_dict(),
            'best_result': best_result.to_dict(),
            'best_score': best_score,
            'all_results': [r.to_dict() for r in all_results],
            'iterations': len(all_results),
            'optuna_study': {
                'best_trial': _best_trial_num,
                'best_value': _best_value,
                'best_params': _best_params
            }
        }'''

content, ok = apply_patch(content, OLD_A, NEW_A,
    "guard best_result None + study.best_trial ValueError when all trials pruned")
all_ok = all_ok and ok

if all_ok:
    write(TARGET, content)
    new_lines = len(content.splitlines())
    print(f"  Lines: {original_lines} → {new_lines} (+{new_lines - original_lines})")

print("\n" + "=" * 55)
if all_ok:
    print("✅ PATCH APPLIED")
    print()
    print("NOTE ON SMOKE TESTS:")
    print("  The root issue is pruning kills all trials at 5M seeds in higher")
    print("  seed ranges. Two options for smoke testing:")
    print()
    print("  Option A — Disable pruning for smoke test (recommended):")
    print("    ssh rzeus \"sed -i 's/\\\"enable_pruning\\\": true,/\\\"enable_pruning\\\": false,/' \\")
    print("        ~/distributed_prng_analysis/agent_manifests/window_optimizer.json\"")
    print()
    print("  Option B — Reset coverage tracker to force seed_start=0 for smoke test:")
    print("    ssh rzeus \"cd ~/distributed_prng_analysis && python3 -c \\\"")
    print("    import sqlite3; conn = sqlite3.connect('prng_analysis.db');")
    print("    conn.execute(\\\\\\\"DELETE FROM exhaustive_progress\\\\\\\");")
    print("    conn.commit(); print('Coverage tracker reset')\\\"\"")
    print()
    print("  Recommend Option A — disable pruning, re-enable before production run.")
else:
    print("⚠️  PATCH FAILED")
    sys.exit(1)
