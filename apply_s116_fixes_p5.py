#!/usr/bin/env python3
"""S116 Fix p5 - fix study_name param being clobbered"""
import sys

with open('window_optimizer_bayesian.py', 'r') as f:
    content = f.read()

old_block = (
    '        _resume = False\n'
    '        _resumed_completed = 0\n'
    '        study_name = f"window_opt_{int(time.time())}"\n'
    '        storage_path = f"sqlite:////home/michael/distributed_prng_analysis/optuna_studies/{study_name}.db"\n'
    '\n'
    '        if resume_study:\n'
    '            # specific study_name takes priority over auto-select\n'
    '            if study_name:\n'
    '                _candidate_dbs = [f"optuna_studies/{study_name}.db"]\n'
)

new_block = (
    '        _resume = False\n'
    '        _resumed_completed = 0\n'
    '        _fresh_study_name = f"window_opt_{int(time.time())}"\n'
    '        _fresh_storage_path = f"sqlite:////home/michael/distributed_prng_analysis/optuna_studies/{_fresh_study_name}.db"\n'
    '\n'
    '        if resume_study:\n'
    '            # specific study_name takes priority over auto-select\n'
    '            if study_name:\n'
    '                _candidate_dbs = [f"optuna_studies/{study_name}.db"]\n'
)

if old_block in content:
    content = content.replace(old_block, new_block)
    print("  ✅ 1: Renamed local study_name -> _fresh_study_name")
else:
    print("  ❌ 1: Block not found"); sys.exit(1)

old_fallback = (
    '            if not _resume_found:\n'
    '                print(f"   \U0001f4ca No resumable study found \u2014 starting fresh")\n'
    '        study = optuna.create_study(\n'
    '            study_name=study_name,\n'
    '            storage=storage_path,\n'
)

new_fallback = (
    '            if not _resume_found:\n'
    '                print(f"   \U0001f4ca No resumable study found \u2014 starting fresh")\n'
    '        if not _resume:\n'
    '            study_name = _fresh_study_name\n'
    '            storage_path = _fresh_storage_path\n'
    '        study = optuna.create_study(\n'
    '            study_name=study_name,\n'
    '            storage=storage_path,\n'
)

if old_fallback in content:
    content = content.replace(old_fallback, new_fallback)
    print("  ✅ 2: Assign fresh name/path only when not resuming")
else:
    # Try without unicode
    idx = content.find('            if not _resume_found:')
    if idx >= 0:
        print(f"  ⚠️  Context: {repr(content[idx:idx+150])}")
    print("  ❌ 2: fallback block not found"); sys.exit(1)

with open('window_optimizer_bayesian.py', 'w') as f:
    f.write(content)

print("✅ Done")
