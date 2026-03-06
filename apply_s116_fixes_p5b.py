#!/usr/bin/env python3
"""S116 Fix p5b - fix fallback block with blank line"""
import sys

with open('window_optimizer_bayesian.py', 'r') as f:
    content = f.read()

old_fallback = (
    '            if not _resume_found:\n'
    '                print(f"   \U0001f4ca No resumable study found \u2014 starting fresh")\n'
    '\n'
    '        study = optuna.create_study(\n'
    '            study_name=study_name,\n'
    '            storage=storage_path,\n'
)

new_fallback = (
    '            if not _resume_found:\n'
    '                print(f"   \U0001f4ca No resumable study found \u2014 starting fresh")\n'
    '\n'
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
    with open('window_optimizer_bayesian.py', 'w') as f:
        f.write(content)
    print("✅ Done")
else:
    print("  ❌ 2: block not found"); sys.exit(1)
