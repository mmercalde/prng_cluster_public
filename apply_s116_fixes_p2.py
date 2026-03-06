#!/usr/bin/env python3
"""
S116 Fix Script Part 2
Fix: Add --study-name CLI argument to window_optimizer.py argparse
     and pass study_name through all call sites
"""
import sys

print("Fix: window_optimizer.py - add --study-name CLI arg and pass through call sites")

with open('window_optimizer.py', 'r') as f:
    content = f.read()

# 1: Add --study-name after --resume-study argparse block
old_arg = """    parser.add_argument('--resume-study', action='store_true',
                       help='Resume most recent incomplete Optuna study DB instead of starting fresh. '
                            'Skips warm-start enqueue if study already has trials. '
                            'Default: False (fresh study every run).')
    parser.add_argument('--test-both-modes', action='store_true',"""

new_arg = """    parser.add_argument('--resume-study', action='store_true',
                       help='Resume most recent incomplete Optuna study DB instead of starting fresh. '
                            'Skips warm-start enqueue if study already has trials. '
                            'Default: False (fresh study every run).')
    parser.add_argument('--study-name', type=str, default='',
                       help='Optuna study DB name to resume (e.g. window_opt_1772507547). '
                            'Empty string = auto-select most recent incomplete study. '
                            'Only used when --resume-study is set.')
    parser.add_argument('--test-both-modes', action='store_true',"""

if old_arg in content:
    content = content.replace(old_arg, new_arg)
    print("  ✅ 1: Added --study-name to argparse")
elif '--study-name' in content:
    print("  ✅ 1: --study-name already in argparse")
else:
    print("  ❌ 1: argparse anchor not found"); sys.exit(1)

# 2: Pass study_name through all call sites that have resume_study
import re

# Replace all occurrences of resume_study=getattr(args, 'resume_study', False)
# that are NOT already followed by study_name
old_call = "resume_study=getattr(args, 'resume_study', False)"
new_call = "resume_study=getattr(args, 'resume_study', False),\n            study_name=getattr(args, 'study_name', '')"

count = content.count(old_call)
if count > 0:
    content = content.replace(old_call, new_call)
    print(f"  ✅ 2: Patched {count} call site(s) to pass study_name")
elif "study_name=getattr(args, 'study_name'" in content:
    print("  ✅ 2: study_name already passed at call sites")
else:
    print("  ❌ 2: call site pattern not found"); sys.exit(1)

with open('window_optimizer.py', 'w') as f:
    f.write(content)

print("✅ Fix complete\n")
print("Verify:")
print("  python3 -c \"import py_compile; py_compile.compile('window_optimizer.py', doraise=True); print('Syntax OK')\"")
