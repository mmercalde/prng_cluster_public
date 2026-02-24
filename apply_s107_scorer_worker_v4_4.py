#!/usr/bin/env python3
"""
apply_s107_scorer_worker_v4_4.py
S107 Step 2 v4.4 patch — remove orphaned bc_stat reference from _log_trial_metrics

Fix: v4.3 removed bc_stat from run_trial but left a reference in the metrics dict
     inside _log_trial_metrics, causing NameError on every trial.

Patch: 1 change to scorer_trial_worker.py
"""
import sys, ast, shutil, datetime

TARGET = 'scorer_trial_worker.py'
BACKUP = f'{TARGET}.s107_v4_4_backup_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'

OLD = "        'bc_stat'     : round(bc_stat, 2) if bc_stat is not None else None,\n"
NEW = ""  # remove entirely

print(f"=== S107 scorer_trial_worker.py v4.4 patcher ===")

content = open(TARGET).read()

# Pre-flight
if OLD not in content:
    print("ABORT: target string not found - file may already be patched or has changed")
    sys.exit(1)

count = content.count(OLD)
if count != 1:
    print(f"ABORT: expected exactly 1 occurrence, found {count}")
    sys.exit(1)

# Backup
shutil.copy2(TARGET, BACKUP)
print(f"Backup: {BACKUP}")

# Apply
patched = content.replace(OLD, NEW)
open(TARGET, 'w').write(patched)
print("Patch 1/1: removed orphaned bc_stat line from _log_trial_metrics dict")

# AST check
try:
    ast.parse(patched)
    print("AST: OK")
except SyntaxError as e:
    print(f"AST FAIL: {e}")
    shutil.copy2(BACKUP, TARGET)
    print("Rolled back")
    sys.exit(1)

# Post-flight
result = open(TARGET).read()
if 'bc_stat' in result:
    print("WARNING: bc_stat still present in file - check manually")
else:
    print("Post-flight: bc_stat fully removed - OK")

print("=== v4.4 patch complete ===")
