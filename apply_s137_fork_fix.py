#!/usr/bin/env python3
"""
S137 patch: Fix n_parallel=2 pickle error in window_optimizer_integration_final.py

Root cause: _mp.set_start_method('spawn', force=True) at line 829 forces spawn
start method. Python spawn requires all objects passed to Process(target=...) to
be picklable. _partition_worker is a local nested function — unpicklable with spawn.

History: S125 confirmed n_parallel=2 working. S127 stale-file commit re-introduced
the broken version. 'fork' is Linux default, avoids pickle entirely, and is safe
here since we're on Linux only (Zeus + rigs all Ubuntu).

Fix: Change 'spawn' to 'fork'.
"""

import shutil
from pathlib import Path

TARGET = Path('window_optimizer_integration_final.py')
BACKUP = Path('window_optimizer_integration_final.py.bak_s137_fork')

assert TARGET.exists(), f"ERROR: {TARGET} not found"

shutil.copy2(TARGET, BACKUP)
print(f"Backup: {BACKUP}")

content = TARGET.read_text()
original = content

OLD = "                _mp.set_start_method('spawn', force=True)\n"
NEW = "                _mp.set_start_method('fork', force=True)  # S137: fork avoids pickle on local fn\n"

assert content.count(OLD) == 1, f"Anchor not found exactly once (found {content.count(OLD)})"
content = content.replace(OLD, NEW)

assert content != original
TARGET.write_text(content)
print("✅ Patch applied: spawn → fork")

import ast
try:
    ast.parse(content)
    print("✅ Syntax check passed")
except SyntaxError as e:
    print(f"❌ SYNTAX ERROR: {e}")
    shutil.copy2(BACKUP, TARGET)
    raise

assert "'fork'" in content
print("✅ Sanity check passed")
