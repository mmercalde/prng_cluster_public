#!/usr/bin/env python3
"""
S137 patch B: Fix AttributeError in _partition_worker line 645

Error: module 'os' has no attribute 'dirname'
  File "window_optimizer_integration_final.py", line 645, in _partition_worker
  _sys.path.insert(0, _os2.dirname(_os2.abspath(__file__)))

With fork, __file__ is not reliably available inside the nested function scope.
Fix: use hardcoded project path instead, same as every other worker in this codebase.
"""

import shutil
from pathlib import Path

TARGET = Path('window_optimizer_integration_final.py')
BACKUP = Path('window_optimizer_integration_final.py.bak_s137_pathfix')

assert TARGET.exists(), f"ERROR: {TARGET} not found"

shutil.copy2(TARGET, BACKUP)
print(f"Backup: {BACKUP}")

content = TARGET.read_text()
original = content

OLD = "                import sys as _sys, os as _os2\n                _sys.path.insert(0, _os2.dirname(_os2.abspath(__file__)))\n"
NEW = "                import sys as _sys\n                _sys.path.insert(0, '/home/michael/distributed_prng_analysis')  # S137: hardcoded, fork-safe\n"

count = content.count(OLD)
assert count == 1, f"Anchor not found exactly once (found {count})"

content = content.replace(OLD, NEW)
assert content != original

TARGET.write_text(content)
print("✅ Patch applied: __file__ path → hardcoded project path")

import ast
try:
    ast.parse(content)
    print("✅ Syntax check passed")
except SyntaxError as e:
    print(f"❌ SYNTAX ERROR: {e}")
    shutil.copy2(BACKUP, TARGET)
    raise

assert "distributed_prng_analysis" in content
print("✅ Sanity check passed")
