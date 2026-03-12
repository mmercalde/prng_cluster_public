#!/usr/bin/env python3
"""
S137 patch D: Fix UnboundLocalError — 'bounds' not initialized in n_parallel > 1 branch

Error:
  UnboundLocalError: local variable 'bounds' referenced before assignment
  File "window_optimizer_integration_final.py", line 901
  if hasattr(bounds, 'session_options')

Root cause: bounds = SearchBounds.from_config() is only set in the n_parallel==1 path
(line ~957). The n_parallel>1 path references bounds.session_options at line 901
when building _best_cfg2 from the best Optuna trial params.

Fix: Initialize bounds alongside survivor_accumulator just before worker launch.
"""

import shutil
from pathlib import Path

TARGET = Path('window_optimizer_integration_final.py')
BACKUP = Path('window_optimizer_integration_final.py.bak_s137_bounds')

assert TARGET.exists(), f"ERROR: {TARGET} not found"

shutil.copy2(TARGET, BACKUP)
print(f"Backup: {BACKUP}")

content = TARGET.read_text()
original = content

OLD = """            # S137: Initialize accumulator here so it exists when workers merge results
            survivor_accumulator = {'forward': [], 'reverse': [], 'bidirectional': []}"""

NEW = """            # S137: Initialize accumulator and bounds here so they exist in n_parallel path
            survivor_accumulator = {'forward': [], 'reverse': [], 'bidirectional': []}
            bounds = SearchBounds.from_config()  # S137: needed for session_options after best trial"""

count = content.count(OLD)
assert count == 1, f"Anchor not found exactly once (found {count})"

content = content.replace(OLD, NEW)
assert content != original

TARGET.write_text(content)
print("✅ Patch applied: bounds initialized before worker launch")

import ast
try:
    ast.parse(content)
    print("✅ Syntax check passed")
except SyntaxError as e:
    print(f"❌ SYNTAX ERROR: {e}")
    shutil.copy2(BACKUP, TARGET)
    raise

assert "S137: needed for session_options" in content
print("✅ Sanity check passed")
