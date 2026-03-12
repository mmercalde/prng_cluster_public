#!/usr/bin/env python3
"""
S137 patch C: Fix UnboundLocalError — survivor_accumulator not initialized
in n_parallel > 1 branch of window_optimizer_integration_final.py

Error:
  UnboundLocalError: local variable 'survivor_accumulator' referenced before assignment
  File "window_optimizer_integration_final.py", line 879
  print(f"      Forward:       {len(survivor_accumulator['forward'])}")

Root cause: survivor_accumulator is initialized at line 947 (n_parallel==1 path),
but the n_parallel>1 branch jumps straight to collecting from result_queue and then
printing accumulator stats at line 879 — without ever initializing it.

Fix: Initialize survivor_accumulator = {'forward': [], 'reverse': [], 'bidirectional': []}
just before the worker launch block.
"""

import shutil
from pathlib import Path

TARGET = Path('window_optimizer_integration_final.py')
BACKUP = Path('window_optimizer_integration_final.py.bak_s137_accumulator')

assert TARGET.exists(), f"ERROR: {TARGET} not found"

shutil.copy2(TARGET, BACKUP)
print(f"Backup: {BACKUP}")

content = TARGET.read_text()
original = content

OLD = """            # Divide trials and launch worker processes
            # ----------------------------------------------------------------
            _trials_per_worker = [max_iterations // n_parallel] * n_parallel"""

NEW = """            # Divide trials and launch worker processes
            # ----------------------------------------------------------------
            # S137: Initialize accumulator here so it exists when workers merge results
            survivor_accumulator = {'forward': [], 'reverse': [], 'bidirectional': []}

            _trials_per_worker = [max_iterations // n_parallel] * n_parallel"""

count = content.count(OLD)
assert count == 1, f"Anchor not found exactly once (found {count})"

content = content.replace(OLD, NEW)
assert content != original

TARGET.write_text(content)
print("✅ Patch applied: survivor_accumulator initialized before worker launch")

import ast
try:
    ast.parse(content)
    print("✅ Syntax check passed")
except SyntaxError as e:
    print(f"❌ SYNTAX ERROR: {e}")
    shutil.copy2(BACKUP, TARGET)
    raise

assert "S137: Initialize accumulator here" in content
print("✅ Sanity check passed")
