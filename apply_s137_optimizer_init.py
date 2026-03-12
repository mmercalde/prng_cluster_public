#!/usr/bin/env python3
"""
S137 patch E: Fix UnboundLocalError — 'optimizer' not initialized in n_parallel > 1 branch

Error:
  UnboundLocalError: local variable 'optimizer' referenced before assignment
  File "window_optimizer_integration_final.py", line 931
  optimizer.save_results(results, output_file)

Root cause: optimizer = WindowOptimizer(self, dataset_path) is only set in the
n_parallel==1 path (line ~955). The n_parallel>1 path calls optimizer.save_results()
at line 931 after building the results dict from the best Optuna trial.

Fix: Initialize optimizer alongside survivor_accumulator and bounds before worker launch.

Note: This is the last known unbound variable in the n_parallel>1 path.
survivor_accumulator and bounds were fixed in patches C and D respectively.
"""

import shutil
from pathlib import Path

TARGET = Path('window_optimizer_integration_final.py')
BACKUP = Path('window_optimizer_integration_final.py.bak_s137_optimizer')

assert TARGET.exists(), f"ERROR: {TARGET} not found"

shutil.copy2(TARGET, BACKUP)
print(f"Backup: {BACKUP}")

content = TARGET.read_text()
original = content

OLD = """            # S137: Initialize accumulator and bounds here so they exist in n_parallel path
            survivor_accumulator = {'forward': [], 'reverse': [], 'bidirectional': []}
            bounds = SearchBounds.from_config()  # S137: needed for session_options after best trial"""

NEW = """            # S137: Initialize accumulator, bounds, optimizer so they exist in n_parallel path
            survivor_accumulator = {'forward': [], 'reverse': [], 'bidirectional': []}
            bounds = SearchBounds.from_config()      # S137-D: needed for session_options after best trial
            optimizer = WindowOptimizer(self, dataset_path)  # S137-E: needed for save_results"""

count = content.count(OLD)
assert count == 1, f"Anchor not found exactly once (found {count})"

content = content.replace(OLD, NEW)
assert content != original

TARGET.write_text(content)
print("✅ Patch applied: optimizer initialized before worker launch")

import ast
try:
    ast.parse(content)
    print("✅ Syntax check passed")
except SyntaxError as e:
    print(f"❌ SYNTAX ERROR: {e}")
    shutil.copy2(BACKUP, TARGET)
    raise

assert "S137-E: needed for save_results" in content
print("✅ Sanity check passed")
