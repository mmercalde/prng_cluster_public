#!/usr/bin/env python3
"""
fix_s156_remove_v1_block.py

The Zeus file has BOTH v1 and v2 cleanup blocks simultaneously.
The previous patch found [S156-BANDAID v2] and exited early,
leaving the v1 block with broken f-strings intact.

This script:
1. Finds the v1 block by its unique marker '# [S156] Pre-spawn cleanup'
2. Finds the end of the v1 block by 'cleanup_done'
3. Removes ONLY the v1 block
4. Validates syntax
5. Commits
"""

import ast
import shutil
from pathlib import Path

TARGET = Path("/home/michael/distributed_prng_analysis/persistent_worker_coordinator.py")
BACKUP = TARGET.with_suffix(".py.bak_before_v1_removal")

content = TARGET.read_text()
lines = content.splitlines(keepends=True)

print(f"Total lines: {len(lines)}")

# Find v1 block start and end
v1_start = None
v1_end = None

for i, line in enumerate(lines):
    if '# [S156] Pre-spawn cleanup' in line and v1_start is None:
        v1_start = i
        print(f"Found v1 start at line {i+1}: {line.rstrip()}")
    if v1_start is not None and 'cleanup_done' in line:
        # Find the end of the try block - look for the sleep(2) after
        for j in range(i, min(i+10, len(lines))):
            if '_time.sleep(2)' in lines[j] and 'BANDAID' not in lines[j]:
                # Check this isn't the v2 sleep
                v1_end = j
                print(f"Found v1 end at line {j+1}: {lines[j].rstrip()}")
                break
        if v1_end is not None:
            break

if v1_start is None:
    print("v1 block not found - may already be removed")
    # Verify syntax
    try:
        ast.parse(content)
        print("✅ File syntax OK")
    except SyntaxError as e:
        print(f"❌ Syntax error at line {e.lineno}: {e.msg}")
    exit(0)

if v1_end is None:
    print("ERROR: Found v1 start but not end")
    exit(1)

print(f"Removing v1 block: lines {v1_start+1} to {v1_end+1}")

# Remove the v1 block
new_lines = lines[:v1_start] + lines[v1_end+1:]
new_content = ''.join(new_lines)

# Validate
try:
    ast.parse(new_content)
except SyntaxError as e:
    print(f"ERROR: Syntax error after removal at line {e.lineno}: {e.msg}")
    print(f"Text: {e.text}")
    exit(1)

# Check v2 is still present
if '[S156-BANDAID v2]' not in new_content:
    print("ERROR: v2 block not found after removal - aborting")
    exit(1)

# Backup and write
shutil.copy2(TARGET, BACKUP)
print(f"Backup: {BACKUP}")
TARGET.write_text(new_content)

# Final verify — ast.parse + py_compile (TB requirement)
try:
    ast.parse(TARGET.read_text())
except SyntaxError as e:
    print(f"ERROR: Post-write syntax error (ast): {e}")
    shutil.copy2(BACKUP, TARGET)
    print("Restored backup")
    exit(1)

import subprocess
result = subprocess.run(
    ["python3", "-m", "py_compile", str(TARGET)],
    capture_output=True, text=True
)
if result.returncode != 0:
    print(f"ERROR: py_compile failed:\n{result.stderr}")
    shutil.copy2(BACKUP, TARGET)
    print("Restored backup")
    exit(1)

print("✅ v1 removed, v2 retained, ast + py_compile both OK")
print("\nVerify with:")
print(f"  grep -n 'S156-BANDAID' {TARGET}")
print(f"  grep -n 'pkill -9 -f sieve_gpu_worker' {TARGET}")
