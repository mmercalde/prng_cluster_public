#!/usr/bin/env python3
"""
S137 patch: Add missing 'import os' to window_optimizer_integration_final.py

Bug: os.path.exists/getmtime/basename/splitext used at lines 760,763,785
but 'import os' is absent from the file entirely.
Only surfaces when n_parallel>=2 hits the fresh-study path in optimize_window().
"""

import shutil
from pathlib import Path

TARGET = Path('window_optimizer_integration_final.py')
BACKUP = Path('window_optimizer_integration_final.py.bak_s137_pre')

assert TARGET.exists(), f"ERROR: {TARGET} not found — run from ~/distributed_prng_analysis/"

shutil.copy2(TARGET, BACKUP)
print(f"Backup: {BACKUP}")

content = TARGET.read_text()
original = content

OLD = "from typing import Dict, Any, List, Tuple\nimport json\n"
NEW = "from typing import Dict, Any, List, Tuple\nimport json\nimport os\n"

assert content.count(OLD) == 1, "Anchor not found exactly once"
content = content.replace(OLD, NEW)

assert content != original
TARGET.write_text(content)
print("✅ Patch applied: 'import os' added at module level")

import ast
try:
    ast.parse(content)
    print("✅ Syntax check passed")
except SyntaxError as e:
    print(f"❌ SYNTAX ERROR: {e}")
    shutil.copy2(BACKUP, TARGET)
    raise

assert 'import os' in content
print("✅ Sanity check passed")
