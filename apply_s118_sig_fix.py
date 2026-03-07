#!/usr/bin/env python3
"""
apply_s118_sig_fix.py
=====================
Adds enable_pruning to optimize_window() signature.
G2b (S118) wired it to BayesianOptimization() inside the function but
forgot to add it to the function signature — causing TypeError on call.
"""
import sys, shutil
from pathlib import Path
from datetime import datetime

REPO = Path('.')
filepath = 'window_optimizer_integration_final.py'
path = REPO / filepath

src = path.read_text()

old = """                        n_parallel: int = 1):  # S115 M1"""
new = """                        n_parallel: int = 1,
                        enable_pruning: bool = False):  # S115 M1 / S118 sig fix"""

if old not in src:
    if 'enable_pruning: bool = False' in src:
        print("⏭  SKIP — already applied")
        sys.exit(0)
    print("❌ FAIL — anchor not found")
    sys.exit(1)

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
bak = Path(str(path) + f'.bak_s118sig_{ts}')
shutil.copy2(path, bak)
print(f"   💾 Backup: {bak.name}")

path.write_text(src.replace(old, new, 1))
print("✅ APPLIED — enable_pruning added to optimize_window() signature")

import ast
try:
    ast.parse(path.read_text())
    print("✅ AST OK")
except SyntaxError as e:
    print(f"❌ AST FAIL: {e} — restoring backup")
    shutil.copy2(bak, path)
    sys.exit(1)
