#!/usr/bin/env python3
"""
apply_s118_gaps1_2.py
=====================
Closes the remaining two --enable-pruning forwarding gaps in window_optimizer.py.

GAP 1: main() parses --enable-pruning but never passes it to run_bayesian_optimization()
GAP 2: run_bayesian_optimization() accepts enable_pruning but never passes it to optimize_window()

Gap 3 (optimize_window signature) was already closed by apply_s118_sig_fix.py.

Anchors verified against public repo (mirrors Zeus at c930b6e per S117).
"""
import sys, shutil, ast
from pathlib import Path
from datetime import datetime

REPO = Path('.')
FILE = 'window_optimizer.py'
path = REPO / FILE
src  = path.read_text()

PATCHES = [
    {
        "label": "GAP 1 — main() → run_bayesian_optimization()",
        "old": "            study_name=getattr(args, 'study_name', '')  # NEW: Pass through\n        )",
        "new": (
            "            study_name=getattr(args, 'study_name', ''),  # NEW: Pass through\n"
            "            enable_pruning=getattr(args, 'enable_pruning', False),  # S118\n"
            "            n_parallel=getattr(args, 'n_parallel', 1)  # S118\n"
            "        )"
        ),
    },
    {
        "label": "GAP 2 — run_bayesian_optimization() → optimize_window()",
        "old": (
            "        resume_study=resume_study,\n"
            "        study_name=study_name\n"
            "    )"
        ),
        "new": (
            "        resume_study=resume_study,\n"
            "        study_name=study_name,\n"
            "        enable_pruning=enable_pruning,  # S118\n"
            "        n_parallel=n_parallel  # S118\n"
            "    )"
        ),
    },
]

backed_up = False
all_ok    = True

for p in PATCHES:
    if p["old"] not in src:
        if p["new"].split("\n")[0].strip().rstrip(",") in src:
            print(f"  SKIP  {p['label']} (already applied)")
        else:
            print(f"  FAIL  {p['label']} — anchor not found")
            all_ok = False
        continue

    if not backed_up:
        ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
        bak = Path(str(path) + f'.bak_s118g12_{ts}')
        shutil.copy2(path, bak)
        print(f"  BAK   {bak.name}")
        backed_up = True

    src = src.replace(p["old"], p["new"], 1)
    print(f"  OK    {p['label']}")

if not all_ok:
    sys.exit(1)

try:
    ast.parse(src)
    print("  AST OK")
except SyntaxError as e:
    print(f"  AST FAIL: {e}")
    sys.exit(1)

path.write_text(src)
print("\n✅ Both gaps closed. Run verify_pruning_s118.py to confirm.")
