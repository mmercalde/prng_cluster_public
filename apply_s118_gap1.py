#!/usr/bin/env python3
"""
apply_s118_gap1.py
==================
Closes Gap 1: main() bayesian call site does not forward
enable_pruning or n_parallel to run_bayesian_optimization().

Only the bayesian branch needs it — random/grid/evolutionary
don't use pruning. Gap 2 and Gap 3 already confirmed applied.

Anchor verified from live public repo (c930b6e).
"""
import sys, shutil, ast
from pathlib import Path
from datetime import datetime

path = Path('window_optimizer.py')
src  = path.read_text()

OLD = (
    "            resume_study=getattr(args, 'resume_study', False),\n"
    "            study_name=getattr(args, 'study_name', '')  # NEW: Pass through\n"
    "        )\n"
    "\n"
    "        print(\"\\n✅ Bayesian optimization complete!\")"
)

NEW = (
    "            resume_study=getattr(args, 'resume_study', False),\n"
    "            study_name=getattr(args, 'study_name', ''),  # NEW: Pass through\n"
    "            enable_pruning=getattr(args, 'enable_pruning', False),  # S118\n"
    "            n_parallel=getattr(args, 'n_parallel', 1)  # S118\n"
    "        )\n"
    "\n"
    "        print(\"\\n✅ Bayesian optimization complete!\")"
)

if OLD not in src:
    if "enable_pruning=getattr(args, 'enable_pruning', False)" in src:
        print("SKIP — already applied")
    else:
        print("FAIL — anchor not found")
        print("Run: grep -n 'study_name=getattr' window_optimizer.py")
        sys.exit(1)
    sys.exit(0)

bak = Path(str(path) + f".bak_s118g1_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(path, bak)
print(f"  BAK  {bak.name}")

patched = src.replace(OLD, NEW, 1)

try:
    ast.parse(patched)
except SyntaxError as e:
    print(f"  AST FAIL: {e}")
    sys.exit(1)

path.write_text(patched)
print("  OK   Gap 1 closed — enable_pruning+n_parallel forwarded from main()")
print("  AST OK")
print("\nAll 3 gaps now closed. Run:")
print("  python3 verify_pruning_s118.py --trials 6 --max-seeds 2000000")
