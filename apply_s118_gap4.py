#!/usr/bin/env python3
"""
apply_s118_gap4.py
==================
Fixes the final optuna_trial forwarding gap in WindowOptimizer.optimize().

ROOT CAUSE:
  window_optimizer_bayesian.py line 320 calls:
    result = objective_function(config, optuna_trial=trial)

  But the objective() wrapper inside WindowOptimizer.optimize() is:
    def objective(config: WindowConfig) -> TestResult:
        return self.test_configuration(config, seed_start, seed_count)

  It doesn't accept or forward optuna_trial — TypeError on every trial.

FIX (one patch, window_optimizer.py):
  Add optuna_trial=None to objective() and forward it to self.test_configuration().

WHY ONLY ONE PATCH:
  optimize_window() monkey-patches optimizer.test_configuration = test_config.
  test_config already accepts optuna_trial=None (S115 P8).
  Calling self.test_configuration(config, ss, sc, optuna_trial=trial) maps correctly:
    config=config, ss=seed_start, sc=seed_count, optuna_trial=trial  ✅
  The test_configuration_func path is dead code when monkey-patch is active — no change needed there.

Anchor verified from live public repo (c930b6e, mirrors Zeus per S117).
"""
import sys, shutil, ast
from pathlib import Path
from datetime import datetime

path = Path('window_optimizer.py')
src  = path.read_text()

OLD = (
    "        def objective(config: WindowConfig) -> TestResult:\n"
    "            return self.test_configuration(config, seed_start, seed_count)"
)

NEW = (
    "        def objective(config: WindowConfig, optuna_trial=None) -> TestResult:  # S118\n"
    "            return self.test_configuration(config, seed_start, seed_count,\n"
    "                                           optuna_trial=optuna_trial)  # S118"
)

if OLD not in src:
    if "optuna_trial=optuna_trial)  # S118" in src:
        print("SKIP — already applied")
    else:
        print("FAIL — anchor not found")
        sys.exit(1)
    sys.exit(0)

bak = Path(str(path) + f".bak_s118g4_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(path, bak)
print(f"  BAK  {bak.name}")

patched = src.replace(OLD, NEW, 1)

try:
    ast.parse(patched)
    print("  AST OK")
except SyntaxError as e:
    print(f"  AST FAIL: {e}")
    sys.exit(1)

path.write_text(patched)
print("  OK   objective() now accepts and forwards optuna_trial")
print("\nRun: python3 verify_pruning_s118.py --trials 6 --max-seeds 2000000")
