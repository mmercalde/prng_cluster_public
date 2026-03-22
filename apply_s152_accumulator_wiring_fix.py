#!/usr/bin/env python3
"""
apply_s152_accumulator_wiring_fix.py
======================================
Fix: _survivor_accumulator not passed from BayesianOptimization to OptunaBayesianSearch.

ROOT CAUSE
----------
In window_optimizer_integration_final.py:
    strategy._survivor_accumulator = survivor_accumulator  # [S149]

`strategy` is a BayesianOptimization instance.

But BayesianOptimization.search() immediately delegates to:
    self.optuna_search.search(...)

`self.optuna_search` is an OptunaBayesianSearch instance.

Inside OptunaBayesianSearch.search() (window_optimizer_bayesian.py line 650):
    _survivor_acc = getattr(self, '_survivor_accumulator', None)

`self` here is OptunaBayesianSearch — NOT BayesianOptimization.
The attribute was set on BayesianOptimization but read from OptunaBayesianSearch.
They are different objects. The accumulator is always None. NPZ never written.

FIX
---
In BayesianOptimization.search(), before delegating to self.optuna_search.search(),
copy _survivor_accumulator onto self.optuna_search:

    if hasattr(self, '_survivor_accumulator'):
        self.optuna_search._survivor_accumulator = self._survivor_accumulator

This is a one-line fix with zero architectural change.

Files patched
-------------
  window_optimizer.py

Backup: window_optimizer.py.bak_s152_accum_wire
"""

import shutil
import sys
from pathlib import Path

DRY_RUN = "--dry-run" in sys.argv

TARGET = Path("window_optimizer.py")
BACKUP = Path("window_optimizer.py.bak_s152_accum_wire")

OLD_DELEGATE = '''\
        if self.optuna_search:
            # Use real Optuna implementation
            return self.optuna_search.search(objective_function, bounds, max_iterations, scorer,
                                             resume_study=resume_study, study_name=study_name,
                                             trse_context_file=trse_context_file,
                                             trial_history_context=trial_history_context)'''

NEW_DELEGATE = '''\
        if self.optuna_search:
            # [S152] Wire _survivor_accumulator through to OptunaBayesianSearch
            # BayesianOptimization.search() delegates immediately — accumulator must
            # be copied onto the inner search object or getattr(self,...) finds None.
            if hasattr(self, '_survivor_accumulator'):
                self.optuna_search._survivor_accumulator = self._survivor_accumulator
            # Use real Optuna implementation
            return self.optuna_search.search(objective_function, bounds, max_iterations, scorer,
                                             resume_study=resume_study, study_name=study_name,
                                             trse_context_file=trse_context_file,
                                             trial_history_context=trial_history_context)'''


def apply():
    src = TARGET.read_text()

    if "[S152] Wire _survivor_accumulator" in src:
        print("⚠️  Already patched — aborting.")
        return

    if OLD_DELEGATE not in src:
        print("❌ Anchor not found.")
        return

    patched = src.replace(OLD_DELEGATE, NEW_DELEGATE, 1)

    if DRY_RUN:
        print("=== DRY RUN — no files written ===")
        print(f"  [S152] wire present: {'[S152] Wire _survivor_accumulator' in patched}")
        return

    shutil.copy2(TARGET, BACKUP)
    print(f"✅ Backup: {BACKUP}")
    TARGET.write_text(patched)
    print(f"✅ Patched: {TARGET}")
    print()
    print("Fix: _survivor_accumulator now copied from BayesianOptimization")
    print("     to OptunaBayesianSearch before search() delegation.")
    print("NPZ checkpoint will now fire after every trial with survivors.")


if __name__ == "__main__":
    apply()
