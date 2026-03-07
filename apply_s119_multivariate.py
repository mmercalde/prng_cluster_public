#!/usr/bin/env python3
"""
S119: Enable multivariate=True on TPESampler in window_optimizer_bayesian.py
==============================================================================

WHY:
  multivariate=True makes TPE model correlations between parameters jointly
  rather than treating each dimension independently. For our 7-parameter search
  space (window_size, offset, session_idx, skip_min, skip_max, fwd_threshold,
  rev_threshold), parameters interact — e.g. a narrow skip range only matters
  with a small window. Independent TPE misses these correlations.

  Optuna's own benchmarks show consistent improvement with multivariate=True
  in low-dimensional spaces (7 params qualifies).

SAFETY CHECKS PERFORMED:
  ✅ Search space is STATIC — skip_max lower bound = max(skip_min=0..10, 10) = 10
     always. No dynamic distributions → no IndependentSampling warnings.
  ✅ Tested with Optuna 4.4.0 (exact Zeus version): 15-trial study, 0 warnings
     beyond the one-time ExperimentalWarning at sampler creation.
  ✅ ExperimentalWarning is suppressed inline — will not pollute run logs.
  ✅ Anchor validated 1x exact match against live Zeus code (post-S119 Gap4b/5 patch).
  ✅ multivariate=True not already present in file.

CHANGE: 1 line added to TPESampler() call in OptunaBayesianSearch.search()
"""

import os
import shutil
from datetime import datetime

ZEUS_BASE = os.path.expanduser("~/distributed_prng_analysis")
FILEPATH  = os.path.join(ZEUS_BASE, "window_optimizer_bayesian.py")
TS        = datetime.now().strftime("%Y%m%d_%H%M%S")

OLD = """\
        # Create study with TPE sampler
        sampler = TPESampler(
            n_startup_trials=self.n_startup_trials,
            seed=self.seed
        )"""

NEW = """\
        # Create study with TPE sampler
        # S119: multivariate=True — models param correlations jointly (window_size↔skip_max etc.)
        # Safe: search space is static (skip_max lower bound always=10), Optuna 4.4.0 tested.
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.filterwarnings('ignore', message='.*multivariate.*')
            sampler = TPESampler(
                n_startup_trials=self.n_startup_trials,
                seed=self.seed,
                multivariate=True   # S119
            )"""


def main():
    print("=" * 60)
    print("S119: TPESampler multivariate=True")
    print("=" * 60)

    if not os.path.exists(FILEPATH):
        print(f"❌ File not found: {FILEPATH}")
        return

    with open(FILEPATH) as f:
        content = f.read()

    # Pre-flight checks
    count = content.count(OLD)
    already = content.count('multivariate=True')

    print(f"\nPre-flight:")
    print(f"  Anchor found: {count}x  {'✅' if count == 1 else '❌'}")
    print(f"  Already patched: {already}x  {'✅ clean' if already == 0 else '⚠️  already present'}")

    if count != 1:
        print("\n❌ Anchor mismatch — aborting")
        return
    if already > 0:
        print("\n⚠️  multivariate=True already in file — nothing to do")
        return

    # Backup
    bak = FILEPATH + f".bak_s119_mv_{TS}"
    shutil.copy2(FILEPATH, bak)
    print(f"\n  📦 Backup: {os.path.basename(bak)}")

    # Apply
    with open(FILEPATH, 'w') as f:
        f.write(content.replace(OLD, NEW))
    print(f"  ✅ Patch applied")

    # Verify
    with open(FILEPATH) as f:
        result = f.read()
    ok = 'multivariate=True' in result and OLD not in result
    print(f"  {'✅' if ok else '❌'} Verification: {'PASSED' if ok else 'FAILED'}")

    print(f"\n{'=' * 60}")
    if ok:
        print("✅ Done. TPE will now model parameter correlations jointly.")
        print("   ExperimentalWarning suppressed inline — logs stay clean.")
        print()
        print("Next: push to GitHub, then run verify test:")
        print("  git add window_optimizer_bayesian.py")
        print("  git commit -m 'S119: TPESampler multivariate=True for correlated 7-dim search space'")
        print("  git push")
        print()
        print("  python3 verify_pruning_s118.py --trials 6 --max-seeds 2000000")
    print("=" * 60)


if __name__ == '__main__':
    main()
