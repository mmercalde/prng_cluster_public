#!/usr/bin/env python3
"""
apply_s101_scorer_worker_v3_5.py
=================================
Applies two mandatory bug fixes to scorer_trial_worker.py.
Team Beta approved 2026-02-20.

Patch A (line 354): Remove random.seed(42) — replace with per-trial seed
Patch B (lines 416-424): Replace neg-MSE objective with Spearman rank correlation

Usage:
    python3 apply_s101_scorer_worker_v3_5.py
    python3 apply_s101_scorer_worker_v3_5.py --target /path/to/scorer_trial_worker.py
    python3 apply_s101_scorer_worker_v3_5.py --dry-run
"""

import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# PATCH DEFINITIONS — exact string match against commit 4e340de
# ─────────────────────────────────────────────────────────────────────────────

PATCH_A_OLD = '            random.seed(42)  # Reproducible sampling'

PATCH_A_NEW = '            random.seed(params.get(\'optuna_trial_number\', 0))  # v3.5: per-trial seed'

PATCH_B_OLD = '''        # Calculate MSE
        holdout_mse = float(torch.nn.functional.mse_loss(
            torch.tensor(y_pred_holdout),
            torch.tensor(y_holdout)
        ))
        
        accuracy = -holdout_mse  # Negative MSE (higher is better)
        
        logger.info(f\"Holdout MSE: {holdout_mse:.6f}, Accuracy (NegMSE): {accuracy:.6f}\")'''

PATCH_B_NEW = '''        # v3.5: Spearman rank correlation — correct objective for ranking
        # Team Beta Mod 1: guard both y_pred AND y_holdout for degeneracy
        # Team Beta Mod 2: runtime SciPy import guard (best-effort, non-fatal)
        try:
            from scipy.stats import spearmanr
            _scipy_available = True
        except ImportError:
            _scipy_available = False
            logger.error('scipy not available on this worker — accuracy = -1.0')

        if not _scipy_available:
            accuracy = -1.0
        else:
            y_pred_arr    = np.array(y_pred_holdout)
            y_holdout_arr = np.array(y_holdout)

            if np.std(y_pred_arr) < 1e-12:
                accuracy = -1.0
                logger.warning('Degenerate NN: all predictions identical. rho = -1.0')
            elif np.std(y_holdout_arr) < 1e-12:
                accuracy = -1.0
                logger.warning('Degenerate scorer: y_holdout constant. rho = -1.0')
            else:
                correlation, p_value = spearmanr(y_pred_arr, y_holdout_arr)
                accuracy = float(correlation) if not np.isnan(correlation) else -1.0
                logger.info(f'Holdout Spearman rho: {accuracy:.6f}  (p={p_value:.4f})')'''

HEADER_OLD = 'scorer_trial_worker.py (v3.4 - Holdout Sampling Fix)'

HEADER_NEW = '''scorer_trial_worker.py (v3.5 - Spearman Objective + Per-Trial Sampling)
==================================================
v3.5 (2026-02-20):
- BUG FIX: Replace neg-MSE objective with Spearman rank correlation
  (MSE collapsed to constant on low-variance score distributions — S101)
- BUG FIX: Remove random.seed(42) — replaced with per-trial seed
  (seed=42 locked all 100 trials to identical 450 seeds, 2.6% pool coverage)
  New: random.seed(params[\'optuna_trial_number\']) — unique per trial,
  stable for retries, ~93% survivor pool coverage across 100 trials
- Team Beta Mod 1: guard y_holdout degeneracy (constant scorer output)
- Team Beta Mod 2: runtime SciPy import guard, best-effort non-fatal'''


# ─────────────────────────────────────────────────────────────────────────────
# APPLY
# ─────────────────────────────────────────────────────────────────────────────

def apply(target: Path, dry_run: bool = False):
    content = target.read_text()
    original = content
    results = {}

    # ── Patch A ───────────────────────────────────────────────────────────────
    if PATCH_A_OLD in content:
        content = content.replace(PATCH_A_OLD, PATCH_A_NEW, 1)
        results['patch_a'] = 'APPLIED'
        print('✅  Patch A — random.seed(42) removed, per-trial seed installed')
    else:
        results['patch_a'] = 'NOT FOUND'
        print('❌  Patch A — target string not found (already patched or file changed)')

    # ── Patch B ───────────────────────────────────────────────────────────────
    if PATCH_B_OLD in content:
        content = content.replace(PATCH_B_OLD, PATCH_B_NEW, 1)
        results['patch_b'] = 'APPLIED'
        print('✅  Patch B — neg-MSE replaced with Spearman + degeneracy guards')
    else:
        results['patch_b'] = 'NOT FOUND'
        print('❌  Patch B — target string not found (already patched or file changed)')

    # ── Header bump ──────────────────────────────────────────────────────────
    if HEADER_OLD in content:
        content = content.replace(HEADER_OLD, HEADER_NEW, 1)
        results['header'] = 'APPLIED'
        print('✅  Header — version bumped to v3.5')
    else:
        results['header'] = 'NOT FOUND'
        print('⚠️   Header — v3.4 string not found (manual version bump may be needed)')

    # ── Write ────────────────────────────────────────────────────────────────
    failed = [k for k, v in results.items() if v == 'NOT FOUND' and k != 'header']
    if failed:
        print(f'\n❌  ABORT — critical patches not found: {failed}')
        print('    File not modified. Verify target file is scorer_trial_worker.py at commit 4e340de.')
        sys.exit(1)

    if content == original:
        print('\n⚠️   No changes made — file may already be patched.')
        sys.exit(0)

    if dry_run:
        print('\n[DRY RUN] No file written.')
        return

    backup = target.with_suffix(f'.py.bak_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    shutil.copy2(target, backup)
    print(f'\n📦  Backup written: {backup.name}')

    target.write_text(content)
    print(f'✅  Patched file written: {target}')
    print('\n── Verification ─────────────────────────────────────────────────────')

    # Quick sanity checks on written file
    written = target.read_text()
    checks = [
        ('random.seed(42) removed from code',
         not any(ln.strip().startswith('random.seed(42)')
                 for ln in written.splitlines())),
        ('optuna_trial_number seed present', 'optuna_trial_number' in written),
        ('spearmanr present',              'spearmanr' in written),
        ('scipy import guard present',     '_scipy_available' in written),
        ('y_holdout degeneracy guard',     'y_holdout constant' in written),
        ('neg-MSE removed',               'Negative MSE' not in written),
    ]
    all_ok = True
    for desc, ok in checks:
        status = '✅' if ok else '❌'
        print(f'  {status}  {desc}')
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print('✅  All verification checks passed. scorer_trial_worker.py is v3.5.')
        print('\nNext steps:')
        print('  1. scp scorer_trial_worker.py to all worker rigs')
        print('  2. Run 10-trial smoke test:')
        print('     PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 2 --end-step 2')
        print('  3. Verify trial accuracy values differ across trials')
        print('  4. Proceed with S102 full run')
    else:
        print('❌  Some checks failed — review output above before proceeding.')
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Apply scorer_trial_worker.py v3.5 patches')
    parser.add_argument('--target', type=Path,
                        default=Path('scorer_trial_worker.py'),
                        help='Path to scorer_trial_worker.py (default: ./scorer_trial_worker.py)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be patched without writing')
    args = parser.parse_args()

    if not args.target.exists():
        print(f'❌  File not found: {args.target}')
        sys.exit(1)

    print(f'Target: {args.target.resolve()}')
    print(f'Mode:   {"DRY RUN" if args.dry_run else "APPLY"}\n')
    apply(args.target, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
