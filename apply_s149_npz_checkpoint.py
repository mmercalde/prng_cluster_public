#!/usr/bin/env python3
"""
apply_s149_npz_checkpoint.py
==============================
Fix: NPZ accumulator writes per-trial instead of end-of-run only.

Problem:
  The survivor_accumulator is held in memory across all 50 trials and only
  written to bidirectional_survivors_all.npz at the very end of the run.
  A crash at trial 49 loses all accumulated survivors.

Fix:
  1. create_incremental_save_callback() in window_optimizer_bayesian.py
     gains an optional `survivor_accumulator` parameter.
     When provided, save_best_so_far() writes an NPZ checkpoint after
     each trial that produced survivors.

  2. The callback creation in BayesianOptimization.optimize() passes
     the survivor_accumulator from the closure.

  3. window_optimizer_integration_final.py passes survivor_accumulator
     to create_incremental_save_callback() via the BayesianOptimization
     instance before calling optimize().

Design:
  - Checkpoint write is non-fatal — wrapped in try/except, never halts trial
  - Atomic write: write to .tmp then rename (same pattern as config write)
  - Merge policy: best per-seed score wins (same as end-of-run write)
  - Only writes if trial has survivors (skips pruned/zero-survivor trials)
  - Does not replace the end-of-run write — both paths remain active

Files changed:
  1. window_optimizer_bayesian.py
     - create_incremental_save_callback(): add survivor_accumulator param
     - save_best_so_far(): add NPZ checkpoint write block
     - BayesianOptimization.optimize(): pass survivor_accumulator to callback

  2. window_optimizer_integration_final.py
     - Set optimizer._survivor_accumulator before calling optimize()

Usage:
  python3 apply_s149_npz_checkpoint.py [--dry-run]
"""

import argparse
import shutil
import os
import sys

DRY_RUN = False


def log(msg):
    print(msg)


def backup(path, tag="s149_npz"):
    if DRY_RUN:
        log(f"  [DRY-RUN] would create backup {path}.bak_{tag}")
        return
    bak = f"{path}.bak_{tag}"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        log(f"  backup → {bak}")


def replace_exact(content, old, new, label):
    if old not in content:
        log(f"  ERROR: anchor not found — {label!r}")
        log(f"         (already patched, formatting drifted, or upstream refactor)")
        return content, False
    count = content.count(old)
    if count != 1:
        log(f"  WARNING: {count} occurrences of {label!r} — replacing first only")
    result = content.replace(old, new, 1)
    log(f"  patched: {label}")
    return result, True


def patch_bayesian(path):
    log(f"\n[FILE 1] {path}")
    backup(path)

    with open(path) as f:
        content = f.read()

    ok_count = 0

    # Fix 1: Add survivor_accumulator parameter to create_incremental_save_callback
    content, ok = replace_exact(
        content,
        'def create_incremental_save_callback(\n    output_config_path: str = "optimal_window_config.json",\n    output_survivors_path: str = "bidirectional_survivors.json",\n    total_trials: int = 50,\n    trial_history_context: dict = None,  # [S140b]\n):',
        'def create_incremental_save_callback(\n    output_config_path: str = "optimal_window_config.json",\n    output_survivors_path: str = "bidirectional_survivors.json",\n    total_trials: int = 50,\n    trial_history_context: dict = None,  # [S140b]\n    survivor_accumulator: dict = None,   # [S149] per-trial NPZ checkpoint\n):',
        "create_incremental_save_callback signature — add survivor_accumulator"
    )
    ok_count += ok

    # Fix 2: Add NPZ checkpoint write inside save_best_so_far, after the config save
    # Inject after the [SAVE] Trial print line
    content, ok = replace_exact(
        content,
        '            print(f"[SAVE] Trial {trial.number}: config saved (best={study.best_value:.0f} @ trial {study.best_trial.number})")',
        '''            print(f"[SAVE] Trial {trial.number}: config saved (best={study.best_value:.0f} @ trial {study.best_trial.number})")

            # [S149] Per-trial NPZ checkpoint — write accumulator after each trial with survivors
            if survivor_accumulator is not None:
                try:
                    _bidi = survivor_accumulator.get('bidirectional', [])
                    if _bidi:
                        import numpy as _np_ckpt
                        import os as _os_ckpt
                        _SKIP_ENC = {'constant': 0, 'variable': 1}
                        _PRNG_ENC = {
                            'java_lcg': 0, 'java_lcg_reverse': 1,
                            'mt19937': 2, 'xorshift128': 4, 'lcg32': 6, 'minstd': 8,
                        }
                        _accum_npz = 'bidirectional_survivors_all.npz'
                        _binary_npz = 'bidirectional_survivors_binary.npz'

                        # Build arrays from current accumulator
                        def _dedup(lst):
                            seen = {}
                            for s in lst:
                                seed = s['seed']
                                if seed not in seen or s.get('score', 0) > seen[seed].get('score', 0):
                                    seen[seed] = s
                            return list(seen.values())

                        _deduped = _dedup(_bidi)

                        # Merge with prior NPZ if exists
                        _prior_seeds = set()
                        _merged = {s['seed']: s for s in _deduped}
                        if _os_ckpt.path.exists(_accum_npz):
                            try:
                                _prior = _np_ckpt.load(_accum_npz)
                                _prior_arr = _prior['seeds']
                                _prior_scores = _prior.get('score', _np_ckpt.zeros(len(_prior_arr)))
                                for i, seed in enumerate(_prior_arr):
                                    _prior_seeds.add(int(seed))
                                    if int(seed) not in _merged or float(_prior_scores[i]) > _merged[int(seed)].get('score', 0):
                                        # Keep prior — higher score or not in current
                                        _merged[int(seed)] = {'seed': int(seed), 'score': float(_prior_scores[i]),
                                                              '_from_prior': True}
                            except Exception:
                                pass

                        _all = list(_merged.values())
                        _seeds = _np_ckpt.array([s['seed'] for s in _all], dtype=_np_ckpt.uint64)
                        _scores = _np_ckpt.array([s.get('score', 0.0) for s in _all], dtype=_np_ckpt.float32)

                        # Atomic write
                        _tmp = _accum_npz + '.ckpt.tmp'
                        _np_ckpt.savez_compressed(_tmp, seeds=_seeds, score=_scores)
                        _os_ckpt.replace(_tmp, _accum_npz)

                        # Write binary NPZ for Steps 2-6
                        _fwd_mr = _np_ckpt.array([s.get('forward_match_rate', 0.0) for s in _all], dtype=_np_ckpt.float32)
                        _rev_mr = _np_ckpt.array([s.get('reverse_match_rate', 0.0) for s in _all], dtype=_np_ckpt.float32)
                        _tmp_bin = _binary_npz + '.ckpt.tmp'
                        _np_ckpt.savez_compressed(_tmp_bin, seeds=_seeds,
                                                  forward_match_rate=_fwd_mr,
                                                  reverse_match_rate=_rev_mr,
                                                  score=_scores)
                        _os_ckpt.replace(_tmp_bin, _binary_npz)

                        _new = len(_seeds) - len(_prior_seeds)
                        print(f"[S149-CKPT] Trial {trial.number}: NPZ checkpoint written "
                              f"({len(_seeds):,} total, +{_new} new seeds)")
                except Exception as _ckpt_err:
                    print(f"[S149-CKPT] Warning: checkpoint write failed (non-fatal): {_ckpt_err}")''',
        "save_best_so_far — add NPZ checkpoint write"
    )
    ok_count += ok

    # Fix 3: Pass survivor_accumulator to create_incremental_save_callback in optimize()
    content, ok = replace_exact(
        content,
        '        _incremental_callback = create_incremental_save_callback(\n            output_config_path="optimal_window_config.json",\n            output_survivors_path="bidirectional_survivors.json",\n            total_trials=max_iterations,\n            trial_history_context=_th_context\n        )',
        '        _survivor_acc = getattr(self, \'_survivor_accumulator\', None)  # [S149] per-trial checkpoint\n        _incremental_callback = create_incremental_save_callback(\n            output_config_path="optimal_window_config.json",\n            output_survivors_path="bidirectional_survivors.json",\n            total_trials=max_iterations,\n            trial_history_context=_th_context,\n            survivor_accumulator=_survivor_acc,  # [S149]\n        )',
        "BayesianOptimization.optimize — pass survivor_accumulator to callback"
    )
    ok_count += ok

    log(f"\n  window_optimizer_bayesian.py: {ok_count}/3 patches applied")

    if not DRY_RUN:
        with open(path, "w") as f:
            f.write(content)
        log(f"  wrote {path}")
    else:
        log(f"  [DRY-RUN] would write {path}")

    return ok_count == 3


def patch_integration(path):
    log(f"\n[FILE 2] {path}")
    backup(path)

    with open(path) as f:
        content = f.read()

    ok_count = 0

    # Fix 4: Set _survivor_accumulator on optimizer before optimize() call
    content, ok = replace_exact(
        content,
        '        optimizer.test_configuration = test_config',
        '        optimizer.test_configuration = test_config\n        optimizer._survivor_accumulator = survivor_accumulator  # [S149] per-trial NPZ checkpoint',
        "wire survivor_accumulator onto optimizer instance"
    )
    ok_count += ok

    log(f"\n  window_optimizer_integration_final.py: {ok_count}/1 patches applied")

    if not DRY_RUN:
        with open(path, "w") as f:
            f.write(content)
        log(f"  wrote {path}")
    else:
        log(f"  [DRY-RUN] would write {path}")

    return ok_count == 1


def verify(base_dir):
    log("\n[VERIFY]")
    errors = []

    bayesian = open(os.path.join(base_dir, "window_optimizer_bayesian.py")).read()
    integration = open(os.path.join(base_dir, "window_optimizer_integration_final.py")).read()

    checks = [
        ("bayesian: survivor_accumulator param in signature",
         "survivor_accumulator: dict = None,   # [S149]" in bayesian),
        ("bayesian: NPZ checkpoint write present",
         "S149-CKPT" in bayesian),
        ("bayesian: survivor_acc passed to callback",
         "_survivor_acc = getattr(self, '_survivor_accumulator', None)" in bayesian),
        ("integration: _survivor_accumulator wired onto optimizer",
         "optimizer._survivor_accumulator = survivor_accumulator" in integration),
    ]

    for label, condition in checks:
        status = "✓" if condition else "✗"
        log(f"  {status}  {label}")
        if not condition:
            errors.append(label)

    if errors:
        log(f"\n  FAIL — {len(errors)} error(s)")
        return False
    log("\n  PASS — all checks green")
    return True


def main():
    global DRY_RUN
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()
    DRY_RUN = args.dry_run

    print(f"{'[DRY-RUN] ' if DRY_RUN else ''}S149 Per-Trial NPZ Checkpoint Fix")
    print("Writes NPZ accumulator after each trial — no more end-of-run data loss risk")
    print("=" * 60)

    r1 = patch_bayesian(os.path.join(args.base_dir, "window_optimizer_bayesian.py"))
    r2 = patch_integration(os.path.join(args.base_dir, "window_optimizer_integration_final.py"))

    if not DRY_RUN:
        passed = verify(args.base_dir)
    else:
        log("\n[VERIFY] skipped in dry-run mode")
        passed = r1 and r2

    print("\n" + "=" * 60)
    if r1 and r2 and passed:
        print("✓ S149 NPZ checkpoint fix COMPLETE")
        print()
        print("Behavior after fix:")
        print("  - After each trial with survivors: NPZ written atomically")
        print("  - Crash at trial 49: survivors from trials 1-48 are safe")
        print("  - End-of-run write still runs: both paths active")
        print("  - Pruned/zero-survivor trials: no write (no-op)")
        print()
        print("Commit:")
        print("  git add window_optimizer_bayesian.py window_optimizer_integration_final.py")
        print("  git commit -m 'fix(s149): per-trial NPZ checkpoint — eliminate end-of-run data loss'")
        print("  git push origin main && git push public main")
        print()
        print("Then resume Run 1:")
        print("  bash sweep_run1.sh --resume")
        return True
    else:
        print("✗ Fix INCOMPLETE — review errors above")
        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
