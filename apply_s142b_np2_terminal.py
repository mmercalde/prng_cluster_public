#!/usr/bin/env python3
"""
apply_s142b_np2_terminal.py
S142-B — TB Fix: NP2 terminal + canonical trial history backfill

TB Ruling:
  The NP2 block is not terminal. After it completes, the single-process
  OptunaBayesianSearch.search() path runs, reinitializes survivor_accumulator,
  and completes remaining Optuna trials. _worker_obj can never own all trials.
  Canonical step1_trial_history must be written from the final shared study.

Two patches to window_optimizer_integration_final.py:

Patch 1 — Backfill step1_trial_history from _fin_study (canonical write).
  Inserted in NP2 block before the "Falls through" comment.
  Iterates all COMPLETE trials, resolves session string, writes with
  run_id=f"step1_{prng_base}_{seed_start}_backfill". Covers all trials
  regardless of execution path.

Patch 2 — Guard single-process search path.
  Replace `if False: pass  # indent anchor` with `_np2_complete = n_parallel > 1`.
  Then wrap the survivor_accumulator reinitialization and optimizer.optimize()
  call with `if not _np2_complete:` so NP2 survivors are not wiped.
  The shared dedup+save survivor block still runs for both paths.
  For n_parallel=1: _np2_complete=False, single-process path runs normally.
  For n_parallel>1: _np2_complete=True, single-process search skipped,
    NP2 survivor_accumulator preserved for dedup+save block.
"""

import shutil
import os
import sys

BASE = os.path.expanduser("~/distributed_prng_analysis")
TARGET = os.path.join(BASE, "window_optimizer_integration_final.py")

# ─── Patch 1: Backfill trial history from shared study ───────────────────────

OLD_P1 = """            optimizer.save_results(results, output_file)
            # Falls through to the dedup+save survivor block below
            # (that block reads survivor_accumulator directly, not 'results')


        if False: pass  # indent anchor"""

NEW_P1 = """            optimizer.save_results(results, output_file)

            # [S142-B] Canonical trial history backfill from shared study
            # _worker_obj writes are opportunistic. This is the authoritative write.
            print(f"\\n   [TRIAL_HISTORY] Backfilling from shared study ({len(_fin_study.trials)} trials)...")
            try:
                from database_system import DistributedPRNGDatabase as _DBFILL
                _db_fill = _DBFILL()
                _si_opts = (bounds.session_options
                            if hasattr(bounds, 'session_options')
                            else [['midday'], ['evening'], ['morning']])
                _fill_written = 0
                _fill_skipped = 0
                for _ft in _fin_study.trials:
                    if _ft.state.name != 'COMPLETE':
                        _fill_skipped += 1
                        continue
                    _fparams = _ft.params or {}
                    _fsi = _fparams.get('session_idx', 0)
                    _fsess_raw = (_si_opts[_fsi]
                                  if isinstance(_si_opts, list) and _fsi < len(_si_opts)
                                  else ['unknown'])
                    _fsess = (','.join(_fsess_raw)
                              if isinstance(_fsess_raw, (list, tuple))
                              else str(_fsess_raw))
                    _db_fill.write_step1_trial(
                        run_id=f"step1_{prng_base}_{int(seed_start)}_backfill",
                        study_name=_mp_study_name,
                        trial_number=int(_ft.number),
                        prng_type=str(prng_base),
                        seed_range_start=int(seed_start),
                        seed_range_end=int(seed_start + seed_count - 1),
                        params={
                            'window_size':       _fparams.get('window_size'),
                            'offset':            _fparams.get('offset'),
                            'skip_min':          _fparams.get('skip_min'),
                            'skip_max':          _fparams.get('skip_max'),
                            'time_of_day':       _fsess,
                            'forward_threshold': _fparams.get('forward_threshold', 0.49),
                            'reverse_threshold': _fparams.get('reverse_threshold', 0.49),
                        },
                        trial_score=float(_ft.value or 0.0),
                        forward_survivors=0,
                        reverse_survivors=0,
                        bidirectional_survivors=int(_ft.value or 0),
                        pruned=False
                    )
                    _fill_written += 1
                print(f"   [TRIAL_HISTORY] Backfill complete: "
                      f"{_fill_written} written, {_fill_skipped} skipped (PRUNED)")
            except Exception as _fill_e:
                print(f"   [TRIAL_HISTORY] Backfill failed (non-fatal): {_fill_e}")

            # Falls through to the dedup+save survivor block below
            # (that block reads survivor_accumulator directly, not 'results')
            print(f"\\n[NP2] EXIT NP2 PATH — entering shared dedup+save survivor block")

        # [S142-B] NP2 terminal flag — prevents single-process search from running
        _np2_complete = n_parallel > 1  # True when NP2 block ran above"""

# ─── Patch 2: Guard single-process survivor_accumulator reinit + search ──────

OLD_P2 = """        survivor_accumulator = {
            'forward': [],
            'reverse': [],
            'bidirectional': []
        }

        optimizer = WindowOptimizer(self, dataset_path)
        bounds = SearchBounds.from_config()
        trial_counter = {'count': 0}"""

NEW_P2 = """        # [S142-B] Skip single-process search when NP2 already ran
        if not _np2_complete:
            print(f"[SINGLE] ENTER SINGLE-PROCESS SEARCH PATH")
        else:
            print(f"[NP2] Single-process search path SKIPPED (n_parallel={n_parallel})")

        if not _np2_complete:
            survivor_accumulator = {
                'forward': [],
                'reverse': [],
                'bidirectional': []
            }

        if not _np2_complete:
            optimizer = WindowOptimizer(self, dataset_path)
            bounds = SearchBounds.from_config()
            trial_counter = {'count': 0}"""

# ─── Patch 3: Guard optimizer.optimize() call ────────────────────────────────

OLD_P3 = """        results = optimizer.optimize(
            strategy=strategy,
            bounds=bounds,
            max_iterations=max_iterations,
            scorer=BidirectionalCountScorer(),
            seed_start=seed_start,
            seed_count=seed_count,
            resume_study=resume_study,              # S116-Bug5 confirmed
            study_name=study_name,                  # S116-Bug5 confirmed
            trse_context_file=trse_context_file,    # S123 TRSE thread
            trial_history_context=_trial_history_ctx  # [S140b]
        )

        optimizer.save_results(results, output_file)

        print(f\"\\n{'='*80}\")
        print(\"OPTIMIZATION COMPLETE\")"""

NEW_P3 = """        if not _np2_complete:  # [S142-B] skip single-process search for NP2
            results = optimizer.optimize(
                strategy=strategy,
                bounds=bounds,
                max_iterations=max_iterations,
                scorer=BidirectionalCountScorer(),
                seed_start=seed_start,
                seed_count=seed_count,
                resume_study=resume_study,              # S116-Bug5 confirmed
                study_name=study_name,                  # S116-Bug5 confirmed
                trse_context_file=trse_context_file,    # S123 TRSE thread
                trial_history_context=_trial_history_ctx  # [S140b]
            )

            optimizer.save_results(results, output_file)

        print(f\"\\n{'='*80}\")
        print(\"OPTIMIZATION COMPLETE\")"""


def main():
    print("=" * 60)
    print("S142-B — NP2 terminal + canonical trial history backfill")
    print("=" * 60)

    with open(TARGET, 'r') as f:
        src = f.read()

    patches = [
        ("P1-backfill",    OLD_P1, NEW_P1),
        ("P2-surv-guard",  OLD_P2, NEW_P2),
        ("P3-optim-guard", OLD_P3, NEW_P3),
    ]

    for label, old, new in patches:
        if old not in src:
            print(f"ERROR: anchor '{label}' not found.")
            print(f"  First 80 chars: {repr(old[:80])}")
            sys.exit(1)
        count = src.count(old)
        if count > 1:
            print(f"ERROR: anchor '{label}' appears {count} times")
            sys.exit(1)
        print(f"  Anchor '{label}': OK (unique)")

    bak = TARGET + ".bak_s142b"
    if not os.path.exists(bak):
        shutil.copy2(TARGET, bak)
        print(f"Backup: {bak}")
    else:
        print(f"Backup already exists: {bak}")

    patched = src
    for label, old, new in patches:
        patched = patched.replace(old, new, 1)

    with open(TARGET, 'w') as f:
        f.write(patched)

    lines_before = src.count('\n')
    lines_after  = patched.count('\n')
    print(f"Patch applied: {lines_before} → {lines_after} lines (+{lines_after - lines_before})")

    with open(TARGET) as f:
        n = sum(1 for _ in f)
    print(f"Final line count: {n}")

    # Syntax check
    import subprocess
    result = subprocess.run(
        ["python3", "-c", f"import ast; ast.parse(open('{TARGET}').read()); print('AST: CLEAN')"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print(f"AST ERROR: {result.stderr.strip()}")
        sys.exit(1)

    print("\n✅ Patch complete.")
    print("\nExpected after re-run:")
    print("  Log: '[TRIAL_HISTORY] Backfill complete: 8 written, 2 skipped'")
    print("  Log: '[NP2] Single-process search path SKIPPED'")
    print("  DB:  8 rows, 0 NULL sessions, run_id ending in _backfill")


if __name__ == "__main__":
    main()
