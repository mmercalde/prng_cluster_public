# Stage 3 (3a + 3b + Beta R1 fixes): full after-pass, 2026-11-21 (CPU only, ssh/scp/rsync stubbed, sequential)

- **Before:** the 2026-11-20 run on a clean `b2efdef` (`logs/pwc_stage3b_v1/before/`). Kept, not re-run.
- **After (R1):** the combined tree. That is Stage 3b, plus agent 2's Stage 3a files (sha256 20/20 equal to pwc-offset), plus the two Beta R1 LIVE fixes. Logs: `logs/pwc_stage3b_v1/after_r1/`; runner: `logs/pwc_stage3b_v1/run_suites_r1.sh`.
- **Suites:** 79. These are the 77 of `logs/pwc_stage3b_v1/suites.txt`, plus `tests/test_pwc_stage3a.py` and `tests/test_pwc_stage3_r1_fixes.py`. `tests/test_s172_staging_partb.py` was not run. Elapsed: 2,773 s.
- **Live tree:** untouched.
  - The main checkout's status and HEAD (`b2efdef`) are the same before and after.
  - The `optuna_studies/` listing is unchanged.
  - The `~/miner_staging` mtime is unchanged after every suite (`2026-11-19 23:24:44`).
  - There were 25 stubbed network calls, all refused by the stub, the same number as the before pass.

## Result: 72 of 75 common suites have identical exit codes and failing-check sets. 3 differ; none is a behaviour change.

| Suite | rc before → after | What changed | Class |
|---|---|---|---|
| `tests/test_gate12_cleantree_admission.py` | 0 → 1 | W-NO-WEAKENING: it compares the producer, finalizer and `.gitignore` with HEAD (`:941-951`). `window_optimizer_integration_final.py` is uncommitted | Working-tree tripwire. Clears on commit (by reading; Beta C-Q2 NO OBJECTION, review `:109`) |
| `tests/test_s172_phase4_coordinator.py` | 0 → 1 | Gate 22 names only the uncommitted Stage 3 `.py` files | Working-tree tripwire. Clears on commit |
| `tests/test_s172_miner_seed_start.py` | 1 → 1 | G-R5-ROUTES fails at the same point, the dataset pointer missing in a worktree, under its new label (owner-approved PWC-line edit) | Same failure, new label |

New suites (all rc 0): `test_pwc_backup_stage3` 14/14 + 18/18; `test_compare_engine_outputs` 13/13; `test_pwc_stage3a` 6/6 + 3/3; `test_pwc_stage3_r1_fixes` 4/4 + 5/5.

Compared with the 2026-11-20 after-pass (`../before_after_tally.md`), `test_persistent_worker_harness.py` is green again, 25/25. T20 was edited only by owner approval.

Some tally lines differ although the failing-check sets do not: timing text, log order, and `test_prove_unchanged_outside` 29 → 49 cases (Stage 3a). The full table is `logs/pwc_stage3b_v1/after_r1_table.md`, and the check-by-check comparison is `before_after_failing_checks_r1.txt`.
