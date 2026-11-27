# X-W1-RUNNER - Alpha closing note (no second round)

Date: 2026-11-26. From: Team Alpha. To: Team Beta. Owner ruling: one Beta round on part 1.

Unit W part 1 (the why-report runner) lands on main on top of 8128dcb with the owner's yes. Six files: scripts/diagnostics/why_report_runner.py, tests/test_why_report_runner.py, why_report_flags.py, tests/test_why_report_flags.py, tests/why_flags_mutants.py, .gitignore. Nothing calls the runner yet; part 2 (the WATCHER switch, default off) is separate and owner-landed.

## What was done with the R1 review

Each finding was checked against the code first. All were right.

- Q1 lost failure: FIXED. The FAILED log append is protected; the original reason is recorded; the log error goes in the result and to stderr. Gate W1.log_append_failure.
- Q1 surviving workers: FIXED. The child's process group is killed and the child reaped on every ending (exit, failure, signal, time limit, interrupted wait). Gate W1.child_group_cleanup.
- Q2 unreadable attempts file: BUILT. Flags and attachments stay readable, attempts_error is reported, the runner will not run a flag it cannot record, and the CLI exits 1. Gate W1.attempts_unreadable.
- Q3 pooled-only grade: FIXED. A pooled-only grade is analysed over the dataset sessions the walk-forward recorded; a report with no graded draws fails with the new reason no_graded_draws and the flag stays open. Gates W1.pooled_only, W1.fail.no_graded_draws. Your Run H origin 0 setup now gives 594 graded draws, not empty contrasts.
- Q3 analysis override: FIXED. Only the declared analysis keys may be overridden; anything else is refused before any file is made (CLI exit 2). Gate W1.analysis_keys.
- Q3 grade binding: BUILT. Each grade dir is checked against the origin-pinned summary digest and its arm, pool size, dataset and block. Gate W1.grade_binding (8 tamper cases).
- Q4 bytecode and out root: FIXED. No bytecode from the plain CLI; the out root is checked against data/ before anything is created. Gates W1.cli_no_bytecode, W1.out_root_boundary.
- Q6: BUILT, including the G-Q4-SCALER refusal carried through the runner as a child. Gate W1.g_q4_scaler.

## Proof (fresh clone of 8128dcb, only the six files copied in, processor only, one suite at a time)

- Runner: 35/35 gates, 11/11 mutants. Flags: 32 passed. Benching: 30 passed. Flag mutants 25/25. Benching mutants 21/21.
- Logs: ~/tfm_worktrees/_proof/w1_8128dcb_logs/ on VM101.

## Limits, stated plainly

- Real-block runs: Run H block 0 only, lightgbm and neural_net. CatBoost, XGBoost and the family-arm fallback were not run on a real block.
- The out-of-memory kill is shown with a stand-in child.
- The flag does not yet record which model was flagged; that comes with part 2.
- Until the loop is wired, most live flags will end no_graded_block and stay open.

No reply is asked for.
