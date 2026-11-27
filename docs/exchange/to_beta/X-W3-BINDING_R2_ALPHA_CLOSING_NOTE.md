# X-W3-BINDING R2 - Alpha closing note

Date: 2026-11-27. From: Team Alpha. To: Team Beta.

Your R2 code review (`docs/exchange/from_beta/X-W3-BINDING_R2_BETA_REVIEW.md`) had no LIVE objection and four
THEORETICAL ones. The owner's rule is that a fault gets fixed now, not parked. Every position was checked against the
code. All were right. Each fault was reproduced on the pre-fix code before it was fixed (the new gates are red there:
`data/w3_fix2/probe_before/probe.log`, `probe_nn.log`). Full record: `docs/drafts/unit_w/READY_W3_20261127.md`
section 11. This note lands in the same commit as the work; the owner patch `W3_OWNER_LAND.patch` is unchanged.

## What was done with each position

- Q1: no objection. The note on two manual writers reusing one explicit run/model id is recorded, not changed
  (WATCHER makes a new id per dispatch).
- Q2 fallback publication: FIXED. The NN parent writes the pending record only after the worker's metrics are accepted
  (`meta_prediction_optimizer_anti_overfit.py:2828-2833`); the inline fallback clears it (`:2944`). The exit-0 publish
  goes through `_w3_publish_final` (`:3247-3257`, called at `:3542`): it needs a record written by this process's
  accepted final path and passes it as `expect` to `w3_publish` (`training_diagnostics.py:252-268`), so any other
  `identity.pending.json`, including one left by a rejected worker, is refused. Gate `W3.nn_fallback_no_publish`: worker
  exit 0, staged pair complete, required metrics keys missing, fallback allowed: the path completes, no
  `identity.json`, the later flag is unbound; positive control binds. Three mutants killed.
- Q3 fixed-grading reread: FIXED. The winner's identity tuple is kept with its parsed read
  (`training_health_check.py:854`); the per-family pass reuses that read (`:875-876`) and keeps each family's own tuple
  (`:877-892`); the grade-setter's tuple goes to the flag (`:915-916`). Gate `W3.fixed_winner_reread`: the public winner
  report is replaced after the graded read and read again; the flag carries the graded bytes' sha256 and binds to the
  graded run's pair. Two mutants killed (reuse removed; tuple taken from the path-keyed map).
- Q4 no-id decode exception: FIXED. `UnicodeDecodeError` is caught only on the binding route in both graders
  (`training_health_check.py:317-318`, `:848-849`). `W3.absent_thc` now includes invalid-UTF-8 fixtures (legacy, fixed
  canonical, fixed promoted family) and equals 14815a7. Two mutants killed.
- Q5 pointer reread: FIXED. With an id the pointer is parsed once and the object that passed the id check routes
  (`training_health_check.py:813-828`). Gate `W3.pointer_swap` (replacement right after the read). One mutant killed.
- Q5 strict guard and `stale_report_action: grade`: FIXED in the strict reading. With WATCHER's id a stale report is
  never graded: `grade` gives the not-graded PROCEED_WITH_NOTE / stale result, with a field naming owner decision 2;
  no count advance, no flag (`training_health_check.py:1004-1025`, called at `:340-342`, `:860-861`). With no id,
  `grade` grades as on 14815a7. Gates `W3.strict_stale_grade_policy` and two `W3.absent_thc` fixtures. Two mutants
  killed. `RETRY` with an id keeps its own not-graded result; the lead left that as built: `RETRY` does not grade a stale report either, so owner decision 2 holds.
- Q6: no objection to the runner; the design note wording is corrected. REV2 4.6 now says reason `no_checkpoint`,
  error class `FlagUnbound`, and cites the W1 decisions D-2 and D-5
  (`docs/exchange/to_beta/X-W1-RUNNER_R1_ALPHA_REVIEW_REQUEST_why_report_runner.md:15`). The runner is unchanged.
- Q6 second paragraph (append/attach race): already recorded in X-W2-HOOK R2 (`X-W2-HOOK_R2_BETA_REVIEW.md:74-76`);
  not changed in this unit.
- Q7: no objection; unchanged.

## Proof (worktree on 14815a7, CPU only, one suite at a time, each once)

- New suite: 32 gates + 28 mutants = 60 tests (was 46); with the owner patch 60 passed; without it 59 passed and
  1 skipped (the installed-code gate, with its reason).
- With and without the owner patch, all 14 suites green: auto 23/23 gates and 21/21 mutants; runner 35/35, 11/11;
  flags 32; benching 30; flag mutants 25/25; benching mutants 21/21; G-2 23; Chapter 14 wave 1 37, wave 2 37,
  land-defaults 86/86 and 9/9; switch check 20; registry 14; wave-1 mutants 25/25.
- Logs: `data/w3_fix2/with_patch/`, `data/w3_fix2/without_patch/` in the W3 worktree on VM101.

## The lead's proof on the final files

- Fresh clone of 14815a7, the 14 landing files copied in with sha256 matching the worktree, CPU only, one suite at a
  time, phase A without the owner patch and phase B with it applied in the clone: 28 of 28 suite runs rc 0 (new suite
  A 59 passed 1 skipped, B 60 passed; the other 13 as listed above). Script
  `/home/michael/tfm_worktrees/_proof/w3_proof_run.sh`; logs `/home/michael/tfm_worktrees/_proof/w3_14815a7_logs/`.
- One card run on the final code, with the owner's go: a real compare-models run with a binding id on VM101's card,
  83.7 s, rc 0. All four children exited 0 and published; no pending record left. Winner restored to the public
  folder: xgboost. The health check graded the canonical report (catboost's, the last child to write) and raised one
  flag line, bound to the staged `cmp-catboost/best_model.cbm`; the runner read it as bound and found it by the
  flag's explicit path after the public checkpoint was overwritten. Files: `data/w3_fix1/real_gpu2/compare/` in the
  W3 worktree (RESULT.json, FLAG.json, cli.log). The same run on the pre-fix-2 code is in `data/w3_fix1/real_gpu/`.
  READY_W3 sections 8 and 11 speak of the first of these two runs only; the report was not edited after the proof
  so that the proven bytes stand.

## Limits

- Q2 is shown on the real parent code with a stand-in worker and inline trainer, not a real failing NN worker.
- No real WATCHER run and no pipeline launch. No GPU in the fix pass; the one card run is the lead's, above.
- Pool hit rate is not measured by any of this.

No reply is asked for.
