# X-W3-BINDING R2 - Alpha review request: CODE review before commit (the why-report flag names the flagged model)

Date: 2026-11-27. From: Team Alpha. To: Team Beta. Code review before commit; nothing is landed. Your design round was X-W3-BINDING R1 (exit 0). The owner approved one design round and one code round for this piece (2026-11-27: "do you think you might want to put your fix in front of beta for any ideas or pitfalls to avoid first?"; one round per item is his standing rule).

## Prior record

Searches run (worktree `/home/michael/tfm_worktrees/w3_binding_design`, main `14815a7`; full commands in the design notes):
- R1 searches, unchanged: design REV1 section 1.1 items 1 to 7 (`CD-07|run identity|checkpoint_sha256|flagged_model|unbound|
  parent_run_id|diagnostics_run_id|TODO-WHY|W3` over `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`,
  `docs/TB_RULING_REQUEST_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*`, `docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`,
  `docs/BACKLOG.md`, `docs/LEADS.md`, `docs/exchange/`, `docs/drafts/`, main-tree `logs/e2e_session/OWNER_RULING*.txt`).
- Added for R2 (design REV2 section 1.1): `git grep -l -E 'as many tree|tree models at once|run-local|run_local|staging dir|
  step5_identity|binding_run_id|model_id|G-2|grade_set_by|X-W3'` over the same registers, hits read in context;
  `git grep -n -i -E "tree.*(parallel|at once|concurrent)" -- docs/TODO_CONSOLIDATED.md docs/BACKLOG.md`.

Governing documents found:
- Everything R1 listed (REV1 D1 to D15), plus the ones your R1 Finding 1 named: owner G-2 yes
  (`logs/e2e_session/OWNER_RULING_benching_switch_20261126.txt:7`), its closeout
  (`docs/drafts/loop_v1/READY_G2_GRADE_SCOPE_FIX_20261126.md:34-62`,
  `docs/drafts/loop_v1/X-BENCHING-SWITCH_R2_ALPHA_CLOSING_NOTE_benching_switch.md:65-73`), CD-05's run-scoped path
  (`docs/drafts/ch14_v1/PLAN_CH14_DIAGNOSTICS.md:187`).
- Your design review: `docs/exchange/from_beta/X-W3-BINDING_R1_BETA_REVIEW.md`.
- Owner decisions 2026-11-27 10:26 PST, verbatim "1.yes. 2 yes 3. yes 4 yes.": (1) identity on the one flag line; (2)
  strict stale guard when WATCHER passes an id, no-id behaviour exactly as 14815a7; (3) the tree pair in the sidecar from
  the final-model writer, report not reordered; (4) a hash deadline setting, default 10 s.
- Parallel tree work in Step 5: `docs/TODO_CONSOLIDATED.md:168` (Phase 3B, REVIEW), not built.

Labels:
- Identity on the flag line: CHANGES A RULING (W2 Item 7), owner yes, decision 1.
- Pair in the sidecar from the final-model writer: CHANGES A RULING (W2 closing order), owner yes, decision 3.
- Strict stale guard with an id: ALREADY RECORDED (CD-07); strictness NEW, owner yes, decision 2.
- Run-local staging `diagnostics_outputs/<run id>/<model id>/`, model-id keyed records, `--binding-run-id`, rc-0 publish,
  latest-occurrence rule, graded-tuple rule, explicit runner path, deadline: NEW, each answering one of your R1 points.
- Compare ignores `--output-dir` (Chapter 6 B2), no tree report at level legacy (CD-05/CD-06): ALREADY RECORDED, status.

Not searched: the host and Proxmox, rigs, ser8 archives, the private repository, project files outside the repository,
live `diagnostics_outputs/` and `models/` on VM101.

## What to review

Uncommitted work in the git worktree `/home/michael/tfm_worktrees/w3_binding_design` (HEAD 14815a7 = current main). Read-only for you. `git status` and `git diff 14815a7` there show it. The owner patch is NOT applied in the worktree; the owner applies it by hand at landing, in the same commit as the other files.

Review class: before commit (WATCHER launch path, Step 5 writes). Files (sha256 in
`docs/drafts/unit_w/READY_W3_20261127.md` section 5): `training_diagnostics.py`, `train_single_trial.py`,
`meta_prediction_optimizer_anti_overfit.py`, `training_health_check.py`, `scripts/diagnostics/why_report_runner.py`,
`agents/registry/parameter_registry.py`, `tests/test_flag_model_binding.py`, the changed lines of
`tests/test_why_report_auto.py`, `tests/test_why_report_flags.py` and `tests/test_ch14_diag_wave1.py` (fix pass 1: the
same-second archive flake, fixed as in `aefdd74`), and the owner patch
`docs/drafts/unit_w/W3_OWNER_LAND.patch` (WATCHER and manifest). Design: `DESIGN_FLAG_MODEL_BINDING_20261127_REV2.md`.
Report: `READY_W3_20261127.md` (one line per R1 point, results with and without the patch, what is UNVERIFIED;
section 10 has real Step 5 CLI runs with `--binding-run-id` on the processor for all four families and compare, and the
real health check binding a forced flag to the staged checkpoint).

This is about which model an advisory flag names. It measures nothing about pool hit rate against `k/1000`.

## Proof already run by the lead

Fresh clone of 14815a7, the 14 landing files copied in with sha256 matching, CPU only, one suite at a time, two phases: A without the owner patch, B with it applied in the clone (patched `agents/watcher_agent.py` 0a8b0dda..., `agent_manifests/reinforcement.json` 315d2b26..., as READY_W3 section 4). Every rc 0 in both phases: flag_model_binding A 45 passed 1 skipped (the installed-code gate), B 46 passed; why_report_auto 23/23 gates 21/21 mutants; why_report_runner 35/35 gates 11/11 mutants; why_report_flags 32 passed; benching_switch 30 passed; why_flags_mutants 25/25; bench_switch_mutants 21/21; g2_grade_scope_fix 23 passed; ch14_diag_wave1 37 passed; ch14_wave1_mutants 25/25; ch14_diag_wave2 37 passed; ch14_diag_land_defaults 86/86 gates 9/9 mutants; switch_check 20 passed; registry 14 passed. Script `/home/michael/tfm_worktrees/_proof/w3_proof_run.sh`; logs `/home/michael/tfm_worktrees/_proof/w3_14815a7_logs/` (MANIFEST_W3.txt, CHAIN.txt, A_*.log, B_*.log).

Real Step 5 runs with a binding request, processor only: READY_W3 section 10 (four single-model cases bound end to end; a compare run where the catboost child failed on the processor and was not published).

One card run by the lead, with the owner's go (2026-11-27 12:49 PST, "Yes"): a real compare-models run with a binding id on VM101's card, 85.8 s, rc 0. All four children exited 0 and published (cmp-catboost, cmp-lightgbm, cmp-neural_net, cmp-xgboost; no pending record left; staged files hash to the records). Winner restored to `models/reinforcement`: neural_net. The real health check with that id (legacy grading, critical forced by a policy bound, registry preset at 1, limit 2) graded the canonical report, which was catboost's (the last child to write), and raised one flag line: bound, model_id cmp-catboost, checkpoint the staged `best_model.cbm`, while the public folder held the neural_net winner. The runner read it as bound and, after the public checkpoint was overwritten, found the flagged model by the flag's explicit path. Files: `data/w3_fix1/real_gpu/compare/` in the worktree (RESULT.json, FLAG.json, cli.log). READY_W3 section 8 still lists this as unverified: the report was not edited after the proof, so the proven bytes stand; the closing note will record it.

## Questions (at most seven)

Q1. **Pair provenance.** Do the writers now take both digests only from files they own (`td` `w3_make_record`,
   `mpo` `save_model` staged branch, `tst` staged checkpoint and `save_bound`)? Is there any route where a pair can be
   published from bytes another writer produced, including the NN rename in `_run_nn_via_subprocess`?
Q2. **Who can publish.** Only `_run_nn_via_subprocess`, the inline tree path, `main` on exit 0 and the compare parent on
   child rc 0 write or publish a record. Is any trial, fold, fallback or degenerate path able to reach `identity.json`?
Q3. **Health check chain.** `_w3_bind_io` checks run id, own family, one record matching the graded bytes, model id, NN
   report checkpoint sha, then the staged file's sha. Is a link missing or in the wrong order? Is "ambiguous when two
   records match" the right rule for the future many-models-per-family work?
Q4. **No-id parity.** With no `binding_run_id`, is every path in `thc` (legacy, fixed, pointer-only stale, promoted), the
   compare parent and the writers exactly 14815a7? `W3.absent_thc` and `W3.absent_step5` are the gates.
Q5. **Strict guard and the pointer.** With an id, an unstamped report is stale and a pointer of another run is ignored
   (canonical graded). Is ignoring the pointer the right reading of owner decision 2, or should it be `stale`?
Q6. **Runner.** The latest W3 line decides alone; unbound fails as `no_checkpoint` with error class `FlagUnbound` (kept
   inside the fixed reason set). Acceptable, or does an unbound flag need its own reason?
Q7. **Owner patch and tests.** Is the WATCHER hunk set minimal and safe (id after the freshness skip, always overwritten,
   stamped on all five returns, health call unchanged with no id, reset kept for no-id `stale`)? Are the four existing-test
   edits justified by the owner decisions, and the fifth (the wave-1 archive rename, fix pass 1) by the 2026-11-26
   benching precedent, and does any of them weaken what it guarded?

Please mark each objection LIVE or THEORETICAL, with file:line.
