# X-W1-RUNNER  R1  REVIEW_REQUEST  Unit W1: the why-report runner (TODO-WHY-1, trigger half)

- **Status:** sent 2026-11-26 by the lead on the owner's instruction (22:43 PST, "one round on beta part one"). One round. Built by two Opus agent passes; the lead has read the runner, the why_report_flags.py diff and the second-pass diff in full, and the READY note. The lead has NOT read the test file line by line.
- **Review class:** BEFORE-COMMIT.
- **Base commit:** `a85adb1` (main).
- **Where to read:** detached worktree `~/.local/state/api-agents/worktrees/20261126-213336-build-w1` at `a85adb1`, changes in the working tree. READY note with both passes: `docs/drafts/unit_w/READY_W1_20261126.md` there (read its "Second pass" section first). sha256 (first 16): `scripts/diagnostics/why_report_runner.py` 35c58fc013a28a97 (new); `tests/test_why_report_runner.py` 6918228ebecc08c9 (new); `why_report_flags.py` eb87890b5bcca779; `.gitignore` 09429f54a64991ec; `tests/test_why_report_flags.py` 368d92bd4c827d12; `tests/why_flags_mutants.py` 450dcf3cc5e674e2.
- **Allowed answers per question:** `NO OBJECTION` / `OBJECTION` (with `LIVE` or `THEORETICAL`) / `SUGGESTION` / `NEEDS EVIDENCE`.
- **Scope:** real-world cases on this system as it is today. W1 is a stand-alone script; NOTHING calls it until W2 (the WATCHER hook, default off, hand-landed by the owner) exists. W2 is NOT under review; its proposed diff is in the worktree as text only.

## Prior record

- Plan: `~/tfm_worktrees/_staging/ll78/PLAN_LAND_LL7_LL8_20261126.md` sections 6 and 9 (outside the repository). Owner decision D4 there (stage 2 WATCHER is a named step). TODO-WHY-1: `docs/LEADS.md:891-897`.
- The missing `why_report_failed` state was recorded in `docs/exchange/to_beta/X-LL7-LL8_R1_ALPHA_REVIEW_REQUEST_ll7_ll8.md:74`.
- Your X-BENCHING-SWITCH R1 (`docs/exchange/from_beta/X-BENCHING-SWITCH_R1_BETA_REVIEW.md:39`, `:81-97`) reviewed the flag files, the WATCHER look-out and the git-ignore gate.
- Lead decisions for this unit (given to the agent, told to the owner 21:25-22:30 PST): D-1 the report uses the latest graded walk-forward block whose frozen checkpoint sha256 equals the model's; D-2 no usable checkpoint is failure `no_checkpoint`; D-3 one attempt per flag per call, no retry loop; D-4 advisory, changes no setting; D-5 a fixed set of eleven failure reasons; the votes matrix is rebuilt from the block's unlearned arm for every family (second pass).
- Searches the agent ran are listed in the READY note ("Prior record", both passes). Not searched: the host, ser8 archives, the private repository, the rigs.

## What it does

For one open why-report flag: resolve the model's checkpoint; find the matching graded walk-forward block under the given `--wf-root` dirs; check every recorded input by sha256; rebuild the votes matrix with `e2e_vote_npz.py` (child process); write an LL-8 spec; run LL-8 (`scripts/diagnostics/attribution_cycle.py`, unchanged) as a child process in its own session with one deadline; on success attach the report with its sha256 so the flag closes; on any failure append a failed attempt so the flag shows `why_report_failed` with a reason and is listed first by `summary_lines()`. `why_report_flags.py` is changed additively: `why_report_attempts.jsonl`, `record_attempt()`, `flag_status()`, optional `report_sha256`/`extra` on `attach()`; the new keys appear only when the attempts file exists.

## What was run (VM 101, CPU only)

- `tests/test_why_report_runner.py`: 25/25 gates, 6/6 mutants (final files). `tests/test_why_report_flags.py`: 32 passed. `tests/test_benching_switch.py`: 30 passed. `tests/why_flags_mutants.py`: 25/25.
- Real walk-forward block 0 of run H (`~/tfm_worktrees/e2e-map/data/e2e_runH_wf_00_base_smoke`, read-only, synthetic flag, scratch flags root): lightgbm and neural_net each end `status ok` with a report attached; the votes matrix from the unlearned arm is byte-identical to the one from the lightgbm family arm (sha256 3fc2150bd927...). 38 s and 46 s; peak about 0.76 GB and 1.0 GB.
- NOT run: the fresh-copy proof on main (comes after this round); anything under WATCHER.

## Known limits (stated so you do not have to find them)

1. **The flag line carries no checkpoint** (`training_health_check.py:1239-1257`), so the runner uses the sidecar that is live when it runs (`--models-dir`, default the Step 6 manifest dir). Recording the model at flag time needs plumbing through 3 to 4 call levels and the WATCHER call site (READY note, second pass, change 2). The lead puts it in W2, which already needs a hand-landed WATCHER edit.
2. **A live Step 5 model will usually match no walk-forward block** (the walk-forward trains its own models per origin), so on live flags `no_graded_block` is the expected result until the loop wiring (owner's item 6) decides where graded draws for a live model come from. Not measured on live data; from the code.
3. The runner's spec carries no Step 6 evidence, so LL-8 reports `step6_match.checked: false`.
4. The default analysis settings (`ANALYSIS_DEFAULTS`) come from the lead's untracked hand spec of 2026-11-24.
5. The `killed` reason is produced by a stand-in command, not a real out-of-memory kill. The fallback to the family arm for votes is shown only on the synthetic fixture.
6. The attached report is never re-checked after attach; the attachments file is append-only.

## Questions

**Q1. Failure surfacing.** Can any path close a flag without a report, lose a failure without an attempt line, or leave a child process running after the time limit?

**Q2. `why_report_flags.py`.** With no attempts file, is every existing caller's result unchanged (WATCHER's `watcher_run_start` / `watcher_on_health`, the CLI, notifications)? With an attempts file, can the new keys break any existing reader?

**Q3. Block and input selection.** Given known limits 1 and 2, can the runner today attach a report that explains a different model or different draws than it says it does? Is rebuilding the votes matrix from the unlearned arm sound for every family?

**Q4. Advisory only.** Does the runner write anything outside its out dir and the append-only records, or change any setting?

**Q5. Landing W1 before W2.** Is anything in W1 unsafe to have on main while nothing calls it?

**Q6. Tests.** Any gap that matters on a real run?
