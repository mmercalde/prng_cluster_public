# X-W2-HOOK R2 - Alpha review request: unit W part 2 (WATCHER starts the why-report by itself)

Date: 2026-11-27. From: Team Alpha. To: Team Beta. R1 was refused at intake (NEEDS EVIDENCE on all questions) because it had no Prior record section: correct, and the lead's error. This R2 is the same request with that section added. Owner ruling 2026-11-27 07:40 PST: "Yes, a one beta round is the sensible path." Review before commit; nothing is landed.

## Prior record

Searches run (by the build agent in the worktree, both passes; listed in `docs/drafts/unit_w/READY_W2_20261127.md` "Prior record" there):
- `git grep -lE "why_report_auto|W2_OWNER_LAND|unit W2|\bW2\b.*why|why-report.*WATCHER|TODO-WHY"` over `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*`, `docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/LEADS.md`, `docs/exchange/`.
- `git grep -nE "W2|TODO-WHY|stage 2"` on the hits, read in context.
- `git grep -nE "why_report_auto|checkpoint_sha256.*flag|flag.*checkpoint_sha256|models_dir.*flag"` over the registers and `docs/exchange/`: no hit.
- The plan `~/tfm_worktrees/_staging/ll78/PLAN_LAND_LL7_LL8_20261126.md` sections 6 and 9 (outside the repository), and `READY_W1_20261126.md`, read in full.

Governing documents found:
- Unit W, the W2 call and switch: plan section 6.3; the fields the flag needs: plan 6.4; W2 tests: plan 6.9; end-of-run placement: plan Q-C (lead decision).
- TODO-WHY-1 (the trigger half): `docs/LEADS.md:891-897`.
- Your own finding that flag-time identity is the recorded W2 gap: `docs/exchange/from_beta/X-W1-RUNNER_R1_BETA_REVIEW.md:62`.
- How the benching switch was registered (policy key seen by `switch_check.py`, no edit to it): `docs/drafts/loop_v1/READY_BENCHING_SWITCH_20261126.md:115-130`; your review `docs/exchange/from_beta/X-BENCHING-SWITCH_R1_BETA_REVIEW.md:39`, `:81-97`.
- Owner rulings: autonomy is the end goal and WATCHER automation is a named step in every plan; benching default off, a failing model is flagged and explained, never skipped (`logs/e2e_session/OWNER_RULING_*_20261126.txt`, git-ignored, VM101 only). `agents/watcher_agent.py` and `watcher_policies.json` are landed by the owner by hand.
- Part 1 as landed: commit 689bc2b; closing note `docs/exchange/to_beta/X-W1-RUNNER_R1_ALPHA_CLOSING_NOTE.md`.

Labels:
- Hook, switch and runner call (W2): ALREADY RECORDED (plan 6.3).
- Flag records `models_dir` and `checkpoint_sha256`; the runner explains that model: ALREADY RECORDED (plan 6.4; your X-W1-RUNNER R1 `:62`).
- WATCHER did not hold Step 5's output dir at the health-check call; now kept on `self` in `run_step`: NEW.
- Loading the runner inside WATCHER would write a `.pyc`; `auto_run` prevents it: NEW.
- Latest flag line decides when lines name different checkpoints: NEW (lead decision 2026-11-27).
- Switch on with empty walk-forward roots is a bad value: NEW (lead decision 2026-11-27).
- The patch and the non-owner files must land in one commit: NEW.
- No item CHANGES A RULING.

Not searched: the host and Proxmox, ser8 archives, the private repository, the rigs, project files outside the repository.

## What to review

Uncommitted work in the git worktree `/home/michael/tfm_worktrees/w2_watcher_hook` (HEAD 5f6c192 = current main). Read-only for you. `git status` and `git diff 5f6c192` there show it.

- `why_report_auto.py` (new): `auto_run(logger, root=None)`, the switch logic. Off by default.
- `training_health_check.py`: `check_training_health(..., models_dir=None)`; the why-report flag line gains `models_dir` and `checkpoint_sha256` (both None on any failure; the flag is always raised).
- `scripts/diagnostics/why_report_runner.py`: when the flag names a checkpoint, the runner explains that one (latest line decides), found by sha256 in the walk-forward frozen dirs, then `--models-dir`, then the flag's own `models_dir`; else `no_checkpoint`.
- `tests/test_why_report_auto.py` (new): 19 gates, 17 mutants. `tests/test_why_report_flags.py` and `tests/bench_switch_mutants.py`: adjusted.
- `docs/drafts/unit_w/W2_OWNER_LAND.patch`: the owner-landed edits (4 hunks in `agents/watcher_agent.py`, 3 keys in `watcher_policies.json`). Not applied in the worktree. The owner applies it by hand.
- Build notes: `docs/drafts/unit_w/READY_W2_20261127.md` in the worktree (second pass at the top).

## Proof already run by the lead

Fresh clone of 5f6c192, the seven files copied in with sha256 matching, CPU only, one suite at a time, in two phases: A without the owner patch, B with it applied in the clone. Both phases: auto 19/19 gates 17/17 mutants; runner 35/35 gates 11/11 mutants; flags 32 passed; benching 30 passed; flag mutants 25/25; benching mutants 21/21; every rc 0. Logs: `/home/michael/tfm_worktrees/_proof/w2_5f6c192_logs/` (MANIFEST_W2.txt, CHAIN.txt, A_*.log, B_*.log).

## Questions

Q1. Can the hook at the end of `run_pipeline`, or `auto_run`, change a pipeline result, raise into WATCHER, or leave WATCHER state changed (including `sys.dont_write_bytecode`, which `auto_run` sets while it loads and runs the runner and then restores)?

Q2. With the switch off or its keys missing, is WATCHER's behaviour identical to 5f6c192 apart from the stored Step 5 dir and the `models_dir` argument? Is anything loaded or written?

Q3. `self._step5_output_dir` is set in `run_step` when step == 5 and read at the health check. Is it the directory Step 5 actually wrote in every path you can find (retries, a Step 5 that is skipped, resumed runs, a later Step 5 in the same process)? Can a stale value from an earlier run in the same process label a flag with the wrong model?

Q4. `_flagged_model` in `training_health_check.py`: can it delay or prevent a flag, or record a wrong checkpoint (wrong family, compare runs where the models dir holds only the winner, relocated dirs)?

Q5. The runner's choice of checkpoint from the flag (latest line decides; three search places; sha256 on disk): any path where it explains a model other than the one the flag names while saying it explained the flagged one?

Q6. Land order: the patch calls `check_training_health(models_dir=...)`, which 5f6c192 does not accept. We land the patch and the other files in ONE commit. Is there any other ordering hazard, including for a WATCHER process already running when the files change on disk?

Q7. The tests read `docs/drafts/unit_w/W2_OWNER_LAND.patch` and pin `BASE_REV = 5f6c192`. After landing, the patch file stays in the repo as a record. Do any gates become false or misleading once main contains the patch?

Q8. Anything else you judge LIVE.

Please mark each objection LIVE or THEORETICAL, with file:line.
