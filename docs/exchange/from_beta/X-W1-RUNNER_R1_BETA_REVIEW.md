# X-W1-RUNNER · R1 · Team Beta review · 2026-11-26

## Prior record

- I checked both prior-record sections in the READY note and ran `rg -n -i` for `why.report|why_report|attribution.cycle` and for checkpoint and grade provenance, failure and notification handling, and pooled sessions. The searches covered the project file catalog, `docs/TB_RULING_*`, `docs/TB_RULING_REQUEST_*`, `docs/PROPOSAL_*`, chapters and audits, consolidated TODO, backlog, leads, exchange, and the loop drafts. I read the learning-loop section of `CLAUDE.md`, the relevant owner logs, the W1/W2 plan, and the neighboring Beta reviews. I checked the local Run H development archive. Live run state, the rigs, Proxmox, ser8 archives, the private remote, and W2's proposed implementation were outside the review.
- **Finding 1 · ALREADY RECORDED.** The request's prior record omits the underlying owner rulings and the earlier LL-8 finding on historical ranking. The owner directed that poor health trigger a why-report while retaining the family, agreed to the two WATCHER stages, and deferred the queue until real reports arrive. The LL-8 review records why a checkpoint pin alone leaves historical Step 6 scoring unverified. These govern the interpretation of Q3 and the dormant landing in Q5. `logs/e2e_session/OWNER_RULINGS_pending_list_20261124.txt:20-28`, `logs/e2e_session/OWNER_RULING_benching_switch_20261126.txt:3-6`, `logs/e2e_session/OWNER_RULING_ll78_plan_20261126.txt:2-5`, `docs/exchange/from_beta/X-LL7-LL8_R1_BETA_REVIEW.md:52-64`.
- **ALREADY RECORDED.** The stage-1 flag interface, D-1 through D-5, the missing flag-time checkpoint, the expected `no_graded_block` on live checkpoints, the lack of Step 6 evidence, the unlearned vote source, and the attempt-file ignore rule are in the request and READY. The run-time checkpoint substitution is an acknowledged limit. The plan's original binding and failure tests are in §§6.4, 6.6–6.7 and 6.9. `docs/LEADS.md:889-897`, `docs/exchange/to_beta/X-W1-RUNNER_R1_ALPHA_REVIEW_REQUEST_why_report_runner.md:10-16,28-35`, `docs/drafts/unit_w/READY_W1_20261126.md:9-18,32-91,322-328`, `~/tfm_worktrees/_staging/ll78/PLAN_LAND_LL7_LL8_20261126.md:196-225`.
- **NEW.** Failure-log I/O can bypass recording, a killed child can leave workers alive, an unreadable attempts file disables the existing flag read, a pooled-only block can close a flag with empty contrasts, analysis overrides can change the declared draw range, and plain CLI invocation writes bytecode outside the output directory. **CHANGES A RULING.** None.

I read detached base `a85adb196adf703c500a4c0cbdfae505d560fe18` in `~/.local/state/api-agents/worktrees/20261126-213336-build-w1` from 22:45 to 22:49 PST. Main was at the same commit. All six submitted file hashes matched Alpha's request at first read and recheck.

`git status --porcelain` at first read in the detached worktree

```text
 M .gitignore
 M tests/test_why_report_flags.py
 M tests/why_flags_mutants.py
 M why_report_flags.py
?? data/
?? docs/drafts/unit_w/
?? scripts/diagnostics/why_report_runner.py
?? tests/test_why_report_runner.py
```

`git status --porcelain` at first read on main

```text
 M docs/exchange/DEFERRED.md
 M docs/exchange/LEDGER.md
?? docs/exchange/from_beta/X-1B-STEP6-NN_R1_BETA_REVIEW.md
?? docs/exchange/to_beta/X-1B-STEP6-NN_R1_ALPHA_REVIEW_REQUEST_step6_nn_refusal.md
?? docs/exchange/to_beta/X-W1-RUNNER_R1_ALPHA_REVIEW_REQUEST_why_report_runner.md
```

## Positions

| Question | Position | Severity |
|---|---|---|
| Q1 · Failure surfacing | OBJECTION | LIVE — A runner-log write failure can lose an attempt, and a killed vote child can leave workers running past the deadline. |
| Q2 · Flag reader compatibility | SUGGESTION | — |
| Q3 · Block and input selection | OBJECTION | LIVE — A supported pooled-only archive closes the flag with zero draw contrasts, and an analysis override can change the analyzed range while the attachment names the original block. |
| Q4 · Advisory write boundary | OBJECTION | LIVE — The documented plain CLI creates Python bytecode outside the allowed output and record paths. |
| Q5 · Dormant W1 on main | NO OBJECTION | — |
| Q6 · Tests | SUGGESTION | — |

## Q1 · Failure surfacing

- On the normal W1 path, closure requires `ok`, an existing report, the expected checkpoint hash and a second check of the report's SHA before appending the attachment. If recording the subsequent `ok` attempt fails, the report is already attached and the runner exposes `attempt_record_error`. Deletion after attachment remains the request's recorded limit. `scripts/diagnostics/why_report_runner.py:428-456`, `why_report_flags.py:144-145,218-238`.
- **Verified lost failure.** I injected `ENOSPC` into the `FAILED` log append after `no_checkpoint`. `run_for_flag` raised and left the flag open with no attempt line. The single-flag CLI calls it directly. The all-open wrapper recorded `internal_error` in the same probe and lost the original diagnostic reason. Protect the secondary log append so the original reason reaches `record_attempt` whenever the ledger is writable. If that append also fails, retain the original reason and expose the publication error in stderr and the result. `scripts/diagnostics/why_report_runner.py:457-483,495-508,543-549`.
- **Verified surviving worker.** A stand-in child started a worker in the same process group, then killed itself with `SIGKILL`. The runner returned `rc=-9` after 0.10 s, and the worker was still sleeping beyond the 0.50 s deadline. I killed the scratch worker. The real vote builder forks workers. `run_child` kills the group when its deadline branch runs. Reaping the leader first takes the early exit and skips group cleanup. Clean up the process group when a child fails or the wait is interrupted, and reap the direct child. The submitted timeout and killed tests cover direct-child outcomes. `scripts/diagnostics/why_report_runner.py:268-295,298-325`, `scripts/diagnostics/e2e_vote_npz.py:35,67-71`, `scripts/diagnostics/e2e_miss_report.py:455-472`, `tests/test_why_report_runner.py:577-583`.

## Q2 · Flag reader compatibility

With the attempts file absent, the base-comparison gate passed for the reader, open flags, summary, list formats and manual attach. The in-tree WATCHER hooks still use flag IDs, `open` and existing fields. Normal attempt records add fields to list JSON and a failure line to the summary. Notification retry and reminder selection still use the existing open flag and notification records. The reminder text stays generic. Plan §6.9 assigns notification of the failed state to W2's integration proof. `tests/test_why_report_runner.py:310-369`, `why_report_flags.py:147-163,203-205,242-269,323-330,349-415,433-449`, `agents/watcher_agent.py:117,2581,3035`.

An unreadable attempts file becomes a dependency of every `read_flags` caller. With an existing flag and a mode-000 attempts file in scratch, `--all-open --json` logged `PermissionError` and exited 0 with `[]`. The source also routes that error through the run-start hook's outer catch before it can list flags or send a reminder. Keep the base flags and attachment data readable when attempt history is unavailable, surface that condition explicitly, and let the runner return a nonzero outcome when it cannot enumerate work. The existing runner test explicitly expects an empty list on reader failure. `why_report_flags.py:76-92,149-163,369-389`, `scripts/diagnostics/why_report_runner.py:486-493,548-557`, `tests/test_why_report_runner.py:635-662`.

## Q3 · Block and input selection

- **Pooled-only grades.** Walk-forward supports `--grade-sessions pooled`, and records the dataset's actual sessions. W1 selects the pooled grade directory as a fallback and then removes `pooled` from the analysis sessions without substituting those actual sessions. I exposed only the real pooled grade for Run H origin 0 in scratch metadata and ran both real children. The grade had 594 outcomes. W1 returned `ok`, closed the flag, and attached a report with `contrasts: {}` and `c2: {}`. Use the dataset sessions when analyzing pooled-only grades, or leave the flag open with a reason if no graded draws enter any report cell. `scripts/diagnostics/e2e_walk_forward.py:435-437,653-659,730-734,1235-1243`, `scripts/diagnostics/why_report_runner.py:185-198,254-260,351-355,428-447`, `scripts/diagnostics/attribution_cycle.py:520-540,597-621,646-648`.
- **Analysis can replace provenance.** The `--analysis-json` option merges its object into the entire spec, including the block and grade paths. I reused the rebuilt Run H vote matrix, selected the regular grades, and passed `blocks={"elsewhere":[0,2]}` through the API used by the CLI. W1 reported `ok` and attached range `[3808,4402)`. LL-8 produced only the two `elsewhere` cells, each with zero draws. Limit overrides to declared analysis fields and keep the selected block, grade directories, sessions and model bindings derived from the archive. `scripts/diagnostics/why_report_runner.py:336-357,392-395,437-444,524-542`, `scripts/diagnostics/attribution_cycle.py:597-607`.
- **Model and grade binding.** With unmodified inputs, the selected frozen checkpoint and LL-8's loaded checkpoint are pinned to the bytes resolved at invocation. Flag-time identity remains the recorded W2 gap. The archived sidecar still supplies the feature order, and `step6_match.checked` is false without historical score evidence. For grades, checking a `per_draw.jsonl` against its adjacent `SHA256SUMS` proves local integrity. To establish that the grade belongs to the selected arm and draw data, verify the origin-pinned grader summary, its per-draw output digest, and its arm, pool, dataset and block bindings. The producer already saves the needed summary digest and the grader records those bindings. `training_health_check.py:1239-1258`, `scripts/diagnostics/why_report_runner.py:131-156,179-198,222-251,336-353,431-433`, `models/inference_bundle.py:236-264`, `scripts/diagnostics/attribution_cycle.py:478-508,547-557`, `scripts/diagnostics/e2e_walk_forward.py:1295-1312`, `scripts/diagnostics/e2e_grader.py:1097-1102,1331-1342`.
- **Unlearned votes.** The source rule is sound for all four families on driver-produced archives with the shared survivor rows, generator configuration, skips and block coordinates. The driver passes the same vote inputs to both arms, and the full vote matrix is made from each survivor's sequence before rank is applied. The rebuild checks the unlearned arm's written pools, and LL-8 checks survivor row order. This establishes the vote-value matrix under that common configuration. The family ranking and its Step 6 parity are separate evidence. Alpha exercised LightGBM and NN on the real block. I independently ran the real unlearned rebuild in the pooled probe. Real CatBoost, XGBoost and family-arm fallback runs are **UNAVAILABLE** in this review. `scripts/diagnostics/e2e_walk_forward.py:1167-1216`, `scripts/diagnostics/e2e_miss_report.py:402-437,443-473`, `scripts/diagnostics/e2e_vote_npz.py:49-91`, `scripts/diagnostics/attribution_cycle.py:509-518`, `docs/drafts/unit_w/READY_W1_20261126.md:113-132,149-150`.

These findings concern which observations the diagnosis uses. They establish no direction for hit rate. A change prompted by a report still needs an equal-`k` walk-forward comparison on draws unseen by the sieve against the active baseline and `k/1000`. `~/tfm_worktrees/_staging/ll78/PLAN_LAND_LL7_LL8_20261126.md:227-228`, `CLAUDE.md:43-66`.

## Q4 · Advisory write boundary

The settings hash gate passed. The explicit W1 report artifacts and LL-8 cache roots are under their configured outputs, and the stores append attempts and attachments. The documented `python3 scripts/diagnostics/why_report_runner.py` invocation created `__pycache__/why_report_flags.cpython-310.pyc` beside the module in a clean scratch copy with zero flags and bytecode enabled. Both children use `-B`. The parent imports the flag reader with the interpreter's bytecode setting, and the W1 suite disables bytecode and calls the CLI with `-B`. Make the supported CLI honor the write boundary and test it as documented. Also validate the resolved output root against the plan's `data/` boundary before creating any files. As written, W1 creates its own output tree before LL-8 rejects an external output root. `scripts/diagnostics/why_report_runner.py:43-50,67-70,268-277,381-387,398-415`, `scripts/diagnostics/attribution_cycle.py:358-364,421-438,705-728`, `why_report_flags.py:95-98,192-200,227-238`, `tests/test_why_report_runner.py:47-49,668-670,699-715`.

## Q5 · Dormant W1 on main

The current WATCHER entry points call the stage-1 hooks and the runner only executes its CLI from its main guard. The attempts path is ignored, and the absent-file path preserves ordinary flag operations. This position covers the dormant main path. Q1, Q3 and Q4 apply to W1's usable stand-alone command today. WATCHER integration evidence is **UNAVAILABLE** for this W1 review. `agents/watcher_agent.py:117,2581,3035`, `scripts/diagnostics/why_report_runner.py:560-561`, `why_report_flags.py:147-163`, `.gitignore:127-132`.

## Q6 · Tests

Add controls for the reproduced paths in Q1–Q4. The most consequential assertion is that a supported pooled-only origin supplies its graded draws to the report before the flag closes. Check that a denied attempt-history read preserves the stage-1 summary, a child with a surviving worker is cleaned up, and the original attempt reason survives a log append failure. Check the documented entry point with bytecode enabled. Plan §6.9 also names a W1 G-Q4-SCALER case. W1's `cycle_refused` gate currently uses reversed vote seeds. The LL-8 suite covers the missing historic scaler path itself. Carry that refusal through the W1 subprocess and assert the recorded reason and the open flag. `tests/test_why_report_runner.py:562-583,635-681,699-715`, `tests/test_ll8_attribution_cycle.py:671-672`, `~/tfm_worktrees/_staging/ll78/PLAN_LAND_LL7_LL8_20261126.md:223-225`.

## Checks run and evidence read

I activated the Torch venv and ran the submitted sources on CPU in `/tmp/beta-w1.j0fK12/`. The W1 runner suite passed 25/25 gates and killed 6/6 mutants. The flag and benching suites passed 62 tests combined. The flag mutation runner reported 25/25 including its clean control. I ran the scratch log, permissions, process, CLI and Run H probes described above. The pooled-only probe used Run H origin 0's existing LightGBM checkpoint, survivors, votes inputs and pooled grade, with scratch summaries and a synthetic flag. I read W1's source and test paths, the vote and LL-8 call chain, the walk-forward and grader producers, and Alpha's READY results. A real WATCHER invocation, live flag match rates, an actual out-of-memory kill and the post-landing main proof are **UNAVAILABLE**.
