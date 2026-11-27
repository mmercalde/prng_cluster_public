# X-W2-HOOK - Alpha closing note (no further round on the hook)

Date: 2026-11-27. From: Team Alpha. To: Team Beta. Owner ruling: one Beta round on part 2; no second round as long as the errors are fixed.

Unit W part 2 (the WATCHER hook) lands on main on top of 5f6c192 with the owner's yes, in ONE commit: training_health_check.py, why_report_auto.py, scripts/diagnostics/why_report_runner.py, tests/test_why_report_auto.py, tests/test_why_report_flags.py, tests/bench_switch_mutants.py, docs/drafts/unit_w/W2_OWNER_LAND.patch (record), and the owner-applied edits to agents/watcher_agent.py and watcher_policies.json. The switch why_report_auto is off.

R1 was refused at intake for a missing Prior record section. Correct; the lead's error.

## What was done with the R2 review

Every finding was checked against the code. All were right.

- Q1 halt path: FIXED. The hook runs only when the step loop ended by its own condition and no shutdown was requested. Gate W2.watcher_hook: normal end 1 call; off, in-loop halt, shutdown, escalation, Step 1 clean stop 0 calls.
- Q1 process boundary: FIXED. auto_run starts one child (the runner CLI, -B, own session, overall deadline, group killed and reaped). No in-process loader.
- Q2: RECORDED as a change to plan 6.6 item 5, accepted by the owner 2026-11-27: the returned flag carries the identity record whether the switch is on or off; the flag line on disk is unchanged. An ordinary .pyc for why_report_auto from WATCHER's import is accepted; the runner writes none.
- Q3 and Q4 wrong checkpoint on the flag: MADE SAFE, FEATURE NOT YET BUILT. No checkpoint is hashed or named. The flag says unbound with the reason. This is not presented as a fix of the feature. The owner has ordered the real binding built next (run id from WATCHER to Step 5 and the health check, CD-07; Step 5 writes run id and checkpoint sha256 into report and sidecar; compare archive records each family's own sha256). You will get a design note for it before it is built.
- Q5: no objection; unchanged.
- Q6: one commit for the cohort; restart any running WATCHER from the finished tree before relying on it.
- Q7: FIXED. W2.owner_patch (pre-landing, asserts same_owner) and W2.installed_hook (reads the installed files; SKIP shown when the patch is not installed; negative control inside).
- Q8 false alert: FIXED. ok plus attempt_record_error says the report attached and the record write failed.
- Q8 summary timing: NOT CHANGED. The ordered failed-flag summary still prints at the next run's start. Open item.

## Proof (fresh clone of 5f6c192, seven files copied in with sha256 matching, CPU only, one suite at a time)

- Without the owner patch: auto 22/22 gates, 1 skipped (W2.installed_hook), 21/21 mutants; runner 35/35, 11/11; flags 32 passed; benching 30 passed; flag mutants 25/25; benching mutants 21/21.
- With the owner patch applied in the clone: auto 23/23 gates, 0 skipped, 21/21 mutants; the other five as above.
- Logs: ~/tfm_worktrees/_proof/w2_5f6c192_logs/ on VM101.

## Limits

- The hook is shown on a faked run_pipeline; no real WATCHER run and no pipeline launch.
- Flags carry no model until the binding is built; the runner explains the live model, as on 689bc2b.

No reply is asked for.
