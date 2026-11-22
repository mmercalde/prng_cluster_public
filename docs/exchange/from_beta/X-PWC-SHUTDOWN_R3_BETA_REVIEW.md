# X-PWC-SHUTDOWN · R3 · Team Beta review

**Date** 2026-11-21 PST. **Base read** `ab2d330bb5bd74deed716c823f2cfb9efd7fa60c` in `pwc-backup`. The five file hashes and the R2 review hash match the closing note. This is the post-landing record for Michael.

`git status --porcelain` at intake

```text
?? data/
?? docs/exchange/to_beta/X-PWC-SHUTDOWN_R3_ALPHA_CLOSING_NOTE_stage4.md
```

## Prior record

I read R1 and R2, checked Alpha's cited passages in the redeploy plan and SQLite addendum, and searched the Stage 4 drafts, exchanges, `TB_RULING_*`, `PROPOSAL_*`, `CHAPTER_*_AUDIT.md` and `TODO_CONSOLIDATED.md` for export and digest failures, wrong-tree or wrong-token cleanup, and mutation controls. I read the saved confirmation evidence on VM101. Rigs, Proxmox, ser8 archives, the private repository and live ledgers were outside the checks.

| Label | Record |
|---|---|
| ALREADY RECORDED | R2's positions, the redeploy plan's four-file `6d75f71` export, and the retained-WAL rule. The last two are now cited in `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:49-50`. The governing sources omitted in R3 are listed in Finding 1. |
| NEW | The R3 runbook and mutation-test changes, the saved confirmation evidence, and two runbook failure cases reproduced below. |
| CHANGES A RULING | None. |

## Positions

| Question | Position | Severity |
|---|---|---|
| R3-Q7 A2b receipts, A4b export, C5b status and re-probe, C6 | OBJECTION | LIVE — A partial export can be copied over a deployed rig worker even after A4b reports a digest mismatch. |
| R3-Q8 Mutant construction outside the detector and self-test | NO OBJECTION | — |
| R3-E1 Rig confirmation run as evidence for the shutdown fix | NO OBJECTION | — |

## Finding 1 · Governing prior record omitted from the closing note

**ALREADY RECORDED.** Add the verification integrity standard for mutant independence and cleanup observations, and the provisioning contract for A2b's on-target digest and retained evidence. Alpha's prior-record section names R2 and its Finding 1 sources. These additional sources govern the current questions and were listed in R1 Finding 1 and the runbook. `docs/VERIFICATION_INTEGRITY_STANDARD.md:30-47`, `:68-107`, `docs/RUNTIME_DATASET_PROVISIONING_CONTRACT.md:82-90`, `:115-126`, `:156-167`, `docs/exchange/from_beta/X-PWC-SHUTDOWN_R1_BETA_REVIEW.md:55-57`, `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:46-47`.

## Reasoning

**R3-Q7.**

- **A2b.** The success and failing-retry controls passed. The first attempt wrote both current log hashes. The retry returned `A2B_STATUS=1` and removed both previous receipts. This closes the R2 retry case. `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:262-295`, `:300-309`, `tests/test_stage4_miner_launch.py:282-328`.
- **A4b.** The export uses the run-tree commit and the one worker file. The clean control reached all three stub copies after a matching digest. With `tar` stubbed to leave a partial file and return 2, the same block printed `export MISMATCH - STOP`, reached the first backup and copy stubs, and exited 0 after the post-copy mismatch. In an ordinary shell, the pre-copy check only prints the stop message. A5 can refuse a subsequent launch, but the worker file can already have been overwritten. Require a successful export and matching digest before entering the copy loop. Return failure when any rig step fails. `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:393-405`, `:413-421`.
- **C5b.** The classifier passed all ten cases. With the intended log, the stubbed fallback selected the correct token and printed an all-clear only after a fresh `NONE`. `FOUND` and SSH exit 255 escalated. A second check set `$TP` to a missing directory while the shell was in another run tree. The failed `cd` left the shell in that tree, the reap stub received its token, and the block printed an all-clear. That can select a leftover worker from a previous run. Require successful entry to the intended tree and a readable run log before generating or sending any reap command. `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:610-632`, `:645-672`, `tests/test_stage4_miner_launch.py:331-351`.
- **C6.** The runbook now qualifies the NPZ ordinal check by row presence and dates the historical no-log extraction. The zero-row tests confirmed that the current extractor requires the log and rejects renamed pass files and substring matches. `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:685-706`, `tests/test_extract_pwc_bundle.py:564-609`.

**R3-Q8.** `run_mutant` builds and loads the mutant before the detector handler. The self-test passed for both whitespace-broken anchors and the non-unique anchor. I also checked the actual detector output. Mutant A reached W1 with 61 connects and 59 reconnect sleeps. Mutant B reached W3's `never exited` assertion at the 500-sleep guard. The clean controls passed. This closes the R2 mutation-setup objection. `tests/test_pwc_shutdown_reap.py:123-139`, `:148-221`.

**R3-E1.** The saved parity JSONs show PWC **39/39 MATCH** and gate12 **30/30 MATCH**. All three worker digests match the commit bytes. The confirmation tree is at `b9b990c`. Its four pass files each account for 10,738/10,738 complete chunks with zero failures and timeouts. The run log records the launch token, then transport stop at 16:13:27, then `exited` for all three rigs at 16:13:30. The classifier requires a grace probe that found no workers with this token and a successful final probe. This supports clean shutdown of the confirmation run's worker set. `~/stage4_run_pwc2/logs/stage4/pwc_run.log:143-144`, `:86245-86253`, `persistent_worker_coordinator.py:290-316`, `:1829-1845`.

The all-PWC `pgrep` on the rigs and VM101 is lead-reported. Raw probe output and the rig kill, survivor and unavailable paths are **UNAVAILABLE** as rig evidence. The log also shows a separate WATCHER post-step GPU cleanup returning SSH 255 for `.120`, `.154` and `.162`. It uses the default config and addresses different from the three successful PWC reaps. `docs/RESULT_PWC_BACKUP_STAGE4_20261121.md:127-143`, `~/stage4_run_pwc2/logs/stage4/pwc_run.log:86238-86251`, `agents/watcher_agent.py:1887-1906`, `gpu_cleanup.py:112-159`.

## Finding 2 · Stale landing status in the result record

The result record still describes the R2 runbook and test changes as uncommitted in two places. The changes are in `ab2d330` alongside the result record. `docs/RESULT_PWC_BACKUP_STAGE4_20261121.md:119-120`, `:187-188`, `docs/exchange/to_beta/X-PWC-SHUTDOWN_R3_ALPHA_CLOSING_NOTE_stage4.md:12-17`.

## Checks run

I activated `~/venvs/torch` and ran the affected suites from a scratch checkout of `ab2d330`. The PID-isolated runs at UID 1000 gave shutdown **13/13** and Stage 4 launcher **24/24**, each with zero unavailable. The extractor gave **45/45**. An initial run of the two PID-isolated suites mapped Michael to UID 0 inside a private user namespace. I repeated both at UID 1000 and used those results here. The A4b and C5b checks used local stubs. I read the saved rig artifacts locally and made no rig connections. Unseen-draw prediction-pool hit rate against `k/1000` is **UNAVAILABLE** in this evidence. `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md:9-16`.
