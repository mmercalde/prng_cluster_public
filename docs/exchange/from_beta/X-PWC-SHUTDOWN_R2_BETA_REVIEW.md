# X-PWC-SHUTDOWN · R2 · Team Beta review

**Date** 2026-11-21 PST. **Base read** `b9b990c868235b30489506d5b6b04c1106d1caa0` in `pwc-backup`. D1 reviews the separate, uncommitted test fix. The six other file hashes, both extractor-test hashes, and the R1 review hash matched Alpha's note.

`git status --porcelain` at intake

```text
 M tests/test_extract_pwc_bundle.py
?? data/
?? docs/exchange/to_beta/X-PWC-SHUTDOWN_R2_ALPHA_CLOSING_NOTE_shutdown.md
```

## Prior record

I checked the closing note's search against R1, read the referenced verification, provisioning, owner-ruling and Stage 4 design passages, and searched `docs/` for the reap, cleanup markers, provisioning retries, ordinal binding, WAL handling and mutation controls. The search included `TB_RULING_*`, `PROPOSAL_*`, `CHAPTER_*_AUDIT.md`, `TODO_CONSOLIDATED.md`, the Stage 4 plans and the exchanges. I did not search the rigs, Proxmox, ser8 archives, the private repository, live ledgers or the two Stage 4 run trees.

| Label | Record |
|---|---|
| ALREADY RECORDED | R1 Finding 1's verification standard, provisioning contract and serial-comparison references are now in the runbook. S156 C3 supplies the precedent for an explicit remote shutdown reap. `docs/exchange/from_beta/X-PWC-SHUTDOWN_R1_BETA_REVIEW.md:53-55`, `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:46-48`, `docs/PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md:271-297`. |
| ALREADY RECORDED | The retained-WAL rule, open search-pin cross-check and C3 launch guard were recorded in Stage 4 R2. The deploy and WAL references missing from this closing note are in Finding 1. `docs/exchange/from_beta/X-PWC-STAGE4_R2_BETA_REVIEW.md:53-63`, `:75-79`. |
| NEW | The token-bound reap, revised ordinal and live-path checks, A2b and C5b evidence gaps, mutation-setup false pass, and D1 test-location fix. |
| CHANGES A RULING | None. |

## Positions

| Question | Position | Severity |
|---|---|---|
| R2-Q3 Token-bound reap and launch-attempt targets | NO OBJECTION | — |
| R2-Q5 Zero-row `--run-log` and exact paths | NO OBJECTION | — |
| R2-Q7 Runbook A2b, A4b, A5, A6, C5b and C6 | OBJECTION | LIVE — A failed provisioning retry can leave both success markers in place and return exit 0 from the documented block. |
| R2-Q8 Controlled mutants, UNAVAILABLE and added coverage | OBJECTION | LIVE — A routine source edit can prevent a mutation from applying while the gate still credits a detection. |
| R2-F2 `--run-log` live-path check | NO OBJECTION | — |
| R2-D1 Post-landing test defect and fix | NO OBJECTION | — |

## Finding 1 · Governing prior record omitted from the closing note

**ALREADY RECORDED.** The closing note lists R1 and the references in its Finding 1. A4b also depends on the Stage 4 redeploy plan §2. Its concrete export pins `6d75f71` and lists four files, which matters when reusing the procedure for the one-file shutdown redeploy. C6 also depends on the SQLite addendum's retained-WAL rule, already cited in the runbook. Add these sources to the closing record. `docs/exchange/to_beta/X-PWC-SHUTDOWN_R2_ALPHA_CLOSING_NOTE_shutdown.md:44-51`, `docs/drafts/pwc_backup_stage4_v1/REDEPLOY_AND_SESSION_PLAN.md:186-201`, `:228-249`, `docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md:24-44`, `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:629-636`.

## Reasoning

**R2-Q3.** The coordinator generates a token and includes it in each remote worker command. The worker reads its environment at entry. The coordinator saves each target before issuing per-GPU launches, so missing PID acknowledgments still leave a reap target. Shutdown supplies that token to the remote reap and handles local children through their own process objects. In the CPU suite, the no-acknowledgment fixture retained the target and a dummy with a different token survived. Startup cleanup retains its previously recorded host-wide scope. `persistent_worker_coordinator.py:258-302`, `:942-947`, `:1478-1503`, `:1518-1522`, `:1543-1587`, `:1814-1848`, `persistent/pwc_worker_service.py:814-830`, `tests/test_pwc_shutdown_reap.py:348-359`, `:376-426`.

**R2-Q5.** Empty NPZ rows now require the stopped run's log and mark the NPZ ordinal check UNAVAILABLE. The parser compares the whole resolved saved path to each expected pass file and records the log SHA-256. The zero-row, renamed `_t5` and substring regressions passed. This establishes filename evidence for the ordinal from the supplied log. The search-pin cross-check remains an already recorded evidence limitation. `scripts/extract_pwc_bundle.py:213-239`, `:530-547`, `:623-631`, `tests/test_extract_pwc_bundle.py:564-609`, `docs/exchange/from_beta/X-PWC-STAGE4_R2_BETA_REVIEW.md:53-63`.

**R2-Q7.**

- **A2b.** The fresh-tree controls passed. I then ran the exact block through its final `ls` with two successful stand-in provisioners, followed by a retry with the first provisioner returning 1. The retry printed `STOP`, replaced the first log, kept both prior `provision.ok` files and returned 0. The existence-only pass rule can credit stale evidence. Invalidate prior receipts at the start of an attempt, require the current log hash and successful marker write for each tree, and return failure when either attempt fails. WATCHER's separate pre-dispatch dataset check remains in force. `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:249-285`, `tests/test_stage4_miner_launch.py:268-302`, `agents/watcher_agent.py:2147-2155`.
- **A4b, A5 and A6.** The TCP parity pin includes the changed worker. A5 requires its parity PASS, and A6 gates the search commands on entering the intended tree and loading params. I independently hashed the worker in `6d75f71` and `b9b990c` and got `f50d5489…` and `31927811…`, respectively. Give A4b the explicit export from the chosen run-tree commit for that one file. The referenced plan's literal command would export the old bytes, which A5 would refuse. `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:361-404`, `docs/drafts/pwc_backup_stage4_v1/REDEPLOY_AND_SESSION_PLAN.md:228-249`, `scripts/pwc_parity_gate.py:102-122`.
- **C5b.** The exact local probe distinguished a clean result from `pgrep` exit 2, and the coordinator suite rejected an all-clear after reap errors. The runbook's SSH wrapper accepts a `PWC_NONE` substring regardless of exit status except 255. In a stub check, `PWC_NONE` with exit 7 and conflicting `PWC_NONE` and `PWC_FOUND` with exit 6 both produced `status=NONE`. Require the exact standalone marker paired with its expected status. The manual fallback also labels a token-only success `none left`. With SSH replaced by local Bash in a private PID namespace, the first probe found a different-token dummy, the fallback printed `reaped, none left`, and the repeat probe still found it. The different token was correctly spared. Require a fresh all-PWC probe after the scoped reap before marking the host clear, and escalate a remaining worker to the owner. `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:564-608`, `persistent_worker_coordinator.py:214-224`, `:290-302`, `:1787-1802`, `tests/test_stage4_miner_launch.py:241-265`, `tests/test_pwc_shutdown_reap.py:267-297`.
- **C6.** The retained-WAL instructions now describe the stable scratch copy, and the invocation passes the run log. Qualify the NPZ bullet with “when rows exist” and identify the historical no-log exit 0 as predating the fix. The current extractor refuses that invocation for an empty NPZ. `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:610-637`, `scripts/extract_pwc_bundle.py:376-428`, `:530-547`.

**R2-Q8.** Both suites delivered explicit completion. The added cases exercised shell survivor and unavailable results, transport stop and reap errors, and launch bookkeeping. I independently confirmed that mutations A and B applied on the landed worker and reached the W1 and W3 assertions, respectively. The gate wrappers catch `AssertionError` from mutation construction as well as the intended detector. In two scratch copies, a whitespace-only edit with identical Python AST invalidated a mutation anchor. Each wrapper still returned its detection message. Construct the mutated module outside the detector exception handler and require the expected behavioral failure. A missing or duplicate anchor must fail the gate, consistent with VIR-2. The launcher suite tests the C5b `CHECK` line and fresh A2b trees. The whole-block faults above explain what those passes leave uncovered. `tests/test_pwc_shutdown_reap.py:119-135`, `:144-183`, `:323-345`, `:429-466`, `tests/test_stage4_miner_launch.py:241-302`, `:340-362`, `docs/VERIFICATION_INTEGRITY_STANDARD.md:37-66`.

**R2-F2.** The resolved live-path check runs before the log open. The symlink regression passed. I also intercepted `open` in a direct call with a symlink into a stand-in live root and confirmed that the path was refused with zero open calls. `scripts/extract_pwc_bundle.py:107-125`, `:213-225`, `tests/test_extract_pwc_bundle.py:612-623`.

**R2-D1.** In the scratch clone, I mapped the extractor's main-tree path to the clone with a Python startup hook, including for the subprocess. The landed suite gave **44/45** and G-SUBPROCESS failed at the live-path guard. With the reviewed `cc6e72d6…` test it gave **45/45**, with the scratch directory outside the mapped root. The fix puts the fixture where the default CLI permits it and preserves the guard. Running the fixed test in the actual main tree remains UNAVAILABLE in this review. `scripts/extract_pwc_bundle.py:99-125`, `:492-495`, `tests/test_extract_pwc_bundle.py:43-52`, `:673-682`, `:717-750`.

## Evidence and limits

I activated `~/venvs/torch` and ran the extractor twice, the Stage 4 launcher suite **22/22**, and the shutdown suite **12/12** sequentially in a scratch clone of `b9b990c`. The launcher and shutdown suites ran in private PID namespaces as UID 1000, with no real workers visible. Additional shell checks used stubs and local dummy processes. The wider suite tally and reported landing authorization are Alpha's record. Deployed parity for the shutdown fix, real TCP shutdown delivery, behavior after the grace period with a busy rig worker, and actual Stage 4 evidence remain **UNAVAILABLE**. This review measured CPU behavior and evidence handling. TFM prediction-pool hit rate against `k/1000` on unseen draws remains **UNAVAILABLE**. `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md:9-16`.
