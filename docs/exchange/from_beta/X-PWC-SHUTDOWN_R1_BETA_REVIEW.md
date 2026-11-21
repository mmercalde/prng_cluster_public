# X-PWC-SHUTDOWN · R1 · Team Beta review

## Prior record

I checked Alpha's prior record, read the cited lifecycle, reconnect, search-record and Stage 4 material, and searched `docs/` for worker ownership and overlapping runs, lost launch acknowledgments, cleanup visibility, ordinal and evidence binding, dataset provisioning and runbook logging. The search included `TB_RULING_*`, `PROPOSAL_*`, `CHAPTER_*_AUDIT.md`, `TODO_CONSOLIDATED.md`, the owner rulings, the Stage 3 and Stage 4 plans, and the exchanges. I did not search the rigs, Proxmox, ser8 archives, the private repository, live ledgers or the actual Stage 4 run trees.

| Label | Record |
|---|---|
| ALREADY RECORDED | S96B provides for clean planned exit. S156 C3 calls for a remote shutdown reap, and S163-KARG records the 60-refusal retry budget. `docs/PROPOSAL_S96B_PERSISTENT_GPU_WORKERS_v1_0.md:64-70`, `docs/PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md:271-297`, `docs/SESSION_CHANGELOG_20260419_S163KARG_P2.md:71-76`. |
| ALREADY RECORDED | The explicit-refusal path, lost PID acknowledgment and GPU telemetry in terminal paths were previously reviewed. The startup kill's host-wide scope is also recorded. `docs/exchange/from_beta/X-PWC-STAGE2_R1_BETA_REVIEW.md:76-88`, `docs/exchange/from_beta/X-PWC-STAGE1_R1_BETA_REVIEW.md:46-50`. |
| ALREADY RECORDED | The governing documents missing from the request's prior record are in Finding 1. The R2 generation-file guard and C3 launch findings, the retained-WAL documentation correction and the open search-pin binding are in `docs/exchange/from_beta/X-PWC-STAGE4_R2_BETA_REVIEW.md:53-63`, `:75-79`. |
| NEW | Worker exit on explicit shutdown, READY accounting and the TCP shutdown reap. The zero-row ordinal and log-token checks, A2b and C5b evidence failures, and shutdown-suite reporting and baseline issues are addressed below. |
| CHANGES A RULING | None. |

**Date** 2026-11-21 PST. **Base compared** `6eafad9df098e13202941cff95c2ef3f4cf268af`. **Worktree HEAD** `6d75f71d7b637c7a329429aef3b0ba29a806a355` in `pwc-backup`. I compared each existing review path to its blob at the requested base. The shutdown test is new there. All seven SHA-256s matched the request at intake and after testing.

`git status --porcelain` at intake

```text
 M docs/exchange/DEFERRED.md
 M docs/exchange/LEDGER.md
 M persistent/pwc_worker_service.py
 M persistent_worker_coordinator.py
 M scripts/compare_engine_outputs.py
 M tests/test_compare_engine_outputs.py
?? data/
?? docs/drafts/pwc_backup_stage4_v1/
?? docs/exchange/from_beta/X-PWC-STAGE4_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-PWC-STAGE4_R2_BETA_REVIEW.md
?? docs/exchange/to_beta/X-PWC-SHUTDOWN_R1_ALPHA_REVIEW_REQUEST_shutdown_and_stage4_fixes.md
?? docs/exchange/to_beta/X-PWC-STAGE4_R1_ALPHA_REVIEW_REQUEST_stage4_tooling.md
?? docs/exchange/to_beta/X-PWC-STAGE4_R2_ALPHA_CLOSING_NOTE_DRAFT.md
?? docs/exchange/to_beta/X-PWC-STAGE4_R2_ALPHA_CLOSING_NOTE_stage4_tooling.md
?? scripts/extract_miner_bundle.py
?? scripts/extract_pwc_bundle.py
?? tests/test_extract_miner_bundle.py
?? tests/test_extract_pwc_bundle.py
?? tests/test_pwc_shutdown_reap.py
?? tests/test_stage4_miner_launch.py
```

## Positions

| Question | Position | Severity |
|---|---|---|
| Q1 Planned shutdown ends the worker | NO OBJECTION | — |
| Q2 `_had_session` from READY | NO OBJECTION | — |
| Q3 Can the reap hit a non-leftover worker | OBJECTION | THEORETICAL — A second PWC coordinator must start on a rig while the first is still shutting down, outside the serialized Stage 4 procedure. |
| Q4 Can RANGE-MINER be affected | NO OBJECTION | — |
| Q5 Extractor record ordinal | OBJECTION | THEORETICAL — Accepting a foreign pass set requires mixed or modified evidence outside the prescribed fresh single-trial tree. |
| Q6 R2-Q5 pre-read live-path check | NO OBJECTION | — |
| Q7 Runbook corrections, incl. R2 Finding 2 | OBJECTION | LIVE — A2b loses both provisioning logs in a fresh tree, and C5b can present an unsuccessful process probe as an empty result. |
| Q8 Tests | OBJECTION | LIVE — The suite fails on clean HEAD once the fix lands and can count an unexecuted shell test as passed. |

## Finding 1 · Governing prior record omitted from the request

**ALREADY RECORDED.** Add `docs/VERIFICATION_INTEGRITY_STANDARD.md:30-47`, `:68-107` for cleanup observation, test skips and reporter safety. Beta already required the standard for PWC cleanup in `docs/exchange/from_beta/X-PWC-STAGE1_R1_BETA_REVIEW.md:42-44`. Add `docs/RUNTIME_DATASET_PROVISIONING_CONTRACT.md:82-90`, `:115-126`, `:156-167` for A2b's on-target verification and retained per-node evidence. A2b cites that contract in the runbook, outside Alpha's request's prior record. The overlap classification and evidence scope also depend on Michael's serial Stage 4 comparison and deferral of parallel trials, and on the design's same-job and comparison criteria. `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:47-50`, `docs/drafts/pwc_backup_stage3_v1/PLAN.md:299-325`.

## Reasoning

**Q1.** Explicit shutdown is the right terminal signal. Transport stop sends a `shutdown` frame. Both worker receive sites set the exit flag, and the worker entry point returns after handling it. Reuse across passes uses the same coordinator. A planned restart should launch a new pool. Workers that miss the frame can still need the reap. The earlier warning about GPU telemetry on terminal paths applies especially before init because the exit heartbeat still samples CuPy before closing. W1/W2 stub that call. `persistent/pwc_transport_tcp.py:241-261`, `persistent/pwc_worker_service.py:187-221`, `:260-274`, `:418-432`, `:673-696`, `:814-830`, `persistent_worker_coordinator.py:2775-2808`, `:2974-2975`, `docs/exchange/from_beta/X-PWC-STAGE2_R1_BETA_REVIEW.md:84-88`.

**Q2.** Marking the session after sending READY covers a disconnect from the ensuing job loop. READY here means the worker sent the frame. The coordinator separately registers the receipt. The budget is about five minutes of connection refusals at the configured cadence. Timeouts and other errors can take longer, and a successful handshake resets the count. Keeping initial retries is acceptable for this change. It preserves startup tolerance, and a definitive hello rejection already exits. A launched process that never reaches READY still needs either successful teardown or a later cleanup. `persistent/pwc_worker_service.py:180-244`, `:308-316`, `:365-395`, `persistent/pwc_transport_tcp.py:562-572`, `tests/test_pwc_shutdown_reap.py:119-130`.

**Q3.** The rig list is launch-scoped, while the match is host-wide. The old listener closes before the reap. A second coordinator can therefore bind and start workers during the first one's grace period if an operator overlaps two launches. The shutdown reap can then kill them. Existing startup cleanup is already host-wide. VM101 uses the captured `Popen` objects. For the prescribed run, retain exclusive use of the rigs until cleanup completes. Before permitting overlap, scope cleanup to verified process identities, for example PID with launch token or process start time. The current worker IDs can recur across launches. A bare PID can be reused. Record launch attempts separately from confirmed dispatch counts. If every acknowledgment on one rig is lost, the current reap omits that rig, even though a launch may have occurred. This is a consequence of the already recorded lost-acknowledgment case. `persistent/pwc_transport_tcp.py:241-261`, `persistent_worker_coordinator.py:214-224`, `:267-281`, `:1453-1478`, `:1497-1528`, `:1544-1587`, `:1614-1622`, `:1759-1807`, `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:10-11`, `:458-462`, `docs/exchange/from_beta/X-PWC-STAGE2_R1_BETA_REVIEW.md:84-88`.

**Q4.** The counted miner's local and remote launch commands execute `miner/range_miner_worker.py`. Their normal arguments do not match the shutdown reap's PWC pattern. The miner integration returns before the PWC dispatch call, and the reap is gated on a PWC TCP transport. The checked miner paths and fleet launcher match the base. This answer covers the new shutdown reap. `scripts/launch_fleet_manual.sh:344-351`, `:393-403`, `window_optimizer_integration_final.py:2024-2046`, `persistent_worker_coordinator.py:267-281`, `:1759-1767`.

**Q5.** The producer uses the integration's pre-incremented record ordinal for the pass filenames and candidate rows. Discovery and the NPZ row check are correct for populated arrays. The chunk link records trial counts and the generation but carries no ordinal. With zero NPZ rows, the other checks constrain pass shape, interval, time and thresholds. In a scratch fixture with an empty NPZ, I renamed all four files from `_t1` to `_t5` and added a phase 3 survivor while preserving mtime and an empty bidirectional set. The extractor granted credit and accepted the changed phase count. The original log rejected it. A log with paths ending `other-<expected basename>.bak` passed because the regex checks a substring with a word boundary. Run C6 with the actual stopped run's log and require it for zero-row bundles used as comparison evidence. Match the complete saved path token and report the empty NPZ check as unavailable for ordinal binding. If evidence is gathered from separate sources, retain hashes of the pass files and the log alongside the credited publication in a run receipt. Logs alone only establish what filenames were reported. `window_optimizer_integration_final.py:1239-1254`, `:2073-2075`, `:3273-3316`, `persistent_worker_coordinator.py:2607-2625`, `:2808`, `:2835`, `:2880`, `:2905`, `utils/step1_search_ledger.py:897-905`, `scripts/extract_pwc_bundle.py:176-227`, `:307-328`, `:467-470`, `:512-525`, `:541-569`.

**Q6.** The pre-read guard closes the R2 symlink case. Both targets are resolved by `refuse_live` before the sidecar is opened or the NPZ is hashed. The sidecar regression passed. Finding 2 covers the new input. `scripts/extract_pwc_bundle.py:119-125`, `:137-162`, `tests/test_extract_pwc_bundle.py:525-546`.

**Q7.**

- **C3.** R2 Finding 2 is closed for the documented invocation. The exact block refused missing, malformed and RANGE-MINER params and a failed directory change with no `nohup`. The valid control passed JSON through to the stub. `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:476-501`, `tests/test_stage4_miner_launch.py:193-272`.
- **A2 and A6.** Copying the ignored schema serves the post-publication result save. The explicit `PYTHONPATH=.` serves the direct search CLI imports. A6 should also chain its search commands to a successful `cd` and read of params because the CLI opens a cwd-relative database. The proposed schema copy and search commands were checked by reading. `.gitignore:41`, `core/results_manager.py:112-115`, `integration/sieve_integration.py:313-337`, `window_optimizer_integration_final.py:3708-3710`, `agents/watcher_agent.py:812-830`, `database_system.py:44-46`, `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:211-240`, `:346-361`.
- **A2b.** Provisioning per run-tree path is correct and needs the stated owner go. The snippet runs before A3 creates either `logs/stage4` directory. In two fresh scratch directories, the exact loop with a successful provisioner stub printed two `rc=0` lines and `tee` emitted two missing-directory errors. Neither log existed. Create the log directories before the rig write and check both provisioner and log-writer results before proceeding. Record success for each tree. The driver returns 0 when every declared node passes, while WATCHER still enforces its execution set before dispatch. `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:246-274`, `scripts/provision_dataset_fleet.py:85-95`, `:109-148`, `miner/dataset_authority.py:846-859`, `agents/watcher_agent.py:2112-2154`.
- **C5 and C6.** The `_t1` explanation and C6's explicit log path fit the fresh run. Apply the Q5 evidence limits. C6 still needs the R2 documentation correction for a retained WAL, which the extractor reads from a stable scratch copy. A5 still says the parity-governed files are unchanged. Put the redeploy prerequisite before A5/C2 and identify the changed worker pin there. `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:337-344`, `:464-470`, `:506-513`, `:546-566`, `scripts/extract_pwc_bundle.py:374-416`, `scripts/pwc_parity_gate.py:107-122`.
- **C5b.** The exact probe block printed `none` for local and remote `pgrep` exit 2 with stubs. Two other rigs with SSH exit 255 produced no per-rig result. Require a working probe and an explicit result on every host. Treat an SSH failure, absent result or probe error as UNAVAILABLE and stop. The fallback loop only contacts rigs, so give VM101 survivors an explicit stop-and-escalate branch. Require the success marker as well as exit 0 on each manual reap. Verify exclusive use of each rig before a host-wide kill. The coordinator's own probe-checked tail distinguishes no-match from observation failure. Its unconditional `All workers shut down` message can follow an error. Use the per-host results as the cleanup evidence. `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:515-544`, `:684-686`, `persistent_worker_coordinator.py:208-224`, `:1759-1769`, `:1781-1814`.

**Q8.** W1–W4 and the old-worker control exercise the worker loop. C2 verifies transport stop precedes cleanup, per-host results and local wait and kill. It does not assert concurrent overlap or exercise launch bookkeeping, a throwing transport stop, or failed local kill or wait. C1 supplies survivor and unavailable statuses directly to the classifier. C3 exercises the clean and kill paths in Bash. Add shell-level fault controls for a surviving process and unavailable probe. I confirmed with stubs that C3's skip is printed as PASS and that W-MUTANT fails when handed the reviewed source as HEAD. Give C3 an explicit UNAVAILABLE outcome on skip, and build the mutant from a fixed pre-fix source or controlled mutation so a clean post-landing checkout still exercises the check. Real TCP shutdown delivery, a busy chunk through the grace window and rig parity remain UNAVAILABLE. `tests/test_pwc_shutdown_reap.py:47-161`, `:178-254`, `:264-291`, `persistent_worker_coordinator.py:1544-1587`, `:1759-1814`.

## Finding 2 · The new run-log input bypasses the live-path guard

**NEW · THEORETICAL.** With the prescribed C6 path in the stopped scratch run tree there is no live-path access. In a fixture an override symlink led into a configured stand-in live root, and extraction opened the target and granted credit. Apply `refuse_live(run_log, "--run-log", live_paths)` before the first open and add a symlink regression. `scripts/extract_pwc_bundle.py:119-125`, `:213-219`, `:480-488`, `:518-520`.

## Evidence and limits

I activated `~/venvs/torch` and ran the three suites sequentially in a scratch archive of `6eafad9` overlaid with the seven reviewed files. The extractor passed **41/41**, the launcher passed **19/19**, and the shutdown suite passed **9/9**. For the reported shutdown run I used a private PID namespace with a private `/proc`, verified that the test initially saw only PID 1, and ran as UID 1000. An initial isolated trial used a namespace-local UID 0 mapping. Neither namespace could see host processes during C3. The test signalled only scratch dummies. I also ran the scratch cases in Q5, Q7, Q8 and Finding 2. The remaining production behavior was read at the cited lines.

Alpha's reported Stage 4 artifacts, wider suite results and deployment observations are relayed evidence. I did not run a worker, launch either engine, open a live ledger, provision a rig or verify the actual bundles. Rig shutdown timing, final deployed parity and on-target provisioning remain **UNAVAILABLE** in this review. The reported exit 3 establishes equality only for fields that were compared. Prediction-pool hit rate against `k/1000` on unseen draws remains **UNAVAILABLE**. `scripts/compare_engine_outputs.py:478-498`, `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md:9-16`.
