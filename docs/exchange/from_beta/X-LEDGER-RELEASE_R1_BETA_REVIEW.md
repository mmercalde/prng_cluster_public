## Prior record

1. **Finding 1 · TO-DO · ALREADY RECORDED.** The packet cites the August displacement report. The governing review records the verified backup sequence and limits its ratification to that one migration. The L7 ruling also governs why Phase 5 must release all references before staged files are removed and capacity is released. Carry both into the owner’s record for this one-off recovery. The later WAL ruling documents how committed rows can remain in the WAL after the sampler exits. `docs/TB_RULING_BRIEF_I_PRODUCTION_SHAPE_FAILURE.md:200`, `:206`, `:208`, `docs/TB_BINDING_RULINGS_S172_PHASE4.md:72`, `:84`, `:108`, `docs/CLAUDE_CODE_REPORT_STEP1_CREDIT_WAL.md:56`, `:72`, `:120`.
2. **ALREADY RECORDED.** The August report documents run-prefixed staging names and the verified relocation of staged files out of active staging. `docs/CLAUDE_CODE_REPORT_PROD_SHAPE_RUN_SCOPING.md:40`, `:42`, `:46`.
3. **NEW.** The scoped hand release of Run I’s 3,554 reservations. **CHANGES A RULING:** none. Michael decides whether to apply the one-off recovery.

Searched `docs/` and the exchange with `rg` for ledger displacement, backups and WAL, held reservations, staging capacity, abort discharge and credit, including `TB_RULING_*`, `PROPOSAL_*`, the chapter audits and `TODO_CONSOLIDATED.md`. Searched the local launch and coordinator paths for reservation readers, writers, cleanup and run identity. Ser8, private remotes, the Claude project, rigs, Proxmox and live ledger contents were outside the search.

# X-LEDGER-RELEASE · R1 · BETA REVIEW

- Date 2026-11-18 PST.
- Submitted base read `5605ed35298ac50cfcb06423cf5090c029a85f46`. Detached worktree HEAD `3579088b8e9c2d902252d626baae15bf0647907f`. The reviewed coordinator, sink, credit and verifier sources match the submitted base.
- Initial `git status --porcelain`

```text
 M docs/CHAPTER_13_IMPLEMENTATION_PROGRESS_v1_6.md
 M docs/LEADS.md
 M docs/SELFPLAY_ARCHITECTURE_PROPOSAL_v1_0.md
 M docs/TRIANGULATED_FUNCTIONAL_MIMICRY_VERIFIED_v1_0.md
 M docs/exchange/DEFERRED.md
 M docs/exchange/LEDGER.md
 M miner/range_miner_worker.py
 M scripts/launch_fleet_manual.sh
?? docs/CLAUDE_CODE_INSTRUCTIONS_RIG_GPU_FAULT_20261115.md
?? docs/CLAUDE_CODE_PLAN_RIG_ARM_V1_V2.md
?? docs/CLAUDE_CODE_PLAN_RIG_GPU_PROBE.md
?? docs/CLAUDE_CODE_REPORT_GCVM_LOCATED_20261116.md
?? docs/CLAUDE_CODE_REPORT_MINER_NO_TORCH.md
?? docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md
?? docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT_DECODE_20261115.md
?? docs/CLAUDE_CODE_RUNBOOK_RIG_FAULT_RECORDER_AB.md
?? docs/RIG_POWER_CONTROL_TUYA_REFERENCE.md
?? docs/SESSION_CHANGELOG_20261115_RIG_GPU_FAULT.md
?? docs/SESSION_CHANGELOG_20261116_RIG_FAULT_RECORDER.md
?? docs/TFM_SESSION_HANDOFF_20261118.md
?? docs/drafts/
?? docs/exchange/from_beta/X-CACHE-POLICY-D6_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-CACHE-POLICY_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-CACHE-POLICY_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-CACHE-POLICY_R3_BETA_REVIEW.md
?? docs/exchange/from_beta/X-GCVM-LOCATED_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-GCVM-LOCATED_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-GCVM-LOCATED_R3_BETA_REVIEW.md
?? docs/exchange/from_beta/X-MINER-NO-TORCH_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-PWC-CACHE-RESET_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R3_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-PROBE_R1_BETA_REVIEW.md
?? docs/exchange/to_beta/X-CACHE-POLICY-D6_R1_ALPHA_REVIEW_REQUEST_cache_policy_v7.md
?? docs/exchange/to_beta/X-CACHE-POLICY_R1_ALPHA_REVIEW_REQUEST_cache_policy_v4.md
?? docs/exchange/to_beta/X-CACHE-POLICY_R2_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-CACHE-POLICY_R3_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-CUPY-CACHE-RESET_R1_ALPHA_REVIEW_REQUEST_launcher_cache_reset.md
?? docs/exchange/to_beta/X-GCVM-LOCATED_R1_ALPHA_REVIEW_REQUEST_fault_address_reading.md
?? docs/exchange/to_beta/X-GCVM-LOCATED_R2_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-GCVM-LOCATED_R3_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-LEDGER-RELEASE_R1_ALPHA_REVIEW_REQUEST_hand_release_run_I.md
?? docs/exchange/to_beta/X-MINER-NO-TORCH_R1_ALPHA_REVIEW_REQUEST_cleanup_torch_flag.md
?? docs/exchange/to_beta/X-PWC-CACHE-RESET_R1_ALPHA_REVIEW_REQUEST_pwc_cache_reset_v1.md
?? docs/exchange/to_beta/X-RIG-GPU-FAULT_R1_ALPHA_REVIEW_REQUEST_fault_evidence_and_plan.md
?? docs/exchange/to_beta/X-RIG-GPU-FAULT_R2_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-RIG-GPU-FAULT_R3_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-RIG-GPU-PROBE_R1_ALPHA_REVIEW_REQUEST_torch_arms_probe_controller.md
?? scripts/rig_fault_map.py
?? scripts/rig_fault_recorder.py
?? scripts/rig_gpu_recorder.py
?? scripts/rig_seam_controller.py
?? scripts/rig_seam_probe.cleanup.sha256
?? scripts/rig_seam_probe.py
?? tests/test_miner_no_torch.py
?? tests/test_rig_fault_map.py
?? tests/test_rig_fault_recorder.py
?? tests/test_rig_seam_probe_controller.py
```

| Question | Position | Severity |
|---|---|---|
| Q1 · Is the hand release safe on this cluster? | NO OBJECTION | — |

## Q1 · Yes, under Alpha’s five conditions

The update uses the same two field changes as `release_reservation` and scopes them to held rows for Run I. I compared the 17:44 copy with Alpha’s post-update copy. Exactly 3,554 rows changed, all for Run I, and only `status` and `released_at` differed. All other rows and the schema matched. Both copies passed `integrity_check` and `foreign_key_check`. The global admission count went from `(591863 bytes, 3554 files)` to `(0, 0)`. `miner/range_miner_coordinator.py:1514`, `:1550`, `logs/ledger_sim/sim.py:4`, `:9`, `:12`, `logs/ledger_sim/sim.log:3`, `:8`.

Both later trials’ persisted terminal reasons show `staging_capacity_timeout` at 6,528 held files out of 6,528. The limit uses a global held count. In an isolated fixture with the production `reserve()` method, 3,554 held rows left room for 2,974 fresh reservations. After release, all 6,528 fit. That verifies the capacity mechanism. The console job counts and historical reservation totals measure different things from the held count at the timeout. `miner/range_miner_coordinator.py:1506`, `:1514`, `:1527`, `:5131`, `:5160`, `/tmp/X-LEDGER-RELEASE_R1_beta.cN7jQF/check.log`.

- **BLOCKS if the live transaction disagrees with the copy.** Run the global admission query inside the `BEGIN IMMEDIATE` transaction after the scoped update, before `COMMIT`. Require both 3,554 changed rows and `(0, 0)` globally, then verify again after commit. Roll back on either mismatch. To reproduce, start with all 3,554 Run I rows and add one held row for a different run. The scoped update still changes 3,554, but a fresh run needing all 6,528 slots stops at 6,527. A separate fixture with 3,553 Run I rows verified that rolling back preserves every row. `miner/range_miner_coordinator.py:1514`, `:1527`, `/tmp/X-LEDGER-RELEASE_R1_beta.cN7jQF/check.py`, `/tmp/X-LEDGER-RELEASE_R1_beta.cN7jQF/check.log`.
- **BLOCKS if the backup omits a nonempty WAL or cannot be read back.** While the stack is stopped and the sampler is closed, capture a stable database and any existing `-wal` and `-shm`, hash source and copies, and verify a WAL-aware read on a separate scratch copy. Record absent sidecars as absent. To reproduce, checkpoint a synthetic WAL database, commit a marker change with autocheckpoint disabled, then copy the main file alone and the main file with its WAL. Both copies pass `integrity_check`. Only the paired copy shows the new committed value. `window_optimizer_integration_final.py:1443`, `:1454`, `:1463`, `:1496`, `docs/TB_RULING_BRIEF_I_PRODUCTION_SHAPE_FAILURE.md:204`.
- **TO-DO in this recovery.** Keep the stack stopped through both the SQL and file archive. The Phase 5 sink holds Run I’s manifests in process memory, and a fresh production invocation creates a fresh run ID. That is why a dead Run I can be handled offline under the L7 invariant. Move the exact `distributed_config_t1_34a6727c__` prefix, including `.tmp.<pid>` files, into a new dated directory outside active staging. Record a filename and checksum manifest, verify the destination against it, then check that the prefix is absent from active staging before relaunch. If interrupted after SQL commit, finish the file archive under downtime using the manifest and the committed ledger state. `window_optimizer_integration_final.py:1804`, `:1808`, `:1924`, `miner/range_miner_npz_writer.py:1139`, `:1178`, `:1280`, `miner/range_miner_coordinator.py:4372`, `:4565`, `:4723`, `:8913`, `:12539`.
- **TO-DO in the maintenance record.** Record the operator, time, run ID, before/after counts, backup and archive paths, and that Run I’s `running` trial and stripe/shard rows were preserved as crash evidence. B8 will pass after the hand release, and C1/C2 can pass after archiving if their other checks are clear. Credit still requires a committed miner result and complete, run-scoped sweep evidence. B1 will still fail for Run I. Preserve the pre-repair evidence for any later assessment of Run I. `tests/gate_s172_prod_shape.py:229`, `:239`, `:322`, `:399`, `window_optimizer_integration_final.py:1479`, `:1487`, `:1522`, `utils/step1_search_ledger.py:1044`, `:1057`.

**Verification.** Activated `~/venvs/torch`. Opened both Alpha copies with `mode=ro&immutable=1` after confirming absent or empty WALs, and checked that their file metadata stayed unchanged. Ran the row and schema comparison, the in-memory admission and rollback cases, and the synthetic WAL backup check. Read Alpha’s simulation source and output without rerunning its update. Live ledger counts, process quiescence at the time of maintenance, and the live file archive are **UNAVAILABLE** to this review. Michael owns the maintenance decision and operation.
