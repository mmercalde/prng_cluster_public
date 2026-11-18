## Prior record

1. **Finding 1 · ALREADY RECORDED, omitted governing references.** The D6-I1 certification record identifies the preserved dispatch semantics and launcher surface. Carry forward the direct Run 4 pinned-geometry ruling and the Step 1 search proposal, which govern the required operator-originated pins, credited range and treatment of a completed search. The request cites the no-torch report for the D6 count. Its deployment and Run 4 result entries also apply to the checklist. `docs/SESSION_CHANGELOG_20260815_S183.md:41`, `:114`, `:183`, `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:204`, `docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md:291`, `docs/CLAUDE_CODE_REPORT_MINER_NO_TORCH.md:190`, `:195`.
2. **ALREADY RECORDED.** Cache-policy R3 sets the D6 repair and first-run evidence requirements. GCVM R3 sets the counted-path join and parked environment read. The Run 4 checklist, Route A R2, delegated-execution plan and S172 completion plan govern the live launch and acceptance. `docs/exchange/from_beta/X-CACHE-POLICY_R3_BETA_REVIEW.md:76`, `:93`, `docs/exchange/from_beta/X-GCVM-LOCATED_R3_BETA_REVIEW.md:88`, `docs/RUN4_LAUNCH_CHECKLIST.md:348`, `:510`, `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:145`, `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:55`, `:118`, `:175`, `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:56`, `:317`.
3. **NEW.** The v7 fixture repair and independent checks below. The WS-7 literal changes as requested in R3, and its per-GPU requirement remains tested. Flag the replacement of the certified suite's literal for Michael.

Searched the exchange, drafts, leads and session reports, `docs/TB_RULING_*`, `docs/PROPOSAL_*`, chapter audits and `docs/TODO_CONSOLIDATED.md` with `rg` for D6-I1, WS-7, cache-policy coverage, Run 4 parity and credit, Route A and timing evidence. Checked the pinned launcher objects and the submitted base in Git. Ser8, private remotes, the Claude project, live ledgers, rigs and Proxmox were outside the search.

# X-CACHE-POLICY-D6 · R1 · BETA REVIEW

- Date 2026-11-18 PST.
- Submitted base read `048fd47c92932962c4379c94b787852df8fdb7c7`. Worktree HEAD read `3579088b8e9c2d902252d626baae15bf0647907f`.
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
?? docs/SESSION_CHANGELOG_20261115_RIG_GPU_FAULT.md
?? docs/SESSION_CHANGELOG_20261116_RIG_FAULT_RECORDER.md
?? docs/drafts/
?? docs/exchange/from_beta/X-CACHE-POLICY_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-CACHE-POLICY_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-CACHE-POLICY_R3_BETA_REVIEW.md
?? docs/exchange/from_beta/X-GCVM-LOCATED_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-GCVM-LOCATED_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-GCVM-LOCATED_R3_BETA_REVIEW.md
?? docs/exchange/from_beta/X-MINER-NO-TORCH_R1_BETA_REVIEW.md
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
?? docs/exchange/to_beta/X-MINER-NO-TORCH_R1_ALPHA_REVIEW_REQUEST_cleanup_torch_flag.md
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
| Q1 · Landability | NO OBJECTION | — |
| Q2 · D6 repair | NO OBJECTION | — |
| Q3 · First counted run checklist | SUGGESTION | — |

## Q1 · LANDABLE on the submitted base

The verdict covers the four diffs, three scripts, cache test and Route A fixture at the submitted hashes. All ten listed artifacts and four target hashes matched. All four diffs applied with zero fuzz on temporary copies. The applied D6 file and patched Route A matched the supplied copies byte for byte. The three previously reviewed production diffs, three scripts and fixture also matched v6 byte for byte. The detached worktree HEAD precedes the no-torch launcher context. A zero-fuzz dry run against that HEAD's launcher failed two hunks, as previously recorded. Land against `048fd47` or an exact matching target. `docs/drafts/cache_policy_v7/README.md:23`, `docs/drafts/cache_policy_v7/launch_fleet_manual.diff:3`, `:67`, `docs/exchange/from_beta/X-GCVM-LOCATED_R3_BETA_REVIEW.md:86`.

The landed-layout cache suite passed **111/111** using the patched launcher, both patched entry points and the three scripts. The shallow installed copy parsed `--landed --help` successfully. Calling that copy in pre-landing mode reported the missing draft layout. Draft discovery now occurs in the pre-landing branch. The D6 blocker is closed by the evidence in Q2. Michael decides whether to land. `docs/drafts/cache_policy_v7/tests/test_cache_policy_v7.py:31`, `:67`, `:800`.

## Q2 · The D6 repair

The diff leaves the historical anchor pins, red assertions and mutant bodies intact. The two pinned launcher objects match their full SHA-256 pins and have no cache-policy branch. The new fixture selects `shared`. WS-7 checks the shared per-GPU path and the remote command's use of it. WS-2f ties each remote ACK PID to exactly one `CUPY_CACHE_DIR` in its NUL-delimited environment and requires the expected directory to exist for each of the two distinct slots. I observed both correct paths with the D6 fixture and reproduced the collapsed-path failure by substituting GPU 0 in the shared branch. Alpha's retained logs show the unpatched baseline **82/82**, v6 with the unmodified suite **71/82**, and v7 **84/84** in all three runs, including WS-2f, both historical red arms and M4–M6. See `logs/cache_policy_v7_evidence/` for the completion counts. `docs/drafts/cache_policy_v7/test_s172_d6_integration_repair.diff:18`, `:40`, `:67`, `docs/drafts/cache_policy_v7/d6/test_s172_d6_integration_repair.py:610`, `:650`, `:1028`, `:1480`, `:1539`.

The counted policy runs the same detachment, settle and dispatch-status logic, with cache prep before the worker starts, a process-environment check before the ACK and record verification before dispatch completion. The landed cache suite exercises real stand-in processes and an executable SSH pipe. It covers the default policy while stand-ins are alive, the Zeus control, prep and environment failure, record refusal, and both entry-point slices through join and freeze. Its sentinel and liveness gates are stand-ins. D6's zero-REGISTER observation and immediate remote exit-status assertion still run on the shared path. I separately exercised the D6 SSH pipe with the policy unset and a fixture-local cache root and hostname. The clean case returned 0 with distinct paths and a verified TSV. Immediate remote exit 9 was preserved for both slots, the launcher returned 1 and the TSV was absent. This leaves the real loopback assertion in D6 and the policy-specific failures in the landed cache suite. I found no landing blocker in that division of coverage. `scripts/launch_fleet_manual.sh:335`, `:373`, `docs/drafts/cache_policy_v7/launch_fleet_manual.diff:79`, `:104`, `:120`, `docs/drafts/cache_policy_v7/d6/test_s172_d6_integration_repair.py:1042`, `:1315`, `docs/drafts/cache_policy_v7/tests/test_cache_policy_v7.py:240`, `:287`, `:585`, `:678`.

## Q3 · First counted run checklist

The three R3 corrections are present. I compared the patched external wrapper directly with patched Gate 12 and found a `--params`-only difference. All **17** values and JSON types match `RUN4_PARAMS`. The README carries forward the full Run 4 checklist and specifies timing collection before dispatch, the 25 nonce-matched worker logs, the verified pre-sampler freeze, and the footprint JSON hashed into the whole-run bundle. Retain the installed wrapper digest, parameter comparison and committed WATCHER CLI proof as separate evidence. `docs/drafts/cache_policy_v7/README.md:84`, `:110`, `:124`, `:149`, `docs/RUN4_LAUNCH_CHECKLIST.md:320`, `:348`, `:376`, `docs/drafts/cache_policy_v7/gate12_launch.diff:76`, `docs/drafts/cache_policy_v7/run4_launch.diff:76`, `miner/range_miner_worker.py:2202`, `:2214`, `:2223`.

Two clarifications are needed before using this as an execution checklist.

1. **Reconcile the historical parity and search steps with the actual pre-run state.** The referenced checklist expects a 27/30 coordinator-only mismatch and the original upper-half gap. The selected base changes a governed worker file, and the no-torch report already records 30/30 deployment of `048fd47` and a credited Run 4 attempt. Take the parity comparison from the committed tree used by the installed wrapper. Use the measured table to decide whether committed-HEAD deployment is needed and require a fresh 30/30 at admission. Have Michael confirm the authorized search, exact run scope and a runnable gap before dispatch. WATCHER generates no Step 1 run for a completed search. Current parity and search state are **UNAVAILABLE** in this review. `docs/RUN4_LAUNCH_CHECKLIST.md:134`, `:160`, `:202`, `:304`, `:337`, `docs/CLAUDE_CODE_REPORT_MINER_NO_TORCH.md:190`, `:195`, `scripts/gate12_parity_gate.py:174`, `agents/watcher_agent.py:2239`, `gate12_launch.sh:79`.
2. **Place the cache preflight before Run 4 step 13.** README item 1 includes the launch, then items 2–4 describe prerequisites. Put the landed tests, installed wrapper comparison and timing collection setup before step 13 in the written sequence. Carry forward the committed-tree checks in Route A R2's pre-launch sequence, including the infrastructure items still applicable under Michael's rulings. The delegated plan removes power caps as a general prerequisite for unattended runs. `docs/drafts/cache_policy_v7/README.md:90`, `:110`, `:124`, `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:145`, `docs/TODO_CONSOLIDATED.md:80`, `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:20`.

## Checks and limits

Activated `~/venvs/torch`. Ran the hash and patch checks, the landed-layout cache suite, the landed-layout mutation checks (**22/22 detected**), the shallow-layout checks, the Route A parameter comparison and the focused D6 fixture checks in temporary directories. Full independent D6 execution is **UNAVAILABLE** because the sandbox denied loopback socket creation. The 84/84 results above are from Alpha's retained logs. The full live entry points, committed-tree Route A provenance run, live timing and cache footprint are **UNAVAILABLE** here. This review establishes an offline landing position. Acceptance of a counted result requires the run-scoped production-shape verdict, publication, search and coverage confirmation, complete hashed evidence and Michael's decision under the recorded procedure. `docs/RUN4_LAUNCH_CHECKLIST.md:510`, `:533`, `:552`, `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:317`.
