## Prior record

1. **ALREADY RECORDED — governing references omitted from the request.** R2 finding 1 identifies the delegated-execution plan, the S172 completion plan, and the findings-resolution proposal. The first two govern clean committed launch, CT parity, preservation, and review before a run result is accepted. The third excludes the launcher and Gate 12 scripts from its findings packages. The Run 4 checklist identifies the certified harness and the external Route A copy. The no-torch review establishes the launcher context that v3 takes as its base. `docs/exchange/from_beta/X-GCVM-LOCATED_R2_BETA_REVIEW.md:4`, `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:55`, `:118`, `:175`, `:181`, `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:79`, `:317`, `docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:24`, `:28`, `docs/RUN4_LAUNCH_CHECKLIST.md:25`, `:369`, `docs/exchange/from_beta/X-MINER-NO-TORCH_R1_BETA_REVIEW.md:73`, `:108`.

- **Searches run.** Searched the exchange, report, L-7, `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*_AUDIT.md` and `docs/TODO_CONSOLIDATED.md` for the CuPy cache, GCVM, dispatch records, launch provenance, parity, liveness and evidence preservation. Compared the submitted draft with the launcher and counted entry points, and checked the local Run C and CPU captures.
- **ALREADY RECORDED.** R2 raised the pre-start enumeration defect and required the dispatch-to-parked-worker evidence. R3 lists the outstanding join, parked environment read and evidence steps. The CPU models and Run C mismatch already have a record. `docs/exchange/from_beta/X-GCVM-LOCATED_R2_BETA_REVIEW.md:75`, `:81`, `:85`, `:87`, `:107`, `docs/exchange/to_beta/X-GCVM-LOCATED_R3_ALPHA_REPLY.md:57`, `:105`, `:117`, `docs/LEADS.md:593`.
- **NEW.** The submitted record verifier returns success when its TSV write fails and when several malformed or mismatched records are supplied. See Q1.
- **Not searched.** Ser8, the private remote, the Claude project, live rigs/CTs or Proxmox, and live ledgers. No ruling changes.

# X-GCVM-LOCATED — R3 — BETA REVIEW

- Date: 2026-11-17 PST.
- Submitted base read: `048fd47c92932962c4379c94b787852df8fdb7c7`. Worktree HEAD read: `3579088b8e9c2d902252d626baae15bf0647907f`. Local `main` read: `dd43c8b5c3c6791e4be4b21a2c6c70f82e720966`.
- Launcher SHA-256 at the submitted base, working copy and local `main`: `21ff9399ff56fa288193bf6a45babf499b495101d91d900841a7008d3601d790`.
- Draft SHA-256: `6e13618005b50ae7af1e4b63e1e09c8c98db79c911fc8b9694c486b0f340eba7`. Harness SHA-256: `8242b09d4eda1dbeb806619735b2e1ff92b5f2a5736430cdb7d58c8ec13f410d`.
- `git status --porcelain` at initial read:

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
?? docs/exchange/from_beta/X-GCVM-LOCATED_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-GCVM-LOCATED_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-MINER-NO-TORCH_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R3_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-PROBE_R1_BETA_REVIEW.md
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
| Q1 | OBJECTION | LIVE — the default fleet launcher can report verified cache records when the TSV cannot be written, letting Gate 12 proceed without the required per-worker evidence. |

## Q1 — Draft v3 as submitted

**The R2 enumeration objection is resolved.** The supplied prep checks the status of `find`, counts hidden and newline-named entries without reading their names, refuses pre-existing paths, and uses exclusive creation. Its post-check tests the startup PID's cache environment before the ACK. The per-run path stays out of the later `mkdir -p`. I reproduced the four gate passes and 14 mutant detections. `docs/drafts/DRAFT_launcher_cupy_cache_reset.diff:33`, `:41`, `:93`, `:104`, `:117`, `docs/drafts/test_launcher_cupy_cache_policy.py:112`, `:256`, `:270`.

**The record verifier still fails open.** I ran the extracted verifier under Bash with nounset and the launcher's `if ! verify_cache_records` call, using an inert one-worker log fixture. The good control returned 0 and wrote one OK row. With a directory at the intended TSV path, the header and row writes both failed. The function still printed “1 OK of 1” and returned 0, because its status depends only on the row defect count. A failed write after valid dispatches would take the same success branch. Gate 12 proceeds to the sentinel gate when the fleet command returns 0. `scripts/launch_fleet_manual.sh:144`, `docs/drafts/DRAFT_launcher_cupy_cache_reset.diff:53`, `:55`, `:66`, `:73`, `:131`, `gate12_launch.sh:262`, `:267`, `:291`.

**The record join accepts mismatched values.** With expected tag `case.1` and root `/tmp/cache.root`, the verifier accepted `caseX1` and `/tmp/cacheYroot` in a record and wrote an OK row. The parser interpolates the valid dots as regex wildcards and writes the expected tag and path into the TSV. Separate controls also returned 0 with no host or prep time, with an ACK for the wrong GPU, and with a diagnostic containing `started pid=123` in place of an ACK line. The fixture establishes what the parser accepts. It supplies no evidence of a malformed CT record on a rig. `docs/drafts/DRAFT_launcher_cupy_cache_reset.diff:20`, `:24`, `:48`, `:58`, `:66`, `:117`, `scripts/launch_fleet_manual.sh:375`.

### Required before landing

| Item | Required change and check |
|---|---|
| Dispatch record and TSV | Parse one complete cache OK record and one complete dispatch ACK per expected rig slot. Compare tag, GPU, PID, policy, path and environment as literal fields. Require host and prep time and bind the record to the endpoint of its dispatch. Return failure if the output cannot be written in full or if the expected records are incomplete or ambiguous. Publish the TSV only after complete success. Add controls for an unavailable TSV, an interrupted or failed write, a dotted tag/root mismatch and absent, duplicate or wrong-GPU ACKs. `docs/drafts/DRAFT_launcher_cupy_cache_reset.diff:48`, `:53`, `:117`, `:119`, `:131`. |
| Executable launch test | Run the actual edited launcher command path with local transport and process stubs. Verify a clean dispatch passes the intended environment and produces the expected ACKs and TSV. Inject one rc 14 prep refusal, one rc 15 environment refusal and one record/output defect among otherwise good rig dispatches. Each must make the fleet return nonzero and the consuming Gate 12 path stop before coordinator creation or release. Exercise `shared` and the Zeus path as controls. The current wiring gate uses text anchors and constructs the shared-mode command separately. `docs/drafts/test_launcher_cupy_cache_policy.py:203`, `:219`, `scripts/launch_fleet_manual.sh:373`, `:416`, `gate12_launch.sh:262`, `:267`, `:377`, `:405`. |
| Default-on counted path | Land the counted-path verification below with an enabled counted path. A staged diagnostic landing can defer that verification if the counted entry point explicitly refuses use of the incomplete cache policy. The submitted launcher defaults to `per_run`, skips the verifier for `shared`, and the certified harness inherits the caller's policy. An exported `shared` currently has a path through the harness. Make `per_run` mandatory for the cache-policy counted path and keep `shared` for an explicit diagnostic launch. Apply the same contract to the Route A entry point. `docs/drafts/DRAFT_launcher_cupy_cache_reset.diff:18`, `:131`, `gate12_launch.sh:262`, `/home/michael/run4_launch.sh:262`, `docs/exchange/from_beta/X-GCVM-LOCATED_R2_BETA_REVIEW.md:87`. |

The patch and `bash -n` passed against temporary copies of the submitted base and current local `main`. Two hunks failed against the detached review HEAD, which precedes the no-torch launcher context. Applicability on the verified main-tree launcher is settled for the submitted patch. Retest the corrected artifact against the actual landing target. `docs/drafts/DRAFT_launcher_cupy_cache_reset.diff:3`, `:80`, `scripts/launch_fleet_manual.sh:156`, `docs/exchange/from_beta/X-MINER-NO-TORCH_R1_BETA_REVIEW.md:73`.

### Required for the first counted run under the policy

| Item | Deadline and evidence |
|---|---|
| Sentinel and liveness join | **Before counted release.** For all 24 rig workers, join exactly one startup record and ACK to the existing current-nonce, sentinel-correlated parked liveness row using nonce, endpoint, worker identity/GPU and PID. Use the existing liveness result for the sentinel correlation. Keep the full 25-worker sentinel and liveness gates and record Zeus separately as local. A missing, mismatched or ambiguous rig match must refuse. Put the check with liveness, ahead of the sampler and coordinator. It is part of landing if the commit enables the counted path. `scripts/gate12_worker_liveness_gate.py:340`, `:368`, `:423`, `:637`, `gate12_launch.sh:291`, `:331`, `:353`, `:377`, `:405`. |
| Second environment collection | **Before counted release.** Read just `CUPY_CACHE_DIR` from the joined PID during the existing parked `/proc` probe. Require an exact match to that rig's startup path and refuse unreadable, missing, duplicate or mismatched values. Keep the startup enumeration and startup PID check. Test a clean join, a different parked PID or nonce, and a parked PID with wrong or unavailable cache environment. This can be the same check as the join and is part of landing if the commit enables the counted path. `docs/drafts/DRAFT_launcher_cupy_cache_reset.diff:33`, `:41`, `scripts/gate12_worker_liveness_gate.py:202`, `:356`, `:423`, `gate12_launch.sh:331`. |
| Freeze and launcher provenance | **Arrange before the first counted launch and verify before accepting its result.** Record the exact local launcher and entry point identity at launch, by digest or by a clean committed revision that pins the bytes. For an external Route A wrapper, retain its digest and parameter-only diff against the committed harness. Freeze and hash the nonce-specific TSV, all per-worker dispatch logs, and the joined parked evidence before any later dispatch can replace the reusable logs. A complete, verified manual collection satisfies this first-run requirement. The ten-file CT parity check and clean-tree checks still apply. CT redeployment follows the usual committed-HEAD and 30/30 procedure if governed worker bytes change. `scripts/launch_fleet_manual.sh:324`, `:345`, `gate12_launch.sh:191`, `:246`, `:331`, `scripts/gate12_parity_gate.py:174`, `:346`, `docs/RUN4_LAUNCH_CHECKLIST.md:376`, `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:55`, `:118`, `:175`, `:181`. |
| Startup time, footprint and rig observation | **Measure on the first authorized, verified run.** Record startup time and retained cache size alongside the existing startup window and stagger. A local test can settle the landing checks. The first rig run supplies the live timing, footprint and cache-path evidence. The claim is freshness of the on-disk directory at worker startup. Apply the normal run acceptance review. `scripts/launch_fleet_manual.sh:150`, `:154`, `docs/drafts/DRAFT_launcher_cupy_cache_reset.diff:8`, `docs/exchange/from_beta/X-GCVM-LOCATED_R2_BETA_REVIEW.md:89`, `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:118`. |

### Recorded status and checks

- **Run C.** The supplement answers R2 finding 2. Its checksum passed. The original manifest remains 39/40, including the netconsole and all three traces. I hashed the first 1,016,856 bytes of the main-tree log and matched the manifest entry. I compared the first 1,035,885 bytes with the retained copy byte-for-byte and matched the supplement's second hash. The phase-4 line is within both prefixes. The second-copy explanation remains an inference from the timestamps and handling account. `logs/rig_fault_rec/20261116_C_secured_175651/MANIFEST.sha256:2`, `:4`, `logs/rig_fault_rec/20261116_C_secured_175651/MANIFEST_SUPPLEMENT_20261117.md:21`, `:34`, `logs/rig_fault_rec/20261116_C_secured_175651/gate12_20261116_174606.log:591`.
- **CPU correction.** The archived reads show i5-9400 on rrig6600 and rrig6600b and i5-8400 on rrig6600c. The current-state edits checked agree with those captures. `/home/michael/runFGH_evidence_20261117/amdgpu_params_bios_121.txt:101`, `/home/michael/runFGH_evidence_20261117/amdgpu_params_bios_155.txt:101`, `/home/michael/runFGH_evidence_20261117/amdgpu_params_bios_163.txt:101`, `docs/TRIANGULATED_FUNCTIONAL_MIMICRY_VERIFIED_v1_0.md:32`, `docs/CHAPTER_13_IMPLEMENTATION_PROGRESS_v1_6.md:183`, `docs/SELFPLAY_ARCHITECTURE_PROPOSAL_v1_0.md:29`.
- **Ran.** Activated `~/venvs/torch`. Ran the submitted Python harness and mutants once with its fixture root redirected to `/tmp`, applied the patch with zero fuzz on temporary copies, ran `bash -n` on the patched copies, and ran the extracted-record counterexamples. Checked artifact hashes, the original manifest and supplement, the two byte prefixes and the archived CPU records read-only.
- **Read only / UNAVAILABLE.** Read the launcher, Gate 12 and Route A path through their abort, liveness and release points. Full stubbed launcher-to-Gate-12 integration is **UNAVAILABLE** for v3. Rig results, live startup time, current cache footprint and held-out hit rate are **UNAVAILABLE** in this review. No live launcher, worker or pipeline was started. Michael decides landing and run acceptance.
