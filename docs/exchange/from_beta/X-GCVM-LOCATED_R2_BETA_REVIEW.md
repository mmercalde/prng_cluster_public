## Prior record

- **Searches run, 2026-11-17.** Read Alpha's reply, the revised report, L-7, the V1/V2 plan, the November 15 decode, and Beta's earlier GCVM and rig-fault reviews. Searched `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, the exchange, and the local GPU-fault register for GCVM, cold/warm caches, launch admission, parity, nonces and cache freshness. Cross-checked the draft against the launch path, and read the retained run evidence listed below.
- **Finding 1 — ALREADY RECORDED, omitted from the request's governing record.** The delegated-execution plan governs committed-tree parity and fresh launch nonces (§3), environment and gate changes (§4), counted-run acceptance (§6), and the fault stop and evidence freeze (§8). The S172 completion plan lists the applicable launch and evidence gates and distinguishes launch from acceptance (§3.A and §7). The findings-resolution proposal also designates the launcher as no-touch for its findings packages (§0). These scopes matter for Q1–Q3. This review changes no ruling. (`docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:55`, `:69`, `:118`, `:181`; `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:79`, `:317`; `docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:24`, `:28`.)
- **ALREADY RECORDED.** L-7 has the 8/8 versus 4/4 tally and both crossed boot/cache outcomes. The register contains the historical counterexamples. R1 already kept translation and hardware visibility open. The earlier cache-reset draft is superseded. (`docs/LEADS.md:539`, `:565`, `:579`, `:590`; `docs/exchange/from_beta/X-GCVM-LOCATED_R1_BETA_REVIEW.md:97`, `:123`; `docs/exchange/to_beta/X-CUPY-CACHE-RESET_R1_ALPHA_REVIEW_REQUEST_launcher_cache_reset.md:26`.)
- **NEW.** Independently reproduced the timing table and compared archived cache contents. The draft's listing-error behavior and the Run C manifest mismatch are new review observations. The hardware correction is already recorded in L-7 and corroborated by the local captures. (`docs/LEADS.md:593`; `/home/michael/runFGH_evidence_20261117/amdgpu_params_bios_121.txt:101`, `amdgpu_params_bios_155.txt:101`, `amdgpu_params_bios_163.txt:101` in the same directory.)
- **Not searched.** Ser8, the private repository, the Claude project and owner-channel transcript, live rigs/CTs/Proxmox, and live ledgers. The exact installed driver source and hardware flush/dispatch observations are **UNAVAILABLE** in this review.

# X-GCVM-LOCATED — R2 — BETA REVIEW

- Date: 2026-11-17, America/Los_Angeles.
- Requested run base: `048fd47c92932962c4379c94b787852df8fdb7c7`.
- Worktree HEAD read: `3579088b8e9c2d902252d626baae15bf0647907f`, plus the submitted working-tree files. The launcher and Gate 12 entry point match the requested base before the draft. The local worker hashes to `2bca2a083262a523801739bc453744c92a1937e38eaa745a95eaaf13baf2ca20`. The retained A–H source-digest bundles report that hash locally and on each CT.
- Reviewed draft SHA-256: `58f9de35ce8425880c87b3c3270ba2fff09f4303459cd1388a54d0e82f0d827b`.
- `git status --porcelain` at initial read:

```text
 M docs/LEADS.md
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
?? docs/exchange/from_beta/X-MINER-NO-TORCH_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R3_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-PROBE_R1_BETA_REVIEW.md
?? docs/exchange/to_beta/X-CUPY-CACHE-RESET_R1_ALPHA_REVIEW_REQUEST_launcher_cache_reset.md
?? docs/exchange/to_beta/X-GCVM-LOCATED_R1_ALPHA_REVIEW_REQUEST_fault_address_reading.md
?? docs/exchange/to_beta/X-GCVM-LOCATED_R2_ALPHA_REPLY.md
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
| Q1 | NO OBJECTION | — |
| Q2 | OBJECTION | LIVE — A CT directory-listing error can be recorded as verified emptiness and let a counted launch proceed with an unverified cache. |
| Q3 | SUGGESTION | — |
| Q4 | NO OBJECTION | — |

## Q1 — Fresh per-worker cache for counted runs

No objection to using a verified fresh on-disk CuPy cache for each ROCm rig worker on each counted launch. The unit in L-7 is one fleet attempt, with the relevant outcome on rrig6600/rrig6600c. It records eight clean cold attempts and four faulting warm attempts. Run C demonstrates that the fault can occur with a warm cache on the first run after boot. E, G and H show that later runs on the same boot can complete cleanly with fresh caches. Same-boot attempts share history and the set is small, so the table supports an operating choice for this launch shape with residual fault risk. The 11-15 boot/cache assignments on the affected CTs carry the caveat in the decode report. (`docs/LEADS.md:544`, `:565`, `:584`, `:590`; `docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT_DECODE_20261115.md:150`, `:157`.)

The retained D–H console logs each record Step 1 credit. The retained C netconsole independently records the phase-4 fault, and the D/E and F–H netconsole windows contain no GCVM/page-fault lines in the relevant time windows. F–H have zero recorded trace-ring overruns, commit overruns and drops on all hosts. Those are observed run outcomes. For a future counted run, apply the usual launch, stop, coverage, publication and acceptance requirements and include the worker-level cache evidence from Q2. A recurrence with verified fresh caches on the current production path should reopen the mitigation assessment. Michael decides the operating policy and counted acceptance follows its usual evidence review. Held-out hit rate is **UNAVAILABLE** here. (`/home/michael/runCDE_evidence_20261116/runD/console.log:978`, `/home/michael/runCDE_evidence_20261116/runE/console.log:913`; `/home/michael/runFGH_evidence_20261117/runF/console.log:1112`, `/home/michael/runFGH_evidence_20261117/runG/console.log:894`, `/home/michael/runFGH_evidence_20261117/runH/console.log:843`; `logs/rig_fault_rec/20261116_C_secured_175651/netconsole_all_rigs.log:2185`; `logs/rig_fault_rec/20261117_F/runF/trace_loss_counters_post.txt:1`, corresponding G/H files; `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:118`, `:181`; `AGENTS.md:21`.)

## Q2 — Launcher landing requirements

**Fix before landing.** The draft requires plain `mkdir`, then tests only the output of `ls -A`. When the listing fails without stdout, the test sees an empty string and the CT command emits `entries=0`. I applied the draft to a temporary launcher copy, ran `bash -n`, and exercised the extracted policy and remote prep under Bash with temporary paths and an inert marker. An injected listing error printed its diagnostic, emitted the zero-entry record, reached the marker and returned 0. Fresh, pre-existing, shared, bad-policy, bad-tag and injected nonempty controls reached their expected outcomes. Check both enumeration status and emptiness, and make any entry count, including hidden entries, explicit. A fixed nonempty marker from a successful enumeration avoids mistaking a filename made entirely of newlines for empty output. Keep the per-run cache out of the later generic `mkdir -p` call so only the verified creation path can supply it. Make the prep refusal visible as remote rc 14 with a reason. (`docs/drafts/DRAFT_launcher_cupy_cache_reset.diff:14`, `:42`, `:45`, `:54`; `scripts/launch_fleet_manual.sh:373`, `:382`.)

| Gate | What the positive and refusal controls must prove | Mutant the gate should kill |
|---|---|---|
| Policy and isolation | Unset defaults to `per_run`. A valid nonce selects a distinct path per CT/GPU; a different nonce selects a new path. A nonce-free manual dispatch produces a fresh tag. Invalid mode/tag refuses before dispatch. Explicit `shared` preserves the old per-GPU path and its files. Zeus retains its local cache. | Default to `shared`, bypass validation, omit tag or GPU from the per-run path, or apply the rig path to Zeus. |
| CT preparation | The submitted command creates an empty directory in the CT fixture and passes that exact path as `CUPY_CACHE_DIR` to an inert worker stand-in. A repeated nonce, a pre-existing file/directory/symlink, creation failure, a hidden entry, and an enumeration error refuse with rc 14 before the stand-in can start. | Change exclusive creation to `mkdir -p`, bypass the nonempty check, ignore enumeration status, or give the stand-in the old/default path. |
| Counted launch and evidence | Exercise the shell argument/env path and downstream abort with local stubs. One worker's prep failure makes fleet dispatch fail before coordinator creation and release. Successful dispatch still requires the existing sentinel and liveness checks. A missing, stale, duplicate, shared or path-mismatched rig record cannot establish a per-run cache launch. | Ignore a single worker refusal, accept a launch ACK alone as cache verification, or omit the nonce/worker/path join. |

Test exit status, side effects, the emitted evidence, and the stand-in's effective environment. Run clean controls alongside fault injections so an unavailable test path cannot earn mutant credit. The host dispatch already aggregates remote failures, and Gate 12 aborts a failed fleet launch before sentinel verification. Keep that ordering and the existing post-dispatch barriers. (`scripts/launch_fleet_manual.sh:286`, `:324`, `:370`, `:416`; `gate12_launch.sh:261`, `:267`, `:291`, `:331`, `:405`.)

**Parity.** Keep the ten governed project files and the 30/30 requirement. The parity gate checks those deployed file digests. The draft changes the VM101 launcher, outside the CT import closure, so a launcher-only commit calls for the normal clean-tree and parity gates with no additional CT source deployment solely for this hunk. Pin the local launcher revision or digest in the run bundle. If the implementation adds evidence fields to a governed worker file, deploy it from committed HEAD and require 30/30 again. (`scripts/gate12_parity_gate.py:156`, `:174`, `:306`; `gate12_launch.sh:191`, `:246`; `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:55`, `:59`.)

**Run evidence.** Require a fresh CT-side success record for each of the 24 ROCm workers, carrying the run nonce or tag, CT/host, worker GPU ordinal, policy, exact cache path, successful exclusive creation, successful pre-start enumeration with zero entries, and time. Join it to that dispatch's PID/ACK and the current-nonce sentinel/liveness identity. Verify the same worker PID received the expected `CUPY_CACHE_DIR` before release, for example by collecting just that variable during the existing parked-process probe. This can reuse the existing dispatch and liveness evidence paths. Retain the records in the run's frozen evidence because the dispatch filenames are reused. Record Zeus separately as local. An unverified rig row leaves the per-run condition unverified, and an explicit `shared` arm supplies diagnostic evidence only for this mitigation. A later cache listing or a global policy banner establishes neither the pre-start directory state nor the worker's effective path. (`docs/drafts/DRAFT_launcher_cupy_cache_reset.diff:34`, `:45`, `:58`; `scripts/launch_fleet_manual.sh:287`, `:324`, `:344`; `scripts/gate12_worker_liveness_gate.py:335`, `:364`; `gate12_launch.sh:331`; `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:175`.)

Scope the launcher claim to a fresh on-disk directory at worker startup. Startup already runs CuPy kernels, and the sieve retains kernels by family inside a process. Any claim about a particular phase compiling requires phase-level evidence. Keep the existing stagger and startup window and capture startup time on the first verified run. Old directories accumulate during a CT's uptime. Record the footprint and leave cleanup to the existing operating procedure, outside worker startup. (`docs/drafts/DRAFT_launcher_cupy_cache_reset.diff:7`, `:11`; `miner/range_miner_worker.py:1718`; `sieve_gpu_worker.py:124`; `scripts/launch_fleet_manual.sh:150`, `:154`, `:352`.)

## Q3 — Add the driver-side KFD flush decision record

For one warm diagnostic run, choose the driver-side record. Preserve the existing timing, workload and cache arm. For the phase-4 buffer and code mappings, and any preceding unmaps relevant to those VAs, record host monotonic time, BDF, process and VM/PASID identity, map/unmap identity, update/last-flush sequence values, skip reason or invalidate domain and target, and request/result if observable. Include completion status when the installed path exposes it and label it **UNAVAILABLE** otherwise. Join to the retained map/PTE/interrupt/fault rows. Compare the first faulting VM/card with workers in the same run that reached the same phase on each host. Record loss and whether the instrumented warm timings still overlap the existing warm measurements. The submitted timing script measures buffer map to code map. A map-to-interrupt interval includes work beyond dispatch, and the actual user-mode dispatch instant remains unobserved here. (`logs/rig_fault_rec/allgaps.py:30`, `:36`; `docs/CLAUDE_CODE_REPORT_GCVM_LOCATED_20261116.md:188`, `:251`.)

- A skip, failed invalidate, or missing expected completion in the faulting VM narrows the driver lead, particularly if matched healthy VMs show a different decision for comparable updates. A common skip on healthy workers is weak discrimination. A verified invalidate for the relevant VM/domain after the last relevant update weakens the specific missing-flush account. An unobserved completion or a clean warm run leaves causation open.
- The packet's driver reading distinguishes a flush-path call from a branch that can skip. Until measured, describe successful maps as returning after the reported decision path, with invalidate issuance and completion unobserved. Run C's second fault was a read at a previously unmapped VA. It weighs against a still-usable cached translation satisfying that particular access, while the initiating fault and VM visibility elsewhere remain open. Likewise, use “no covering mapping in the retained capture” for B's first fault pages. (`docs/exchange/to_beta/X-GCVM-LOCATED_R2_ALPHA_REPLY.md:148`, `:156`, `:169`; `logs/rig_fault_rec/20261116_C_secured_175651/netconsole_all_rigs.log:2205`; `logs/rig_fault_rec/20261116_C_secured_175651/stream/trace_192.168.3.163.log:62839`, `:63047`; `docs/exchange/from_beta/X-GCVM-LOCATED_R1_BETA_REVIEW.md:97`.)
- A delay is a timing intervention for a later comparison. If testing whether cold compilation's delay matters, place the treatment relative to the interval actually being tested, since the present contrast is before code mapping. Forced scratch allocation changes allocations and timing together. Either intervention could change recurrence without isolating the driver mechanism. Driver recording availability on the installed stack is **UNAVAILABLE** locally. Any driver change falls outside delegated rig authority. (`docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:65`, `:69`.)

## Q4 — Contradictions and limits

- **Historical GCVM.** The PWC-era rows challenge a universal cache explanation. They are outside the present phase-4 cohort. C-15 and C-34 are early/first-launch events, C-47 was trial 2 and the register classifies it as later by trial, and C-86 is a young-boot rejoin inferred from uptime with the run log deleted. Report those as boot/launch observations with the cache-state limits of the retained record. They leave room for more than one fault mechanism. (`/home/michael/tfm_worktrees/gpu-fault-history/docs/REGISTER_GPU_FAULT_PRIOR_WORK.md:3276`, `:3280`, `:3285`, `:3298`, `:3347`; `docs/LEADS.md:518`, `:582`.)
- **Card change.** The retained Run C fault episode has no `06:00.0` fault record, and does show rrig6600c `16:00.0` and `19:00.0`. Run C is one faulting fleet attempt. This weakens a fixed-card requirement. Card-specific susceptibility and the affected rig's missing tail remain open. (`logs/rig_fault_rec/20261116_C_secured_175651/netconsole_all_rigs.log:2185`, `:2191`; `docs/CLAUDE_CODE_REPORT_GCVM_LOCATED_20261116.md:233`, `:251`.)
- **Identical archives.** I hashed the 64 per-worker `.hsaco` members in each of the nine F–H tars and the retained November 15 rrig6600b donor tar. All ten name/hash inventories match, with eight distinct program hashes each. The phase-4 entry is `b8081567240a7871d1df7db2966c403c34ea208265c94ce30f7c8291c6730d7f`. This weighs against a differing on-disk object as the explanation and is compatible with different load/timing behavior. The direct per-CT byte histories for B and at fault time in C, and the bytes actually consumed by the GPU, remain **UNAVAILABLE**. (`/home/michael/runFGH_evidence_20261117/cache_manifest_after_F_121.txt:3`; `docs/CLAUDE_CODE_PLAN_RIG_ARM_V1_V2.md:55`, `:76`; `docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT_DECODE_20261115.md:240`.)

## Remaining finding

2. **NEW — Run C frozen coordinator log hash mismatch.** `sha256sum -c MANIFEST.sha256` returned 39 matches out of 40. The listed `gate12_20261116_174606.log` expects `e67a3d10…` and the frozen file hashes to `777abb49…`. Its phase-4 dispatch line agrees with the separately retained main-checkout log. The raw netconsole and all three trace files match their manifest digests and support the C fault observation. Preserve the existing manifest, identify the version of the coordinator log it describes, and document the versions and hashes in a supplement before treating that log as frozen evidence. (`logs/rig_fault_rec/20261116_C_secured_175651/MANIFEST.sha256:2`, `:4`; `logs/rig_fault_rec/20261116_C_secured_175651/gate12_20261116_174606.log:591`; `/home/michael/distributed_prng_analysis/logs/gate12_20261116_174606.log:591`.)

## Checks and availability

- Activated `~/venvs/torch`. Ran patch applicability, syntax checking of a temporary patched copy, and seven local shell fixture cases. Six gave the intended result. The listing-error case reproduced the false success above. Full launcher/gate integration and the proposed mutants are **UNAVAILABLE** for this draft. No launcher, worker or pipeline was started.
- Reran `allgaps.py` under `TZ=America/Los_Angeles` with output in a temporary directory. Its JSON matched byte-for-byte and reproduced the A–H table, including 23 C workers. Read-only checks covered the Run C manifest, fault lines, D–H console credit messages, F–H trace-loss counters, archived objects, parity manifests and CPU model captures. (`logs/rig_fault_rec/allgaps.py:41`; `/home/michael/distributed_prng_analysis/logs/gate12_20261117_170045_source_digests.json:51`.)
- Ran `python -B tests/test_beta_relay.py`. **34/34 PASS**, including the completion sentinel. Rig access, live ledgers, hardware flush decisions, AQL dispatch observations, and a warm diagnostic run are **UNAVAILABLE**. Beta wrote only this review in the repository.
