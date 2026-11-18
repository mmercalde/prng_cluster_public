## Prior record

1. **Finding 1 · ALREADY RECORDED, omitted governing reference.** Carry forward the Route A routing review and its separate pre-launch proof of the seven approved pins through the WATCHER operator CLI. The Run 4 checklist cites that proof and also calls for a parameter-only diff of the installed wrapper. Both checks still apply before the first counted Run 4 with the cache policy. The earlier routing position bounds the conclusions from Run 4 to operational evidence. `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:23`, `:150`, `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:118`, `:168`, `docs/RUN4_LAUNCH_CHECKLIST.md:348`, `:376`.
2. **ALREADY RECORDED.** R1 records the four landing items and the first-run timing and footprint evidence. The GCVM review records the per-launch cache position. L-7 records 8/8 clean cold and 4/4 faulted warm on the two affected rigs in phase 4. The delegated-execution plan and S172 completion plan govern clean committed launch, parity, preservation and acceptance. `docs/exchange/from_beta/X-CACHE-POLICY_R1_BETA_REVIEW.md:74`, `:81`, `:97`, `docs/exchange/from_beta/X-GCVM-LOCATED_R2_BETA_REVIEW.md:87`, `:89`, `docs/LEADS.md:590`, `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:55`, `:118`, `:175`, `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:79`, `:317`.
3. **NEW.** The v5 implementation and the reproduced landed-layout, Route A pin-drift, footprint timeout and encoder-status cases below. No ruling changes.

Searched the exchange, drafts, L-7, the GCVM report, Run 4 checklist, `docs/TB_RULING_*`, `docs/PROPOSAL_*`, chapter audits and `docs/TODO_CONSOLIDATED.md` for CuPy cache policy, launch and Route A provenance, freeze and parity, park/release deadlines, and the specific failure cases below. Compared the draft against the launch targets and the sentinel and liveness producers. Ser8, private remotes, the Claude project, live ledgers, rigs and Proxmox were outside the search.

# X-CACHE-POLICY · R2 · BETA REVIEW

- Date 2026-11-17 PST.
- Submitted base read `048fd47c92932962c4379c94b787852df8fdb7c7`. Worktree HEAD read `3579088b8e9c2d902252d626baae15bf0647907f`. Local `main` read `d606eb725f872a62690591a4b82c1b3aa2af9c3f`.
- All eight submitted SHA-256 values and all three launch-target hashes matched. The two repository targets matched the submitted base and local `main`. The three diffs applied to copies with zero fuzz and passed `bash -n`. The patched Route A baseline matched the supplied fixture byte for byte. Its diff against patched Gate 12 contained only the `--params` block. `docs/exchange/to_beta/X-CACHE-POLICY_R2_ALPHA_REPLY.md:40`, `:47`, `docs/drafts/cache_policy_v5/tests/fixtures/run4_launch.sh:465`.
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
?? docs/exchange/from_beta/X-GCVM-LOCATED_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-GCVM-LOCATED_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-GCVM-LOCATED_R3_BETA_REVIEW.md
?? docs/exchange/from_beta/X-MINER-NO-TORCH_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R3_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-PROBE_R1_BETA_REVIEW.md
?? docs/exchange/to_beta/X-CACHE-POLICY_R1_ALPHA_REVIEW_REQUEST_cache_policy_v4.md
?? docs/exchange/to_beta/X-CACHE-POLICY_R2_ALPHA_REPLY.md
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
| Q1 · Landability | OBJECTION | LIVE — the test installed under the repository's `tests/` directory computes its scratch path outside the checkout and can fail before any landed checks run. |
| Q2 · First-run items | OBJECTION | LIVE — one stalled post-run SSH measurement can terminate the collector without the required `INCOMPLETE` report. |
| Q3 · What remains | SUGGESTION | — |

## Q1 · Landing

| R1 item | v5 result |
|---|---|
| Git and hash status | **Satisfied.** Both counted entry points status-check HEAD and porcelain output before dispatch and reject a hash command that exits nonzero. The injected Git and hash failures refused before any dispatch log on both routes. Correct the README's claim that the provenance file includes an individual footprint script digest, or add the digest and cover it. The committed revision already pins the in-repo script's bytes. `docs/drafts/cache_policy_v5/gate12_launch.diff:23`, `:30`, `:41`, `docs/drafts/cache_policy_v5/run4_launch.diff:23`, `:30`, `:41`, `docs/drafts/cache_policy_v5/tests/test_cache_policy_v5.py:668`, `:677`, `docs/drafts/cache_policy_v5/README.md:42`. |
| NUL-safe environment and exact argv | **The original parser objections are resolved.** Startup counts the named entries and whole-value matches with NUL framing and checks both grep statuses. Parked comparison splits bytes on NUL, requires exactly one complete cache value and checks exact nonce token adjacency in both argv sources. The supplied malformed, unreadable and non-UTF-8 controls passed. One residual transport issue is below. `docs/drafts/cache_policy_v5/launch_fleet_manual.diff:49`, `docs/drafts/cache_policy_v5/scripts/gate12_cache_join_gate.py:78`, `:83`, `:187`, `:197`, `:209`, `docs/drafts/cache_policy_v5/tests/test_cache_policy_v5.py:361`, `:413`. |
| Post-rename invalidation | **Satisfied for the R1 reproducer and fallback.** A directory `fsync` failure gave verifier rc 2 and removed the official path. Injecting an unlink failure moved it to `.INVALID`. If both operations fail the verifier reports the invalidation failure and still returns 2, which the launcher treats as fleet failure. `docs/drafts/cache_policy_v5/scripts/cache_records_verify.py:107`, `:119`, `:167`, `docs/drafts/cache_policy_v5/launch_fleet_manual.diff:138`, `docs/drafts/cache_policy_v5/tests/test_cache_policy_v5.py:182`. |
| Landed harness and SSH stub | **Partially satisfied.** The stub protects fixture paths containing `/tmp/`. Both entry-point slices exercise the real join and freeze, verify the manifest and refuse before the sampler on join and freeze failures. The test still derives its scratch path from the draft layout even with `--landed ROOT`. Installing it at `/home/michael/distributed_prng_analysis/tests/test_cache_policy_v5.py` makes it try to create `/logs`. Installing it in this worktree's `tests/` makes it try `/home/logs`. I confirmed the path calculation and kept all writes in a temporary tree. Fix scratch-directory creation independently of draft discovery, then rerun the suite with the test and fixture installed under `tests/`, using an ordinary writable checkout and explicit `--landed`. **This blocks landing the package as submitted.** `docs/drafts/cache_policy_v5/tests/test_cache_policy_v5.py:27`, `:54`, `:201`, `:615`, `:662`, `:689`, `:759`. |

The parked probe returns the reader's `PIPESTATUS[0]` and discards the encoder's status. A local Bash fixture made the encoder emit valid base64 and then exit 1. The probe reported transport rc 0 and three section statuses of 0, and the resulting payloads satisfied all three parsers. **THEORETICAL** on the evidence here. This required an injected encoder fault. Preserve both statuses if revising the probe. The parked command also sends the whole environment. R1 called for selecting `CUPY_CACHE_DIR=` on the CT while retaining the NUL boundaries. That narrows the payload and still lets the Python checker reject duplicates. `docs/drafts/cache_policy_v5/scripts/gate12_cache_join_gate.py:48`, `:199`, `:209`, `docs/exchange/from_beta/X-CACHE-POLICY_R1_BETA_REVIEW.md:77`.

**Before the first counted run.** Run landed mode against the installed Route A for cache-gate coverage, then verify the wrapper's digest, parameter-only diff and exact approved values separately. Landed mode only reads the supplied Route A and runs its syntax and cache-gate slice. Changing the copied wrapper's `warm_start_offset` from 25 to 26 still passed **108/108**. The parameter block is downstream of the slice. An explicit comparison with the reviewed patched fixture plus the checklist's Route A pre-launch provenance check covers the gap. This verification is a counted-run requirement. The submitted Route A diff and fixture themselves match. `docs/drafts/cache_policy_v5/tests/test_cache_policy_v5.py:54`, `:71`, `:625`, `:662`, `docs/drafts/cache_policy_v5/tests/fixtures/run4_launch.sh:465`, `docs/RUN4_LAUNCH_CHECKLIST.md:348`, `:376`.

## Q2 · Sentinels and footprint

- **Sentinel capture satisfies R1 Q3.** The join uses the liveness row's log path, requires one matching nonce, PID and worker, parses the UTC timestamp and records the line and same-CT `sentinel_epoch - t_prep`. The real worker supplies the identity fields and UTC prefix. Both entry-point slices froze the sentinel TSV under a passing `SHA256SUMS` before reaching the sampler marker. `miner/range_miner_worker.py:1946`, `:2157`, `:3004`, `scripts/gate12_worker_liveness_gate.py:242`, `docs/drafts/cache_policy_v5/scripts/gate12_cache_join_gate.py:95`, `:190`, `:215`, `:234`, `docs/drafts/cache_policy_v5/gate12_launch.diff:80`, `docs/drafts/cache_policy_v5/run4_launch.diff:80`, `docs/drafts/cache_policy_v5/tests/test_cache_policy_v5.py:662`.
- **Footprint measurement and placement satisfy the R1 design.** The collector reads paths and endpoints from the frozen TSV and labels apparent and allocated bytes separately. The fixture verified both against local `du` and checked the retained-directory count. The entry points print a separate post-run command after creating the manifest. Its exclusive-create JSON can sit in the freeze directory without being a member of `SHA256SUMS`. `docs/drafts/cache_policy_v5/scripts/cache_footprint.py:30`, `:54`, `:67`, `:73`, `docs/drafts/cache_policy_v5/tests/test_cache_policy_v5.py:514`, `docs/drafts/cache_policy_v5/gate12_launch.diff:83`, `:101`, `docs/drafts/cache_policy_v5/run4_launch.diff:83`, `:101`.
- **Fix the timeout in the first-run evidence path.** A nonzero SSH exit is converted to an unmeasured row, while `subprocess.TimeoutExpired` escapes before any JSON is written. I injected that exception at the SSH call and confirmed there was no output file. Catch it at the row boundary, include the affected endpoint and path with a timeout reason, continue measuring the other rows, and write `INCOMPLETE` with exit 1. Test a timeout alongside a successful row. This is required before relying on the collector for the first counted run. It does not affect launch admission. `docs/drafts/cache_policy_v5/scripts/cache_footprint.py:39`, `:59`, `:69`, `:73`.

## Q3 · First counted run and acceptance

- Keep the clean committed harness, installed Route A checks above, GPU and parity gates, and 24 joined rig records plus the local Zeus liveness evidence. This cache-policy package adds no file in the ten-file CT parity list. If governed worker bytes change in the landed commit, the existing committed-HEAD deployment and 30/30 verification apply. Verify the freeze before accepting the result and retain the separate post-run footprint JSON. `scripts/gate12_parity_gate.py:174`, `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:55`, `:59`, `:118`, `docs/exchange/from_beta/X-CACHE-POLICY_R1_BETA_REVIEW.md:81`, `docs/drafts/cache_policy_v5/scripts/gate12_cache_join_gate.py:148`, `:229`.
- Include the cache join and freeze duration in the existing first-run timing report with per-slot startup, configured stagger and time from first park to actual release. The join adds 24 serial SSH probes with a 20-second timeout per probe. The worker's parking deadline is 900 seconds on these launch paths and starts independently for each worker. The maximum observed `SESSION_RELEASED.waited_s` across the fleet gives the longest successful wait using worker-local elapsed times. An expired worker emits `SESSION_RELEASE_TIMEOUT`. Live headroom is **UNAVAILABLE**. `docs/drafts/cache_policy_v5/scripts/gate12_cache_join_gate.py:43`, `:165`, `:194`, `docs/drafts/cache_policy_v5/gate12_launch.diff:61`, `docs/drafts/cache_policy_v5/run4_launch.diff:61`, `scripts/launch_fleet_manual.sh:150`, `miner/range_miner_worker.py:2203`, `:2214`, `:2223`.

## Checks and limits

Activated `~/venvs/torch`. Ran the unmodified suite from an isolated copy under `/tmp`. Reproduced **109/109 pre-landing**, **108/108 against a simulated landed tree** with the test and fixture under `tests/`, and **19/19 mutants detected**. The simulated checkout was deliberately deep enough to keep the miscomputed scratch path under the temporary root. The separate wrong-pin run also passed **108/108**. Ran the focused footprint-timeout and encoder-status injections locally. The suite used inert processes, stub SSH and stub `pkill`. Only this review was written in the worktree.

Read the source of the real gates, worker sentinel and release paths. Full live Gate 12 and Route A, rig timing and footprint, and held-out pool hit rate are **UNAVAILABLE**. Michael decides landing and counted-run acceptance.
