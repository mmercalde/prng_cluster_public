## Prior record

**Searches run in this session (2026-11-15):**
- Read `AGENTS.md`, the request, its incorporated plan's Prior record, the companion report's applicable T4 sections, and X-RIG-GPU-FAULT R1–R3 reviews/replies. The incorporated record contains actual searches and labels; this request meets the prior-record requirement.
- Ran `rg -n -i` for `seam|NCPROOF|torch|unresolved|kill -0|heartbeat|FW|power|phase.4` across `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*_AUDIT.md`, and `docs/TODO_CONSOLIDATED.md`; then searched those sources and the exchange record for recorder silence/staleness, missing first datagrams, missing pins, shutdown timeouts and SSH 255. Search captures are under `/tmp/X-RIG-GPU-PROBE_R1_beta/` and `/tmp/rig-probe-prior-search.txt`.
- Read governing passages in the completion/delegated-execution plans, relevant historical hits including the free-all-blocks proposal, S163KARG incident timeline and netconsole runbook. Read the external-worktree register's classification, onset, pre-PWC, torch-history, era-difference and recording passages. No additional missed governing document established.
- Re-read the four cited git commits and the January cleanup source; checked `S163_MEM_DEBUG` history and current launcher/worker sites. Searched current miner/PWC/sieve/registry paths and callers of `get_pytorch_gpu_reference`/`_get_torch`. Read all three submitted scripts and the test suite with line numbers, and the production executor, cleanup, payload builder, protocol and launch seams. Read the supplied plan/targets JSON and saved recorder test6.

| Record / item | Label and disposition |
|---|---|
| `AGENTS.md`; `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:19`; `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:24`, `:69`, `:183`, `:203` | ALREADY RECORDED. Beta reviews; Michael decides. Freeze reproduction remains owner-operated. |
| X-RIG-GPU-FAULT R1–R3: missing observation, stop controller, torch-control interpretations, runtime attribution, passive recorder and lock | ALREADY RECORDED requirements/positions. Implementation-specific counterexamples below are NEW evidence, not rediscovery of those requirements. |
| Register: no pre-April kernel capture; gated TCP-PWC torch; UNKNOWN August host loss; launch-relative onset timing | ALREADY RECORDED facts. Preserve their limits. |
| Recorder freshness, shutdown/SSH failure handling, first-datagram silence and FW denominator in the submitted controller | NEW implementation defects, Q7/Q8. |
| Arm definitions, reduced probe and fleet flag | NEW proposal/implementation; no ruling changed by this review. |

**Not searched / not executed:** register Parts A–E in full; private repository; ser8 archives; rig/CT/Proxmox filesystems; Zeus; installed runtime binding/thunk implementation. No remote connection, GPU import/initialization, real worker/probe, pipeline, recorder main, remote sampler, host write, package operation, live-state write, commit or push. No ledger opened. Historical raw logs were not independently recounted. Runtime fidelity and real GPU behavior are **UNAVAILABLE** here. Socket-based local tests are also **UNAVAILABLE** because this sandbox rejects socket creation; details below.

# X-RIG-GPU-PROBE — R1 — BETA REVIEW

- Date: 2026-11-15, America/Los_Angeles.
- Base read: `3579088b8e9c2d902252d626baae15bf0647907f`.
- Scope: submitted uncommitted plan/probe/controller/tests, recorder changes and document propagation. Fleet worker flag remains design-only.
- All eight submitted file hashes match the request. Initial `git status --porcelain`:

```text
 M docs/LEADS.md
 M docs/exchange/LEDGER.md
?? docs/CLAUDE_CODE_INSTRUCTIONS_RIG_GPU_FAULT_20261115.md
?? docs/CLAUDE_CODE_PLAN_RIG_GPU_PROBE.md
?? docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R3_BETA_REVIEW.md
?? docs/exchange/to_beta/X-RIG-GPU-FAULT_R1_ALPHA_REVIEW_REQUEST_fault_evidence_and_plan.md
?? docs/exchange/to_beta/X-RIG-GPU-FAULT_R2_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-RIG-GPU-FAULT_R3_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-RIG-GPU-PROBE_R1_ALPHA_REVIEW_REQUEST_torch_arms_probe_controller.md
?? scripts/rig_gpu_recorder.py
?? scripts/rig_seam_controller.py
?? scripts/rig_seam_probe.cleanup.sha256
?? scripts/rig_seam_probe.py
?? tests/test_rig_seam_probe_controller.py
```

| Question | Position | Severity |
|---|---|---|
| Q1 | NEEDS EVIDENCE | — |
| Q2 | NO OBJECTION | — |
| Q3 | SUGGESTION | — |
| Q4 | SUGGESTION | — |
| Q5 | OBJECTION | LIVE — The proposed E1 comparison's advertised power uses an UNKNOWN host loss as a demonstrated GCVM outcome. |
| Q6 | SUGGESTION | — |
| Q7 | OBJECTION | LIVE — Recorder loss and unreachable probes can be accepted without complete observation or a durable recovery barrier on this fleet. |
| Q8 | OBJECTION | LIVE — The gates omit the recorder-loss and shutdown-failure paths that invalidate the intended first test. |
| Q9 | OBJECTION | LIVE — The owner commands use the controller with the unresolved Q7 defects, so the first series is not correctly bounded as written. |

The ladder and arm interventions are useful. Q7/Q8 need implementation corrections before the proposed first use. This is Beta's review position, not a launch authorization or a root-cause conclusion.

## Q1 — History supported; exact operation-at-onset is not established

The torch-history table and “presence alone does not separate the eras” are supported by the re-read git anchors and the register's explicit lack of pre-April capture (`docs/CLAUDE_CODE_PLAN_RIG_GPU_PROBE.md:70`, `:78`; register `:3038`, `:3080`). Current TCP-PWC gating is explicit (`sieve_filter.py:139`); its service calls that cleanup after job completion (`persistent/pwc_worker_service.py:437`, `:462`). The current forced-debug launcher sites exist at `launch_s174.py:380` and `launch_s174_baseline_pool8_25k.sh:88`. The small history detail at plan `:89`–`:90` reverses the chronological order of the two April-15 commits: `57eb048` is 15:50; `ba86d77` is 16:52.

The proposed per-event worker-path/env/call-state check is appropriate (`docs/CLAUDE_CODE_PLAN_RIG_GPU_PROBE.md:97`). Include process identity, earlier jobs/chunks in that process, the event-time source version and evidence for the env value; an env value alone is insufficient to prove a call occurred or never occurred.

**NEEDS EVIDENCE applies to the onset table's strength.** The register establishes launch-relative timing (`REGISTER_GPU_FAULT_PRIOR_WORK.md:2747` in the external worktree). The single restart timeline establishes handshake → init → import → job on one worker (`docs/TB_Incident_Report_rrig6600c_S163KARG.md:74`). Neither establishes the faulting operation, fresh module state or absence of earlier torch calls for each of those ten events. Plan `:109` labels one inference, but `:112` then calls “first launch of a kernel module newly loaded in that process” a recorded pattern in both engines. Use **“onset near worker startup in those PWC events; module/load/launch and torch-call state UNKNOWN per event”** until the event check resolves them. Likewise, today's dispatch-to-GCVM interval brackets dispatch, not a measured per-worker launch instruction (`:108`). This preserves the temporal observation without promoting a mechanism inference to a recorded fact.

## Q2 — Accept the ladder and the bounded geometry claim

No objection to R1 → R2 → R3, fixed screens, non-reproduction wording, or testing other arms at the lowest reproducing rung (`docs/CLAUDE_CODE_PLAN_RIG_GPU_PROBE.md:131`, `:144`). FW is a separate warm-up reproduction and must not silently count as a clean seam attempt when deciding to climb; Q7 addresses accounting.

The builder reads the actual named stripe ranges, checks the dataset SHA, calls the production payload builder, and partitions by the production cap (`scripts/rig_seam_controller.py:103`, `:113`, `:119`, `:125`, `:128`). The probe iterates the full supplied range list through the production executor (`scripts/rig_seam_probe.py:415`, `:517`, `:522`). The supplied gpu6 plan contains 68 ranges totaling 67,108,864 seeds for each phase. Thus **real stripe geometry and production execution path** are supported by source and the supplied artifact. Actual matching survivor volume, kernel arguments and behavior remain **UNAVAILABLE**, not verified by geometry alone. The reduced-history disclosure remains necessary (`scripts/rig_seam_probe.py:12`).

## Q3 — Useful arms; retain the runtime-attribution limit

The source preserves GC and both pool frees across arms and changes torch import/calls (`scripts/rig_seam_probe.py:209`, `:227`, `:256`, `:274`, `:281`, `:284`). P1 places one torch cleanup after each completed phase (`:525`). The plan's contrast table correctly retains the initialization/sync/cache bundle and synchronization ambiguity (`docs/CLAUDE_CODE_PLAN_RIG_GPU_PROBE.md:167`). These interventions differ in torch-related instrumentation and timing as well as torch execution, so the rate comparison applies to the instrumented configuration.

A new fd appearing during a torch API call associates the open with that call interval. **It does not identify the bound runtime copy from maps/fds alone.** `runtime_identity()` collects process-wide paths/fds, without an opener stack or bound-symbol attribution (`scripts/rig_seam_probe.py:146`). Keep the library-structure explanation at plan `:182` as an unverified hypothesis, not the first attempt's automatic runtime verdict. Both initialization attribution and later device use remain UNOBSERVED unless the first real attempt supplies stronger evidence. No installed-library inspection establishing that attribution was available in this review.

The claimed snapshot coverage is also broader than the implementation: P snapshots only its first cleanup and combines the post-sync/cache snapshot (`scripts/rig_seam_probe.py:233`, `:243`, `:250`); P1 takes only a post-phase-call snapshot (`:289`, `:300`). Narrow the prose or add the promised per-call brackets, especially around P1's first availability/sync calls. “M never invokes torch.cuda” is supported; “import cannot initialize a runtime” is not established merely by not calling that API (`:265`).

## Q4 — Inventory acceptable; correct the deletion coordinates

No additional active torch import/call found on the specified miner or sieve paths. The miner cleanup is the relevant block, called by the executor finally (`miner/range_miner_worker.py:1060`, `:1331`). Registry torch imports are lazy (`prng_registry.py:84`, `:111`, `:4141`); the named helper's external callers found are the scorer and tests. SSH-PWC and reverse-script cleanup are unconditional (`sieve_gpu_worker.py:84`, `:348`; `reverse_sieve_filter.py:73`); TCP-PWC is gated as above.

Correct “delete one 8-line block `:1059-1066`”: the complete torch try/except is **seven lines, `miner/range_miner_worker.py:1060`–`:1066`**. Line 1059 is the preceding GC exception handler's `pass`; deleting it would leave that handler empty. The size estimate is otherwise small and `.get()` already blocks the CuPy result path (`:1270`). This is a coordinate correction, not a newly discovered runtime defect.

## Q5 — Arithmetic correct; distinguish the E1 rate from host-loss history

No objection to the one-flag design, default preserving today's behavior, with separate worker review when built (`docs/CLAUDE_CODE_PLAN_RIG_GPU_PROBE.md:230`). Retaining production phases 1–3 is appropriate for that full-history comparison: changing their sub-stripe counts changes the cleanup exposure (`miner/range_miner_worker.py:1331`). Arbitrary shorter geometry does not preserve full memory history merely because family names remain the same.

I independently recomputed the displayed power values by enumerating binomial outcomes and applying the submitted Fisher function (`scripts/rig_seam_controller.py:84`). At 20/arm, powers are **0.9941** for 1/2 vs 0, **0.8485** for 1/3 vs 0, and **0.2313** for 1/6 vs 0; the five displayed 8/12/20 rows agree after rounding. The proposed 10/29/19/49/39 budgets cross 80% at the stated sizes; each immediately preceding size is below 80%. These are conditional model calculations.

**Objection:** plan `:243` explicitly includes the August UNKNOWN host loss in its 2/6, then `:252` calls 1/3 “today's record rate” for an E1 endpoint. The register keeps that event UNKNOWN (`REGISTER_GPU_FAULT_PRIOR_WORK.md:2735`); four historical compute completions also do not establish four fully observed E1-negative windows. The lower row's “1 in 6 ... UNKNOWN ... excluded” (`docs/CLAUDE_CODE_PLAN_RIG_GPU_PROBE.md:245`) is arithmetically misdescribed: excluding one observation changes the denominator; retaining it as unknown gives a range.

Keep 1/6, 1/3 and 1/2 as **planning scenarios**, carry unknown coverage explicitly, and describe 20+20 as a fixed exploratory budget with scenario-dependent power. It has only ~23% power for elimination at the 1/6 scenario. A larger budget is the owner's decision; this objection requires correcting the claim, not mandating 39+39. Boot dependence, incomplete observations and multiple screened contrasts further limit an unconditional power guarantee.

The durations are planning estimates, not measurements of this probe. The owner-run full-fleet clean timing sums to roughly 16–17 min; the cited fault interval of ~34 min is recovery **after onset**, so ~40 min is a rough estimate rather than a demonstrated start-to-ready duration (`docs/CLAUDE_CODE_PLAN_RIG_GPU_PROBE.md:255`). The reduced attempt's 30-second window overlaps phase-4 execution: the controller computes it from the first launch exit, not after the full stripe (`scripts/rig_seam_controller.py:426`). Do not present these estimates as hard limits.

## Q6 — Production path is credible; strengthen pinning and trace claims

The real backend calls production detect/warm, checks BDF before warm/stripe load, replaces the exact cleanup global used by the executor, and wraps the executor launch and kernel lookup (`scripts/rig_seam_probe.py:365`, `:381`, `:383`, `:386`, `:391`, `:398`; `miner/range_miner_worker.py:1331`). One invocation constructs one backend and executes one attempt (`scripts/rig_seam_probe.py:495`, `:499`). BDF API compatibility remains explicitly unverified; there was no GPU test here.

The P replacement matches the production cleanup's normal call order and best-effort exception groups on reading (`scripts/rig_seam_probe.py:227`; `miner/range_miner_worker.py:1051`). Its pin protects against production source drift only when the adjacent file exists: `_expected_cleanup_sha()` returns None on an absent file (`scripts/rig_seam_probe.py:427`), and `if want and got != want` then skips refusal (`:377`). A local check confirmed the missing-file return. Fail closed on absent/empty/malformed pins and test the real-backend check with stub modules. The supplied pin is present and matches today; this is a hardening suggestion, not evidence that today's P source differs.

The fake backend injects stub torch into both cleanup factories and uses stub pools; its operations do not import the real GPU libraries (`scripts/rig_seam_probe.py:312`, `:323`, `:333`, `:556`, `:581`). No GPU-capable execution was needed for the review checks.

**Trace limitation:** real execution wraps launch, lookup and whole execute; it does not separately trace allocation/H2D or D2H `.get()` (`scripts/rig_seam_probe.py:398`, `:415`; `miner/range_miner_worker.py:1200`, `:1270`). The fake backend's `d2h_get` marker (`scripts/rig_seam_probe.py:341`) has no real-backend counterpart. Also `_get_kernel` constructs/caches a `RawKernel` object (`sieve_gpu_worker.py:126`); lookup completion alone does not prove actual device-module load completion. Implement the promised finer trace, or explicitly report the unresolved execute interval and withdraw “every GPU-touching operation”/fine operation-at-fault claims (`scripts/rig_seam_probe.py:41`; `docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:511`). These limits do not disqualify a correctly observed E1 reproduction.

## Q7 — Correct the following controller paths before use

**1. Recorder coverage can silently disappear and still produce C.** The watcher handles only newly appended sample rows; it has no freshness deadline if the recorder dies, stalls or stops writing (`scripts/rig_seam_controller.py:291`). E3 elapsed time is checked only when another UNOBSERVED row arrives (`:300`). Final coverage requires no UNOBSERVED rows and merely **one OBSERVED row from any target host** (`:528`). A recorder that emits once then dies, or never observes a second host, can therefore satisfy the clean predicate. My direct call to the submitted classifier returned **C** with one old OBSERVED event. This is not a socket/runtime test; it verifies the inadequate predicate.

Track last fresh samples **per required host**, check staleness on every watchdog iteration including complete file silence, and require coverage through the declared window. Preserve stopped/failed-channel events and backlog/sequence limits. Missing coverage must produce I, never C. Optional `--recorder` currently removes the coverage requirement altogether (`:529`, `:609`); counted three-channel attempts must not silently use that bypass.

**2. SSH failure is treated as confirmed process death; timeout can erase the recovery record.** The alive check maps every nonzero result to “gone” (`scripts/rig_seam_controller.py:446`), including SSH 255 for an unreachable CT. Executing the exact shutdown AST with fake subprocess results of 255 produced **`unresolved=[]`**, after TERM and one alive check, with no KILL attempt. That is the realistic hung-host case this controller exists to handle. A 30-second subprocess timeout is also uncaught (`:443`, `:446`), so execution can exit before appending `attempts.jsonl` (`:457`). The next invocation's refusal reads only that file (`:365`); an interrupted shutdown can leave no recovery barrier.

Use explicit remote responses to distinguish absent PID from transport failure, catch/report every escalation timeout, continue best-effort stopping of other targets, and persist an in-progress/recovery-required attempt before dispatch. Finalize it in exception/interrupt paths; refuse incomplete prior attempts. A lost hello/PID must be unresolved by target identity, not silently skipped (`:439`). An `end` UDP record is emitted before process exit (`scripts/rig_seam_probe.py:542`), so skipping OS-death verification for every `ended` target (`scripts/rig_seam_controller.py:440`) also overstates “confirmed gone.”

**3. The first heartbeat can be missing forever until the much longer step/deadline limit.** With a hello but no UDP record, `last_rx.get(k, now)` resets the effective baseline each loop (`scripts/rig_seam_controller.py:311`). The exact watch loop, with fake clock/tails, produced **no trigger after 6 seconds with a 1-second silence limit**. Initialize freshness at launch/hello and require the first datagram within the limit. Later heartbeat receipt shows that heartbeat thread is alive, not GPU progress; the external operation deadline remains necessary.

**4. Stop/withhold coverage and recovery flags are incomplete.** The initial process-launch loop bypasses `Attempt.send()` and launches every target even after a trigger (`scripts/rig_seam_controller.py:388`). The check inside `send()` releases its lock before socket dispatch (`:320`, `:328`), allowing a trigger between checking and sending. Serialize the issue/trigger decision and stop issuing initial launches after a trigger. The normal command wait/deadline and later seam-slot withholding are otherwise explicit (`:336`, `:420`).

Refusal works for successfully recorded recovery-needed rows (`:366`), but `hello_timeout`, `probe_error` and `probe_refused` are excluded from the recovery predicate (`:546`), even though the controller's contract says “after any trigger.” At least unknown process/host state must latch the barrier. `classify()` can also return C while passed a nonempty unresolved list (`:531`, `:536`); my direct check reproduced that. Define and enforce whether C covers just the seam or the full successful attempt, and never let unresolved cleanup authorize another attempt.

**5. FW is being counted as a known negative in the E1 range.** `summarize()` includes FW in B, excludes it from I, then reports `[F/B,(F+I)/B]` (`scripts/rig_seam_controller.py:569`). A one-row FW series yielded **`B=1, F=C=I=0, FW=1, e1_range=[0,0]`**. That attempt stopped before observing the seam. Keep FW separately, but treat its seam outcome as unobserved in the attempted-budget range, or explicitly exclude it and report a separate reached-seam denominator. Do not infer zero seam risk from warm-up faults.

The numerical upper-bound/Fisher functions pass the existing CPU gate (`tests/test_rig_seam_probe_controller.py:239`). The VM101 receipt-clock basis is consistent (`scripts/rig_seam_controller.py:277`, `:333`). Pre-proof checks age/IP and post-proof checks after exit dispatch (`:472`, `:491`); **failed pre-proof does not withhold GPU work**, it only invalidates the eventual class (`:386`, `:388`, `:531`). Make proof a dispatch prerequisite. Post-proof bounds should be measured from the advertised wait start: the current timeout starts at exit dispatch and includes shutdown/escalation time (`:495`). Summary lists strata but pools them for Fisher (`:579`, `:582`); expose per-stratum counts and label any pooled comparison's assumptions.

## Q8 — Good positive controls; important surviving classes remain

The 12 named gates cover the listed happy path, arms, explicit fault/UNOBSERVED rows, one hung operation, a trace gap, proof absence, resume, statistics and source digest (`tests/test_rig_seam_probe_controller.py:268`). There are **13 pytest test functions**, including the mutant aggregate (`:349`, `:363`), so “12 passed” needs its exact selection/transcript explained.

The recorder fixture continually writes rows (`tests/test_rig_seam_probe_controller.py:71`); it never tests death after one sample, no first sample, per-host missing coverage, or a last UNOBSERVED followed by complete silence. The shutdown fixture uses local `kill` (`:115`); no SSH transport-error/timeout class or durable incomplete-attempt barrier is exercised. The fault gate asserts an empty unresolved list, not independent disappearance of every recorded process (`:192`). None covers FW range accounting. Add these classes and mutants keyed to the corrected behavior.

**Constructible surviving class:** change the real BDF mismatch guard at `scripts/rig_seam_probe.py:388` to `if False`. Every executable controller gate selects `--backend fake` (`tests/test_rig_seam_probe_controller.py:105`), whose BDF check simply echoes the expected identity (`scripts/rig_seam_probe.py:326`); the pin gate inspects only production cleanup (`tests/test_rig_seam_probe_controller.py:258`). Thus the suite has no route to detect removal of the real BDF guard. This is established by reachability inspection, **not a claimed executed survivor**. Use stubbed real-backend dependencies for that guard, pin rejection and cleanup replacement.

## Q9 — Retain the first screen; repair the controller before owner use

R1/P on the suspect BDF with a declared eight-attempt screen is the right first rung, conditional on Q7/Q8 corrections. The owner-only deployment, netconsole arming, recovery outlet/console and network preparation are appropriate (`docs/exchange/to_beta/X-RIG-GPU-PROBE_R1_ALPHA_REVIEW_REQUEST_torch_arms_probe_controller.md:127`). The supplied targets file exists locally; I did not deploy it or the probe.

The owner also needs:
- A successful **fresh** pre-proof and live recorder sample from every required host confirmed by the controller before bring-up, plus a working channel-failure stop. Merely binding port 6667 does not prove sender delivery.
- A checked production dependency/payload contract, not just the worker file: the probe imports the registry/kernel/builder paths through the executor (`miner/range_miner_worker.py:1129`, `:1150`). Retain dataset/pin hashes and actual observed BDF with the attempt. Current commands hash probe and worker only (request `:141`); no remote parity was independently verified here.
- A local dry-run of the failure/recovery paths, including unreachable CT and controller interruption, with the durable recovery record checked before another attempt. An owner procedure for clearing that specific record and rechecking mapping/proof must accompany it.
- Explicit shell failure visibility: the suggested run is piped to `tee` (request `:159`), so use `pipefail` or capture the controller exit status if shell automation relies on success. Owner/manual invocation remains mandatory.

After those corrections, the first real attempt is also a pilot for the explicitly unverified HIP BDF keys, trace completeness and actual production survivor/argument fidelity. Record those outcomes before interpreting the eight-attempt screen; no such result exists yet.

## Findings outside the numbered questions

No additional missed governing document or unrelated new finding established. Recorded-status updates:

1. The recorder now holds an exclusive lifetime flock per output directory (`scripts/rig_gpu_recorder.py:461`), resolving the earlier PID-publication race for that scope. Passive mode selects the dedicated sampler (`:490`), which omits the dynamic sensor queries (`:179`). Saved test6 independently recounts as 18 OBSERVED passive samples and final `unresolved=[]` (`logs/rig_gpu_recorder/test6/rig_gpu_20261115_160509.jsonl:1`). The reported simultaneous-start test was not independently rerun. Different output directories still mean different lock scopes; this is not a system-wide recorder singleton.
2. Q13/Q2 wording propagation is already recorded, not a new finding. No historical UNKNOWN event is upgraded by this review.

## Execution and availability record

Activated `source ~/venvs/torch/bin/activate`; used `python3 -B` and disabled pytest cache. Copied only the necessary submitted scripts/test/source into `/tmp/X-RIG-GPU-PROBE_R1_beta/` so the existing suite's `logs/` fixtures could not write into the repository (`tests/test_rig_seam_probe_controller.py:343`).

- Existing full test file: **2 passed, 11 failed due to unavailable socket creation**, beginning at the harness's `socket.socket()` (`tests/test_rig_seam_probe_controller.py:44`), `PermissionError: [Errno 1] Operation not permitted`. This is an environment limitation, not eleven established product failures. Statistics and cleanup-pin gates passed. Of the aggregate mutant checks, M5 was DETECTED; eight others were INVALID because their clean controls hit the socket restriction. Independent “12 gates/9 mutants passed” verification is **UNAVAILABLE**.
- Socket-free actual-function/AST checks: `/tmp/X-RIG-GPU-PROBE_R1_beta/checks.py`, `socket_free_results.json`. Verified the concrete classifier, SSH-255, uncaught-timeout, initial-UDP-silence, FW-range and missing-pin behaviors described above. These substituted clock/tails/subprocess responses; no socket, SSH, GPU library, worker or recorder main ran.
- Power enumeration results: `/tmp/X-RIG-GPU-PROBE_R1_beta/check_results.json`. Recounted saved recorder test6; read the supplied plan/targets; verified all eight submitted hashes.
- Real GPU execution, kernel-argument/count fidelity, runtime-copy attribution, remote stop/disappearance and fault reproduction remain **UNAVAILABLE**, never passes or zero faults.

Only this review was written inside the repository. No submitted implementation/document was edited. No counted-run result is accepted and no root cause is claimed.

Final verification: all eight submitted hashes remained unchanged; all 63 explicit file:line anchors resolve; the positions table contains exactly Q1–Q9 with required LIVE explanations. The only repository status addition from this review is this reply file.
