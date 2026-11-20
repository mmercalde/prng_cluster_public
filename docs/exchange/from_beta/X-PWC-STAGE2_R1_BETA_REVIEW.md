# X-PWC-STAGE2 · R1 · Team Beta review

Read 2026-11-20 PST at base `ee394047ddb7c3f9d7aadc0e4baf1d9cc474a0e6`.

`git status --porcelain` at intake

```text
 M persistent/pwc_transport_tcp.py
 M persistent/pwc_worker_service.py
 M persistent_worker_coordinator.py
 M test_persistent_worker_harness.py
 M tests/test_pwc_backup_step1.py
 M tests/test_pwc_stage1.py
 M tests/test_s172_phase5_d3_25_candidate_ingress.py
 M tests/test_s172_threshold_propagation.py
?? data/
?? docs/drafts/pwc_backup_stage2_v1/
?? docs/exchange/to_beta/X-PWC-STAGE2_R1_ALPHA_REVIEW_REQUEST_real_world_scope.md
?? tests/test_pwc_backup_stage2.py
```

| Question | Position | Severity |
|---|---|---|
| Q1 · Threshold contract | NO OBJECTION | — |
| Q2 · Phase scheduling | NO OBJECTION | — |
| Q3 · Five strategies on the rigs | NEEDS EVIDENCE | — |
| Q4 · Offset refusal | NO OBJECTION | — |
| Q5 · Stale-worker probes | NO OBJECTION | — |
| Q6 · Revocation and reconnects | OBJECTION | LIVE — A normal TCP reconnect can overlap old-handler cleanup and have its ready state cleared and its active lease requeued or failed. |
| Q7 · Exit on explicit refusal | SUGGESTION | — |
| Q8 · Run 4 RANGE-MINER | NO OBJECTION | — |
| Q9 · Shared suites | NO OBJECTION | — |

**Disposition.** Q6 holds the Stage 2 commit from Beta's review position. The reconnect fixture reproduced the cleanup race, and Finding 2 records a separate wrong-result path through late TCP results. I found no change to Run 4 RANGE-MINER runtime behavior on a clean checkout. The existing counted admission gate refused the dirty review checkout. Michael decides whether the package lands. `persistent/pwc_transport_tcp.py:628-671`, `gate12_launch.sh:125-139`, `:465-473`, `window_optimizer_integration_final.py:1937-1955`.

## Prior record

- **Searches run.** Searched `docs/` with `rg`, including `TB_RULING_*`, `PROPOSAL_*`, `CHAPTER_*_AUDIT.md`, `TODO_CONSOLIDATED.md`, the file catalog, and prior exchanges. Looked for S147/B1/Q0/Q2, S156/S182, process visibility, the stable envelope, the anchor/phase ABI, clean-tree and Run 4 admission, reconnect and lease ownership, timeouts, and late result correlation. Read Alpha's request, package, owner rulings, and applicable code. Reviewed the governing omissions in Finding 1 first.
- **ALREADY RECORDED.** The prior record identifies the owner decisions, S147 pruning, the threshold quarantine, VIR-1/VIR-2/VIR-5, and the three Stage 1 carryovers. The per-run token, exact threshold echo, extra Java forward-hybrid argument, PWC launch plumbing, and search credit are recorded open items. `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:14-34`, `docs/VERIFICATION_INTEGRITY_STANDARD.md:30-47`, `:104-118`, `docs/exchange/from_beta/X-PWC-STAGE1_R1_BETA_REVIEW.md:58-81`, `docs/drafts/pwc_backup_stage2_v1/README.md:278-286`, `:453-486`.
- **CHANGES A RULING.** The package replaces S147 Q2's one strategy with all five under Michael's default rule. The request leaves F5 with Michael. `docs/STEP1_EXECUTION_FLOW_AND_PRUNING_S147.md:176`, `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:14-23`, `docs/drafts/pwc_backup_stage2_v1/README.md:251-255`.
- **NEW.** The reconnect cleanup interleaving, the specific cross-phase late-result delivery, and the GPU telemetry call before the new refusal exit are addressed below. The checks pinned the first two interleavings in local fixtures.
- **Scope of search.** The rigs, Proxmox, external project records, and live staging and ledgers were outside the search.

## Reasoning

### Q1 · Threshold contract

The trial passes each directional threshold into both keys. On the TCP path the worker forwards `phase2_threshold` and `sieve_filter` coerces it to a float, supplies it to the hybrid call, and rounds the report after the call. The coordinator rejects a missing or mismatched hybrid threshold up front, and a completed chunk without an echo fails the contract. My CPU check exercised the `sieve_filter.execute_sieve_job` path with GPU methods stubbed. It saw `0.73125` at the hybrid call and `0.7312` in the echo. I found no code path on this route that changes the numeric threshold in transit and escapes the check. `persistent_worker_coordinator.py:1933-1949`, `:2053-2067`, `:2287-2306`, `:2359-2366`, `:2691-2692`, `:2716-2717`, `persistent/pwc_worker_service.py:541-565`, `adaptive_thresholds.py:95-119`, `sieve_filter.py:583-599`, `tests/test_pwc_backup_stage2.py:268-298`.

Four places are a genuine observation limit. I verified that `0.74996` and `0.75004` both report `0.7500` while their `float32` cutoffs sit on opposite sides of `0.75`. A match rate of `0.75` therefore distinguishes them. PWC also applies the original threshold in a host-side comparison. That counterexample requires a separate defect that changes the threshold before the sieve call. Keep the unrounded echo in the recorded follow-up before relying on the report for exact certification. `persistent_worker_coordinator.py:560-577`, `sieve_filter.py:432-437`, `:464-479`, `docs/drafts/pwc_backup_stage2_v1/README.md:484-486`.

### Q2 · Phase scheduling

For the Run 4 base family `java_lcg`, the zero-forward and zero-variable-forward fixtures both called all four passes in both-mode, and constant-only called 1–2. There is an intended result change beyond cost. With zero constant-forward survivors, the constant intersection remains empty, but P3 and P4 can produce a variable intersection and the integration receives an unpruned trial. With zero variable-forward survivors, P4 can populate the reverse diagnostic map and completion record while the variable intersection stays empty. The CPU gate verifies scheduling and egress shape. It does not establish GPU completion. `gate12_launch.sh:468-472`, `persistent_worker_coordinator.py:2621-2655`, `:2673-2726`, `:2759-2775`, `window_optimizer_integration_final.py:1987-2007`, `tests/test_pwc_backup_stage2.py:343-354`.

Qualify the parity claim to base families. PWC retains the `_hybrid` suffix guard that the shared `_step1_required_phases` helper uses. `workflow_stages_for` itself checks only `test_both_modes`. No Run 4 discrepancy follows for `java_lcg`. The phase-3 Java ABI question is already recorded. The caller supplies a 16th argument to the 15-argument signature, and its disposition on the deployed HIP path is **UNAVAILABLE**. `persistent_worker_coordinator.py:2544-2553`, `window_optimizer_integration_final.py:1388-1394`, `miner/range_miner_coordinator.py:12046-12062`, `sieve_filter.py:432-463`, `prng_registry.py:1007-1013`, `docs/drafts/pwc_backup_stage2_v1/README.md:278-286`, `:453-460`.

### Q3 · Five strategies on the rigs

The CPU dispatch check confirmed all five strategy records in order. The Java hybrid kernel iterates over the five strategies per seed, with strategy-dependent skip search and early exits. The main GPU result buffers are sized by seeds and window length, and the kernel writes at most one record per seed. Five strategies therefore increase kernel work and may increase the number of records transferred, while the allocation shape does not multiply by five. `persistent_worker_coordinator.py:1966-1982`, `miner/range_miner_worker.py:1227-1236`, `sieve_filter.py:399-428`, `prng_registry.py:1021-1077`, `tests/test_pwc_backup_stage2.py:370-378`.

The existing guard admits six workers at 50,000 seeds or eight at 25,000 per chunk. The historical report recorded an eviction even in the 8/25,000 run and faults at 8/50,000. A longer five-strategy kernel at the 8/25,000 point is a plausible source of evictions or late chunks on these rigs. I have no measured duration, memory-pressure or fault result for that workload. For the owner-run check, measure phase-3/4 chunk times and completion, peak memory and host KFD/MES events at the intended cap and concurrency. Record the 600-second result timeout against the tail of the observed durations. Prediction quality is a separate comparison of the one- and five-strategy configurations on the same unseen draws against `k/1000`. Its direction is unknown. `persistent_worker_coordinator.py:258-260`, `:479-494`, `:1755-1763`, `:2003-2018`, `docs/ROCm_Saturation_Report_S172.md:25-42`, `:68-87`, `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md:52-61`.

### Q4 · Offset refusal

The owner-directed refusal is at the right boundary. The trial validates before constructing the coordinator, and a direct pass repeats the validation before dispatch. `sieve_filter` uses the value to select the draw slice and sends it as a kernel argument, where the miner pins the generator phase to zero. The fixture refused 25 without constructing a coordinator and refused 3 at the pass entry without dispatch. I found no currently relied-on PWC run in the record. `persistent_worker_coordinator.py:344-353`, `:1951-1952`, `:2542-2568`, `sieve_filter.py:174-187`, `:310-311`, `:432-463`, `:534-538`, `miner/range_miner_worker.py:601-605`, `tests/test_pwc_backup_stage2.py:381-401`.

One operational consequence for Stage 3 is that the first sampled nonzero offset can abort an ordinary Optuna study. The search still samples its configured range and passes the exception out of the objective. I confirmed with an in-memory study that the refusal prevented the next valid trial from starting. Until the split lands, a PWC optimization using this path needs offset bounds fixed at zero, including any supplied starting trial. Run 4's anchor 25 remains outside the Stage 2 PWC configuration. `window_optimizer.py:134-143`, `window_optimizer_bayesian.py:537-576`, `:838-840`, `window_optimizer_integration_final.py:2705-2744`, `docs/drafts/pwc_backup_stage2_v1/README.md:473-477`.

### Q5 · Stale-worker probes

I found no evidence of a normal launch here that puts PWC workers outside the cleanup shell's UID or PID namespace. The probe establishes visibility of its own shell and handles target success, no-match and error distinctly. The local launch fixtures confirmed that an unavailable check, a survivor, a missing marker and an exception all suppress the local launch. The isolated process fixtures passed with real `pgrep` and the three broken variants, including the one that can see the control and errors on the target. They also produced a single `reaped:0` on a clean S156 reap. `distributed_config.json:5`, `:14`, `:25`, `:36`, `persistent_worker_coordinator.py:208-255`, `:1287-1305`, `:1313-1359`, `:1472-1498`, `tests/test_pwc_backup_stage2.py:445-517`, `:591-603`.

The shell control covers its own process view. A stale worker moved to another PID namespace or hidden by a differential `/proc` permission policy could be outside that view. I found no such configuration in this cluster's record. The real CT100 SSH path is **UNAVAILABLE**. The legacy SSH startup still logs unavailable S156 evidence and proceeds. The TCP rig kill separately refuses before cache reset or launch. `persistent_worker_coordinator.py:1030-1108`, `:1334-1363`, `docs/VERIFICATION_INTEGRITY_STANDARD.md:104-118`.

### Q6 · Revocation and reconnects

The no-PID launch fixture confirmed that the ID is revoked and excluded from both the online target and the effective pool. An already-registered socket gets a shutdown call, and `ready` and `request_job` check the allowlist. `persistent_worker_coordinator.py:1424-1455`, `:1465-1466`, `persistent/pwc_transport_tcp.py:192-219`, `:514-527`, `tests/test_pwc_backup_stage2.py:606-612`.

**LIVE.** During a TCP interruption, the old handler can enter `finally`, see itself as owner and pop the registry entry. It releases `_workers_lock` before reclaiming leases and clearing last-seen and readiness. In that interval the same worker can reconnect, register, report ready and pull a job. The old handler still has `owner=True`, so its unrestricted reclaim finds the replacement connection's lease. It also deletes the replacement's readiness and last-seen state. I ran the actual handlers and lease methods with two in-process message queues and a barrier just after the registry pop. The replacement had one registered and ready worker and held a valid third-attempt lease. Resuming the old handler left one registered and zero ready, and delivered `max attempts (3) exceeded after worker disconnect` for the replacement's job. Earlier attempts can be requeued while the replacement computes them. A small backup run with only one ready worker can also refuse its next pass at the readiness gate. `persistent/pwc_transport_tcp.py:359-391`, `:480-519`, `:590`, `:650-672`, `persistent_worker_coordinator.py:1988-1997`, `persistent/pwc_worker_service.py:174-214`. AMD PWC session drops are recorded. `docs/TODO_CONSOLIDATED.md:308-311`.

Always reclaim by the disconnecting socket and serialize per-ID cleanup with replacement registration, or key online, ready and last-seen state to the owning connection. Include timeout eviction in the same ownership rule. The submitted cleanup gate makes the new connection ready before closing the old one. Add the opposite interleaving, where the new handler registers after the old one has claimed ownership of cleanup and before it actually finishes. `persistent/pwc_transport_tcp.py:393-416`, `:659-671`, `tests/test_pwc_backup_stage2.py:712-730`.

### Q7 · Exit on explicit refusal

A normal reconnect from a confirmed launch keeps its ID and remains eligible. A process can start successfully on a rig while the SSH reply carrying `PID=` is lost. In that case the coordinator revokes the ID and the worker's next hello is refused. Exit 43 is the intended fail-closed outcome for that unconfirmed slot. I verified one hello, closure and `SystemExit(43)` at `_connect` with a fake socket and heartbeat. The pre-Stage-2 rig code still retries. `persistent_worker_coordinator.py:1404-1409`, `:1424-1455`, `persistent/pwc_transport_tcp.py:449-478`, `persistent/pwc_worker_service.py:342-367`, `ee39404:persistent/pwc_worker_service.py:357-361`.

Keep the refusal path independent of GPU telemetry. It now calls `_emit_heartbeat("shutdown")` before closing or exiting. With the normal heartbeat paths set, that call imports CuPy, samples its pool, and writes and syncs a heartbeat file. The exit test does not establish that ROCm import or the filesystem write will return promptly on a degraded rig. The refusal log already preserves the reason. `persistent/pwc_worker_service.py:65-99`, `:314-318`, `:358-367`, `:633-645`, `:727-747`, `tests/test_pwc_backup_stage2.py:735-765`.

### Q8 · Run 4 RANGE-MINER

I found no changed line that alters Run 4 dispatch, execution or results on a clean checkout. The launcher selects RANGE-MINER, and the miner integration branch returns before the PWC call. The module-scope PWC import succeeds in a fresh process. The ten parity-pinned files matched the base by SHA-256, and the tracked diff has no miner, shared sieve, launch, manifest or WATCHER runtime change. `gate12_launch.sh:314-317`, `:465-473`, `window_optimizer_integration_final.py:57-60`, `:1761-1767`, `:1934-1968`, `scripts/gate12_parity_gate.py:174-185`, `tests/test_pwc_backup_stage2.py:771-778`.

The existing Gate 12 admission command returned 1 against this worktree before the review file was written. It named all 12 intake entries. The launcher tests its production checkout through that same predicate before dispatch. This is the recorded clean-tree restriction, and the request's Gate 22 failure is the corresponding transitional test status. Rig parity and execution from a clean committed Stage 2 checkout are **UNAVAILABLE**. `gate12_launch.sh:79`, `:125-139`, `scripts/gate12_cleantree_gate.py:127-134`, `:232-245`, `window_optimizer_integration_final.py:101-122`, `docs/TB_RULING_FIELD6_IMPLEMENTATION.md:112-120`.

### Q9 · Shared suites

I compared both files against `ee39404` by diff and top-level AST. In the threshold suite, the changes are the PWC oracle, drivers, detectors, `G-PWC-HYBRID` and M3, plus their text. `G-MINER-UNCHANGED`, both optimizer routes and the M1/M2 mutant bodies and registrations match. In D3.25, only `drive_pwc` and G1 changed. Removing the added PWC-only arm restored the whole G1 AST to the base, including the ZMQ assertions. G13 and its mutants match. I found zero differing miner-side lines. `tests/test_s172_threshold_propagation.py:543-551`, `:670-792`, `:805-943`, `tests/test_s172_phase5_d3_25_candidate_ingress.py:316-345`, `:351-402`, `:1038-1175`, `:1328-1358`.

## Findings

1. **ALREADY RECORDED · Governing documents missing from the request's prior record.** Add the Run 4 Route A and Field 6 rulings for Q8/Q9. They govern interpretation of dirty-tree evidence and the post-commit Gate 22 check with the existing allowlist. Add the S156 lifecycle proposal for Q5, the window-anchor sequencing ruling for Q2/Q4, and the ROCm Saturation report that defines the empirical bounds in Q3. The package mentions some of their facts elsewhere. The prior record should name the governing sources. `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:139-153`, `docs/TB_RULING_FIELD6_IMPLEMENTATION.md:112-120`, `docs/PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md:65-77`, `docs/TB_RULING_WINDOW_ANCHOR_SEQUENCING.md:12-21`, `docs/ROCm_Saturation_Report_S172.md:25-42`, `:68-87`.
2. **NEW · LIVE · A late result can be accepted under the next phase's reused job ID.** This path predates Stage 2, whose additional passes expose it across more phase boundaries. The job ID starts at `sieve_000` every pass. On a result timeout the coordinator can submit the same job again while the original worker still computes. If another rig finishes the retry, P2 can begin while the first worker is still finishing P1. The old worker sends its original lease ID, but the transport completes and delivers by job ID alone. With equal directional thresholds and the same chunk interval, both the P2 range and threshold checks accept the P1 result. I reproduced that delivery through the real dispatcher and transport handler with a shortened timeout and synthetic message queues. The active P2 lease was removed and P2 reported the late P1 result as completed. A false forward/reverse intersection can follow. The effect on unseen-draw pool hit rate is unknown. Use job IDs unique across passes and require a matching active lease and connection before retiring the lease or delivering its result. Add a delayed-result regression across a phase boundary. `persistent_worker_coordinator.py:628-650`, `:1737-1814`, `:2053-2058`, `:2093-2105`, `:2654`, `persistent/pwc_worker_service.py:606-615`, `persistent/pwc_transport_tcp.py:261-290`, `:354-357`, `:628-640`, `ee39404:persistent/pwc_transport_tcp.py:601-613`.

## Verification limits

- Independently ran the selected Stage 2 threshold, phase, strategy, offset, SSH refusal, process-probe, mocked launch-accounting and fresh-import gates. Also ran the shared-suite AST comparison, the ten-file pin check, `git apply --reverse --check`, `git diff --check`, the read-only clean-tree command, the two local handler fixtures, an in-process refusal check and an in-memory Optuna check. The process-kill fixtures used a private PID namespace and dummy processes. The attempted socket-pair transfer was **UNAVAILABLE** in this sandbox, so the reconnect and late-result checks used message queues to exercise the actual coordinator handlers.
- Read Alpha's full-suite and mutant results. Independent live CT100 process and handshake behavior, deployed worker exit, real CUDA/HIP kernel execution, five-strategy runtime, and PWC pool hit rate on unseen draws against `k/1000` are **UNAVAILABLE**.
