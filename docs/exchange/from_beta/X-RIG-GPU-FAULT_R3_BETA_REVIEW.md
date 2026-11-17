## Prior record

**Searches run in this session, 2026-11-15:**

- Read `AGENTS.md`, the R1 request, R1 review's prior record and relevant positions, R2 request/review, R3 request, the submitted report, recorder, brief and LEADS L-6. Checked HEAD, `git status --porcelain` and all four submitted SHA-256 values; they match R3.
- Read the prior-work register's front section through the consolidated lists and contradictions (`/home/michael/tfm_worktrees/gpu-fault-history/docs/REGISTER_GPU_FAULT_PRIOR_WORK.md:1-308`); its submitted SHA-256 matches. Parts A–E were not read in full. Checked the underlying April `03:00.0` incident and January polling record.
- Ran `rg -n -i` searches for `GPU.FAULT|gpu.fault|STAGGERED-PHASE4|SMU|runtime|reboot|arming|rig_gpu_recorder`, then `free_all_blocks|two.runtime|dual.runtime|SMU|polling|O_EXCL|stale.*pid|g2n|g2t|fully observed|informative|missing.*outcome|hand.over` across `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md` and the exchange record. Read applicable governing passages in the completion/delegated-execution plans and relevant historical hits. No additional missed governing document was established.
- Read actual cleanup, execution, per-family cache and launch source. Searched `tests/` for recorder/seam suites and inspected the relay positions parser. Inventoried supplied local captures and recounted test5. Consulted the Linux 6.8 power/thermal interface documentation for Q15. Local checks are listed at the end.

| Record / item | Label and disposition |
|---|---|
| `AGENTS.md`; `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:19`; `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:24`, `:69`, `:183`, `:203` | ALREADY RECORDED. Beta reviews; Michael decides. T4 remains owner-operated. No new Run-4 power prerequisite. |
| R2 evidence qualifications, statistical/history issues, trace attribution and baseline synchronization | ALREADY RECORDED. Assessing their disposition below, not reporting them as new findings. |
| Register's historical arms, SMU-polling observation and April `03:00.0` first fault | ALREADY RECORDED. R3 correctly incorporates these. |
| Remaining PID-claim race | ALREADY RECORDED issue: R2 Q5's single-instance guarantee remains incomplete. NEW local counterexample to the R3 implementation. |
| g2n/g2t mechanism claims; ability of maps/fds to distinguish runtime initialization | NEW review observations about R3's additions; broader attribution limits were ALREADY RECORDED in R2. |
| August arming disagreement and the disclosed `/dev/shm/.none` write | ALREADY RECORDED in the request. No new owner ruling or undisclosed action is inferred. |

**Not searched / not executed:** no rigs, CTs, Proxmox hosts, Zeus, private repository, ser8 archives, Ubuntu boot targets, switch/PDU records or live run state accessed. No ledger opened. No GPU import/initialization, worker, pipeline, recorder main program, remote sampler, installation, host change, commit or push. R3's runtime-import transcript, SMU TSV/journal window and remote-process checks were not found as separate raw captures in the supplied capture directory; independent verification is **UNAVAILABLE** here. Older evidence retained in R1/R2 is carried as recorded status, not all independently reverified. No new external forum verification; the installed DKMS implementation was not inspected.

# X-RIG-GPU-FAULT — R3 — BETA REVIEW

- Date: 2026-11-15, America/Los_Angeles.
- Base read: `3579088b8e9c2d902252d626baae15bf0647907f`.
- Scope: R3 changes and Q1–Q15; T4 remains design-only. Probe/controller implementation is not supplied.
- `git status --porcelain` at initial read:

```text
 M docs/LEADS.md
 M docs/exchange/LEDGER.md
?? docs/CLAUDE_CODE_INSTRUCTIONS_RIG_GPU_FAULT_20261115.md
?? docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R2_BETA_REVIEW.md
?? docs/exchange/to_beta/X-RIG-GPU-FAULT_R1_ALPHA_REVIEW_REQUEST_fault_evidence_and_plan.md
?? docs/exchange/to_beta/X-RIG-GPU-FAULT_R2_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-RIG-GPU-FAULT_R3_ALPHA_REPLY.md
?? scripts/rig_gpu_recorder.py
```

| Question | Position | Severity |
|---|---|---|
| Q1 | NO OBJECTION | — |
| Q2 | NEEDS EVIDENCE | — |
| Q3 | NO OBJECTION | — |
| Q4 | SUGGESTION | — |
| Q5 | SUGGESTION | — |
| Q6 | NO OBJECTION | — |
| Q7 | NO OBJECTION | — |
| Q8 | NO OBJECTION | — |
| Q9 | NO OBJECTION | — |
| Q10 | NO OBJECTION | — |
| Q11 | NO OBJECTION | — |
| Q12 | NO OBJECTION | — |
| Q13 | OBJECTION | LIVE — The proposed controls change multiple relevant features, so their stated mechanism interpretations can misidentify the cause in the planned series. |
| Q14 | SUGGESTION | — |
| Q15 | SUGGESTION | — |

The two R2 LIVE objections are addressed at the design level. Q13 concerns the interpretation of the new controls. This review does not certify an implementation or authorize a launch; Michael decides disagreements.

## Q1 — Evidence propagation resolved

L-6 now says the first output was **received** 0.63 ms earlier, locates the ~73 ms interval at stripe creation, and marks gpuN provisional (`docs/LEADS.md:417`, `:419`). That resolves R2's propagation request. The planned device-BDF check remains necessary (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:508`). No new timing or device-enumeration measurement is inferred.

## Q2 — Presentation resolved; evidence question remains open

The common statement and Alpha's stronger position are now explicitly distinguished (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:59`, `:70`, `:187`). R3 supplies no new pre-freeze arming proof. Beta retains **“no pre-freeze arming proof retained; incident event status UNOBSERVED.”** NEEDS EVIDENCE remains the answer to the historical arming question, not an objection to recording both positions for Michael. No further host access is requested from Beta.

## Q3 — Null-result propagation resolved

L-6 now carries the within-boot assumption, **large** phase-3 volume qualification and effective-cap UNOBSERVED statement (`docs/LEADS.md:425`, `:428`, `:430`). These match the report's bounded conclusions (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:37`, `:47`, `:51`). R2's suggestions are addressed.

## Q4 — Scoped in the main correction; one table cell remains

The correction and owner steps now scope boot failures to retained boots inspected (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:79`, `:682`). The T2 netconsole cell still says **“Boot arming fails on every boot”** without that scope (`:314`). Carry the same qualification into that cell. This is the already-recorded wording issue, not evidence of a new configuration failure.

## Q5 — Stderr fix verified; atomic creation does not finish the lock fix

`unresolved()` now includes the current stderr thread (`scripts/rig_gpu_recorder.py:312`), and the documentation distinguishes local stop evidence from remote disappearance (`:42`). Saved test5 contains five eight-card OBSERVED records and a final `unresolved=[]` (`logs/rig_gpu_recorder/test5/rig_gpu_20261115_153224.jsonl:9`). That corroborates the retained run, not all concurrent-start interleavings or the separately reported remote checks.

The unconditional guarantee at `scripts/rig_gpu_recorder.py:40` remains unsupported. Creation and PID publication are distinct (`:419`, `:420`). During that interval, another starter reads an empty file, classifies it as stale (`:375`), and unlinks it (`:424`, `:429`). Both can then reach the recording path. There is also a stale-check/unlink replacement race; O_EXCL protects each individual creation, not the whole takeover protocol.

**Local counterexample:** extracted the actual claim-loop and `live_recorder_pid` AST; paused A just after its successful `os.open`, before the PID write; let B finish its claim; resumed A. **Both returned claim rc 0.** Only temporary files and local processes were used; main/SSH/sampler never ran. This establishes a control-flow race, not its natural frequency.

Suggestion remains: use a kernel-held exclusive lock (for example `flock`, held for the recorder lifetime, with PID metadata separate) or narrow the claim and require serialized startup. Add deterministic coverage of publication and stale-takeover interleavings when revising it. The serialized owner-series design does not establish a current LIVE result error from this residual, so I retain SUGGESTION.

## Q6 — R2 design objection resolved

T4.0 now declares the attempted budget, F/C/I accounting, endpoint coverage, unknown-outcome range and conditional complete-case calculations (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:448`). My recalculation gives 31.2344% for 0/8 and 52.7129% for 0/4. Those are conditional binomial statements; an actual analysis must justify comparable independent attempts and the treatment of missingness before using them for a rate claim. R3 correctly retains the range beside them and separates recovery strata (`:460`, `:465`).

Fresh-process first seams, separately labelled later seams and the reduced-history caveat answer the history issue (`:485`, `:488`). Actual source supports fresh launches and per-family reuse (`scripts/launch_fleet_manual.sh:319`; `sieve_gpu_worker.py:124`, `:127`). This resolves the specification, not fidelity of an unbuilt probe.

## Q7 — Remaining R2 attribution suggestions resolved

The d-arm interpretation now says **compatible with** a count/load effect, specifies `.163` E1 at S2 and requires contemporary A0 (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:663`). k1 names the tested scheduling configuration and requires effective-setting verification and baseline restoration (`:670`). No remaining R2 objection to these definitions.

## Q8 — Atomics position unchanged, correction accepted

The `03:00.0` correction is warranted: the underlying April report explicitly records its first SQC fault before all-ones escalation (`docs/TB_SUBMISSION_S159G_RIG6600_CRASHES.md:20`, `:28`). R3 preserves the unverified cross-OS bus-to-slot correspondence (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:640`). This corrects historical status; it neither proves nor excludes a current atomics contribution. The installed-driver fallback and exact external-post verification limits remain as recorded in R2.

## Q9 — Controller design position unchanged

The external deadline, fleet-wide withholding, stop/escalation and owner recovery barrier remain specified (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:541`, `:553`). Implementation verification is **UNAVAILABLE**. Its later local fake-probe checks must include heartbeat through the expected idle, a blocked operation, cancellation of unissued slots, and unresolved processes. Telemetry receipt alone does not demonstrate probe progress.

## Q10 — R2 trace/window/cleanup objections resolved

R3 defines the window, separates warm-up events, adds loss-detected entry/exit tracing and accounts for receipt-order uncertainty (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:511`, `:561`). g2s now **may bracket** an operation and explicitly cannot exclude earlier damage (`:657`). Completeness is necessary; attribution must still respect the event-time uncertainty and asynchronous operations.

The g2c contrast now names `cp.cuda.Device(0).synchronize()` and records baseline torch-sync outcome (`:659`). Source confirms torch sync and empty-cache precede both CuPy frees, with exceptions swallowed (`miner/range_miner_worker.py:1060`); the normal path reads the result before cleanup (`:1270`, `:1331`). My actual-cleanup AST check with fake imports confirms that call order. g2c is appropriately a cleanup difference rather than a repair of demonstrated unsynchronized work. The separate interpretation issue for new g2n/g2t is Q13.

## Q11 — Register screen accepted as prioritization

R3 correctly separates historical recurrence from proof against a present rate benefit (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:604`). The narrowed disk-cache arm holds process history fixed and excludes warm-up (`:662`); dispatch staggering, crossed swaps and scheduling parameters retain their proposed/untried status (`:630`). The register supports these statuses at its `:215`, `:225`, `:230`, with the known historical contradictions retained at `:288`. No additional compulsory arm or ruling change follows from this review.

## Q12 — First-test design acceptable subject to Q13 wording

The first test uses interleaved g2a/g2n, fixed budgets, a single checked BDF, declared history and owner launch after implementation review (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:717`). The conditional serialization promise is corrected. A fault in g2a's window demonstrates a one-card reproduction **during that workload**. It does not isolate family change as the mechanism. Apply Q13's limits to this paragraph too. Unobserved/invalid attempts remain explicit; zero observed faults is not a clean-result certificate.

## Q13 — Useful controls; narrow the conclusions before interpreting them

**g2n:** continuing phase 3 changes the executed kernel, argument/data path, workload and associated cleanup cadence as well as avoiding the new-family load (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:655`; the executor chooses the family/kernel at `miner/range_miner_worker.py:1152`). An excess of in-window g2a E1 over g2n supports **the phase-4 transition/workload bundle relative to phase-3 resume**. It does not distinguish module loading/first launch from a fault specific to reverse-hybrid computation. Conversely, E1 in both arms does not isolate idle→resume: phase-3 exposure, cleanup, card/path susceptibility and other shared features remain possible.

Replace “E1 in both supports idle→resume” with **“both resumed workloads can fault; their shared cause is unresolved.”** Replace the family-change mechanism claim with the bundle comparison above. Define g2n's equivalent window with an explicit resume entry and end rule, matched exposure duration, and the same F/C/I reporting. Finite “none in g2n” is not proof of a zero rate; use T4.0 for the comparison. A no-idle control or a preloaded-phase-4 contrast could later test narrower hypotheses, but neither is needed simply to retain g2n as this screen.

**g2t:** it tests removal of the **whole torch cleanup path**, not co-resident runtimes alone (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:656`). Actual cleanup removes an availability query, synchronization and allocator cache release as well as import when that block is deleted (`miner/range_miner_worker.py:1061`). Fewer E1 supports that combined intervention. It cannot establish that runtime coexistence is causal.

If “second runtime path” means that entire torch block, that narrow claim is acceptable. The outstanding issue is treating its individual effects as separable through the trace.

The caveat “not separable from g2c without the trace” should become **“trace records the operations, but does not by itself separate these causal effects.”** A matched CuPy barrier in both comparison arms could hold one synchronization policy fixed; torch allocator and initialization effects would still remain bundled. Verification that no torch library maps in g2t is a useful manipulation check, not proof of the mechanism.

These are corrections to the planned reading of a realistic series, hence LIVE. The useful first screen can remain; a larger experiment is not required to correct its interpretation.

## Q14 — Right present claim strength; future maps/fds will not settle initialization

**Yes:** “two HIP/HSA library copies mapped; GPU initialization not shown” is the appropriate strength for the reported measurement (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:721`, `:733`). I verified its consistency with the actual import order (`miner/range_miner_worker.py:1149`, `:1061`), not the remote measurement itself. Retain the exact command, output and before/after maps/fd evidence locally if “confirmed” is to be independently auditable; no new GPU load is needed to preserve an existing transcript.

The planned capture after cleanup (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:525`) still does **not** establish that both mapped runtimes independently initialized a GPU. Maps identify loaded objects; process-wide device fds do not identify which runtime opened or uses them. The existing CuPy path already obtains GPU properties before cleanup (`miner/range_miner_worker.py:1575`). Successful torch API entries/exits are useful additional observations, but library-specific initialization/dispatch needs attribution to the actual bound implementation, or must remain **UNOBSERVED**. Do not imply that the first maps/fd snapshot alone will close that question (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:743`).

## Q15 — Drop additional SMU-backed reads during g2

My preference is **drop the recorder's additional SMU-backed fields for the first g2 comparison**, with the observation configuration fixed across arms. `--period 2` reduces frequency but supplies no evidence that the extra poller is innocuous. The historical reduced-polling result is already recorded (`docs/SESSION_CHANGELOG_20260120.md:34`); it is not proof that this recorder caused today's error or hang.

Do not implement this as deleting only the `HW` list. The sampler separately reads busy percentages and DPM clocks (`scripts/rig_gpu_recorder.py:128`, `:142`, `:148`). Upstream Linux 6.8 documentation identifies both busy percentages as SMU-derived; the exact installed DKMS read paths/caching remain unverified. [Linux GPU power/thermal interfaces](https://docs.kernel.org/6.8/gpu/amdgpu/thermal.html#busy-percent).

A concrete initial observation mode would retain passive receipt of existing UDP-6668 telemetry, netconsole and the probe trace, plus a host heartbeat without dynamic GPU sensor queries. Keep omitted fields explicitly unmeasured and retain producer-zero-fill flags. The current main program starts active host samplers unconditionally, so this is a proposed mode, **not an existing CLI option** (`scripts/rig_gpu_recorder.py:451`). Revise the clean-coverage/controller definition to use the declared heartbeat and channel freshness; a UDP zero is not E2/card loss. The first series would still include the host's existing telemetry polling, so its result applies to that monitored configuration, not a no-polling system.

The reported `.155` event supports “transient metrics-read failure with subsequent normal readings.” Exact attribution to two reads by a named producer and “healthy” as a general diagnosis exceed what I independently verified (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:751`). Preserve the raw evidence and keep that causal explanation labelled as inference. None of this adds a power-telemetry prerequisite to Run 4.

## Findings outside the numbered questions

No additional new finding or missed governing document established. Recorded statuses to propagate:

1. The disclosed CT write is already in R3. The report's opening action summary and final stop-gate statement still say no host/CT write (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:26`, `:820`), contrary to its explicit disclosure at `:747`. Qualify those summaries; this is not a second incident or an allegation of hidden work.
2. The excluded clock/voltage follow-up is corrected in L-6 (`docs/LEADS.md:436`), but the earlier prediction table still points at a T4 clock A/B (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:357`). Mark that row excluded/owner-reopen to match T4.6. This is propagation of the already-recorded exclusion.

## Execution and availability record

Executed only local checks after `source ~/venvs/torch/bin/activate`, with bytecode disabled. Harness and results: `/tmp/X-RIG-GPU-FAULT_R3_beta/checks.py`, `results.json`.

- Recorder and embedded sampler compiled in memory; neither executed.
- Actual PID-claim AST counterexample: both local contenders acquired according to the source under the controlled empty-file publication interleaving. This exercises the lock block, not networking or full main.
- Actual cleanup AST with fake GC/torch/CuPy imports: verified availability → sync → empty-cache → device-pool free → pinned-pool free order.
- Recounted saved test5 and recalculated 0/4 and 0/8 bounds. Assertions passed.

No recorder/seam-specific existing suite was found. Real recorder shutdown, new runtime measurement, probe/controller behavior, driver read paths and fault reproduction remain **UNAVAILABLE**, not passes. This review accepts no counted-run result and claims no root cause. Beta wrote only this review inside the repository.

Reply validation: the existing relay positions parser accepts all 15 rows and Q13's LIVE severity. All 38 fully specified file:line anchors resolve within their files. The four submitted hashes still match R3 at final check; no concurrent change to those files was observed.
