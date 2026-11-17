## Prior record

**Searches run in this session, 2026-11-15:**

- Read `AGENTS.md`, the R1 request/review, the R2 reply, all four submitted files and the LEADS diff. Ran `git rev-parse HEAD`, `git status --porcelain` and SHA-256 checks. All four submitted hashes match R2.
- Ran `rg -n -i 'free_all_blocks|serializ|gpu.fault|freeze.reproduc|hws_max_conc_proc|pool.?6|pool.?8|cwsr_enable|cooldown|jitter|stagger'` over `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*_AUDIT.md` and `docs/TODO_CONSOLIDATED.md`. Searched `docs/` and the exchange record for operation attribution, invalid-cycle denominators, cold/warm history and synchronization before freeing. Read applicable hits, including the delegated-execution plan, S162 Option B/final changelog, free-all-blocks proposal, S159G, S163KARG, S167–169, S170, S172 saturation and S174_D1 records.
- Read the actual executor, cleanup, kernel cache, payload builder, launch and PWC environment paths. Read/parsed the supplied local KFD, atomics/netconsole and August arming/powercap captures and the saved recorder test4 JSONL. Checked the named prior-work register path: still absent at this review.
- Consulted AMD's HIP debugging documentation and CuPy's versioned 13.5.1 API documentation for serialization and blocking copies. Executed only the local checks described at the end; no recorder main program or sampler ran.

| Governing record / item | Label and disposition |
|---|---|
| `AGENTS.md`; `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:24`, `:69`, `:183`, `:203` | ALREADY RECORDED. R1 Finding 1 is addressed; T4 explicitly remains owner-operated. Beta reviews and Michael decides. |
| Completion plan owner rulings, carried forward in R1 | ALREADY RECORDED. This review adds no power-hardening prerequisite to Run 4. |
| R1 Q1–Q4/Q8 evidence qualifications and Q6/Q7/Q9 design objections | ALREADY RECORDED. Status assessed below, not new findings. |
| S162 diagnostics, S163 memory proposal, S170 pacing/peak lock, S172 concurrency observations | ALREADY RECORDED. Historical results justify screening priorities; they do not establish present-day causal exclusions. |
| LEADS L-3 correction | CHANGES A RECORDED POSITION, already raised in R1. Both positions remain for Michael; no owner ruling changed here. |
| Attempted versus observed statistical denominator; repeated g2 module history; g2s attribution; g2c's existing baseline synchronization | NEW observations about the R2 design, with the broader independence/history concerns already recorded in R1 Q6. See Q6/Q10. |

**Not searched / not executed:** no access to rigs, CTs, Proxmox hosts, Zeus, private repository, ser8 archives, Ubuntu boot targets, switch/PDU records or live run state. No ledger was opened in this round; R1's immutable-main-file checks are carried as recorded status, not rerun. No GPU initialization, pipeline, worker, recorder, installation, host change, commit or push. Event-time caps/environment, August pre-freeze arming, physical card identity, and actual probe/controller behavior remain UNAVAILABLE for independent verification here. External operator posts not retrievable in R1 were not newly verified in R2.

# X-RIG-GPU-FAULT — R2 — BETA REVIEW

- Date: 2026-11-15, America/Los_Angeles.
- Base read: `3579088b8e9c2d902252d626baae15bf0647907f`.
- Scope: R2 report, recorder, LEADS and design-only reproduction plan. The probe/controller are not built.
- `git status --porcelain` at initial read:

```text
 M docs/LEADS.md
 M docs/exchange/LEDGER.md
?? docs/CLAUDE_CODE_INSTRUCTIONS_RIG_GPU_FAULT_20261115.md
?? docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R1_BETA_REVIEW.md
?? docs/exchange/to_beta/X-RIG-GPU-FAULT_R1_ALPHA_REVIEW_REQUEST_fault_evidence_and_plan.md
?? docs/exchange/to_beta/X-RIG-GPU-FAULT_R2_ALPHA_REPLY.md
?? scripts/rig_gpu_recorder.py
```

| Question | Position | Severity |
|---|---|---|
| Q1 | SUGGESTION | — |
| Q2 | NEEDS EVIDENCE | — |
| Q3 | SUGGESTION | — |
| Q4 | NO OBJECTION | — |
| Q5 | SUGGESTION | — |
| Q6 | OBJECTION | LIVE — Counting incompletely observed attempts in a binomial denominator and pooling different process histories can produce an unsupported rate conclusion in this proposed series. |
| Q7 | SUGGESTION | — |
| Q8 | NO OBJECTION | — |
| Q9 | NO OBJECTION | — |
| Q10 | OBJECTION | LIVE — The proposed g2 interpretation can assign an operation or an overlap mechanism without a complete trace or a verified synchronization difference from production. |
| Q11 | NO OBJECTION | — |
| Q12 | SUGGESTION | — |

R1 Q4 and Q9 are resolved at the document/design level. R1 Q6 is partly resolved, with the remaining issues below. Most R1 Q7 causal overclaims are corrected. The two LIVE positions concern the design and interpretation before execution; they do not make Beta a launch approver. Michael decides disagreements.

## Q1 — Evidence: main report resolved; propagate the qualifications

T1 now distinguishes stripe creation from launch, receipt spacing from fault simultaneity, main-file shard counts from complete execution history, and observed surviving-rig power from missing failed-rig power (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:190`, `:213`, `:241`). The SATA statement is appropriately a candidate diagnosis (`:341`).

The new KFD captures establish ascending GPU-node location order in all three CTs. For `.164`, node 7 has `location_id=5632`, or `16:00.0`, and node 2 has `1536`, or `06:00.0` (`logs/rig_gpu_fault_20261115/ct_kfd_topology_192.168.3.164.txt:14`, `:44`). They do not independently establish ROCr agent enumeration. Keeping that map provisional and checking the selected device's actual BDF before load resolves the review issue (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:249`, `:482`).

**Suggestion, ALREADY RECORDED qualification:** apply the same receipt wording to `docs/LEADS.md:417`: it still says the card “faulted 0.6 ms earlier.” Say the first fault output was received earlier, and keep gpuN provisional there too. This is propagation of R1's correction, not a newly discovered timing defect.

## Q2 — August arming: evidence supplied, conclusion still unestablished

The raw capture now supplies the previously missing file births and per-boot counts. On `.163`, the persistence files were born at 19:25, boot −3 has zero retained netconsole lines, and that boot's retained journal ends at 18:19 (`logs/rig_gpu_fault_20261115/aug22_arming_and_powercap_192.168.3.163.txt:10`, `:18`, `:44`, `:48`). This corroborates the narrower R1 position; it does not fill the lost tail.

**Beta's position remains:** no pre-freeze arming proof retained; incident kernel-event status UNOBSERVED. Alpha's “not armed” remains a separate position for Michael. The two-position appendix is an appropriate way to record the disagreement (`docs/LEADS.md:288`). It resolves presentation of the disagreement, not the underlying evidence question. “Every current boot failed to arm automatically” is also different from “no ad hoc arming occurred in August.”

The categorical “correct reading” at report `:71` and the August cell at `:185` should be explicitly attributed to Alpha or replaced with the common supported statement. No new host read is requested from Beta.

## Q3 — Null results: resolved in the report, older LEADS wording remains

The revised statements are supported at their declared strength: same recorded trial inputs and boot, large phase-3 survivor volume unnecessary for today's hang, and recurrence despite a configured cap with its effective event-time value unobserved (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:37`, `:47`, `:51`). The new prior-boot “applied” line supplies deployment evidence, correctly not a successful-write readback (`logs/rig_gpu_fault_20261115/aug22_arming_and_powercap_192.168.3.163.txt:55`; script reproduced at report `:332`).

**Suggestion, ALREADY RECORDED qualifications:** carry those exact limits into `docs/LEADS.md:424`, `:426`, `:427`. Those lines still say no static setting discriminates, volume is unnecessary, and the cap is insufficient without the report's assumptions/qualifications. No further evidence is needed merely to use the narrowed wording.

## Q4 — Configuration comparison: resolved

The netconsole row now distinguishes installed persistence files from failed boot arming (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:312`). The supplied `.155` capture adds the surviving host's boot failure and later successful startup (`logs/rig_gpu_fault_20261115/atomics_netconsole_192.168.3.155.txt:190`, `:198`); `.121` and `.163` show the same boot failure at their respective capture `:190`. Scope “every boot” to the retained boots inspected, rather than an assertion about all future or unretained boots.

The environment row now discloses simulated-shell versus live inheritance limits, and the unknown display-stack/firmware comparisons remain explicit (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:313`, `:318`). The corrected PWC DEBUG_TRAP anchor is present in actual source (`persistent_worker_coordinator.py:140`, `:141`, `:145`). No remaining LIVE objection to this comparison.

## Q5 — Recorder: adopted fixes, two remaining suggestions

Verified by source: the remote command uses `-B -I -u`; sampler file access is read-only; stdout reads have a 1 MiB cap; stderr retention is bounded; UDP records carry the producer-zero-fill qualification; and the final exit reflects unresolved local host threads/SSH processes (`scripts/rig_gpu_recorder.py:83`, `:209`, `:220`, `:243`, `:347`, `:452`, `:463`). The late-sample qualification remains explicit (`:18`, `:271`).

I recounted saved test4: **24 OBSERVED, 6 UNOBSERVED with cards null, 25 flagged UDP records, and a final unresolved=[]**. This verifies the retained artifact. The second-instance rc 2 and absence of remote samplers are Alpha's reported execution evidence, not new live checks by Beta (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:406`).

Two suggestions, neither an established LIVE failure:

1. The PID check and PID-file write are separate operations, so the guard covers a later second instance, not simultaneous starts (`scripts/rig_gpu_recorder.py:408`, `:415`). Use an atomic lock if claiming an unconditional single-instance guarantee.
2. The stderr thread is joined with a timeout but is not included in `unresolved()` (`:281`, `:304`). Include it or narrow “every thread.” Local SSH termination also does not by itself certify remote sampler disappearance; retain Alpha's separate remote-process observation as a separate check.

No recorder main program or network function ran during this review.

## Q6 — Statistics and probe fidelity: partly resolved, LIVE issues remain

The fixed attempted budget, abandonment of clean-streak stopping, descriptive 2/6, interleaved controls, replication and S2 non-reproduction wording resolve the principal R1 overclaims (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:437`, `:441`, `:448`, `:520`). Real payloads, worker-specific ranges and separate argument/history/count controls are sound requirements (`:471`, `:480`, `:487`; actual payload/launch at `miner/range_miner_coordinator.py:9331` and `miner/range_miner_worker.py:1268`).

**Remaining statistical issue:** B is explicitly an attempted-cycle budget, cycles may be invalid, but k/B is then given an exact binomial upper bound (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:441`, `:443`, `:446`). An invalid outcome cannot serve as an observed non-fault. Eight attempts with four clean and four unknown outcomes do not provide eight observed non-fault trials. Under the independent common-rate model, 0/4 has a one-sided 95% upper bound of **52.7%**, not 31.2%.

Specify attempted = fault + clean + invalid separately, and report endpoint-specific observation coverage. A demonstrated E1 remains an E1 even if a different observation fails. For unknown E1 outcomes, give the descriptive range F/B through (F+I)/B. An exact binomial interval/Fisher calculation needs a declared independent, comparable observation unit and justified treatment of missingness; excluding informative missing outcomes is not automatically a repair. The published 31.2% value is correct for **eight fully observed independent common-rate cycles**. Repeated process history and recovery-dependent boot changes make those assumptions substantive here, as already noted in R1 Q6.

**Remaining history issue:** “persistent process” plus repeated hand-overs is not yet an unambiguous treatment definition. The design says reverse-hybrid is first loaded at the seam (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:466`), while g2a repeats eight hand-overs (`:587`). Actual `_get_kernel` retains its per-family object and returns it on later calls (`sieve_gpu_worker.py:124`, `:127`, `:131`). My extraction check created each mocked family object only once across eight hand-overs. This checks cache control flow, not actual HIP module timing.

Define whether each attempt reconstructs a declared process history or whether the process persists across attempts. In the latter case, separate the first seam from later seams with reverse-hybrid already used, and align baseline/treatment histories. Also label the omission of production phases 1/2 as a deliberate reduced probe rather than calling the entire runtime history identical. A reduced probe is useful; its non-reproduction still means only non-reproduction. The implementation review must verify the chosen definition, not simply a matching survivor count.

## Q7 — Remaining arms: most R1 corrections resolve; tighten two inferences

The revised b1/b2 limits, fixed ranges, crossed e2 after restoring e1, e1 swap-back and CPU-slot class label substantially resolve R1's objections (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:592`, `:596`, `:597`, `:598`, `:600`). Hardware arms remain available after software non-reproduction (`:612`).

Two suggestions:

- d1 and d2 both clean would be **compatible with** a count/load effect, not by themselves evidence establishing that mechanism (`:594`). The S172 table has finite clean pool-6 runs and a clean pool-8/25k run too; its boundary depends on chunk size as well as concurrency (`docs/ROCm_Saturation_Report_S172.md:31`, `:33`, `:34`). Compare both against contemporary A0 and retain uncertainty. At S2, specify whether a d-arm endpoint concerns `.163` or any host, as R1 requested.
- For k1, fewer E1 supports the tested scheduling configuration, not specifically “KFD/MES concurrency saturation” (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:601`). Verify each setting is supported/effective, restore baseline between parameters, and preserve the distinction between a scheduling intervention and an identified driver/firmware mechanism.

The g2 issues are handled separately in Q10. The remaining software-first ordering is reasonable as a screen.

## Q8 — Atomics: supplied evidence resolves the missing capture

I parsed all three capability captures: eight AMD VGA endpoints each; **only `03:00.0` has ReqEn+**, seven have ReqEn−. The CPU root's positive completer capabilities and PCH negative capabilities are retained (for example `logs/rig_gpu_fault_20261115/atomics_netconsole_192.168.3.155.txt:5`, `:15`, `:67`, `:82`); each journal has seven matching unsupported messages (`:177`). The revised GPU-to-root wording and on-device caveat are appropriate (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:542`, `:549`).

The riser statement is now advice without a verified repair; the KFD firmware lead stays an inference about upstream source rather than certification of the installed DKMS behavior (`:527`, `:534`). Those resolve the requested wording changes. R1's unavailable exact forum-post attributions remain unavailable; no fresh verification of every external quote is claimed.

## Q9 — Stop controller: design objection resolved

The external deadline, controller-issued slots, fleet-wide withholding, stop/escalation with unresolved PIDs, separate E1/E2/E3 and owner recovery barrier answer R1 (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:453`, `:498`, `:501`, `:510`). The plan correctly disclaims recovery of a wedged driver by signal (`:509`).

Implementation verification remains **UNAVAILABLE**, not passed. The future review should exercise cancellation/deadline behavior locally with fake probes, including an expected 200-second idle with an independent heartbeat and a blocked operation. The existing executor can block during its result read or cleanup (`miner/range_miner_worker.py:1270`, `:1063`); telemetry receipt alone must not stand in for completed probe work. This is the stated implementation gate, not an additional permission request.

## Q10 — g2: useful experiment, overstrong attribution and underspecified contrast

**g2a:** a single-card fault establishes that eight active cards are unnecessary for that reproduced event. It leaves card, slot, power, runtime and kernel explanations open. Define the phase/time window: an E1 during phase-3 warm-up is a one-card warm-up reproduction, not a phase-4 seam reproduction. T4.3 currently says “Any E1” yields a seam reproduction (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:518`). Without a no-handover control, even an E1 at the seam supports “one card faulted in this seam workload,” not a unique hand-over mechanism. Q6's cold/later-history distinction also applies.

**g2s:** keeping serialization as a diagnostic instrument is appropriate. S162 Option B recorded recurrence with the flags set; that establishes non-prevention in that run, not impossibility of a present-day rate change (`docs/TB_UPDATE_S162_OPTION_B_RESULTS.md:12`, `:23`, `:33`). AMD documents the flags as a way to bring fault reporting closer to the launching call. That supports using them for attribution, subject to the runtime honoring them. [AMD HIP debugging](https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.2/how_to_guides/debugging.html#general-debugging-tips).

The unconditional promise that the “last completed stamp names the operation in flight at E1” is too strong (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:588`). A completed operation and a currently entered operation are different facts. The specified UDP stamps (`:485`) do not establish a complete, loss-detected entry/return trace or an ordering guarantee against kernel-log receipt. Report **last confirmed completion and last entered operation**, with sequence IDs, device/process identity, return/error status and explicit gaps. Attribute an operation only when that trace supports it; otherwise operation-at-fault is UNOBSERVED. A later fault after a completed free does not alone demonstrate free/kernel overlap. Serialization cannot rule out persistent corruption or a driver defect left by an earlier operation.

**g2b/g2c:** removing only the pool-free calls is a defined intervention, if both device and pinned pools are specified and the rest of cleanup stays fixed (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:589`). It tests that combined release policy, not one pool uniquely. Retained allocation/timing changes are part of the intervention.

For g2c, production already attempts `torch.cuda.synchronize()` before either free (`miner/range_miner_worker.py:1061`, `:1063`, `:1069`), after a blocking result `.get()` on its normal path (`:1270`). CuPy 13.5.1 documents `.get()` as host-blocking by default. [CuPy ndarray.get](https://docs.cupy.dev/en/v13.5.1/reference/generated/cupy.ndarray.html#cupy.ndarray.get).

My actual-cleanup extraction with stub imports verifies the existing sync-before-free call order when torch reports available; when it does not, that sync is skipped. This is not a GPU result. Consequently, “add a device synchronize” (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:590`) is not yet evidence of a distinct outstanding-work barrier. Specify the exact CuPy/device barrier and device, record whether baseline torch synchronization ran/succeeded, and preserve its exception path. A successful extra barrier can test a cleanup difference; it cannot be described as repairing an established unsynchronized production free. These corrections are needed before interpreting a g2 series.

## Q11 — Prior-work screen: reasonable prioritization, no compulsory new arm

I found records supporting the historical results used to deprioritize the listed classes: S162 pool-off/scratch/serialization/seed-cap diagnostics (`docs/SESSION_CHANGELOG_S162_FINAL.md:24`), S170 peak lock with recurrence and 50 ms cooldown failure (`docs/SESSION_CHANGELOG_20260425_S170.md:24`, `:100`, `:166`), and cwsr's adverse/mixed history (`docs/SESSION_CHANGELOG_20260424_S167_S168_S169.md:180`, `:202`; `docs/SESSION_CHANGELOG_20260509_S174_D1.md:43`). The S159G record establishes the three-variable environment deployment (`docs/SESSION_CHANGELOG_20260402_S159G.md:69`); it is not an isolated test of each environment variable.

No objection to dropping these as **historically tried classes from this screen**, with T4.6's explicit different-conditions caveat and owner reopening retained (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:565`). A historical fault rules out absolute prevention in that historical treatment; it does not eliminate a possible current rate benefit. Today's cwsr=1 recurrence addresses necessity of cwsr=0, not the efficacy of switching it to 0; the older adverse results are the reason to deprioritize that arm.

No kept arm must be dropped on the supplied evidence. b1 is a different first-family-launch intervention from first-assignment jitter; h isolates disk-cache state if process history is fixed. Pool-off/raw allocation is different from g2b retaining the pool while removing explicit release. The April removal report conflicts with current source/TODO and is correctly marked a conflict (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:587`; `sieve_gpu_worker.py:91`; `docs/TODO_CONSOLIDATED.md:317`). It does not establish that the exact miner g2b contrast already ran. Recheck the prior-work register when it exists.

## Q12 — First test: reasonable choice, narrow its follow-up promise

g2a is a reasonable low-concurrency first screen after the probe/controller implementation review. Its two basic outcomes are correctly bounded: an E1 in the defined hand-over window demonstrates a one-card reproduction; eight fully observed no-fault attempts are single-card non-reproduction and may motivate A0 (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:651`). This does not establish that g2a is the highest-yield test or that it reproduces every production history.

Replace “a serialised re-run would name the operation in flight” with “a serialised re-run may bracket the operation if the trace is complete.” Apply Q6's attempted/valid/history reporting and Q10's phase attribution. The first test need not be expanded to a fleet run to make these corrections. No launch is performed or authorized by this review.

## Findings outside the numbered questions

No additional new finding. R1 Finding 1 is addressed by the explicit delegated-execution citation and owner-operated boundary (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:423`). The LEADS propagation items above are recorded R1 qualifications, not new defects. In particular, the old clock/voltage follow-up remains at `docs/LEADS.md:432` despite the revised screen; align that status with T4's exclusions rather than treating it as an active voltage experiment.

## Execution and availability record

Executed after `source ~/venvs/torch/bin/activate`, with bytecode writing disabled:

- `/tmp/X-RIG-GPU-FAULT_R2_beta/checks.py`; results in `results.json` in that temporary directory. Recorder and embedded sampler compiled in memory only. Neither executed.
- Actual cleanup and `_get_kernel` functions extracted by AST, with fake torch/CuPy/GC/kernel construction. Verified conditional sync-before-free call order and per-family cache reuse. No real GPU library was imported or device opened by these checks.
- Recounted saved test4, parsed all three capability captures and all three CT location lists, and recalculated the independent-binomial 0/8 and 0/4 upper bounds. All assertions passed. These are local control-flow/artifact checks, not live recorder or probe tests.

No existing recorder/seam-specific suite was found under `tests/`. Real stop behavior, treatment fidelity and fault reproduction are UNAVAILABLE in this design-only review. This round accepts no counted-run result and claims no root cause.

Reply validation: the existing relay positions parser accepted all twelve rows and both LIVE severity cells. All 40 fully specified file:line anchors resolve within their files. The four submitted hashes match R2 at initial and final checks; no concurrent change to those files was observed. Beta wrote only this review in the repository; temporary checks/results are under `/tmp`.
