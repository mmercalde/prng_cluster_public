## Prior record

**Searches run in this session, 2026-11-15:**

- Read the request and repository `AGENTS.md`; ran `git status --porcelain`, `git rev-parse HEAD`, `git diff -- docs/LEADS.md`, and SHA-256 checks of all four submitted files. All four hashes match the request.
- Ran `rg -n 'GPU|rig|power|netconsole|review|fault|atomic|probe|cycle|cache'` over `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*_AUDIT.md`, and `docs/TODO_CONSOLIDATED.md`; then narrower searches for `ROCR_VISIBLE|gpu_index|KFD|HIP_VISIBLE|skip_range|stationar|consecutive|netconsole.*arm|AtomicOps|riser` over that governance set and `docs/exchange/`.
- Searched all `docs/` for `vmbr0.*(exist|abort)|netconsole.{0,50}(fail|boot|persist)` and searched the supplied captures for `Birth|19:25|19:51|location_id|AtomicOps|ReqEn|HSA|netconsole|powercap`. Read the applicable hits and the April incident/session records, August forensic report, LEADS, netconsole runbook, completion plan and delegated-execution plan.
- Read the recorder and the production launch, payload-builder, executor, hybrid argument builder, cleanup and kernel-cache paths. Queried the three specified ledger copies using `mode=ro&immutable=1`. Read the saved recorder JSONL tests without starting the recorder. Checked the linked vendor/operator pages and relevant upstream Linux/Python primary documentation.

| Governing record / item | Label and disposition |
|---|---|
| `AGENTS.md`; `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:19` | ALREADY RECORDED: Beta reviews, Michael decides; power hardening is not a Run-4 prerequisite. |
| `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:65`, `:181`, `:203` | **ALREADY RECORDED governing constraints, omitted from the submitted prior record. Finding 1 below.** Owner-run diagnosis does not automatically authorize delegated reproduction or host writes. |
| S162 isolation/signature; S163KARG; S164/S165/S170; August forensic report | ALREADY RECORDED: recurrence, isolation limitations, rejected Kaspa setting and historical mitigations. Their historical causal conclusions are not independently re-established here. |
| `docs/exchange/from_beta/X-VOLUME-LIMIT_R1_BETA_REVIEW.md:79`; `docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md:360` | ALREADY RECORDED: production payload omits `skip_range`; distinguish trial metadata from effective kernel inputs. |
| LEADS L-3 and report correction | CHANGES A RULING/recorded position, as proposed by Alpha; no owner ruling is changed by this review. Q2 leaves incident arming unestablished. |
| Boot-time netconsole failure in today's supplied journals | NEW to the searched record: persistent configuration did not successfully arm the two failed hosts at boot. Q4. |
| Probe inference, experiment attribution and stopping gaps | NEW review observations about this proposed plan. Q6, Q7, Q9. |

**Not searched / not executed:** no rigs, CTs, Proxmox hosts, Zeus, private repository, ser8 archives, switch/PDU records or Ubuntu boot targets were accessed. No pipeline, worker, GPU initialization, recorder main program, installation, commit or push was run. Raw August stat/journal-query output, CT KFD/env captures and full atomics captures were not found in the evidence directories supplied; their live verification is UNAVAILABLE. This review neither proves a root cause nor accepts a counted run.

# X-RIG-GPU-FAULT — R1 — BETA REVIEW

- Date: 2026-11-15, America/Los_Angeles.
- Base read: `3579088b8e9c2d902252d626baae15bf0647907f`.
- Scope: the submitted uncommitted report, LEADS appendix, recorder and design-only T4 plan.
- `git status --porcelain` at initial read:

```text
 M docs/LEADS.md
?? docs/CLAUDE_CODE_INSTRUCTIONS_RIG_GPU_FAULT_20261115.md
?? docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md
?? docs/exchange/to_beta/X-RIG-GPU-FAULT_R1_ALPHA_REVIEW_REQUEST_fault_evidence_and_plan.md
?? scripts/rig_gpu_recorder.py
```

| Question | Position | Severity |
|---|---|---|
| Q1 | NEEDS EVIDENCE | — |
| Q2 | NEEDS EVIDENCE | — |
| Q3 | NEEDS EVIDENCE | — |
| Q4 | OBJECTION | LIVE — Today's retained boot journals contradict the claim of working netconsole persistence on the failed hosts. |
| Q5 | SUGGESTION | — |
| Q6 | OBJECTION | LIVE — The proposed stopping rule and probe non-reproduction rule can produce unsupported conclusions on this fleet. |
| Q7 | OBJECTION | LIVE — Several confirm/refute rules assign mechanisms that the proposed interventions do not isolate. |
| Q8 | NEEDS EVIDENCE | — |
| Q9 | OBJECTION | LIVE — The proposed probe has no specified external deadline or fleet-wide cancellation of later launches after a fault. |

These are Beta's review positions. Q4 concerns correction of an after-commit-class document; Q6/Q7/Q9 concern the design before execution. They do not turn Beta into a launch or commit approver. Michael decides any disagreement with Alpha.

## Q1 — Evidence

**Supported, within these limits:**

- The retained netconsole lines show `.121` `19:00.0` TCP `0x00841011`, followed by `.163` `16:00.0` SQC `0x00801231`, with the stated permissions and later all-ones values (`/home/michael/run4_attempt2_20261115_131154/logs/netconsole_all_rigs.log:359`, `:370`, `:384`, `:394`, `:404`). The first callback-suppression packets were **0.631 ms apart at the receiver**. That does not measure host-to-host fault simultaneity to sub-millisecond accuracy.
- The ledger main-file view has phase-4 creation time `1794777688.9897048`. Its interval to the first `.121` receipt is **72.997 ms**; to `.163` it is approximately **73.628 ms**. Thus “about 73–74 ms from stripe creation to first received fault output” is supported. **Dispatch-to-fault or kernel-launch-to-fault is not measured.** The report partly makes this distinction at `docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:158`, but the question and first-launch descriptions should keep it throughout. Equal StripeAssign arrival times on both hosts are not established by the coordinator timestamp (`:172`).
- The same main-file view has **60 phase-3 survivors** and **no phase-4 shard rows** for `rrig6600:*` or `rrig6600c:*`. That supports “zero recorded/received shards,” not proof that neither GPU ever completed or attempted a sub-stripe. The copy has a **4,132,392-byte WAL**; immutable queries ignore it. I verified the main-file facts, not WAL-inclusive completeness. Archive/WAL provenance is an already-recorded issue (`docs/exchange/from_beta/X-0001_R1_BETA_REVIEW.md:72`).
- April recurrence is supported as a historical report: S162's first signature and isolation table (`docs/PROPOSAL_S162_RRIG6600C_CRASH_ROOT_CAUSE_v1_0.md:18`, `:44`); S163KARG's three BDFs (`docs/TB_Incident_Report_rrig6600c_S163KARG.md:85`); S164's different preemption/reset event (`docs/SESSION_CHANGELOG_20260419_S164.md:30`). I did not inspect the original April captures.
- Today's **BDF → DRM → VBIOS** association is directly supported, including `.163` DRM card0 = `06:00.0` = L03 (`logs/rig_gpu_fault_20261115/host_192.168.3.163.txt:231`). The production launch selects `ROCR_VISIBLE_DEVICES=N` and device index zero (`scripts/launch_fleet_manual.sh:316`). However, the advertised CT KFD `location_id` capture is absent here. **ROCr ordinal → BDF remains provisional**, as the report correctly acknowledges at `:207`; PID launch order is consistency evidence, not an independent device identification. Keep that qualification on `gpu1/gpu6` experiment instructions too, and verify the visible device's BDF inside the eventual probe before its load.

Two other statements exceed their evidence: phase-3 power exceeding phase-4 power **on every rig** cannot follow when the failed rigs' phase-4 power is missing (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:195`, `:199`); the SATA BadCRC/link/I/O symptoms do not uniquely identify “a cable or port fault” (`:297`). Retain the symptoms and candidate explanation, without that definitive diagnosis.

## Q2 — L-3 correction

**Reopen the old closure; do not replace it with proven “NOT armed.”** The recorded proof packets belong to the later boot, as the existing LEADS timeline already says (`docs/LEADS.md:229`, `:245`). They cannot prove pre-freeze arming. That correction is justified independently of a claim that arming was impossible.

File birth times date the present persistence files, not all possible prior arming. Direct `modprobe netconsole netconsole=...` needs neither file; it is already described in `docs/RUNBOOK_NETCONSOLE_REARM.md:72`. The [Linux netconsole documentation](https://docs.kernel.org/networking/netconsole.html) also describes module-parameter and runtime configfs configuration.

These routes normally produce kernel initialization messages, but the incident boot's retained journal ends **before** the 18:32 failure: `.163` at 18:19 (`logs/rig_gpu_fault_20261115/host_192.168.3.163.txt:216`), `.121` at 18:27 as reported (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:48`). An ad hoc load in the missing tail can leave neither of the two cited persistent files nor a retained journal line. This is an explanation of the evidence gap, not a claim that such a load occurred. The raw stat output and the `grep -c` query results themselves were not supplied in the nine retained capture files.

Suggested disposition: **“No pre-freeze arming proof retained; persistent configuration reportedly created after the freeze; incident kernel-event status UNOBSERVED. The later NCPROOF packets cannot close the not-active branch.”** Today's GCVM event is compatible with the August hang; a matching August signature remains unobserved. Preserve Alpha's stronger position alongside this one for Michael.

## Q3 — Null results 1–3

1. **Identical recorded trial inputs: verified. Identical entire configuration: not verified.** Both trial-context rows match W12/anchor25/phase0/midday/6–99/0.71/0.47 and dataset/residue digests. The previous-boot journals support the shared morning/afternoon boot. There are no per-run effective host-setting or process-environment snapshots. State “same recorded trial inputs, same host boot, different outcomes”; “no static setting explains it by itself” needs the extra assumption that the relevant setting did not change within that boot (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:30`).
2. **Large phase-3 survivor volume is not necessary for the observed phase-4 host hang:** supported by today's 60 and August's documented 13,146,485 (`docs/RIG_CRASH_FORENSIC_20260822.md:304`). This does not establish that the two hangs have the same cause, or eliminate volume as an August contributing factor. Avoid the broader phrase “volume is not necessary” without naming “large phase-3 survivor volume.”
3. **Power-cap non-prevention is plausible but missing event-time treatment evidence.** The retained reads show the installed service/script and post-reboot limits (`logs/rig_gpu_fault_20261115/host_192.168.3.163.txt:178`, `:247`). The script suppresses write errors and logs completion unconditionally (`:199`, `:201`). File installation time alone does not establish successful application before the fault. Provide existing pre-fault cap readbacks or a retained prior-boot application/verification record; otherwise say “recurrence despite configured cap mitigation; effective cap at fault UNOBSERVED.” Even with that evidence, the conclusion concerns this deployed cap, not every cap or the transient-sag hypothesis.

## Q4 — Configuration comparison

**LIVE objection: configuration persistence is being described as successful boot arming.** The report calls netconsole persistent and the TODO stale (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:60`, `:269`). Yet the actual incident-boot kernel journals record:

- `.121`: `vmbr0 doesn't exist, aborting`, then cleanup at 08:27:40 (`/home/michael/run4_attempt2_20261115_131154/hostlogs/journal_kernel_prevboot_192.168.3.121.txt:2104`).
- `.163`: the same failure at 08:27:25 (`/home/michael/run4_attempt2_20261115_131154/hostlogs/journal_kernel_prevboot_192.168.3.163.txt:2102`).
- Later successful arming at 08:56:21 and NCPROOF are retained for both (`.../journal_kernel_prevboot_192.168.3.121.txt:2406`; `.../journal_kernel_prevboot_192.168.3.163.txt:2314`).

The inference is an ordering problem with early module loading and bridge availability. The measured fact is **boot arming failed and later arming succeeded**. No repair was attempted. Change the comparison to “persistence files present; boot arming failed on these two hosts; later session proof succeeded.” The plan's pre-session proof requirement remains appropriate (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:417`). A configured file is not a substitute for proof after the next boot.

The S164 → S165 → S170 progression is useful and correctly withdraws the Kaspa setting (`docs/SESSION_CHANGELOG_20260420_S165.md:22`, `:36`; `docs/SESSION_CHANGELOG_20260425_S170.md:274`). The upstream-port link comparison is supported by the retained chain, and correctly distinguishes the internal 16-GT/s link from the riser path (`logs/rig_gpu_fault_20261115/host_192.168.3.163.txt:478`).

**HSA row:** the explicit non-observation of `/proc/PID/environ` is right. The categorical item “Miner workers start with no HSA_* variables” in the report's labels table is stronger than that row (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:117`, `:274`). Source establishes that the launch does not explicitly set them (`scripts/launch_fleet_manual.sh:314`); it does not exclude inherited/PAM/SSH environment. The advertised CT shell/env captures are absent here. Use “not explicitly set by miner launcher; simulated shell reportedly UNSET; live inheritance UNOBSERVED.” Also correct the PWC anchor to `persistent_worker_coordinator.py:140`, `:141`, **`:145`**; lines 139–141 do not include DEBUG_TRAP.

The comparison should retain unknowns for the older display-stack state versus today's other GPU users, and exact per-card firmware equivalence. The April record explicitly discusses GDM/Xorg (`docs/TB_Incident_Report_rrig6600c_S163KARG.md:233`); the retained `GPU_PROCS` capture is only a post-reboot Python-process listing (`logs/rig_gpu_fault_20261115/host_192.168.3.163.txt:562`), not event-time evidence excluding every GPU client. No additional host reads are requested from Beta.

## Q5 — Recorder

**No objection to its core observation behavior on the submitted evidence.** The sampler's explicit file open is read-only (`scripts/rig_gpu_recorder.py:73`); conversions return null on unreadable fields (`:102`); it does not open a GPU device. The local deterministic check of the actual watchdog function emitted only UNOBSERVED with `cards:null` for a never-answering fake host (`:273`). The retained test1 records independently recount as 45 OBSERVED, 13 UNOBSERVED for the unreachable address and 44 telemetry datagrams. Test2 has five silence records, SIGTERM and a final stopped record. These validate the saved artifacts, not a new fleet execution.

The watchdog does not fabricate observed samples or zeros. **OBSERVED means a received historical sample**, including a delayed one; it must not be read as current host liveness. The source records host time and receipt lag (`scripts/rig_gpu_recorder.py:231`). UDP data are a separate raw channel; its upstream producer substitutes zeros on read errors (`logs/rig_gpu_fault_20261115/host2_192.168.3.163.txt:36`), so do not use those values as independent healthy measurements.

Streaming writes avoid retention proportional to run duration (`scripts/rig_gpu_recorder.py:167`); this is bounded in normal operation on the known hosts. It is not a universal input-size bound: stdout iteration and `stderr.read()` consume entire incoming lines/remaining stderr (`:220`, `:250`). No hostile-input condition was established, so this is no LIVE objection.

PID-based stopping is reasonable for one instance, and the normal shutdown path terminates sessions, joins and flushes (`scripts/rig_gpu_recorder.py:382`). **Suggestions:** refuse a second recorder using the same PID path (`:352` overwrites it); check whether threads/SSH processes actually stopped before reporting completion (`:386` uses a timed join); retain the distinction between normal-stop evidence and a hung/stopped local SSH process. Live cleanup and remote sampler disappearance were not independently tested here.

Host derivation is acceptable for this fleet: endpoint-minus-one is documented in `CLAUDE.md:147` and `rig_profiles_config.json:37`. Calling only the pure derivation function returned `.121/.155/.163` with the correct labels (`scripts/rig_gpu_recorder.py:323`); no connection was made.

**Correct the `-I` claim.** Isolation does not disable bytecode writes, contrary to `scripts/rig_gpu_recorder.py:64` and the report at `:324`. A local temporary-module check with `/usr/bin/python3 -I -u -` produced a `.pyc` and `sys.dont_write_bytecode=False`. [Python documents `-B` for that purpose](https://docs.python.org/3/using/cmdline.html#cmdoption-B). Use explicit `-B` if guaranteeing no bytecode writes is required. This is a THEORETICAL write risk for the existing stdlib-only import set unless a missing/stale writable cache is established; it is not evidence that Alpha wrote a host bytecode file.

## Q6 — Probe, base rate and escalation

**Keep the single-rig non-reproduction rule and S1 → S2 → S3 escalation.** S162's observation is sufficient reason to avoid calling an isolated clean series a fleet pass; it is not a present-day guarantee that isolated runs cannot fault (`docs/PROPOSAL_S162_RRIG6600C_CRASH_ROOT_CAUSE_v1_0.md:21`).

The six phase-4 starts and the four 32/32 versus two 24/32 outcomes are reproducible from the three ledger main-file views. Thus **2/6 is a descriptive historical proportion**. Stationarity/transfer to a probe is disclosed, but the calculation also requires independent comparable cycles and an appropriate failure endpoint. Reboots, process lifetime, cache state and differing historical geometries make those assumptions substantive (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:374`).

**LIVE objection to calling 0/8 “prevents at ~5%.”** At a fixed, known independent baseline probability of 1/3, `(2/3)^8 = 0.039018` is correct. It is not a confidence that the arm prevents failure. Zero failures in eight independent trials leaves a one-sided 95% binomial upper failure probability of **31.23%**. Comparing an observed 2/6 baseline with 0/8 treatment gives a one-sided Fisher probability of **0.1648**, not <0.05. Screening many arms also changes the false-discovery interpretation. Record a fixed attempted-cycle budget, every failure and uncertainty; use “non-reproduction in N cycles” or a bounded rate-reduction claim. Do not retry until an eight-clean streak and discard earlier failures. These are reliability calculations, not TFM predictive-hit-rate measurements.

**LIVE objection to S2's inference.** Zero in eight cycles cannot establish “the probe lacks something production has” (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:431`). It can motivate S3, but remains non-reproduction. Even with perfect fidelity there is a nonzero chance of zero faults; with imperfect fidelity the missing factor need not be coordinator/network/staging.

Calling the real executor is a strong starting point (`miner/range_miner_worker.py:1125`, `:1268`), but **matching survivor counts does not establish seam fidelity**. The proposed fresh process per cycle omits production's prior phases and, for the afternoon failure, a whole earlier trial in the same persistent worker. Production retains kernel objects by family (`sieve_gpu_worker.py:124`); a warm disk cache in a new process does not recreate an already loaded reverse-hybrid module. This directly matters to arm g (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:397`, `:410`, `:488`). Preserve or explicitly vary that process/module history, and record actual allocations, module load, launch and completion times. Match assigned seed ranges to worker identities rather than redistributing ranges after a holdout or swap.

**Already-recorded input distinction:** trial skip 6–99 is metadata, not a delivered `skip_range`; the payload builder omits it (`miner/range_miner_coordinator.py:9331`, `:9358`), while the executor default is 0–16 (`miner/range_miner_worker.py:1164`). For the hybrid seam the prefix uses strategy arrays, not those two skip scalars (`:276`, `:1221`). Do not claim this default by itself changes the hybrid probe. Construct the real phase-specific assignment, with exact residues/digests, thresholds and strategies, instead of treating a `trial_context` row as the payload. Counts are a useful additional control; argument and runtime-history identity are separate controls. The eventual implementation still needs its own review.

## Q7 — Candidate discrimination

**LIVE objection concerns the causal verdicts, not the value of trying these interventions.** “Confirms” should generally become “supports / rate reduction under this configuration.” A fault refutes absolute prevention; it does not usually refute a contributing mechanism. Apply Q6's uncertainty and treatment checks to every row (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:481`).

| Arm | What it can discriminate; required limit |
|---|---|
| b1 | Tests first-launch spacing. Log actual launch/allocation/module timing on all cards. No neighboring launch within ±100 ms refutes a trigger requiring that exact coincidence; it does **not** refute every shared-current-step mechanism. Earlier cards can still be active, and supply recovery time is not measured. The proposed timing claim at report `:483` is too broad. |
| b2 | Tests separation of rig starts, while their 25-second workloads still overlap. A later rig starting alone can still share a loaded circuit/PSU/network resource. Fault under b2 does not exclude cross-rig resource involvement (`:484`). |
| d1/d2 | Sensible suspect-pair versus other-pair holdout. Equal GPU count is not equal electrical load or identical assignment history. Record active BDFs, fixed per-card workload and measured package load; compare both rates against contemporary A0. On S2, `.121` can still fault independently: define whether the endpoint is `.163` or any host (`:485`). |
| d3 | A fault with only two active cards shows full eight-card rig load is unnecessary. It does not prove a card/slot cause or independence from all rig load. Other rigs' activity must be stated. Zero is correctly described as ambiguous (`:487`). |
| g | Separates disk cache states **only if process/module history is held fixed**. Define what is cold, what is reused and whether reverse-hybrid was previously loaded. Fresh directory plus fresh process bundles two histories if A0 is persistent production. Preserve per-GPU cache isolation (`scripts/launch_fleet_manual.sh:317`; `sieve_gpu_worker.py:124`). |
| c1 | A four-variable environment **bundle**, not one factor. Suitable as a mitigation-package screen; a success does not identify which variable matters. Verify values before GPU imports, and label whether each variable is honored by this runtime (`:489`). The historical negative is appropriately qualified. |
| c2 | Appropriate separate ISA-override arm. Verify the resulting device/ISA target and cache identity. A success implicates the override's combined runtime/code-object effects, not one uniquely identified defect (`:490`). |
| a1 | Tests `high` versus auto with clock/voltage/temperature/power observations. Peak SCLK alone does not establish that every voltage/memory/power transition disappeared. One fault refutes absolute prevention; zero does not establish DVFS ramp as necessary (`:491`). |
| a2 | Manual level, SCLK selection and COMPUTE profile are a **bundle**. It may screen the S170 package, but requires separate follow-ups for single-factor attribution. Capture and restore all changed values before the next A0 (`:492`). |
| e1 | Useful card-versus-retained-path crossover when physical labels, auxiliary power connections, workloads and all other settings are recorded. Repeat migration and swap-back evidence is stronger than one fault landing on a new BDF (`:493`). |
| e2 | Useful after e1 establishes a path effect, or with a suitable crossed design. With a suspect card held fixed, a fault staying at its slot still permits **that card** as the cause. “Stays with slot → slot/root port” is otherwise invalid. Riser+cable remains a bundle; riser power/connector and root port are not separated (`:494`). Restore e1 before treating e2 as a single difference from A0. |
| e3 | Tests a power-feed intervention. A lower observed rate supports that intervention; unchanged finite data do not exclude power. Record connector, lead sharing, PSU and all retained paths (`:495`). |
| e4 | Tests Gen1 configuration. Slower links can alter timing/throughput/power as well as signal margin. Improvement does not uniquely demonstrate link-layer margin; recurrence does not eliminate it as a contributor (`:496`). |
| f1 | As the report partly recognizes, tests **CPU-slot/path class**, combining atomics, routing and physical path changes. It cannot isolate atomics. A fault on the CPU slot excludes lack of atomics as a necessary condition for that particular reproduced event only if requester enablement and the new path were checked (`:497`). |

The cheap software-first order is reasonable as a **screening order** (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:510`). I would retain b first for cost, but withdraw the claim that a b fault decisively settles H2. Use contemporary/interleaved A0 controls rather than rerunning A0 only when a recorder session ends (`:478`). Hold boot/cache/process history fixed within a comparison, or explicitly stratify it. Hardware arms should remain available after software non-reproduction too; a d1 improvement is neither necessary nor sufficient to justify physical diagnosis. No new stress geometry or voltage-offset arm is needed for this review.

## Q8 — External claims and atomics

The accessible sources support most of the restrained wording in T4.3:

- [HiveOS thread 52441](https://hiveon.com/forum/t/6600xt-detected-dead/52441) does contain a user's unsuccessful all-riser replacement. [Thread 68383 page 2](https://hiveon.com/forum/t/6600-6600xt-gpu-driver-error-no-temps/68383?page=2) supports default-setting staging, riser advice and the riser-power question. Advice is not a verified successful riser repair; change the report's “riser swap reported” status if it is intended to mean resolution (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:441`). I could not retrieve every quoted post, including the exact #34 dead-BDF and later #46/#48 claims: those exact attributions remain UNAVAILABLE to this review.
- [Thread 54378 page 2](https://hiveon.com/forum/t/gpu-driver-errors-and-6600xt/54378?page=2) supports the one-user Gen1 report. It also contains unsuccessful cable/PSU/riser changes; the source is not a controlled test.
- The [AMD Radeon mGPU page](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-6.4.2/docs/install/native_linux/mgpu.html) supports the quoted lane-width/CPU-slot/atomics guidance and graphics-environment qualification on concurrent workloads. The report appropriately does not establish applicability of the concurrency limit to this headless one-process-per-card fleet.
- The [Ray post](https://discuss.ray.io/t/mitigating-simultaneous-gpu-voltage-spikes-during-initialization/22227) is a single NVIDIA operator's request/workaround, as disclosed. It supplies a candidate explanation, not a measurement of these AMD rigs.

**Atomics interpretation:** `AtomicOpsCap` completer bits, bridge routing support and endpoint `ReqEn` are different facts. Root-port negative completer bits are relevant to GPU-to-root atomic requests, but are not a measurement of every possible GPU atomic operation. The [Linux PCI implementation](https://github.com/torvalds/linux/blob/v6.8/drivers/pci/pci.c#L3674) checks switch routing, root completion capabilities and egress blocking before enabling requester operations. The [AMD explanation](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/conceptual/pcie-atomics.html) likewise distinguishes these roles. Say “PCIe AtomicOp requests to the root are not enabled,” rather than implying no on-device atomics.

The supplied previous-boot journals verify the “PCIE atomic ops is not supported” messages on seven GPUs of `.121/.163` (for example `/home/michael/run4_attempt2_20261115_131154/hostlogs/journal_kernel_prevboot_192.168.3.163.txt:1413`, `:1935`). However, the retained `lspci_*.txt` files contain a device list/tree, and `host2_*.txt` contains filtered AER status: neither retains the advertised `AtomicOpsCap/AtomicOpsCtl` lines. Full three-host capability/control measurements remain **NEEDS EVIDENCE**, not independently verified 21/24.

There is also a useful primary-source lead for the fallback question: [upstream KFD v6.8](https://github.com/torvalds/linux/blob/v6.8/drivers/gpu/drm/amd/amdkfd/kfd_device.c#L185) sets a no-PCI-atomics firmware threshold of 92 for applicable gfx10.3 devices; [its admission check](https://github.com/torvalds/linux/blob/v6.8/drivers/gpu/drm/amd/amdkfd/kfd_device.c#L633) permits sufficiently new firmware rather than universally rejecting such devices. The retained MEC version is `0x7a` = 122 (`logs/rig_gpu_fault_20261115/host_192.168.3.163.txt:522`). **Inference only:** this makes continued KFD operation without PCIe atomics unsurprising. This upstream source is not the installed 6.12.12 DKMS source; its exact implementation and this fleet's runtime fallback mechanism remain UNVERIFIED. No conclusion that absent atomics caused the hang follows.

## Q9 — Hang risk and stopping

The list covers the observed hang, reset/reinitialization history, sysfs-read uncertainty, loss of evidence and powered-off hardware changes (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:517`). Keeping the known bad Kaspa profile and arbitrary voltage offsets excluded is sound (`:504`). Historical audio involvement is a plausible terminal-path concern; today's inaccessible audio function alone does not demonstrate that audio caused or is necessary for the hang (`:522`).

**LIVE design gap:** “run ~25 seconds, then exit” cannot bound a blocking kernel launch/synchronization/cleanup. The executor includes device-to-host reads and cleanup synchronization (`miner/range_miner_worker.py:1270`, `:1331`, `:1063`). The proposed no-retry process rule (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:409`) is not a fleet stop controller. In particular b2 schedules later rigs after the first rig may already have faulted.

Specify before implementation: a VM101 controller with an external wall deadline; cancellation of not-yet-issued launches on first fault/silence/probe error; a best-effort stop of remaining processes with unresolved processes reported; no automatic retry, reset, reboot or next cycle until Michael has recovered and checked the fleet. Do not assume signal delivery can recover a wedged driver. A missing observation must invalidate a clean cycle. Record the failure endpoint separately as first GCVM, bus loss and host silence so an arm that merely changes recovery is not mistaken for preventing the initial fault. These controls address the exact failure and staggered launches proposed here, without adding a new power-hardening prerequisite.

## Findings outside the numbered questions

1. **Missed governing document: delegated execution.** The request/report prior record cites the completion plan but omits `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md`. It explicitly excludes delegated freeze-reproduction classes (`:24`, `:203`), prohibits delegated environment/system changes (`:69`), and ends a session on a silent host or kernel event (`:183`). The report does say recovery and every run are Michael's (`docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md:389`); I read that as an owner-operated diagnostic plan. Add the governing record and keep that operator boundary explicit. A reviewed probe or an existing RUN4 authorization does not itself extend delegated authority to these experiments. This is a prior-record omission, not a demand for another permission step before writing or reviewing the probe.

## Execution and availability record

Executed after activating `~/venvs/torch/bin/activate`, with repository bytecode writes disabled:

- Temporary checks at `/tmp/X-RIG-GPU-FAULT_R1_beta/checks.py`; output `checks.log`, structured results `results.json` in that directory. The first invocation failed because the extraction harness omitted an annotation dependency; corrected in the temporary harness, then the complete check passed.
- Compiled recorder and embedded sampler source in memory; neither main program nor sampler ran. Executed only the actual watchdog with local deterministic fake clock/streams, and the pure host-derivation function. Tested `-I` bytecode behavior on a temporary local module. No SSH/network operation occurred in these checks.
- Recounted all three saved recorder tests; queried the three named ledger copies in immutable mode; reproduced timing, contexts, survivor count, phase-4 rows and reliability calculations. Hashes of those ledger files and existing sidecars were unchanged across the check. The bundle's nonempty WAL means main-file query completeness is not certified.
- Read production code and retained evidence only. GPU fidelity, physical mapping beyond the retained observations, real stop behavior, event-time environment/caps, August arming and missing full capability captures are UNAVAILABLE for independent verification here.

Reply validation: the existing relay positions parser accepted all nine question rows and all four LIVE severity cells. All 60 explicit file:line references resolved within their files. No repository file outside `docs/exchange/from_beta/` was written by Beta. The four submitted file hashes and base HEAD were rechecked before finishing; no concurrent change to those files was observed.
