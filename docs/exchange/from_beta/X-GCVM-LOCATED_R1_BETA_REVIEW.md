## Prior record

**Searches run, 2026-11-16:**

- Read `AGENTS.md`, the request and its incorporated report's Prior record. The incorporation supplies actual searches, so the prior-record gate is satisfied. Searched `docs/` and the exchange for `GCVM|NEVER MAPPED|code.staging|HIP_FORCE_DEV_KERNARG|E28|L-6|L-7|stale TLB|fault.address|ring.buffer`, then `kernarg|HDP|InvalidateCodeCaches|vm_info|tracepoint|read.back|suballoc|fragment|checksum|continuation|common.pid`. Explicitly searched `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, `BACKLOG.md`, `PROJECT_FILE_CATALOG.md`, LEADS and the exchange.
- Read relevant passages of the S172 completion/delegated-execution plans, LEADS L-6/L-7, the decode report, runbook, RIG-GPU-FAULT R1–R3 reviews and the GPU-PROBE review. Checked the separate local prior-work register at `/home/michael/tfm_worktrees/gpu-fault-history/docs/REGISTER_GPU_FAULT_PRIOR_WORK.md:1081`, especially E28–E30, and searched it for kernarg/HDP/TLB/tracepoint/vm_info. No missed governing document was established.
- Read both analysis tools, the recorder tests, actual worker allocation/dispatch and reverse-hybrid kernel source. Recomputed the capture hashes; searched the two faulting hosts' entire streams across all tasks, independently of the mapper's PID filter; compared allocation sequences for all five submitted worker cases in A/B. Read the main checkout's retained A/B coordinator logs and source-digest manifests locally, without opening a ledger. Consulted primary upstream ROCr/ROCclr 6.4.3 sources for loader behavior and argument-placement selection, and upstream amdgpu sources for page units/trace formatting.

| Item | Label / disposition |
|---|---|
| `AGENTS.md`; `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:19`; `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:24` | ALREADY RECORDED authority and operating limits. Beta reviews; Michael decides. |
| Register E28 belief; L-6/L-7; decode/runbook loader discussion | ALREADY RECORDED hypotheses and history, not newly discovered defects. |
| Limits on causal conclusions from controls and finite clean runs | ALREADY RECORDED in RIG-GPU-FAULT R2 Q10 and R3 Q13; applied here to new proposed conclusions. |
| This capture's address/range comparisons | NEW measurements; independently corroborated below with explicit limits. |
| Mapper attribution/continuation omissions; argument-placement premise; overstrong exclusions; manifest defect | NEW observations about this submission, not a new root-cause diagnosis. |
| Owner rulings | No change proposed. |

**Not searched / not executed:** private repository, ser8, rigs/CTs/Proxmox, live process memory, installed driver/runtime binaries and live run state. No ledger opened. No remote connection, GPU initialization, worker/pipeline, recorder main/arm/stream, package operation, host change, commit or push. Exact installed-runtime behavior, GPU-visible bytes/cache state, launch packets and faulting instruction are **UNAVAILABLE** here. The historical register was searched and relevant passages read, not reread in full.

# X-GCVM-LOCATED — R1 — BETA REVIEW

- Date: 2026-11-16, America/Los_Angeles.
- Requested run base: `048fd47c92932962c4379c94b787852df8fdb7c7`.
- Worktree HEAD read: `3579088b8e9c2d902252d626baae15bf0647907f`, plus the submitted working-tree files. These are different bases. The worker and registry bytes read nevertheless match both retained run source-digest manifests: worker `2bca2a08…`, registry `32435fb8…` (`/home/michael/distributed_prng_analysis/logs/gate12_20261116_162311_source_digests.json:50`, `:54`; corresponding `164641_source_digests.json:50`, `:54`). This supports the source-level checks below; it is not whole-run certification.
- `git status --porcelain` at initial read, unchanged before writing this reply:

```text
 M docs/LEADS.md
 M docs/exchange/DEFERRED.md
 M docs/exchange/LEDGER.md
 M miner/range_miner_worker.py
 M scripts/launch_fleet_manual.sh
?? docs/CLAUDE_CODE_INSTRUCTIONS_RIG_GPU_FAULT_20261115.md
?? docs/CLAUDE_CODE_PLAN_RIG_GPU_PROBE.md
?? docs/CLAUDE_CODE_REPORT_GCVM_LOCATED_20261116.md
?? docs/CLAUDE_CODE_REPORT_MINER_NO_TORCH.md
?? docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md
?? docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT_DECODE_20261115.md
?? docs/CLAUDE_CODE_RUNBOOK_RIG_FAULT_RECORDER_AB.md
?? docs/SESSION_CHANGELOG_20261115_RIG_GPU_FAULT.md
?? docs/exchange/from_beta/X-MINER-NO-TORCH_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R3_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-PROBE_R1_BETA_REVIEW.md
?? docs/exchange/to_beta/X-GCVM-LOCATED_R1_ALPHA_REVIEW_REQUEST_fault_address_reading.md
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
?? tests/test_rig_fault_recorder.py
?? tests/test_rig_seam_probe_controller.py
```

| Question | Position | Severity |
|---|---|---|
| Q1 | SUGGESTION | — |
| Q2 | SUGGESTION | — |
| Q3 | OBJECTION | LIVE — The present evidence does not exclude host translation or visibility errors upstream of the observed wild accesses, so the stated exclusion can misdirect this diagnosis. |
| Q4 | NEEDS EVIDENCE | — |
| Q5 | OBJECTION | LIVE — Wrong n_candidates alone cannot produce the reported seed index with the stated grid, and the TCP read has not been attributed to the seeds operand. |
| Q6 | OBJECTION | LIVE — The proposed observations do not eliminate the stated hypotheses, and the device-kernarg premise is not established for these workers. |

This reviews the reading and proposed interpretations. No code is submitted for a commit decision, no counted result is accepted, and no launch is authorized by this review.

## Q1 — Attribution is usable with an explicit distinction between task and VM ownership

The report's proposed combination of task identity and explicit PTE `pid:` is reasonable. The actual mapper does **not** implement that combination: it first selects only `common_pid == fault pid` and then searches those rows (`scripts/rig_fault_map.py:110`, `:117`). Its netconsole parser also extracts the **thread** PID rather than the process PID (`:49`); the retained first-fault lines happen to have equal values (`logs/rig_fault_rec/20261116_AB_secured_165728/netconsole_all_rigs.log:1661`, `:1682`, `:1722`).

A real omitted case exists: task 211508 maps a BO for process 211495, with the PTE event explicitly naming 211495 and VM context `0x10dc` (`logs/rig_fault_rec/20261116_AB_secured_165728/stream/trace_192.168.3.163.log:257154`, `:257155`, `:257158`). I count 3,860 explicit-PID PTE records for 211495 versus 3,845 among its main-task rows. High main-thread coverage therefore does not establish complete attribution. Join thread ownership and explicit VM identity, and use BO/range associations for events without a VM field; do not generally assign a kworker event to a worker by task name.

**Effect on these verdicts:** my all-task, whole-stream searches still find no covering map/PTE event for the three data-fault pages or the second `.121` write page. The omitted rows do not reverse those observations. This is a tool/qualification suggestion, not evidence that these addresses were actually mapped. The `vm_bo_map` inclusive page ranges and 4 KiB conversion are consistent with [upstream amdgpu mapping code](https://cgit.freedesktop.org/drm-misc/commit/?id=6fef2d4c00b5b8561ad68dd2b68173f5c6af1e75); installed DKMS source was not inspected.

## Q2 — Retain the measured ranges; narrow the verdict labels

Let `E = logs/rig_fault_rec/20261116_AB_secured_165728/` for the evidence anchors below.

| Address / event | Independently checked observation |
|---|---|
| `.121` `0x77e7d5826000`, also `0x77e7d17e6000` | No covering map, unmap or PTE update anywhere in that retained host stream, any task. Raw TCP-write records: `E/netconsole_all_rigs.log:1660`, `:1670`. |
| `.163` `0x76a2cd9fb000` | Same absence across the entire retained host stream. T5 contains the anonymous CPU mapping (`E/runB/snaps/T5/192.168.3.163.txt:31029`); it does not establish GPU registration. Raw TCP-read record: `E/netconsole_all_rigs.log:1681`. |
| `.163` `0x76ab421aa000` | Same range absence, but the first printed status is already all ones (`E/netconsole_all_rigs.log:1721`). Retain the low-confidence warning; this is not a reliable first initiating-access diagnosis. |
| SQC `0x76a3e0662000` | Inside the recorded five-page R/W range starting `0x76a3e0660000`, with a covering PTE update and no later covering unmap/update in the retained stream (`E/stream/trace_192.168.3.163.log:257021`, `:257022`). Raw SQC record: `E/netconsole_all_rigs.log:1691`. |

Use **“no covering mapping/PTE event in the retained capture”** for the first three, and **“recorded R/W, non-executable mapping, with no recorded subsequent revocation”** for SQC. “NEVER” and “STILL VALID” need this scope. The mapper derives validity from the last covering map/unmap, not PTE read-back (`scripts/rig_fault_map.py:125`).

The adjacent five-page device-local executable / system-memory R/W pair is strong circumstantial support for the loader-staging identification, consistent with ROCr source, rather than a captured loader symbol/segment identity. The SQC event occurs about 49.5 ms after that card's TCP read; it need not be the initiating fault. Its location supports an instruction request into a recorded data range, not a demonstrated jump mechanism or proof of correct hardware-visible PTEs.

## Q3 — H1 is stronger than the evidence

Agree with: **“No missing/revoked mapping of a known intended buffer is demonstrated by these recorded events.”** Disagree with the general exclusion at `docs/CLAUDE_CODE_REPORT_GCVM_LOCATED_20261116.md:142`.

A host translation/visibility problem affecting a real code, kernarg, descriptor or data page can first corrupt what the GPU consumes, producing a subsequent access to a wild virtual address. The eventual fault address need not equal the upstream page whose contents/translation were wrong. That is a possible chain, not a claim it occurred here. Correct PTE bytes at one launch instant also do not establish the relevant cached translations or visibility throughout execution. Keep H1 open at that scope.

“No ring-buffer drop was recorded” is a bounded observation. The stream reader stamps lines at VM receipt and reports disconnections (`scripts/rig_fault_recorder.py:211`, `:213`, `:215`); the mapper cuts on that receipt timestamp (`scripts/rig_fault_map.py:108`, `:110`). Independent SSH/UDP delivery does not impose event ordering, and no loss marker proves nothing about an untransmitted freeze tail. The snapshots do not capture per-CPU trace loss counters (`scripts/rig_fault_recorder.py:49`, `:54`). My whole-stream range check avoids this particular receipt cutoff, but cannot recover absent events or verify PTE/TLB state. It found no evict/restore events or unsuccessful recorded KFD map completions in either faulting host's stream; this remains event evidence, not a blanket host exoneration.

## Q4 — Matching BO shapes is supported; “identical except” is not established

Creation order matches the actual source: arange, survivors, match rates, strategy IDs, skip sequences (`miner/range_miner_worker.py:1332`, `:1335`, `:1336`, `:1359`, `:1360`). For all five submitted worker cases in both runs, I reproduced the large-BO page sequence **2048, 2048, 1024, 1024, 11776**, followed by the five-page executable code allocation. Examples: `.163` A gpu6 buffer creation begins at `E/stream/trace_192.168.3.163.log:123110`, code at `:123453`; B at `:320215`, code at `:320376`. Sizes/order support the proposed buffer identities, subject to pool rounding/suballocation; trace BO identities alone are not array-pointer telemetry.

For that gpu6 worker, trace-clock time from the first large BO creation to code creation is **612.719 ms in A, 1.277 ms in B**. The analogous three faulting-worker cases show the same large timing contrast. This is measured allocation-to-code timing; actual launch/dispatch visibility is not traced. The healthy `.155` comparison also matches BO shapes but its A interval is 287.193 ms, so the report's A timing range must remain scoped to the faulting-worker cases.

The same-card/same-stripe fact is corroborated locally: `/home/michael/distributed_prng_analysis/logs/gate12_20261116_162311.log:507` associates st3_s23 with rrig6600c:gpu6; `:669` records its completion. The B log at `164641.log:599` associates that stripe with the same worker and no accepted subresults; the B snapshot identifies pid 211495 as gpu6 (`E/runB/snaps/T5/192.168.3.163.txt:30951`). This supports **“that stripe completed on gpu6 in A; gpu6 faulted in B.”** It does not prove equality of loaded instruction bytes, argument bytes, every prior allocation, queue state or device-visible dispatch. All of those remain unmeasured. Use **“same observed BO shapes and stripe identity; timing and carried state differ”**, without an exclusive list of differences.

## Q5 — The offset arithmetic is right; its operand and causal disjunction are not established

Subtracting the proposed seeds base gives offset `0xbfb000`, hence indices 1,570,304–1,570,815 if this page was reached by an eight-byte seeds load. The source guard and seeds reads are at `prng_registry.py:3179`, `:3180`, `:3185`, `:3233`; dispatch uses 256 threads and `(n_seeds + 255)//256` blocks (`miner/range_miner_worker.py:1395`).

With 3,907 blocks, **even removing the n_candidates guard entirely caps the normal thread index at 1,000,191**. A wrong n_candidates alone therefore cannot reach index 1.57 million. A wrong effective pointer, dispatch dimensions/index, executing instructions or some other erroneous access is required under that seeds-load interpretation.

More fundamentally, a TCP read at a page numerically near the seeds BO is not evidence that its operand was candidate_seeds. The kernel also reads residues/strategy arrays and has private storage (`prng_registry.py:3173`, `:3187`, `:3196`, `:3210`). The phase-4 allocation path performs CuPy operations before the sieve dispatch (`miner/range_miner_worker.py:1332`, `:1335`, `:1397`). Retain the report's **kernel identity undetermined** statement. Even after identifying a kernel, distinguishing its seeds read from another operand needs the faulting instruction/effective address or equivalent evidence. The correct present conclusion is “not an in-bounds normal seeds access if that is the operand,” not “this was a seeds overrun.”

## Q6 — Useful hypotheses; replace decisive exclusions with bounded tests

H1–H6 are a working list, not an exhaustive or mutually exclusive partition. In particular, distinguish host-produced argument/dispatch errors, copied-code corruption, stale instruction-cache contents, translation visibility, and actual device execution errors. A matching copied image does not establish that the expected image/descriptor itself is correct.

- **H1:** retain Q3's distinction. PTE read-back tests stored translations at a point, not all hardware-visible translation behavior.
- **H2:** matching C1 read-back does **not** eliminate stale instruction-cache contents. ROCr separately performs copy/flush and cache invalidation in [`RegionMemory::Freeze`, lines 328–348](https://raw.githubusercontent.com/ROCm/ROCR-Runtime/rocm-6.4.3/runtime/hsa-runtime/core/runtime/amd_loader_context.cpp). A read may also perturb the timing/visibility being tested. Compare the actual loaded/relocated segment, not an unadjusted file slice. `/proc/PID/mem` feasibility and debugfs read semantics for the installed driver remain **UNAVAILABLE**, not certified by the upstream loader. A CPU-visible mapping is not by itself proof that the proposed remote read returns the shader's instruction-cache contents.
- **H2 evidence correction:** `dst` is **not empty in the raw trace**. C1's PTE header at `E/stream/trace_192.168.3.163.log:320383` is followed at `:320384` by `{0x1fec64000,0x1fec65000,0x1fec66000,0x1fec67000}`; its final-page destination follows at `:320388`. `trace_for()` parses only event-header lines and discards those continuation lines (`scripts/rig_fault_map.py:65`). Preserve and decode these recorded destinations before declaring physical placement unavailable. They are update operands, not memory read-back; conversion to the proposed debugfs offset still requires the installed driver's address semantics.
- **H3 premise correction:** the flag's declared default `true` does not prove device kernargs. In [ROCclr 6.4.3 `rocsettings.cpp`, lines 84–88 and 219–256](https://raw.githubusercontent.com/ROCm/clr/rocm-6.4.3/rocclr/device/rocm/rocsettings.cpp), the initial implementation is host arguments; default device selection is applied to gfx94x, while the flag-based assignment is conditional on an explicit non-default setting. For gfx1032, the cited source therefore does not support the claimed default placement. The recorded worker environment has no explicit HIP_FORCE_DEV_KERNARG (`E/runB/snaps/T5/192.168.3.163.txt:32041`). Confirm the actual loaded implementation/placement before interpreting `=0` as a treatment; it may leave the baseline placement unchanged. Even a verified placement change with recurrence only excludes that placement as a necessary condition, not all wrong/stale arguments. A read-back likewise does not establish the bytes actually consumed at execution.
- **H4:** the measured BO-size/order difference is contradicted for the checked allocations; arbitrary layout/suballocation/contents differences are not all eliminated.
- **H5:** an adequately repeated crossed card/identity comparison can distinguish associations. One migrated fault does not eliminate intermittent card/path susceptibility or interactions, and moving software identity alone does not move a physical riser/slot.
- **H6:** a clean same-boot cold-cache B does not eliminate carried state. Its effect can interact with fast timing, and recurrence is probabilistic. A fault in that arm demonstrates that warm-cache loading is unnecessary for that event, without separating H5 from H6 or every form of H2/H3. The historical warm-cache counterexamples are already recorded (`docs/LEADS.md:518`). The healthy rig's similar coarse allocator statistics do not prove identical hidden state.

**Which first:** finish the offline continuation/ownership replay and establish actual kernarg placement first; neither needs a GPU experiment. For Michael/Alpha's next authorized rig comparison, prefer **same-boot cold-cache B versus the existing warm-cache B baseline**, preserving workload/history and repeating/recording attempts under the existing stop/recovery rules. It tests the observed timing contrast directly. Treat outcomes as changes in recurrence, with invalid/unobserved attempts explicit. If actual device kernargs are subsequently demonstrated, `HIP_FORCE_DEV_KERNARG=0` becomes a useful narrower second arm. Defer direct code-memory read-back until the address path and what it can exclude are specified. This is prioritization, not a request for Beta to access a rig.

## Findings outside the numbered questions

1. **NEW — evidence-manifest bookkeeping, SUGGESTION.** Of 67 listed entries, **66 match**, including all listed capture payloads. The manifest lists itself at `E/SHA256SUMS.txt:2` with hash `8ed8be6b…`, but its actual hash is `48d464a1a0450eb2b8c59a8edefab2a196cd0999f2f32d6ff66f02617554f287`. `fault_map.txt` is not listed. Record this limitation and generate a non-self-referential manifest including the derived map when Alpha next freezes evidence. This is not evidence of payload alteration and does not negate the independently recomputed ranges. No missed governing document was established ahead of this finding.

## Checks and availability

- Ran `source ~/venvs/torch/bin/activate` then `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_rig_fault_recorder.py`: **6 passed**. The idle-sample fixture exists, so its parser check was exercised. These gates test recorder behavior; they do not test mapping attribution, range validity, runtime placement or hardware visibility (`tests/test_rig_fault_recorder.py:27`, `:126`).
- Ran temporary local capture checks `/tmp/gcvm_beta_audit.py` and `/tmp/gcvm_beta_compare.py`: hashes, all-task range searches, explicit-PID versus task-PID counts, five-worker A/B allocation comparison, and source-digest comparison. No recorded map-completion failure or evict/restore event in the two inspected faulting-host streams. The non-faulting host's five-worker-comparison case was inspected for allocation shapes, not independently audited for every loss/event count.
- Read-only source/evidence review and primary-source consultation as listed above. No real runtime-placement, VRAM read-back, TLB/cache validation, dispatch capture or proposed experimental arm was run: **UNAVAILABLE**.
- No concurrent change observed to the report/tools/tests/worker/registry during this review. Final hashes are recorded below; Beta wrote only this review in the repository.

```text
67f62767f098666defed47764337a09da9dff78cae7e3c0c3150fa375e67a523  docs/CLAUDE_CODE_REPORT_GCVM_LOCATED_20261116.md
b3282c8ca0579f91fee4f22334b95fc3f834588a28152184b6cf935b9e777840  scripts/rig_fault_map.py
7face3040763faa141ed4bf8dd13fd9238d57dae40b0241fddf9ade4048dba2b  scripts/rig_fault_recorder.py
448071e7e47d053cab625bcabed95ea4a3d8bc2ae10d2084cb079de3362bf725  tests/test_rig_fault_recorder.py
2bca2a083262a523801739bc453744c92a1937e38eaa745a95eaaf13baf2ca20  miner/range_miner_worker.py
32435fb82737964dc42a2247cb296117b30a33edbf3990134457a6dc70bdc0b7  prng_registry.py
```

Reply validation: the existing relay positions parser accepted all six rows and the three LIVE severity cells; fully specified local file:line anchors were checked for existence and range.
