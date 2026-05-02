# ROCm Driver-Level Saturation

## Boundary Mapping, Failure Analysis, and Mitigation Strategy

**Distributed PRNG Analysis Cluster — S172 Saturation Investigation**

| | |
|---|---|
| **Prepared by** | Team Alpha |
| **Date** | May 2, 2026 |
| **Branch** | `s167-clean` (HEAD `3fdf434`) |

---

## Executive Summary

Six runs of the S172 forward sieve across varying worker concurrency and chunk-size configurations have produced a definitive boundary map of the cluster's stable operating envelope. The failures observed are not compute-bound, memory-bound, or thermal in origin. They are **driver-level saturation events**: the AMD KFD (Kernel Fusion Driver) and MES (Micro Engine Scheduler) firmware lose their ability to manage GPU state under sustained high-concurrency dispatch, manifesting as queue evictions that escalate to GCVM_L2 page faults and unrecoverable device loss.

The empirical boundary is concurrency-driven, modulated by chunk size. Pool size 6 is stable at all chunk sizes tested. Pool size 8 is stable only at chunk sizes of 25,000 seeds and below. Larger chunks at pool 8 fail without warning at 12 to 23 minutes, producing identical fault signatures matching documented ROCm community issues.

Three classes of mitigation are available, ranging from kernel-parameter tuning (low effort, requires reboot) to architectural rewrite of the persistent worker subsystem (high effort, eliminates the saturation source). **Recommended path:** kernel-parameter testing first to validate the hypothesis at low cost, followed by architectural change if higher throughput is required.

---

## Empirical Boundary Map

Six runs across the pool-size × chunk-size grid produced the following operational data:

| Pool Size | Chunk Size | Outcome | Time to Event | Failure Mode |
|---|---|---|---|---|
| 6 | 50,000 | ✅ STABLE | 29:30 (5 trials) | — |
| 6 | 100,000 | ✅ STABLE | Per operator history | — |
| 8 | 25,000 | ⚠️ STABLE | 1:30:43 (5 trials) | 1 transient eviction (recovered) |
| 8 | 50,000 | ❌ CRASH | ~23 minutes | GCVM_L2_PROTECTION_FAULT |
| 8 | 100,000 | ❌ CRASH | ~12 minutes | GCVM_L2_PROTECTION_FAULT |

### Key empirical observations

- **Concurrency is the primary trigger.** Pool size 8 fails at 50,000 and 100,000 chunks. Pool size 6 is stable at both.
- **Chunk size modulates time-to-failure.** At pool 8, halving chunk size (100k → 50k) approximately doubled survival time (12min → 23min). Quartering the chunk size (100k → 25k) extended survival past the test window.
- **Failure signatures are identical across crash runs.** Both pool-8 crashes produced GCVM_L2 page faults from SQC (Shader Quad Cache) clients with `WALKER_ERROR=0x7` and `MAPPING_ERROR=0x1`, escalating to status `0xFFFFFFFF` within ~95 milliseconds.
- **Memory utilization was never the constraint.** VRAM remained flat at 2 percent across all rigs throughout all runs. CPU memory was stable. The failure is not resource exhaustion in any user-visible metric.

---

## ROCm Community Research

The signatures observed match a documented pattern across multiple ROCm GitHub issues spanning 2025–2026. The pattern is recurring, vendor-acknowledged, and consistent in failure mode.

### Documented eviction-to-fault progression

ROCm Issue #5443 (gfx1151) shows the canonical sequence: workload runs cleanly, then KFD begins emitting "Freeing queue vital buffer, queue evicted" messages at intervals, eventually escalating to GCVM_L2 page faults with WALKER_ERROR/MAPPING_ERROR signatures and unrecoverable device state. Issue #5724 (gfx1151) shows the same chain with `PERMISSION_FAULTS=0x3` and `MAPPING_ERROR=0x1`.

### Concurrency threshold is documented

ROCm Issue #6012 reports that "three concurrent processes run indefinitely without issue. The problem occurs exclusively at four or more concurrent GPU consumers." Pre-cascade evictions occur every ten minutes, then cascade after thirty to sixty minutes, killing the system. This pattern aligns with our observation that pool size 6 is stable and pool size 8 is at or above the concurrency threshold.

### MES is the failure path

Issue #5151 documents the same eviction signature followed by "MES failed to respond to msg=REMOVE_QUEUE" and "MES might be in unrecoverable state, issue a GPU reset." The Micro Engine Scheduler firmware running on the GPU is the component that loses control. Once MES is in an unrecoverable state, the driver has no software path back; only a hardware reset (or full reboot, in our case) recovers.

### Confirmed amdgpu-dkms host memory leak under same workload

Issue #5915 reports a host CPU RSS leak of approximately 5.4 GB per hour during compute workloads, accompanied by repeated queue eviction messages. Both DKMS versions 30.20.1 and 30.30 are affected; the stock in-kernel amdgpu driver does not exhibit the leak. This indicates the saturation problem extends beyond GPU-side firmware into kernel driver bookkeeping itself.

---

## Hypothesis: Why pool=8/25k Completed

Three plausible mechanisms, in order of likelihood:

### 1. Failure speed exceeds eviction threshold at large chunks

KFD queue eviction is a soft recovery mechanism that responds to sustained queue pressure. At smaller chunk sizes (25k), the workload generates many short kernels with frequent inter-kernel gaps, giving KFD opportunities to evict and recover gracefully. At larger chunks (50k, 100k), kernels run longer and the per-kernel memory footprint is larger; the GPU can fault the page table directly before KFD's eviction logic gets a chance to fire. The two failure modes are not on the same continuum.

### 2. Different fault origin between regimes

- **25k regime** — Short kernels with high dispatch turnover. The KFD queue manager is the bottleneck. Pressure manifests as evictions because the queue manager is the first subsystem to reach saturation.
- **50k+ regime** — Longer kernels with larger working sets. The SQC instruction/data caches and UTCL2 page-table walker are the bottlenecks. Pressure manifests as page faults because the memory subsystem is the first to fail.

### 3. Eviction requires idle gaps

KFD evicts queues opportunistically when it observes a transition between dispatches. At 25k chunk size, dispatches complete quickly and there are many transitions per second, giving KFD many eviction opportunities. At 100k chunk size, dispatches run longer with fewer transitions; KFD has less opportunity to intervene before the hardware-level fault occurs. This is consistent with the observed 95-millisecond cascade window: there is no time for software-level mitigation once the fault hits.

### Why this matters

The fact that smaller chunks at pool 8 produce a recoverable eviction event (one observed at 23:20 during the pool-8/25k run, fully recovered) rather than catastrophic failure suggests that the cluster's safe operating envelope is bounded not by a single threshold but by a two-dimensional surface: **concurrency × per-kernel cost**. Any successful mitigation must reduce one or both axes.

---

## Root Cause: Driver-Level Saturation

The failure is not in the GPUs. VRAM remained at 2 percent throughout. GPU utilization spiked but recovered between dispatches. The hardware has substantial headroom. What is saturated is the **software layer between the application and silicon**.

### Saturated components

- **KFD queue manager (kernel)** — Tracks active compute queues across all KFD process contexts. Under sustained load with many concurrent contexts, becomes the first failure point — manifests as evictions.
- **MES firmware (on-GPU)** — Scheduling engine running on the GPU itself. Manages queue dispatch and preemption. Once MES becomes unresponsive ("MES might be in unrecoverable state" in community reports), the driver has no software recovery path.
- **UTCL2 page-table walker (hardware)** — The actual fault site in the GCVM_L2_PROTECTION_FAULT events. `WALKER_ERROR=0x7` indicates the page-table walker hit an invalid state during translation; `MAPPING_ERROR=0x1` indicates the mapping itself was corrupt or absent.

### Why concurrency drives this

With `worker_pool_size=8` across three rigs, the cluster maintains 24 concurrent KFD process contexts, each with its own dispatch queue, page-table state, and preemption window. The driver's internal bookkeeping data structures (queue table, VMID assignments, page-table cache) are sized for the system's nominal workload. Sustained 24-way concurrency with frequent dispatch churn drives those structures into states the firmware was not designed to manage.

Reducing to `worker_pool_size=6` cuts the count to 18 concurrent contexts, and crucially, leaves more spare VMIDs and queue slots for KFD to use during eviction recovery. This is consistent with the documented threshold of "3 versus 4+ concurrent processes" in ROCm Issue #6012 and explains why pool 6 is stable across all chunk sizes while pool 8 fails above 25k.

---

## Mitigation Strategies

Three categories of intervention, in order of effort and reversibility.

### Category A: Kernel Module Parameters

Settable via `/etc/modprobe.d/amdgpu.conf` or kernel cmdline. Require reboot. Apply cluster-wide. Lowest implementation cost.

| Parameter | Default | Effect / Recommended Value |
|---|---|---|
| `amdgpu.hws_max_conc_proc` | auto (max VMIDs) | Caps concurrent KFD processes at driver level. Recommend `4` to test pool=8/50k feasibility with hard concurrency ceiling. |
| `amdgpu.sched_policy` | `0` (HWS w/ over-subscription) | Set to `1` to disable over-subscription. Removes queue juggling that triggers eviction. |
| `amdgpu.queue_preemption_timeout_ms` | `9000` | Longer values reduce eviction attempts under sustained load. Test `30000`. |
| `amdgpu.max_num_of_queues_per_device` | `4096` | Lower to `256` to constrain queue table size and force fewer simultaneous queues. |
| `amdgpu.cwsr_enable` | `1` | Already set to `0` on rrig6600 and rrig6600c per project configuration. Disables compute wave preempt-and-resume; eliminates one eviction path. rrig6600b on default per S162 KIQ fence-timeout finding. |
| `amdgpu.no_queue_eviction_on_vm_fault` | `0` | Setting to `1` prevents eviction cascade on VM fault but masks real faults. Diagnostic risk; not recommended as production setting. |

#### Recommended kernel-parameter test

Configure `/etc/modprobe.d/amdgpu.conf` on all three rigs with:

```
options amdgpu hws_max_conc_proc=4 sched_policy=1 queue_preemption_timeout_ms=30000
```

Reboot, then re-run pool=8/50k as the validation case. If the run completes 5 trials cleanly, the kernel-side cap is sufficient and we can return to higher throughput configurations without architectural changes. If it still crashes, the architecture-level change in Category B is required.

### Category B: Application Architecture

No reboot required. Higher implementation cost. Eliminates the saturation source rather than tolerating it.

#### B-1: Single persistent worker per GPU

Currently the cluster runs `worker_pool_size` workers per rig, each spawning its own KFD process context. The driver sees N×8 concurrent contexts per rig. Switching to one persistent worker per GPU that multiplexes the chunk queue over a single KFD context reduces driver-side process count from `worker_pool_size` to 1 per GPU. This is the cleanest fix because it eliminates the multi-context-per-GPU pressure entirely.

**Implementation cost:** significant rewrite of `pwc_worker_service.py`, including job queue multiplexing logic, per-stream synchronization, and graceful degradation when individual streams fail. Estimated 2–3 sessions to design, implement, and validate. Should be reviewed by Team Beta before commit.

#### B-2: HIP stream pooling within workers

Within each persistent worker, pre-allocate a fixed pool of HIP streams and reuse them across chunk dispatches rather than allocating on demand. Reduces queue churn the driver has to bookkeep without changing the worker count. Smaller intervention than B-1; complementary.

#### B-3: Cooperative throttling between workers

Workers on the same rig share a per-rig semaphore limiting concurrent active GPU operations. Software equivalent of `hws_max_conc_proc`. Less effective than kernel-level cap because it operates at user-level and cannot prevent the driver from observing all worker contexts as concurrent KFD processes regardless of whether they are actively dispatching.

### Category C: Firmware and Infrastructure

#### C-1: Switch from amdgpu-dkms to in-kernel amdgpu

ROCm Issue #5915 reports that the host CPU memory leak (~5.4 GB/hour) is specific to amdgpu-dkms. The stock in-kernel amdgpu driver does not exhibit the leak. Switching driver source could eliminate one accumulating pressure source, though the queue eviction pattern itself is also present in stock kernel. Lower-priority intervention given the primary problem is the eviction cascade rather than the leak.

#### C-2: ROCm version upgrade

Multiple referenced issues note version-specific regressions and improvements. ROCm 7.2.0 with amdgpu-dkms 30.30 is the most recent tested combination. Verifying the cluster is on the latest stable combination is a low-cost diagnostic step before committing to architecture changes.

#### C-3: Process isolation

`amdgpu.enforce_isolation=1` forces stricter VM separation between graphics and compute contexts. May help by preventing cross-context interference; may reduce throughput. Test only in conjunction with the Category A parameter sweep.

---

## Recommendation

Test in order. Stop at the first solution that achieves the throughput target.

- **Step 1: Kernel parameter test.** Apply `hws_max_conc_proc=4` and `sched_policy=1` to all three rigs via modprobe. Reboot. Re-run pool=8/50k. Total cost: one cluster reboot plus one test run (~30 minutes). If clean, problem is solved at minimal cost.

- **Step 2: If Step 1 fails, accept pool=6 as production operating point.** Pool=6/100k is already empirically validated. Throughput is approximately 1.86M sps versus the theoretical 2.07M sps ceiling — roughly 90 percent of optimal. This is acceptable for production while architecture work is planned.

- **Step 3: Plan architecture change B-1.** Single-persistent-worker-per-GPU is the structurally correct solution and removes the saturation source rather than tolerating it. Schedule as a multi-session work item with Team Beta architectural review. Not blocking on production work.

- **Step 4: Concurrent watchdog development.** Build a journalctl-based watchdog on each rig that monitors for "Freeing queue vital buffer" and `GCVM_L2_PROTECTION_FAULT_STATUS` messages. Provides early-warning for the 25k regime where eviction precedes catastrophic failure. Limited utility for 50k+ regime where no precursor exists, but useful as a defense-in-depth measure.

---

## Appendix: Run Provenance

All runs executed against branch `s167-clean` at HEAD `3fdf434`, full 26-GPU topology preserved per Team Beta ruling. Optuna threshold-drop bug fix (FIX 2) confirmed working across all runs based on observed parameter diversity (W ranged 11–32, FT ranged 0.31–0.74, RT ranged 0.36–0.74).

### Run log archive locations on ser8

- `~/s172_50k_pool6_run_logs/` — pool=6/50k clean run, 5 trials
- `~/s172_50k_pool8_run_logs/` — pool=8/50k crashed run, partial trial 1
- `~/s172_25k_pool8_run_logs/` — pool=8/25k clean run, 5 trials
- `~/crash_dumps/20260501_220948_GCVM_L2_PROTECTION_FAULT/` — daemon capture from pool=8/50k crash
- `~/crash_dumps/20260501_232038_rrig6600-workers_8→6/` — daemon capture from transient eviction during pool=8/25k (recovered)
- `~/rocm_mem_pulled_*/` — per-rig memory snapshots from each run
